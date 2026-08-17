// The tracing seam: what the engine records, and what a backend contributes.
//
// TWO TIERS, because the data rates differ by orders of magnitude.
//
//   Tier 1, dispatch timing. Unconditional. The runtime loop consumes it every
//   token, so it cannot route through a sampling profiler and it cannot cost
//   more than a store: record_dispatch() writes one fixed-size record into a
//   preallocated ring with no allocation, no lock and no encoding. The engine
//   supplies the host span itself; a backend contributes the device span
//   through ITrace::resolve_device_span, at batch edges only.
//
//   Tier 2, deep counters. Hardware counters and occupancy, sampled, expensive,
//   and only some backends can do it. This is SELECTION FROM A DECLARED
//   CAPABILITY, not a switch: a backend that cannot measure counters declines
//   and names what is missing, and the optimisation loop proceeds on the
//   timings it does have.
//
// A second backend joins by registering a factory under its name. No caller
// changes — the same shape as probe::register_device_probe and the codec engine
// registry.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/trace/record.hpp"

namespace lse::trace {

enum class Tier : std::uint8_t {
  // Per-dispatch begin/end. Always served: the host half needs no backend at
  // all, and the device half is asked for separately so a backend that has no
  // counter still participates.
  kDispatch,
  // Hardware counters, sampled. Declined by default.
  kDeepCounters,
};

[[nodiscard]] constexpr std::string_view tier_name(Tier tier) noexcept {
  return tier == Tier::kDispatch ? "dispatch" : "deep-counters";
}

// What tier 2 was asked to measure. Counter names are the backend's own
// vocabulary: the engine names what it wants and the adapter answers whether it
// can spell it, rather than the engine encoding one vendor's counter list.
struct DeepRequest {
  std::vector<std::string> counters;
  // Dispatches between samples. 0 means every dispatch, which for hardware
  // counters is normally unaffordable and is why this is here at all.
  std::uint32_t sample_period = 0;
};

class IDeepSession {
 public:
  virtual ~IDeepSession() = default;
  // Counters sampled since the last drain. Empty is a valid answer: sampling
  // may not have landed on a dispatch yet.
  virtual Result<std::vector<CounterSet>> drain() = 0;
};

class ITrace {
 public:
  virtual ~ITrace() = default;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  // The clock every device tick this tracer produces is counted on. Knowable
  // even when no tick is readable — HRX publishes rate, domain and width and
  // still cannot read the counter — and worth having on its own: it is what
  // lets an export declare the clock and what lets a subtraction refuse.
  [[nodiscard]] virtual Result<backend::DeviceClock> device_clock() const = 0;

  // A host tick and a device tick read back to back, which is the only thing
  // that can place an absolute device tick on the host timeline. Declines when
  // the backend cannot read a tick.
  [[nodiscard]] virtual Result<ClockAnchor> clock_anchor() const = 0;

  // Fill `record.device` with the ticks that bracketed this dispatch on the
  // device, or decline naming what is missing. Called at a batch edge, after
  // the work has retired — never per dispatch: a per-packet completion signal
  // costs +14.76%, while reading hardware-written stamps off a signal that was
  // already going to fire costs 25.7 ns.
  //
  // Declining is the correct behaviour for a backend whose runtime stamps only
  // the host clock. It must not substitute one: a host number here becomes a
  // kernel duration in a cost model.
  virtual Status resolve_device_span(DispatchRecord& record) = 0;

  // Ok when this tracer serves `tier`; otherwise a decline that names what is
  // missing, which is the message a caller should print rather than "profiling
  // unavailable".
  [[nodiscard]] virtual Status supports(Tier tier) const = 0;

  // Open a deep-counter session. Non-ok is a decline, not a failure.
  virtual Result<std::unique_ptr<IDeepSession>> open_deep(
      const DeepRequest& request) = 0;
};

using TraceFactory = std::unique_ptr<ITrace> (*)(backend::IBackend&);

void register_tracer(std::string_view backend_name, TraceFactory factory);

// The tracer registered for this backend, or the portable one. Never null.
[[nodiscard]] std::unique_ptr<ITrace> create_tracer(backend::IBackend& backend);

struct TraceRegistrar {
  TraceRegistrar(std::string_view backend_name, TraceFactory factory) {
    register_tracer(backend_name, factory);
  }
};

// What any backend can answer from the IBackend seam alone: the device clock if
// it publishes one, an anchor if it can also read a tick, and a decline for
// everything else. A backend gets this until it registers something better.
[[nodiscard]] std::unique_ptr<ITrace> make_portable_tracer(
    backend::IBackend& backend);

// Tier 1 storage.
//
// One lane per thread, each a fixed-capacity ring for dispatches and a bounded
// vector for host spans. Fixed capacity is the point: the hot path is a masked
// index and a struct store, so it neither allocates nor grows, and the cost of
// tracing does not depend on how long the process has been running. When the
// ring wraps, the oldest records are lost and counted — a timeline wants the
// most recent window, and a dropped count is what stops a truncated trace from
// reading as a complete one.
class Collector {
 public:
  static Collector& instance() noexcept;

  // How much history to hold. A runtime value (like a cache directory), not a
  // switch: tracing is on either way. Rounded up to a power of two. Takes
  // effect for lanes created afterwards; a lane already holding records keeps
  // its ring so a resize cannot lose a run's history mid-run.
  struct Capacity {
    std::size_t dispatches = 1u << 14;  // ~50 tokens of the lemonseed decode
    std::size_t spans = 1u << 12;
    std::size_t counters = 1u << 12;
  };
  void reserve(Capacity capacity) noexcept;
  [[nodiscard]] Capacity capacity() const noexcept;

  // Written once per emitted kernel. Later records for a group already present
  // replace it only when they carry more than it did (a source hash the first
  // one could not reach), so the table cannot lose information it once had.
  void record_group(GroupRecord group);

  void record_counters(CounterSet counters);

  // A consistent-enough copy for export. Allocates, walks every lane, and takes
  // the lane lock, so it is called at a quiescent point — after a synchronize,
  // between tokens — never on the dispatch path.
  //
  // It does not stop the writers: a lane being written during a snapshot may
  // contribute the record currently in flight or not. That is acceptable for a
  // timeline and is stated here so nobody builds an exactly-once accounting on
  // top of it.
  [[nodiscard]] TraceData snapshot() const;

  void clear() noexcept;

  // Ask `tracer` for the device span of every dispatch that does not have one.
  // Returns the tracer's decline on the first record it cannot answer, so the
  // caller prints one honest reason rather than N. `resolved` counts what was
  // filled in before that.
  Status resolve_device_spans(ITrace& tracer, std::size_t* resolved = nullptr);

  // Total dispatches ever recorded, including those the ring has overwritten.
  [[nodiscard]] std::uint64_t dispatches_recorded() const noexcept;

 private:
  Collector() = default;
};

// THE HOT PATH. One relaxed increment for the correlation id, one masked index,
// one fixed-size store. `record.sequence` is assigned here; whatever the caller
// left in it is overwritten.
void record_dispatch(const DispatchRecord& record) noexcept;

// A host span whose ends the caller already has (a loop that reads the clock
// once and uses it twice, which is every existing timing site in the engine).
void record_span(SpanRecord span);

// A scoped host span. Reads steady_clock at both ends — that is the measurement,
// so it is not free and it is not on the dispatch path: token, phase and wait
// spans number in the tens per run.
class Span {
 public:
  Span(SpanKind kind, std::string name) noexcept;
  Span(SpanKind kind, std::string name, GroupId group) noexcept;
  ~Span();

  Span(const Span&) = delete;
  Span& operator=(const Span&) = delete;
  Span(Span&&) = delete;
  Span& operator=(Span&&) = delete;
};

}  // namespace lse::trace
