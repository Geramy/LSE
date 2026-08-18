// What the engine records about its own work, as plain structs.
//
// `probe` measures the hardware once, at admission; `trace` measures the work
// continuously. Two different questions, so two different modules: a
// DeviceProfile answers "what can this part do", a DispatchRecord answers "what
// did this dispatch cost just now". Nothing here encodes anything — encoding is
// export.hpp's job, and a record that had to be serialized to reach a consumer
// in the same process would put a codec on the hot path for no benefit.
//
// Two disciplines carry over from probe/profile.hpp and are not negotiable:
//
//   - A duration nobody could read is UNKNOWN and refuses. It is never filled
//     with a number from a different clock, because at the point of use a host
//     number and a device number are indistinguishable and a cost model would
//     act on the wrong one.
//   - Every timestamp says which clock it came from. Host and device clocks
//     differ in rate and origin, and across machines they are unrelated until
//     correlated.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/core/status.hpp"

namespace lse::trace {

// The identity a dispatch is attributed to.
//
// CARDINALITY, and it is the whole reason this is a struct and not a node
// pointer: one dispatch is one *fusion group* of many nodes. Sibling fusion and
// the lds_fold pass merge nodes into one kernel by design, and a phase group is
// the entire decode graph in a single launch. So the mapping is
//
//     dispatch -> group -> node SET
//
// and never dispatch -> node. A 1:1 field here would be wrong on the first
// fused pair and, in a published append-only schema, permanently wrong.
//
// Two ids, because the engine has two group identities and conflating them
// loses a join:
//   - cache_key is IKernelEmitter::cache_key: the compilation identity, which
//     varies with the device (it mixes wavefront size and the specialized
//     kernel choice). This is the primary group id.
//   - signature is FusionGroup::signature: the shape identity, and the value
//     the kernel's *name* is built from ("lse_fused_<signature>"). It is the
//     join key back to a vendor profiler's kernel-name column, so it travels
//     even though cache_key is what the JIT keys on.
struct GroupId {
  std::uint64_t cache_key = 0;
  std::uint64_t signature = 0;

  [[nodiscard]] bool valid() const noexcept { return cache_key != 0; }
  friend bool operator==(const GroupId&, const GroupId&) = default;
};

// One node of a group, identified the only way a Node can be.
//
// Node carries no id and never has: a shared_ptr address is not stable across
// runs and there is no counter to borrow. So identity here is synthesized —
// position within FusionGroup::nodes, which is stable for a given emit — and
// the kind/shape/primitive text is what makes it recognizable to a human
// reading a timeline. Recorded as a decision, not overlooked: an analysis that
// needs to reach the live Node re-emits the group from its signature.
struct NodeRef {
  std::uint32_t index = 0;
  std::string kind;       // to_string(OpKind)
  std::string shape;      // Shape::to_string()
  std::string primitive;  // Primitive::name, empty for a built-in kind
};

// The group table: written once per emitted kernel, never per dispatch.
//
// This is where the node set lives, so the per-dispatch record stays fixed-size
// and name-free. Cardinality is the number of distinct kernels a process
// compiles (60 for the lemonseed decode graph), against ~330 dispatches per
// token, which is the reason the split exists.
struct GroupRecord {
  GroupId id{};
  std::string entry_name;   // the symbol the device actually ran
  std::string arch;         // what it was compiled for
  std::vector<NodeRef> nodes;
  // FNV of the generated source. Absent, not zero, when the caller could not
  // reach it: the JIT computes it inside get_or_compile and the memory-hit path
  // never sees it, so a record from that path must say "unknown" rather than
  // claim 0 — 0 is also what an empty source hashes to.
  std::optional<std::uint64_t> source_hash;
  std::uint32_t launches = 1;
  bool is_phase = false;
};

// The total of N disjoint host spans, carrying the clock they were read on.
//
// Not a SpanRecord: that is one interval with two ends, and a total over many
// intervals has neither. The clock travels with the number because a bare
// nanosecond count is precisely what let the scheduler's blocking-wait total be
// read as device occupancy for months — at the point of use, a host number and
// a device number are indistinguishable.
struct HostDuration {
  std::uint64_t ns = 0;
  backend::ClockDomain clock = backend::ClockDomain::kHostSteady;

  constexpr void add(std::uint64_t elapsed) noexcept { ns += elapsed; }

  // Same clock only. Adding a steady total to a system-clock total is the same
  // category error as subtracting across two device clocks, so it refuses
  // rather than producing a number nobody can attribute.
  Status add(const HostDuration& other) {
    if (other.clock != clock) {
      return LSE_ERROR(kInvalidArgument, "cannot add a ",
                       std::string(backend::clock_domain_name(other.clock)),
                       " duration to a ",
                       std::string(backend::clock_domain_name(clock)),
                       " total: unrelated clocks");
    }
    ns += other.ns;
    return OkStatus();
  }

  [[nodiscard]] constexpr double seconds() const noexcept {
    return static_cast<double>(ns) * 1e-9;
  }
};

// Both ticks of a device span come from one clock.
//
// A span whose ends were read from different clocks is not a span, so the clock
// is stored once instead of per tick and the two cannot disagree. `clock` is
// filled only by a tracer that actually read both ticks off the device; while a
// backend declines the timestamp capability it stays kUnknown and every
// duration derived from it refuses. That refusal is the correct state of this
// engine today, not a failure to be papered over.
struct DeviceSpan {
  std::uint64_t begin_tick = 0;
  std::uint64_t end_tick = 0;
  backend::DeviceClock clock{};

  [[nodiscard]] bool known() const noexcept { return clock.known(); }

  [[nodiscard]] Result<double> duration_ns() const {
    if (!known()) {
      return LSE_ERROR(kUnimplemented,
                       "no device span on this dispatch: the backend published "
                       "no readable device timestamp");
    }
    return backend::nanoseconds_between(backend::DeviceTimestamp{begin_tick, clock},
                                        backend::DeviceTimestamp{end_tick, clock});
  }
};

// One dispatch: what the host spent handing it over, and what the device spent
// running it, kept apart.
//
// The host span is the submission — recording the packet into the command
// buffer — and on a batched submission path that is nearly all it is. It is
// never reported as the kernel's duration; that is what `device` is for, and
// `device` is unknown until a backend can read its own counter.
//
// MEASURED, so nothing here re-derives it: profiling enabled with no per-packet
// signal costs -0.06% (free), a per-packet interrupt signal costs +14.76%, and
// reading hardware-written start/end stamps off a completion signal costs
// 25.7 ns. Device timestamps are therefore free at batch edges and expensive
// per dispatch, which is why resolve_device_span() is a batch-edge call.
struct DispatchRecord {
  GroupId group{};
  // Monotonic per process. This is the value a tier-2 adapter pushes as its
  // external correlation id, so a vendor profiler's kernel record joins to this
  // row without matching kernel names.
  std::uint64_t sequence = 0;

  std::uint64_t host_begin_ns = 0;
  std::uint64_t host_end_ns = 0;
  // Which clock host_begin_ns/host_end_ns are counted on. Stated in the record
  // rather than known by convention, because the record outlives the process
  // that wrote it.
  backend::ClockDomain host_clock = backend::ClockDomain::kHostSteady;

  DeviceSpan device{};

  // Which part of the dispatch's work items this launch covered. count == 0 is
  // the whole dispatch, which is every launch today.
  backend::WorkRange window{};

  std::uint32_t stream = 0;
  std::uint32_t node_count = 0;
  std::uint32_t bindings = 0;
  std::uint32_t workgroup_count = 0;
  std::uint32_t workgroup_size = 0;
  std::uint32_t elements = 0;
  // Which recording lane this came from — one per thread, assigned by the
  // collector, not by the caller. Host spans from two threads interleave and
  // cannot share one timeline track, so the exporter needs this to keep each
  // thread's nesting its own.
  std::uint32_t thread = 0;
  std::uint8_t device_ordinal = 0;
  bool is_phase = false;

  [[nodiscard]] std::uint64_t host_duration_ns() const noexcept {
    return host_end_ns >= host_begin_ns ? host_end_ns - host_begin_ns : 0;
  }
};

// The nesting levels a timeline needs. Append-only: a viewer built against this
// list must open a trace written by a build that has more of them.
#define LSE_SPAN_KIND_LIST(X)                                              \
  X(kToken, "token")         /* one generated token */                     \
  X(kPhase, "phase")         /* prefill or decode, one staged graph */     \
  X(kGroup, "group")         /* one fusion group, when it is not a launch */ \
  X(kHostWait, "host-wait")  /* a blocking synchronize: submission tail,   \
                                queue drain, execution and interrupt wake, \
                                all in one number and never a device one */ \
  X(kCompile, "compile")     /* emit + JIT for one kernel */               \
  X(kPartition, "partition") /* graph partitioning */

LSE_DECLARE_ENUM(SpanKind, std::uint8_t, LSE_SPAN_KIND_LIST)

// A host span. Low cardinality by construction — one per token, per phase, per
// wait — so it carries its name rather than an index into a table.
struct SpanRecord {
  SpanKind kind = SpanKind::kToken;
  std::string name;
  std::uint64_t begin_ns = 0;
  std::uint64_t end_ns = 0;
  backend::ClockDomain clock = backend::ClockDomain::kHostSteady;
  // Nesting depth at the point the span opened. The exporter reconstructs
  // nesting from (begin, end) anyway; this is carried so a consumer can check
  // that what it was handed really was properly nested.
  std::uint32_t depth = 0;
  // The recording lane, as on DispatchRecord: one per thread, assigned by the
  // collector.
  std::uint32_t thread = 0;
  // Set on a span that belongs to one group, which is what links it to the
  // group table. Zero otherwise.
  GroupId group{};

  [[nodiscard]] std::uint64_t duration_ns() const noexcept {
    return end_ns >= begin_ns ? end_ns - begin_ns : 0;
  }
};

// Counters the engine measures about itself.
//
// Append-only, and deliberately short: a counter with no producer is
// speculation, so this list holds exactly the one PLAN.md requires and names
// the reason it cannot come from anywhere else.
#define LSE_COUNTER_KIND_LIST(X)                                              \
  /* Which expert each routed token actually went to. No clock can see this:   \
     the distribution is a property of the tokens, not of the hardware, and    \
     an external profiler sees identical kernels whatever the routing did.     \
     One atomic per token, not per thread, so it costs nothing. */             \
  X(kRouterExperts, "router experts")

LSE_DECLARE_ENUM(CounterKind, std::uint8_t, LSE_COUNTER_KIND_LIST)

// A distribution, not a scalar derived from one.
//
// CARDINALITY again: the per-bin counts are the datum a future partitioner
// needs, and a skew figure computed at write time throws the distribution away
// and cannot be inverted. So the record carries the histogram and the reader
// computes whatever figure it wants — see routing_skew().
struct CounterSet {
  CounterKind kind = CounterKind::kRouterExperts;
  // Which instance this belongs to: the layer index for a router. Two routers
  // in one model are two CounterSets, never summed.
  std::uint32_t instance = 0;
  std::vector<std::uint64_t> bins;
  // The denominator the bins were drawn from — routed tokens for a router.
  // Recorded rather than summed from the bins: top-k routing puts one token in
  // k bins, so the sum is k*total and only the writer knows k.
  std::uint64_t total = 0;
  std::uint64_t host_ns = 0;
  backend::ClockDomain clock = backend::ClockDomain::kHostSteady;
};

// Peak bin over mean bin: 1.0 is perfectly balanced, N is one hot expert out of
// N. Refuses on an empty or all-zero histogram rather than returning a figure
// no observation supports.
[[nodiscard]] Result<double> routing_skew(const CounterSet& counts);

// CLOCK_MONOTONIC and CLOCK_BOOTTIME read back to back.
//
// Both are host clocks and they are NOT interchangeable: they differ by every
// suspend the machine has had. steady_clock is MONOTONIC, and a timeline format
// that defaults to BOOTTIME will silently misplace MONOTONIC timestamps by that
// difference unless the offset travels with them. So it is measured — never
// computed from an assumption that the two agree, which they do on a box that
// has never slept and nowhere else.
struct HostClockPair {
  std::uint64_t monotonic_ns = 0;
  std::uint64_t boottime_ns = 0;

  [[nodiscard]] bool known() const noexcept {
    return monotonic_ns != 0 && boottime_ns != 0;
  }
};

// Reads both clocks now. Cheap (two vDSO calls) and the only honest source of
// the offset.
[[nodiscard]] HostClockPair read_host_clocks() noexcept;

// Everything a snapshot of the collector holds, and everything the exporter
// consumes. Owning and allocating, so it is built at a quiescent point and
// never on the dispatch path.
struct TraceData {
  std::vector<SpanRecord> spans;
  std::vector<DispatchRecord> dispatches;
  std::vector<GroupRecord> groups;
  std::vector<CounterSet> counters;
  // Taken when this snapshot was, which is as close to the records as anything
  // can be. An exporter without it cannot place a MONOTONIC timestamp on a
  // BOOTTIME timeline.
  HostClockPair host_clocks;
  // Records the fixed-capacity ring overwrote before anyone read them. A
  // non-zero value means the timeline starts later than the run did; reporting
  // it is what keeps a truncated trace from reading as a complete one.
  std::uint64_t dispatches_dropped = 0;
  std::uint64_t spans_dropped = 0;

  [[nodiscard]] bool empty() const noexcept {
    return spans.empty() && dispatches.empty() && groups.empty() &&
           counters.empty();
  }
};

// The group a dispatch was attributed to, or nullptr when the table does not
// hold it. The exporter names a dispatch's slice from this, which is the
// correlation doing its job.
[[nodiscard]] const GroupRecord* find_group(const TraceData& data, GroupId id);

// A host tick and a device tick read back to back, which is the only way an
// absolute device tick can be placed on the host timeline. `uncertainty_ns` is
// the width of the host interval the device read happened inside — measured,
// not assumed, and the reason this is a struct rather than two numbers.
struct ClockAnchor {
  std::uint64_t host_ns = 0;
  backend::DeviceTimestamp device{};
  std::uint64_t uncertainty_ns = 0;

  [[nodiscard]] bool known() const noexcept { return device.valid(); }
};

}  // namespace lse::trace
