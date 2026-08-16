// The cost model the measured profile feeds.
//
// Two questions, in nanoseconds: what does this work cost on that device, and
// what does moving those bytes between these two cost. Everything above —
// whether an op is worth offloading, how to split one across a heterogeneous
// set, which member `auto` should pick — is those two composed.
//
// The model refuses to answer rather than guess. Every term carries the
// Provenance of the measurement behind it, and a cost with an unknown term is
// unknown, not a default: a placement decided from an invented number is worse
// than no placement, because it looks like it was reasoned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/probe/profile.hpp"

namespace lse::probe {

// What an operation costs to run, in the terms a device profile can price.
struct Work {
  // Bytes the kernel reads out of the device's own memory. For decode this is
  // the weight stream and it is what binds.
  std::size_t bytes_streamed = 0;
  // Multiply-accumulates x 2. Zero for a memory-only op, which then needs no
  // matrix-core rate and can be priced on any device with a known roofline.
  std::uint64_t flops = 0;
  // The operand form the work is held in. NOT a format to convert to: the
  // tensor stays in the checkpoint's own format wherever it runs, and a device
  // without a native row for it dequantizes inside the kernel. This selects
  // which rate applies, nothing else.
  math::MatrixElem operand = math::MatrixElem::kF16;
  // Bytes that have to cross a link to run this somewhere other than home.
  std::size_t bytes_moved = 0;
  std::uint32_t launches = 1;

  [[nodiscard]] Work scaled(double fraction) const noexcept;
};

// 2*M*N*K, and the weight stream that feeds it, in the checkpoint's format.
[[nodiscard]] Work matmul_work(std::size_t m, std::size_t n, std::size_t k,
                               DType storage);

struct Cost {
  double ns = 0.0;
  Provenance provenance = Provenance::kUnknown;

  [[nodiscard]] bool known() const noexcept {
    return provenance == Provenance::kMeasured ||
           provenance == Provenance::kDeclared;
  }
  static Cost of(double ns, Provenance p) noexcept { return {ns, p}; }
  static Cost unknown() noexcept { return {}; }
};

// How many items are in flight. Depth is an INPUT to placement, not something
// layered on afterwards: link latency is on the critical path only when there
// is nothing else to do, so a plan chosen at depth 1 is the wrong plan at
// depth 32 and has to be recomputed when the queue fills.
//
// 1 is the latency-bound limit — one request with nothing behind it — which is
// the case a bare "is this operation worth moving" question silently assumes.
inline constexpr std::uint32_t kDefaultQueueDepth = 1;

// The fraction of an `s`-stage pipeline that is idle with `n` items in flight:
// (s-1)/(s-1+n). Eight stages with eight items still idles 47%; thirty-two
// items brings it under 20%. Exact when the stages are balanced, which is what
// makes it worth stating separately from the makespan below.
[[nodiscard]] double bubble_fraction(std::size_t stages,
                                     std::uint32_t depth) noexcept;

// What a placement delivers in steady state, which is the quantity a placement
// decision is actually about.
struct Throughput {
  double items_per_s = 0.0;
  // One item end to end. Rises when work is spread; that is the trade.
  double item_latency_ns = 0.0;
  double bubble = 0.0;
  Provenance provenance = Provenance::kUnknown;

  [[nodiscard]] bool known() const noexcept {
    return provenance == Provenance::kMeasured ||
           provenance == Provenance::kDeclared;
  }
};

// One step of a pipelined placement: work on a device, then bytes handed to
// the next step. The last stage's bytes go back to the first, because the
// answer has to reach whoever asked for it.
struct Stage {
  DeviceId device;
  Work work;
  std::size_t bytes_to_next = 0;
};

// The margin a change has to beat before it is worth making. Background load
// moves throughput 15-25% on this class of machine and a planner that chases
// noise is worse than a fixed heuristic, so a move must win by more than the
// measurement is worth.
inline constexpr double kThroughputMargin = 0.05;

class CostModel {
 public:
  explicit CostModel(const PoolProfile& pool) noexcept : pool_(&pool) {}

  [[nodiscard]] const PoolProfile& pool() const noexcept { return *pool_; }

  [[nodiscard]] Cost compute_cost(const Work& work,
                                  const DeviceId& device) const noexcept;
  [[nodiscard]] Cost transfer_cost(std::size_t bytes, const DeviceId& src,
                                   const DeviceId& dst) const noexcept;

  // Whether this device runs the operand form natively or through a fallback
  // kernel. Both are legal pool members; only the rate differs.
  [[nodiscard]] bool runs_natively(math::MatrixElem operand,
                                   const DeviceId& device) const noexcept;

  // Steady-state rate of a pipelined placement at `depth` items in flight.
  //
  // Stage i costs its own compute plus the transfer it hands on, and the items
  // overlap: makespan(n) = sum(t) + (n-1)*max(t). At depth 1 that is the sum —
  // every transfer on the critical path, which is the latency-bound answer. As
  // depth rises it approaches 1/max(t), where a link that is not the
  // bottleneck costs nothing at all.
  [[nodiscard]] Throughput pipeline_throughput(std::span<const Stage> stages,
                                               std::uint32_t depth) const;

  struct OffloadDecision {
    bool relocate = false;
    Throughput stay;
    Throughput moved;
    Cost transfer;   // the round trip, for a caller that wants the raw number
    std::string_view reason;
  };

  // Whether running this work on `candidate` instead of `home` raises
  // steady-state throughput at this queue depth.
  //
  // NOT "the operation takes longer than some threshold, so move it". Those
  // two rules disagree on a deep queue and this one is the correct one: with
  // items behind it, the send overlaps the previous item's compute and the
  // link latency leaves the critical path entirely. At depth 1 this reduces to
  // the latency-bound comparison, so that case is the n=1 limit rather than a
  // separate rule.
  [[nodiscard]] OffloadDecision should_offload(
      const Work& work, const DeviceId& home, const DeviceId& candidate,
      std::size_t bytes_out, std::size_t bytes_back,
      std::uint32_t depth = kDefaultQueueDepth) const;

  struct Share {
    DeviceId device;
    double fraction = 0.0;
    // Per-item time this member sustains at the requested depth. Equal across
    // every member that got a share — that is what the split is solving for.
    Cost per_item;
  };

  // How to divide one operation across a heterogeneous set, at a queue depth.
  //
  // Not an equal split: the members have different measured rates for this
  // operand — that is the whole point of a pool that mixes devices — so an
  // equal split runs at the pace of the slowest and leaves the rest idle. The
  // shares are the ones that make every member's steady-state per-item time
  // equal, counting each member's dispatch, its roofline, its matrix rate for
  // this operand, and the link that carries its share.
  //
  // Depth enters through each member's own two stages: at depth 1 its transfer
  // and its compute serialize, and at depth the transfer of the next item
  // overlaps the compute of the current one, so a member behind a slow link
  // recovers most of its share as the queue fills.
  //
  // A member whose cost cannot be priced gets nothing and says so with an
  // unknown per-item time. So does one that cannot repay its own fixed cost at
  // any share.
  [[nodiscard]] std::vector<Share> split(
      const Work& work, std::span<const DeviceId> devices,
      const DeviceId& home, std::uint32_t depth = kDefaultQueueDepth) const;

 private:
  // A member's transfer and compute halves at a given share of the work.
  struct Halves {
    double transfer_ns = 0.0;
    double compute_ns = 0.0;
    Provenance provenance = Provenance::kUnknown;
    [[nodiscard]] bool known() const noexcept {
      return provenance == Provenance::kMeasured ||
             provenance == Provenance::kDeclared;
    }
    // Steady-state per-item time, the two halves overlapped the way the
    // pipeline overlaps its stages.
    [[nodiscard]] double per_item_ns(std::uint32_t depth) const noexcept;
  };
  [[nodiscard]] Halves halves(const Work& work, const DeviceId& device,
                              const DeviceId& home, double fraction) const;

  const PoolProfile* pool_;
};

}  // namespace lse::probe
