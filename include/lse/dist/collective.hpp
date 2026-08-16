// The caller-facing collective, and the selection that decides which one runs.
//
// A caller says `all_reduce`. Which algorithm and which wire codec that turns
// into comes from the transport's declared Capabilities and the cost model —
// never from a flag, never from a branch at the call site. A new algorithm
// (an RCCL native all-reduce, a tree, a one-shot) is a new CollectiveAlgo row
// plus a case in `run`, with the caller untouched.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lse/dist/codec.hpp"
#include "lse/dist/transport.hpp"

namespace lse::dist {

enum class CollectiveAlgo : std::uint8_t {
  kNative,   // the transport has its own; delegate
  kRing,     // synthesized over p2p, uncompressed
  kTwoShot,  // segment-owner reduce + all-gather, compressed wire
};

[[nodiscard]] std::string_view to_string(CollectiveAlgo algo) noexcept;

struct CollectivePlan {
  CollectiveAlgo algo = CollectiveAlgo::kRing;
  DType wire = DType::kF16;
  // Modelled wall time in nanoseconds, and why this row won. Carried so
  // `--stats` and the tests can see the decision rather than infer it.
  double predicted_ns = 0.0;
  std::string_view reason;
};

// Everything the selector is allowed to look at. Split out from the transport
// so the model is testable without one.
struct CollectiveCost {
  Capabilities caps;
  Rank world_size = 1;
  std::size_t elems = 0;
  ReduceOp op = ReduceOp::kSum;
};

// `engine` may be null, which means "no codec is available"; the plan is then
// necessarily uncompressed.
[[nodiscard]] CollectivePlan select_all_reduce(
    const CollectiveCost& cost, const CodecEngine* engine) noexcept;

struct CollectiveContext {
  ITransport* transport = nullptr;
  CodecEngine* codec = nullptr;
};

[[nodiscard]] Result<CollectivePlan> plan_all_reduce(
    const CollectiveContext& ctx, const CommBuffer& buf, ReduceOp op);

Status all_reduce(const CollectiveContext& ctx, CommBuffer& buf, ReduceOp op);

// Runs a plan the caller already has. Exists so a test can pin an algorithm
// without the selector; production code calls `all_reduce`.
Status run_all_reduce(const CollectiveContext& ctx, const CollectivePlan& plan,
                      CommBuffer& buf, ReduceOp op);

}  // namespace lse::dist
