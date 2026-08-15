// Mixture-of-Experts routing and expert dispatch.
#pragma once

#include <cstdint>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"

namespace lse::ops {

using graph::Array;

// One expert of a stacked [E, out, in] weight as a plain [out, in] matrix.
// Both lemonseed (w_gate/w_up/w_down) and Qwen3.6 (switch_mlp.*_proj) store
// routed experts stacked this way.
Array expert_slice(const Array& stacked, std::int64_t expert);

// The three matrices of one SwiGLU expert. lemonseed names them w1/w3/w2,
// Qwen3.6 gate_proj/up_proj/down_proj.
struct ExpertWeights {
  Array gate;
  Array up;
  Array down;
};

struct RouteConfig {
  std::int32_t num_experts = 0;
  std::int32_t num_active = 0;
  // Among the top-k, drop experts whose probability falls below
  // (1 - score_band) * top. 1.0 disables the band, i.e. plain top-k, which is
  // what Qwen3.6 uses; lemonseed uses 0.15.
  float score_band = 1.0f;
};

struct ExpertChoice {
  std::int32_t expert = 0;
  float weight = 0.0f;
};

// Per-row top-k over already-softmaxed probabilities, with the optional band
// and renormalization over the kept experts. probs is [.., num_experts].
Result<std::vector<std::vector<ExpertChoice>>> route_topk(
    const Array& probs, const RouteConfig& cfg);

// Runs each row of x through the experts route_topk picked for it and sums the
// weighted results back into a tensor shaped like x. `stacked` holds the routed
// weights as [E, out, in]; routing must have one entry per row of x, in the
// order the weights should accumulate.
Result<Array> dispatch_combine(
    const Array& x, const std::vector<std::vector<ExpertChoice>>& routing,
    const ExpertWeights& stacked);

// router -> softmax -> topk (banded) -> k indexed SwiGLUs. Stays on the
// graph. `router_bias` is the DeepSeek-style aux-loss-free balancing bias;
// leave it invalid to skip it, which is what both reference implementations
// do by default.
Result<Array> routed_experts(const Array& x, const Array& router,
                             const Array& router_bias,
                             const ExpertWeights& stacked,
                             const RouteConfig& cfg);

// Sum of dense SwiGLU experts run on every token. Invalid Array when there are
// none.
Array shared_experts(const Array& x, const std::vector<ExpertWeights>& experts);

// Mixture-of-Depths inference gate: a per-token sigmoid scaling of `value`.
// `bias` may be invalid. Inference only — the training path ranks across the
// whole sequence and is not causal.
Array depth_gate(const Array& x, const Array& weight, const Array& bias,
                 const Array& value);

}  // namespace lse::ops
