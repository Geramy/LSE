// Gated DeltaNet as a system algorithm.
//
// lemonseed and Qwen3.6 run the same math; they differ only in the parameters
// below. Verified against both checkpoints' weight shapes:
//
//                      lemonseed b1.5        Qwen3.6-35B-A3B
//   key/value heads    8 / 8                 16 / 32   (1:2)
//   head dim           64 / 64               128 / 128
//   conv               split q,k,v + bias    fused, no bias
//   A_log, dt_bias     [8]  = key heads      [32] = VALUE heads
//   qkv projection     three matrices        one fused [2048+2048+4096]
//   gate               out_gate              in_proj_z
//
// A model kernel therefore supplies a spec and a weight binding, not an
// implementation.
//
// The gate is NOT a spec dimension: in both models it is sigmoid(W x) applied
// to the head-normed output just before out_proj, and its width is forced to
// equal that output (value_heads * value_head_dim). lemonseed's 512 and
// Qwen's 4096 are the same rule, not two rules.
#pragma once

#include <cstdint>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"

namespace lse::ops {

using graph::Array;

enum class ProjLayout : std::uint8_t { kSplit, kFusedQKV };

struct GatedDeltaNetSpec {
  std::int32_t key_heads = 0;
  std::int32_t value_heads = 0;
  std::int32_t key_head_dim = 0;
  std::int32_t value_head_dim = 0;
  bool conv_bias = true;
  // A_log/dt_bias sized by value heads (Qwen) rather than key heads
  // (lemonseed). Only observable when value_heads != key_heads.
  bool decay_per_value_head = false;
  ProjLayout layout = ProjLayout::kSplit;
  float eps = 1e-6f;       // q/k L2 normalization
  float norm_eps = 1e-6f;  // per-head output RMSNorm
  bool zero_centered_norm = true;
};

// Whichever names a checkpoint uses, the model kernel fills these in.
// Unused slots stay invalid: e.g. fused layouts leave the split members empty.
// Conv weights are [channels, kernel]; conv_*_b is ignored when conv_bias is
// false.
struct GatedDeltaNetWeights {
  Array in_proj_q, in_proj_k, in_proj_v;   // kSplit
  Array in_proj_qkv;                       // kFusedQKV
  Array in_proj_a, in_proj_b;
  Array conv_q_w, conv_q_b, conv_k_w, conv_k_b, conv_v_w, conv_v_b;  // kSplit
  Array conv_w, conv_b;                                              // kFusedQKV
  Array a_log, dt_bias, norm;
  Array gate_proj, out_proj;
};

// Carried across decode steps; all empty on the first call.
//
// The conv tail matters as much as the recurrent state: causal_conv1d always
// zero-pads, so a one-token step would convolve against zeros where the
// previous kernel-1 tokens belong and silently diverge from a full pass. Each
// stream keeps its own tail, [B, kernel-1, width]; the fused layout uses
// conv_qkv alone.
struct GatedDeltaNetState {
  Array recurrent;  // [B, value_heads, value_head_dim, value_head_dim]
  Array conv_q, conv_k, conv_v;
  Array conv_qkv;
};

Result<Array> gated_delta_net(const Array& x, const GatedDeltaNetWeights& w,
                              const GatedDeltaNetSpec& spec,
                              GatedDeltaNetState* state);

}  // namespace lse::ops
