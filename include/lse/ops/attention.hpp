// Gated GQA attention as a system algorithm, plus the shape plumbing every
// attention variant shares.
//
// lemonseed and Qwen3.6 differ only in where the output gate comes from and in
// how much of the head dim RoPE rotates:
//
//                  lemonseed b1.5          Qwen3.6-35B-A3B
//   gate           separate g_proj         second half of a 2x-wide q_proj
//   head dim       64                      256
//   RoPE           full, theta 1e4..5e5    0.25 partial, theta 1e7
#pragma once

#include <cstdint>

#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/rope.hpp"

namespace lse::ops {

using graph::Array;

// [B, T, H*D] -> [B, H, T, D]
Array split_heads(const Array& x, std::int64_t heads, std::int64_t head_dim);

// [B, H, T, D] -> [B, T, H*D]
Array merge_heads(const Array& x);

enum class GateSource : std::uint8_t {
  // Dedicated weight, applied from the layer input: sigmoid(g_proj x).
  kSeparateProj,
  // q_proj is [2 * q_heads * head_dim, hidden]; the upper half is the gate.
  kFusedInQProj,
};

struct GatedAttentionSpec {
  std::int32_t q_heads = 0;
  std::int32_t kv_heads = 0;
  std::int32_t head_dim = 0;
  GateSource gate = GateSource::kSeparateProj;
  graph::MaskKind mask = graph::MaskKind::kCausal;
  std::int32_t window = 0;
  float norm_eps = 1e-6f;
  bool zero_centered_norm = true;
  // Tokens the KV tensors are allocated for. 0 keeps the old growing-concat
  // path, used only by tests that build a cache by hand.
  std::int32_t kv_length = 0;
};

// g_proj stays invalid under kFusedInQProj.
struct GatedAttentionWeights {
  Array q_proj, k_proj, v_proj, o_proj, g_proj;
  Array q_norm, k_norm;
};

// Keys and values for the positions already seen, [B, kv_heads, T, head_dim].
// When `capacity` > 0 the tensors are allocated at that T once and new columns
// overwrite at `used`; otherwise decode concats and T grows every step.
struct AttentionCache {
  Array keys;
  Array values;
  Array pos;
  std::int64_t capacity = 0;
  std::int32_t used = 0;

  [[nodiscard]] std::int64_t length() const noexcept {
    if (capacity > 0) return used;
    return keys.valid() ? keys.shape().dim(2) : 0;
  }
};

// `rope` may rotate fewer channels than head_dim; apply_rope passes the rest
// through, which is what a partial_rotary_factor < 1 needs.
//
// `offset` is the absolute position of x's first token, so RoPE and the causal
// mask agree with what the cache already holds. Pass a null cache to run the
// whole sequence statelessly.
Result<Array> gated_attention(const Array& x, const GatedAttentionWeights& w,
                              const GatedAttentionSpec& spec,
                              const RopeTables& rope, std::int32_t offset,
                              AttentionCache* cache = nullptr);

}  // namespace lse::ops
