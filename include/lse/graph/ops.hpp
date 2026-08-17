// Op builders. Each records a node and returns immediately.
#pragma once

#include <array>
#include <string_view>
#include <vector>

#include "lse/graph/graph.hpp"

namespace lse::graph {

Array add(const Array& a, const Array& b);
Array sub(const Array& a, const Array& b);
Array mul(const Array& a, const Array& b);
Array div(const Array& a, const Array& b);
Array eq(const Array& a, const Array& b);
Array ge(const Array& a, const Array& b);

Array neg(const Array& x);
Array exp(const Array& x);
Array log(const Array& x);
Array sqrt(const Array& x);
Array rsqrt(const Array& x);

Array silu(const Array& x);
Array gelu(const Array& x);
Array sigmoid(const Array& x);
Array tanh_(const Array& x);
Array relu(const Array& x);

Array cast(const Array& x, DType to);
Array clamp(const Array& x, float lo, float hi);

// axis < 0 counts from the end. keepdims retains the reduced axis as size 1.
Array sum(const Array& x, int axis = -1, bool keepdims = false);
Array max(const Array& x, int axis = -1, bool keepdims = false);
Array mean(const Array& x, int axis = -1, bool keepdims = false);
Array softmax(const Array& x, int axis = -1);

Array reshape(const Array& x, Shape shape);
Array transpose(const Array& x, std::vector<int> perm);
Array concat(const std::vector<Array>& parts, int axis);
Array slice(const Array& x, int axis, std::int64_t begin, std::int64_t end);
// Repeats each element along `axis` n times (interleaved), which is what GQA
// head expansion needs.
Array repeat(const Array& x, int n, int axis);

Array matmul(const Array& a, const Array& b);
// x[..., in] @ w[out, in]^T. Weights are stored [out, in], so this avoids
// transposing them on every call. A weight carrying QuantPlanes routes to
// quant_linear instead, so a model kernel says `linear` once and the
// checkpoint decides which contraction runs.
Array linear(const Array& x, const Array& w);
// Logical [out, in] of a weight in whichever format it is stored. The packed
// plane of a group-affine weight counts lanes on its last axis, not weights,
// so a shape check against the config has to ask for this rather than shape().
[[nodiscard]] Shape weight_shape(const Array& w);
// x[..., K] @ W[idx, :, :]^T with W stacked [E, N, K]. `slot` picks a column
// of `idx` [..., k], so MoE runs the k winners without slicing weights.
Array linear_indexed(const Array& x, const Array& w, const Array& idx,
                     int slot = 0);
// The same contraction against a group-affine weight the kernel unpacks in
// register: `packed` is the U32 plane [out, in*bits/32], `scales` and `biases`
// are [out, in/group_size] in the checkpoint's own narrow float. The weight is
// never widened in memory — that is the whole point of the op existing rather
// than a dequantize pass feeding `linear`.
Array quant_linear(const Array& x, const Array& packed, const Array& scales,
                   const Array& biases, int bits, int group_size);
// Row gather out of a group-affine table, dequantized in register. A tied LM
// head makes this unavoidable: the same 4-bit table is read as a matrix by
// lm_head and as rows by embed, and widening it for the gather would widen it
// for both.
Array quant_embedding(const Array& packed, const Array& scales,
                      const Array& biases, const Array& ids, int bits,
                      int group_size);
Array embedding(const Array& table, const Array& ids);

// Row gather/scatter over the last-axis-major layout every op here assumes.
// `rows` holds row indices into x's flattened [N, width] view.
//   gather_rows : x [.., width], rows [n]          -> [n, width]
//   scatter_add_rows : base [.., width] + values [n, width] at rows[i]
// These are what makes a data-dependent dispatch (MoE routing) a handful of
// batched graph ops instead of a host loop over tokens.
Array gather_rows(const Array& x, const Array& rows);
Array scatter_add_rows(const Array& base, const Array& rows,
                       const Array& values);

// Writes `src` into `dst` along `axis` starting at the scalar `begin`.
// The output is dst's shape; the primitive aliases dst's buffer so only
// the written window is touched.
Array overwrite_slice(const Array& dst, const Array& src, int axis,
                      const Array& begin);

// Descending top-k along `axis` (default last). Values are [.., k]; when
// `indices` is non-null it receives the same shape, as f32 positions, matching
// gather_rows. Ties keep the smaller index. score_band < 1 zeros values
// below (1-band)*top and renormalizes the rest (MoE routing).
Array topk(const Array& x, int k, int axis = -1, Array* indices = nullptr,
           float score_band = 1.0f);

// Index of the row maximum over the last axis, as f32 (the engine's index
// convention — see topk). Ties take the smallest index, matching the host
// sampler's argmax exactly. Two nodes: a per-chunk partial reduce to
// [.., nchunks, 2] (value, index) and a final combine to [..] (or [1] for a
// rank-1 input), so greedy decode reads back one float instead of the row.
Array argmax(const Array& x);

// Interleaved-pair RoPE over [B, H, T, D]; cos/sin are [max_T, D].
Array rope(const Array& x, const Array& cos, const Array& sin, int offset);
// `offset` is a 1-element tensor so decode can poke it without a new kernel.
Array rope(const Array& x, const Array& cos, const Array& sin,
           const Array& offset);

// x / sqrt(sum(x^2) + eps) over the last axis.
Array l2_normalize(const Array& x, float eps = 1e-12f);

// log(1 + exp(x)), numerically guarded for large |x|.
Array softplus(const Array& x);

// Causal depthwise conv over the time axis. x [B,T,C], weight [C,K], bias [C].
// Left-pads with K-1 zeros (or `tail`, the previous K-1 inputs) so output[t]
// depends only on inputs <= t.
Array causal_conv1d(const Array& x, const Array& weight, const Array& bias);
// Streaming form: window positions before x's start read `tail` [B,K-1,C]
// instead of the zero pad. Same values, same accumulation order.
Array causal_conv1d(const Array& x, const Array& weight, const Array& bias,
                    const Array& tail);

// The advanced conv window: the last tail-many columns of tail ++ x, without
// materializing the concatenation. Output shape == tail's.
Array conv_tail(const Array& tail, const Array& x);

// Gated delta rule, run per token:
//   S = S*alpha; S += ((v - S k) * beta) k^T; o = S q
// q/k/v are [B,T,H,D], alpha/beta are [B,T,H]. Returns o [B,T,H,D]; `state_out`
// receives the carried S [B,H,D,D] when non-null.
Array gated_delta_step(const Array& q, const Array& k, const Array& v,
                       const Array& alpha, const Array& beta,
                       const Array& state_in, Array* state_out);

enum class MaskKind : std::uint8_t { kNone, kCausal, kSlidingWindow };

// q [B, Hq, T, Dh], k/v [B, Hkv, S, Dh]. GQA is handled internally.
Array sdpa(const Array& q, const Array& k, const Array& v, float scale,
           MaskKind mask, int window = 0, int offset = 0);
Array sdpa(const Array& q, const Array& k, const Array& v, float scale,
           MaskKind mask, int window, const Array& offset);

// Paged form. `k`/`v` are pools [blocks, Hkv, block_size, Dh] rather than one
// contiguous span per sequence, and `table` [rows, stride] says which block
// holds each of a row's positions. `meta` is the per-step descriptor, f32 [3]:
// {first query position, live KV length, real rows}.
//
// The live length is a dispatch value, not a shape: one code object serves
// every sequence length. `rows` likewise — the batch axis is padded to a bucket
// and the rows past `meta[2]` return zero, so a pass is the same kernel however
// many sequences are in it.
Array sdpa_paged(const Array& q, const Array& k, const Array& v, float scale,
                 MaskKind mask, int window, const Array& meta,
                 const Array& table, int block_size);

// Writes `src` [rows, Hkv, T, Dh] into the pool `dst` [blocks, Hkv,
// block_size, Dh] at absolute position meta[0], following `table`. Returns the
// pool: the write aliases it and touches only the positions it covers.
Array kv_page_write(const Array& dst, const Array& src, const Array& meta,
                    const Array& table, int block_size);

// Records a node for a registered primitive. Returns an invalid Array if the
// name is unknown or the arity/shapes do not match.
Result<Array> custom(std::string_view primitive, const std::vector<Array>& inputs,
                     std::array<float, 4> attrs = {});

// x * rsqrt(mean(x^2) + eps) * scale.
// zero_centered selects scale = 1 + weight, which is lemonseed's convention:
// weight is initialized at 0 so the effective scale starts at 1.
Array rms_norm(const Array& x, const Array& weight, float eps,
               bool zero_centered = false);

inline Array operator+(const Array& a, const Array& b) { return add(a, b); }
inline Array operator-(const Array& a, const Array& b) { return sub(a, b); }
inline Array operator*(const Array& a, const Array& b) { return mul(a, b); }
inline Array operator/(const Array& a, const Array& b) { return div(a, b); }

}  // namespace lse::graph
