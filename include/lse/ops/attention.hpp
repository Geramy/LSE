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
#include <span>
#include <vector>

#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/allocator.hpp"
#include "lse/kv/block.hpp"
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
  // Tokens a sequence may reach. 0 keeps the growing-concat path, used only by
  // tests that build a cache by hand.
  std::int32_t kv_length = 0;
};

// g_proj stays invalid under kFusedInQProj.
struct GatedAttentionWeights {
  Array q_proj, k_proj, v_proj, o_proj, g_proj;
  Array q_norm, k_norm;
};

// The paged state of one attention layer: two block pools plus the block lists
// that say which of their blocks hold which positions.
//
// `keys`/`values` are [blocks, kv_heads, kv::kBlockSize, head_dim]. A sequence
// does not own a contiguous span of them — `tables[r]` is row r's ordered block
// list, `alloc` hands blocks out and refcounts them, and the kernels read the
// device image in `table` to turn a position into (block, slot).
//
// Lives in model::MixerState, one per attention layer per sequence set. Every
// attention layer covers the same positions, so the tables agree layer to layer
// even though the pools do not.
struct PagedKvLayer {
  Array keys;
  Array values;
  // [rows, stride] block ids as f32; `stride` is fixed at the engine capacity
  // in blocks so it is a literal in the generated address arithmetic and never
  // varies with how many blocks a sequence currently holds.
  Array table;
  kv::BlockAllocator alloc;
  std::vector<kv::BlockTable> tables;
  // Positions each row will have written once the pending pass lands, one
  // entry per row. Empty means every row reaches the same length, which is what
  // a single sequence needs; a batch of sequences at different lengths sets it,
  // and then the pool is sized to what the rows actually hold rather than to
  // the longest row times the batch.
  std::vector<std::int32_t> row_tokens;
  // Blocks this layer's pool may ever hold. 0 derives it from the engine KV
  // length times the row count, which is the single-sequence answer. A batch
  // driver sets it to the budget it schedules against, and an admission that
  // would pass it is a decision for kv::BlockPolicy rather than an allocation
  // that fails mid-step.
  std::int32_t block_ceiling = 0;
  // `table` has not been re-uploaded since `tables` last changed.
  bool table_dirty = true;

  [[nodiscard]] bool valid() const noexcept {
    return keys.valid() && values.valid() && table.valid();
  }
  [[nodiscard]] std::int32_t stride() const noexcept {
    return table.valid()
               ? static_cast<std::int32_t>(table.shape().dim(table.shape().rank() - 1))
               : 0;
  }
  // Bytes of device pool actually allocated. This is the number paging exists
  // to move: a contiguous cache reserves capacity for every sequence whether it
  // reaches it or not.
  [[nodiscard]] std::size_t pool_bytes() const noexcept;
};

// What an attention call needs to reach its cache. `keys`/`values`/`table` are
// copies of the layer's Arrays (the graph replaces them with the nodes that
// write them, which the caller stores back); the paging bookkeeping is borrowed
// so there is one allocator per layer, not one per call.
//
// With `capacity` == 0 and no `paged` this is the old growing-concat path, kept
// for tests that build a cache by hand.
struct AttentionCache {
  Array keys;
  Array values;
  Array table;
  // Per-step descriptor, kv::step_meta_elems(rows) floats — see kv/block.hpp.
  // One slot for the whole model; see model::HybridLM.
  Array meta;
  PagedKvLayer* paged = nullptr;
  std::int64_t capacity = 0;
  std::int32_t used = 0;
};

// Tops each row's block list up to cover what `layer.row_tokens` asks for —
// or `tokens` positions on every row when it is empty — and re-uploads the
// device table if it changed. Returns true when the pool has to move to a
// bigger rung, which a retained program cannot absorb: the pool buffer changes
// identity, so the caller must rebuild the graph.
//
// Separate from gated_attention because the decode fast path replays a held
// program and never re-records the layer, yet still crosses a block boundary
// every kv::kBlockSize tokens.
Result<bool> extend_paged(PagedKvLayer& layer, std::int32_t tokens);

// Hands row `row`'s blocks back to the pool and empties its list. This is what
// a sequence leaving the batch costs — one table row, not a reallocation — and
// it is also how a preemption frees the blocks the policy asked for.
Status release_row(PagedKvLayer& layer, std::int32_t row);

// The pool rung a set of per-row position counts needs, in blocks. `ceiling` is
// the most the pool may ever hold and is itself a valid rung.
[[nodiscard]] std::int32_t paged_pool_blocks(
    std::span<const std::int32_t> row_tokens, std::int32_t ceiling) noexcept;

// The uniform case: `tokens` positions on each of `rows` rows, with the ceiling
// derived from the engine KV length.
[[nodiscard]] std::int32_t paged_pool_blocks(std::int32_t tokens,
                                             std::int32_t rows,
                                             std::int32_t capacity) noexcept;

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
