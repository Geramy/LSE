#pragma once

#include <cstdint>

namespace lse::ops {
inline constexpr float kFlashPrefillDefaultAlpha = 0.1f;
// BLASST scale is coefficient a in lambda = min(a / live_length, 1).
// FlashPrefill V2 scale is the relative probe-mass threshold alpha in [0, 1].
// Zero exercises the sparse schedule with pruning disabled.
struct SparseAttentionPhase {
  bool blasst = false;
  float scale = 0.0f;
  bool flashprefill = false;
  [[nodiscard]] bool enabled() const { return blasst || flashprefill; }
};
// Query width alone cannot distinguish prompt chunks from speculative verification.
// kTree is a draft tree's verify pass (model::TreeLayout): speculative, and its
// rows are tree nodes rather than consecutive positions.
enum class AttentionExecutionPhase { kDecode, kPrefill, kSpeculative, kTree };

// Block-sparse decode attention (graph::sparse_decode_attention): decode and
// verify passes read the sink pages, the recent window and the far pages
// whose key summaries score highest, up to a token budget.
enum class SparseDecodeMode : std::uint8_t {
  kOff,   // exact attention always (--sparse-decode=off)
  kAuto,  // where the device's measured roofline predicts a gain
  kOn,    // every decode and verify pass with summaries
};
struct SparseDecodeOptions {
  SparseDecodeMode mode = SparseDecodeMode::kOff;
  // Tokens a pass reads: a fixed count, or (budget_share > 0) that share of
  // the pass's KV rung, never below budget_tokens.
  std::int32_t budget_tokens = 0;
  float budget_share = 0.0f;
  // The budget for a pass over a table of `capacity` tokens: whole pages.
  [[nodiscard]] std::int32_t budget_for(std::int64_t capacity) const noexcept {
    if (budget_share <= 0.0f) return budget_tokens;
    const auto share = static_cast<std::int64_t>(static_cast<double>(capacity) * budget_share) / 16 * 16;
    return share > budget_tokens ? static_cast<std::int32_t>(share) : budget_tokens;
  }
  // One page set shared by every key head instead of one per key head.
  bool joint_heads = false;
  // This layer's decode attention stays exact (its summaries are still kept).
  bool exact_layer = false;
  // The engine keeps page summaries for this model (mode is not off, the
  // context can exceed the budget, and the device and KV format qualify).
  [[nodiscard]] bool summaries() const noexcept {
    return mode != SparseDecodeMode::kOff && budget_tokens > 0;
  }
};

struct SparseAttentionOptions {
  SparseAttentionPhase prefill;
  SparseAttentionPhase decode;
  SparseDecodeOptions sparse_decode;
};
// Singleton prompt tails use exact attention; the FlashPrefill schedule needs
// multiple queries. Draft/verify calls always use exact attention at every width.
inline SparseAttentionPhase attention_for_phase(const SparseAttentionOptions& options,
                                                AttentionExecutionPhase phase,
                                                std::int64_t queries) {
  if (phase == AttentionExecutionPhase::kSpeculative ||
      phase == AttentionExecutionPhase::kTree) return {};
  if (phase == AttentionExecutionPhase::kPrefill)
    return queries > 1 ? options.prefill : SparseAttentionPhase{};
  return options.decode;
}
}  // namespace lse::ops
