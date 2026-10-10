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
  std::int32_t budget_tokens = 0;
  // One page set shared by every key head instead of one per key head.
  bool joint_heads = false;
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
