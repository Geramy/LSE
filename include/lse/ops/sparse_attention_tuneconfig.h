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
enum class AttentionExecutionPhase { kDecode, kPrefill, kSpeculative };

struct SparseAttentionOptions {
  SparseAttentionPhase prefill;
  SparseAttentionPhase decode;
};
// Singleton prompt tails use exact attention; the FlashPrefill schedule needs
// multiple queries. Draft/verify calls always use exact attention at every width.
inline SparseAttentionPhase attention_for_phase(const SparseAttentionOptions& options,
                                                AttentionExecutionPhase phase,
                                                std::int64_t queries) {
  if (phase == AttentionExecutionPhase::kSpeculative) return {};
  if (phase == AttentionExecutionPhase::kPrefill)
    return queries > 1 ? options.prefill : SparseAttentionPhase{};
  return options.decode;
}
}  // namespace lse::ops
