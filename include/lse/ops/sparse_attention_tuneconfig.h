#pragma once

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
struct SparseAttentionOptions {
  SparseAttentionPhase prefill;
  SparseAttentionPhase decode;
};
}  // namespace lse::ops
