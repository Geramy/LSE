#pragma once
#include <limits>
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::quant {
inline constexpr float kCenteredMinNormal = std::numeric_limits<float>::min();
inline constexpr float kCenteredMaxBF16 = 3.3895313892515355e38f;
// 64 terms * at most63 centered-code magnitude, plus residual-bias scaling
// and activation rounding. Both unscaled dot and affine products need room.
inline constexpr float kCenteredProductHeadroom =
    std::numeric_limits<float>::max() / 8192.0f;
inline constexpr float kCenteredAccumulatorHeadroom =
    std::numeric_limits<float>::max() / 4.0f;

// Severe cancellation between K64 contributions merits an original ordered
// FP32 recomputation. This is a conservative trigger, not an error certificate.
inline constexpr float kCenteredCancellationRatio = 0.005f;

template<class F> struct CenteredAffine {
  F center, residual_bias;
};
template<class E>
[[nodiscard]] auto centered_affine(E& e, const decltype(e.f32(0))& scale,
                                   const decltype(e.f32(0))& bias) {
  auto center=e.var(0.0f);
  if (auto nonzero=e.when(scale!=0.0f))
    center=math::rint(math::min(e.f32(63.0f),
        math::max(e.f32(0.0f),(e.f32(0.0f)-bias)/scale)));
  const decltype(e.f32(0)) value=center;
  return CenteredAffine<decltype(e.f32(0))>{
      e.let(value),e.let(math::fma(value,scale,bias))};
}
}  // namespace lse::quant
