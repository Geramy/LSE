#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "lse/math.hpp"

namespace lse::math {

// OCP operands used by gfx12, not the incompatible gfx94 FNUZ formats.
// SATFINITE clamps finite overflow; HIP preserves exceptional inputs:
// E4M3 infinity becomes NaN, while E5M2 infinity remains infinity.
// Host NaNs canonicalize to signed 0x7f; device NaN payload/sign is not part
// of the contract. Finite values, signed zero, and E5M2 infinity are exact.
template <MatrixElem Element>
struct Fp8Format {
  static_assert(Element == MatrixElem::kFp8 || Element == MatrixElem::kBf8);
  static constexpr bool e4m3 = Element == MatrixElem::kFp8;
  static constexpr unsigned mantissa_bits = e4m3 ? 3 : 2;
  static constexpr unsigned exponent_bias = e4m3 ? 7 : 15;
  static constexpr unsigned max_finite_code = e4m3 ? 0x7e : 0x7b;
  static constexpr float max_finite = e4m3 ? 448.0f : 57344.0f;
  static constexpr float min_normal = e4m3 ? 0x1p-6f : 0x1p-14f;
  static constexpr float min_subnormal = e4m3 ? 0x1p-9f : 0x1p-16f;
  static constexpr std::string_view pack_key = e4m3 ? "pack4.fp8.ocp" : "pack4.bf8.ocp";
  static constexpr std::string_view value_keys[4] = {
      e4m3 ? "value.fp8.0" : "value.bf8.0",
      e4m3 ? "value.fp8.1" : "value.bf8.1",
      e4m3 ? "value.fp8.2" : "value.bf8.2",
      e4m3 ? "value.fp8.3" : "value.bf8.3"};
};

template <MatrixElem Element>
[[nodiscard]] inline float fp8_value(std::uint8_t bits) {
  using F = Fp8Format<Element>;
  const unsigned magnitude = bits & 0x7f;
  const bool negative = (bits & 0x80) != 0;
  if ((F::e4m3 && magnitude == 0x7f) ||
      (!F::e4m3 && magnitude > 0x7c)) {
    return std::bit_cast<float>((negative ? 0x80000000u : 0u) | 0x7fc00000u);
  }
  if (!F::e4m3 && magnitude == 0x7c)
    return negative ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity();
  const unsigned exponent = magnitude >> F::mantissa_bits;
  const unsigned mantissa = magnitude & ((1u << F::mantissa_bits) - 1);
  const float value = exponent == 0
      ? static_cast<float>(mantissa) * F::min_subnormal
      : std::ldexp(static_cast<float>((1u << F::mantissa_bits) + mantissa),
                   static_cast<int>(exponent) - static_cast<int>(F::exponent_bias) -
                   static_cast<int>(F::mantissa_bits));
  return negative ? -value : value;
}

template <MatrixElem Element>
[[nodiscard]] inline std::uint8_t fp8_bits(float value) {
  using F = Fp8Format<Element>;
  const auto raw = std::bit_cast<std::uint32_t>(value);
  const unsigned sign = raw >> 24 & 0x80;
  const unsigned magnitude = raw & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return static_cast<std::uint8_t>(sign | 0x7f);
  if (magnitude == 0x7f800000u)
    return static_cast<std::uint8_t>(sign | (F::e4m3 ? 0x7f : 0x7c));
  const float positive = std::bit_cast<float>(magnitude);
  if (positive >= F::max_finite)
    return static_cast<std::uint8_t>(sign | F::max_finite_code);
  // Search representable magnitudes, then compare exact dyadic distances in
  // double. This host oracle is independent of GPU bit-shift conversion.
  unsigned lo = 0, hi = F::max_finite_code;
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (fp8_value<Element>(static_cast<std::uint8_t>(mid)) <= positive) lo = mid;
    else hi = mid;
  }
  const double below = double(positive) - double(fp8_value<Element>(static_cast<std::uint8_t>(lo)));
  const double above = double(fp8_value<Element>(static_cast<std::uint8_t>(hi))) - double(positive);
  const unsigned rounded = above < below || (above == below && (lo & 1)) ? hi : lo;
  return static_cast<std::uint8_t>(sign | rounded);
}

template <MatrixElem Element>
[[nodiscard]] inline std::uint32_t pack_fp8(float a, float b, float c, float d) {
  return std::uint32_t(fp8_bits<Element>(a)) |
         (std::uint32_t(fp8_bits<Element>(b)) << 8) |
         (std::uint32_t(fp8_bits<Element>(c)) << 16) |
         (std::uint32_t(fp8_bits<Element>(d)) << 24);
}

template <MatrixElem Element>
[[nodiscard]] inline Val<ir::u32> pack_fp8(const Val<lse::f32>& a,
    const Val<lse::f32>& b, const Val<lse::f32>& c, const Val<lse::f32>& d) {
  return detail::invoke<ir::u32>(Fp8Format<Element>::pack_key, a, b, c, d);
}

template <MatrixElem Element, unsigned Byte>
[[nodiscard]] inline float unpack_fp8(std::uint32_t packed) {
  static_assert(Byte < 4);
  return fp8_value<Element>(static_cast<std::uint8_t>(packed >> (Byte * 8)));
}

template <MatrixElem Element, unsigned Byte>
[[nodiscard]] inline Val<lse::f32> unpack_fp8(const Val<ir::u32>& packed) {
  static_assert(Byte < 4);
  return detail::invoke<lse::f32>(Fp8Format<Element>::value_keys[Byte], packed);
}

}  // namespace lse::math
