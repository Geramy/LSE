#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <span>

#include "lse/kv/cache_dtype.hpp"
#include "lse/math/fp8.hpp"

namespace lse::kv {

template <math::MatrixElem Element>
void pack_cache_vector(std::span<const float> input,
                       std::span<std::uint32_t> output) {
  float maximum = 0.0f;
  for (float value : input)
    maximum = std::max(maximum, std::abs(value));
  const float scale =
      maximum > 0.0f && std::isfinite(maximum)
          ? std::max(maximum / math::Fp8Format<Element>::max_finite,
                     std::numeric_limits<float>::min())
          : 1.0f;
  for (std::size_t i = 0; i < input.size() / 4; ++i)
    output[i] = math::pack_fp8<Element>(
        input[i * 4] / scale, input[i * 4 + 1] / scale,
        input[i * 4 + 2] / scale, input[i * 4 + 3] / scale);
  output[input.size() / 4] = std::bit_cast<std::uint32_t>(scale);
}

[[nodiscard]] inline float unpack_cache_element(CacheDType format,
                                                const std::uint32_t *vector,
                                                std::size_t width,
                                                std::size_t column) {
  const auto bits =
      static_cast<std::uint8_t>(vector[column / 4] >> ((column % 4) * 8));
  const float value = format == CacheDType::kFP8
                          ? math::fp8_value<math::MatrixElem::kFp8>(bits)
                          : math::fp8_value<math::MatrixElem::kBf8>(bits);
  return value * std::bit_cast<float>(vector[width / 4]);
}

} // namespace lse::kv
