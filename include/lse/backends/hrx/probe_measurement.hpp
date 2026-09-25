#pragma once
#include <algorithm>
#include <cstddef>
#include <span>
#include <optional>
#include "lse/core/status.hpp"

namespace lse::backend::hrx_kernels {
// Bound the working set by an observed free budget when available, otherwise
// by a conservative fraction of known device capacity. Neither is a claim
// about the last-level cache; the reported rate is measured streaming traffic.
inline std::size_t streaming_probe_bytes(std::size_t total, std::optional<std::size_t> free) {
  constexpr std::size_t max_bytes = 512u << 20;
  const std::size_t budget = free.has_value() ? *free / 4 : total / 16;
  const std::size_t bytes = std::min(max_bytes, budget);
  return bytes >= (8u << 20) ? bytes : 0;
}
inline Status check_probe_output(std::span<const float> values,
                                 std::size_t payload, float expected) {
  if (payload == 0 || values.size() <= payload) {
    return LSE_ERROR(kInvalidArgument, "probe output has no payload or guard");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float want = i < payload ? expected : -1234.0f;
    if (values[i] != want) {
      return LSE_ERROR(kDeviceError, "probe output mismatch at element ",
                       std::to_string(i));
    }
  }
  return OkStatus();
}
}  // namespace lse::backend::hrx_kernels
