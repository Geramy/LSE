#pragma once

#include <cstdlib>
#include <cstdint>
#include <string_view>

namespace lse::kernels {
// Activation conversion is an accuracy policy, not an instruction capability.
// Only the explicit value "1" enables it; missing or malformed values keep
// floating-point activation math. Genuine integer operations are unaffected.
[[nodiscard]] constexpr bool parse_activation_int8(const char* value) noexcept {
  return value != nullptr && std::string_view(value) == "1";
}

// Latch before the first kernel selection and keep replayed plans consistent.
// Set LSE_HRX_INT8 before starting the process; changing its environment after
// emission cannot safely reinterpret already-compiled or retained programs.
[[nodiscard]] inline bool activation_int8_enabled() noexcept {
  static const bool enabled = parse_activation_int8(std::getenv("LSE_HRX_INT8"));
  return enabled;
}

[[nodiscard]] inline std::uint64_t activation_int8_cache_key(
    std::uint64_t key) noexcept {
  // Include the policy version and value even when the generic primitive name
  // does not change (the dot/float bodies share that primitive).
  key ^= 0x696e7438706f6c31ull;
  key *= 1099511628211ull;
  key ^= static_cast<std::uint64_t>(activation_int8_enabled());
  return key * 1099511628211ull;
}
}  // namespace lse::kernels
