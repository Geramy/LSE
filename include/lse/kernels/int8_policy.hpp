#pragma once

#include <cstdlib>
#include <cstdint>
#include <string_view>

namespace lse::graph {
struct KernelShapes;
}

namespace lse::kernels {
// Missing override permits only a checkpoint-qualified, shape-scoped policy.
// Explicit 1 preserves the diagnostic opt-in; every other supplied value is exact.
enum class ActivationInt8Policy : std::uint8_t { kAutomatic, kExact, kEnabled };
inline constexpr std::int32_t kQwen27BQ4ComputeRevision = 1;
[[nodiscard]] constexpr ActivationInt8Policy parse_activation_int8_policy(
    const char* value) noexcept {
  return value == nullptr                 ? ActivationInt8Policy::kAutomatic
         : std::string_view(value) == "1" ? ActivationInt8Policy::kEnabled
                                          : ActivationInt8Policy::kExact;
}
[[nodiscard]] constexpr bool parse_activation_int8(const char* value) noexcept {
  return value != nullptr && std::string_view(value) == "1";
}
[[nodiscard]] inline ActivationInt8Policy activation_int8_policy() noexcept {
  static const auto policy = parse_activation_int8_policy(std::getenv("LSE_HRX_INT8"));
  return policy;
}
[[nodiscard]] inline bool activation_int8_enabled() noexcept {
  return activation_int8_policy() == ActivationInt8Policy::kEnabled;
}
[[nodiscard]] bool activation_int8_enabled(const graph::KernelShapes&,
                                           ActivationInt8Policy) noexcept;
[[nodiscard]] inline bool activation_int8_enabled(const graph::KernelShapes& s) noexcept {
  return activation_int8_enabled(s, activation_int8_policy());
}

// The exact Q4 decode override is process-wide for the same reason as the
// activation policy. It must also distinguish persistent kernel identities.
[[nodiscard]] inline bool q4_decode_exact_enabled() noexcept {
  static const bool enabled = parse_activation_int8(std::getenv("LSE_Q4_DECODE_EXACT"));
  return enabled;
}

[[nodiscard]] inline std::uint64_t activation_int8_cache_key(std::uint64_t key) noexcept {
  // These two choices change the generated body without changing its generic
  // primitive name, so both belong in the persistent cache identity.
  key ^= 0x696e7438706f6c33ull;
  key *= 1099511628211ull;
  key ^= static_cast<std::uint64_t>(activation_int8_policy());
  key *= 1099511628211ull;
  key ^= static_cast<std::uint64_t>(q4_decode_exact_enabled());
  return key * 1099511628211ull;
}
}  // namespace lse::kernels
