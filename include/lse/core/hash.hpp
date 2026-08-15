// FNV-1a. One definition, because the constants were previously retyped at
// three call sites and a mistyped digit would silently change cache keys.
#pragma once

#include <cstdint>
#include <string_view>

namespace lse {

inline constexpr std::uint64_t kHashSeed = 1469598103934665603ull;
inline constexpr std::uint64_t kHashPrime = 1099511628211ull;

[[nodiscard]] constexpr std::uint64_t hash_mix(std::uint64_t seed,
                                               std::uint64_t value) noexcept {
  return (seed ^ value) * kHashPrime;
}

[[nodiscard]] constexpr std::uint64_t hash_bytes(std::string_view s,
                                                 std::uint64_t seed = kHashSeed) noexcept {
  for (char c : s) seed = hash_mix(seed, static_cast<unsigned char>(c));
  return seed;
}

// Bit pattern, not value: two floats that compare equal but differ in bits
// (0.0 vs -0.0) must not collide in a kernel cache key.
[[nodiscard]] inline std::uint64_t hash_float(std::uint64_t seed, float v) noexcept {
  std::uint32_t bits;
  __builtin_memcpy(&bits, &v, sizeof(bits));
  return hash_mix(seed, bits);
}

}  // namespace lse
