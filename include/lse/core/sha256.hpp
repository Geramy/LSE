#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
namespace lse {
class Sha256 {
 public:
  void update(std::span<const std::byte> bytes) noexcept;
  [[nodiscard]] std::array<std::byte, 32> finish() noexcept;

 private:
  void block(const std::byte* bytes) noexcept;
  std::array<std::uint32_t, 8> state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<std::byte, 64> tail_{};
  std::uint64_t bytes_ = 0;
  std::size_t used_ = 0;
};
[[nodiscard]] std::string sha256(std::span<const std::byte> bytes);
[[nodiscard]] inline std::string sha256(std::string_view text) {
  return sha256(std::as_bytes(std::span(text.data(), text.size())));
}
}
