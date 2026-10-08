// SHA-1, for the git blob id the hub gives a file kept out of LFS:
// sha1("blob <size>\0" + content). Internal to the hub client and its tests.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace lse::hub {

class Sha1 {
 public:
  void update(std::span<const std::byte> bytes) noexcept {
    for (std::byte b : bytes) {
      tail_[used_++] = static_cast<std::uint8_t>(b);
      if (used_ == 64) {
        block();
        used_ = 0;
      }
    }
    bytes_ += bytes.size();
  }
  std::string hex() noexcept {
    const std::uint64_t bits = bytes_ * 8;
    const std::uint8_t pad = 0x80;
    update(std::as_bytes(std::span(&pad, 1)));
    const std::uint8_t zero = 0;
    while (used_ != 56) update(std::as_bytes(std::span(&zero, 1)));
    for (int i = 7; i >= 0; --i) {
      const auto b = static_cast<std::uint8_t>(bits >> (i * 8));
      update(std::as_bytes(std::span(&b, 1)));
    }
    char out[41];
    for (int i = 0; i < 5; ++i) std::snprintf(out + i * 8, 9, "%08x", h_[i]);
    return std::string(out, 40);
  }

 private:
  static std::uint32_t rol(std::uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }
  void block() noexcept {
    std::uint32_t w[80];
    for (std::size_t i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(tail_[i * 4]) << 24) | (std::uint32_t(tail_[i * 4 + 1]) << 16) |
             (std::uint32_t(tail_[i * 4 + 2]) << 8) | std::uint32_t(tail_[i * 4 + 3]);
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; ++i) {
      std::uint32_t f, k;
      if (i < 20) f = (b & c) | (~b & d), k = 0x5a827999u;
      else if (i < 40) f = b ^ c ^ d, k = 0x6ed9eba1u;
      else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8f1bbcdcu;
      else f = b ^ c ^ d, k = 0xca62c1d6u;
      const std::uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d, d = c, c = rol(b, 30), b = a, a = t;
    }
    h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d, h_[4] += e;
  }
  std::uint32_t h_[5]{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};
  std::array<std::uint8_t, 64> tail_{};
  std::size_t used_ = 0;
  std::uint64_t bytes_ = 0;
};


inline std::string git_blob_sha1(std::string_view content) {
  Sha1 h;
  const std::string prefix = "blob " + std::to_string(content.size()) + '\0';
  h.update(std::as_bytes(std::span(prefix.data(), prefix.size())));
  h.update(std::as_bytes(std::span(content.data(), content.size())));
  return h.hex();
}

}  // namespace lse::hub
