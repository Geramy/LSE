#include "lse/core/sha256.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif
namespace lse {
namespace {
constexpr std::uint32_t k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};
std::uint32_t get32(const std::byte* p) {
  return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 8 |
         std::uint32_t(p[3]);
}
}
void Sha256::block(const std::byte* p) noexcept {
  std::uint32_t w[64];
  for (unsigned i = 0; i < 16; ++i) w[i] = get32(p + 4 * i);
  for (unsigned i = 16; i < 64; ++i) {
    auto a = w[i - 15], b = w[i - 2];
    auto s0 = std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3);
    auto s1 = std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  auto a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4], f = state_[5],
       g = state_[6], h = state_[7];
  for (unsigned i = 0; i < 64; ++i) {
    auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    auto t1 = h + s1 + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
    auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    auto t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}
void Sha256::update(std::span<const std::byte> p) noexcept {
  bytes_ += p.size();
  if (used_) {
    auto take = std::min(64 - used_, p.size());
    if (take) std::memcpy(tail_.data() + used_, p.data(), take);
    used_ += take;
    p = p.subspan(take);
    if (used_ == 64) {
      block(tail_.data());
      used_ = 0;
    }
  }
  while (p.size() >= 64) {
    block(p.data());
    p = p.subspan(64);
  }
  if (!p.empty()) {
    std::memcpy(tail_.data(), p.data(), p.size());
    used_ = p.size();
  }
}
std::array<std::byte, 32> Sha256::finish() noexcept {
  const auto bits = bytes_ * 8;
  tail_[used_++] = std::byte{0x80};
  if (used_ > 56) {
    std::fill(tail_.begin() + used_, tail_.end(), std::byte{0});
    block(tail_.data());
    used_ = 0;
  }
  std::fill(tail_.begin() + used_, tail_.begin() + 56, std::byte{0});
  for (unsigned i = 0; i < 8; ++i) tail_[63 - i] = std::byte((bits >> (8 * i)) & 255);
  block(tail_.data());
  std::array<std::byte, 32> out;
  for (unsigned i = 0; i < 8; ++i)
    for (unsigned j = 0; j < 4; ++j) out[4 * i + j] = std::byte((state_[i] >> (24 - 8 * j)) & 255);
  return out;
}
std::string sha256(std::span<const std::byte> bytes) {
  std::array<std::byte, 32> digest;
#if defined(__APPLE__)
  // The portable incremental implementation remains independently testable.
  // Use platform acceleration for multi-gigabyte checkpoint admission.
  CC_SHA256_CTX state;
  CC_SHA256_Init(&state);
  while (!bytes.empty()) {
    const auto count = std::min<std::size_t>(bytes.size(), 1u << 24);
    CC_SHA256_Update(&state, bytes.data(), static_cast<CC_LONG>(count));
    bytes = bytes.subspan(count);
  }
  CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &state);
#else
  Sha256 h;
  h.update(bytes);
  digest = h.finish();
#endif
  std::string out(64, '0');
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < digest.size(); ++i) {
    auto v = std::uint8_t(digest[i]);
    out[2 * i] = hex[v >> 4];
    out[2 * i + 1] = hex[v & 15];
  }
  return out;
}
}
