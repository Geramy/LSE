// The wire, and the only place a byte layout is spelled.
//
// Little-endian, written as-is. The sender states its byte order in the hello
// and a peer that disagrees is refused by name rather than producing plausible
// garbage. This layout is a contract BETWEEN BUILDS — bumping kWireVersion
// breaks every peer on an older engine, which is a deliberate act, not a
// refactor.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lse::comm {

inline constexpr std::uint32_t kFrameMagic = 0x4553'4C43u;  // 'C','L','S','E'
inline constexpr std::uint16_t kWireVersion = 1;

// bit 0 of Hello::flags
inline constexpr std::uint8_t kHelloLittleEndian = 0x01;
// bit 0 of FrameHeader::flags: the last frame of a segmented transfer.
inline constexpr std::uint8_t kFrameLast = 0x01;

// First thing on every lane socket. The 128-bit channel id pairs a lane with
// its sibling at the acceptor, which is what lets one logical channel be two
// physically separate streams.
struct Hello {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint8_t lane;
  std::uint8_t flags;
  std::uint64_t channel_hi;
  std::uint64_t channel_lo;
  std::uint32_t control_ring_bytes;  // what THIS side can receive
  std::uint32_t reserved;
};
static_assert(sizeof(Hello) == 32);
static_assert(std::is_trivially_copyable_v<Hello>);

// `bytes` is this frame's payload; `total_bytes` is the whole transfer's, so a
// kDataOffered event can name a size the receiver can actually allocate for
// even when the link segments. Control frames set total_bytes == bytes.
struct FrameHeader {
  std::uint32_t magic;
  std::uint8_t lane;
  std::uint8_t flags;
  std::uint16_t reserved;
  std::uint32_t tag;
  std::uint32_t bytes;
  std::uint64_t total_bytes;
};
static_assert(sizeof(FrameHeader) == 24);
static_assert(std::is_trivially_copyable_v<FrameHeader>);

inline constexpr std::size_t kHelloBytes = sizeof(Hello);
inline constexpr std::size_t kFrameHeaderBytes = sizeof(FrameHeader);

// The largest control message this build will accept from a peer regardless of
// what that peer advertises: a hostile or confused hello must not make us
// allocate a gigabyte.
inline constexpr std::uint32_t kMaxControlRingBytes = 64u << 20;
inline constexpr std::uint32_t kDefaultControlRingBytes = 64u << 10;

inline constexpr bool host_is_little_endian() noexcept {
  return std::endian::native == std::endian::little;
}

}  // namespace lse::comm
