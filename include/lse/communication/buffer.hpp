// Bytes, and the name of an operation that moves them.
//
// Nothing here encodes anything: the data plane is a pointer, an offset and a
// length, because any encoding on that path is a copy in the bandwidth-critical
// route.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lse::comm {

// A range a transport may DMA. `id` is transport-private; 0 means "no
// registration was needed", which is a legitimate answer and not a failure.
// A transport that does register must tag the instance in the id's high 16
// bits, so that posting a region on a foreign channel is kInvalidArgument
// rather than corruption.
struct Region {
  std::uint64_t id = 0;
  void* host = nullptr;  // null when the bytes are not host-addressable
  std::size_t bytes = 0;

  [[nodiscard]] bool host_addressable() const noexcept {
    return host != nullptr;
  }
};

// An unregistered host range. Valid to post on any transport whose
// Capabilities::registers_memory is false.
[[nodiscard]] inline Region host_region(void* p, std::size_t bytes) noexcept {
  return Region{0, p, bytes};
}

// How memory crosses this seam. A dmabuf fd is how device memory arrives:
// naming backend::DeviceBuffer here would drag the backend registry below this
// module and put GPUDirect registration underneath dist. The fd belongs to the
// caller and must outlive every Region derived from it.
struct RegionRequest {
  void* host = nullptr;
  int dmabuf_fd = -1;  // -1 => host memory
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

// Two independent streams inside one channel. Control never queues behind a
// bulk transfer, which is the whole reason the lane exists and not a flag.
enum class Lane : std::uint8_t { kControl, kData };
enum class Direction : std::uint8_t { kSend, kRecv };

// The name of one posted operation: a channel, and a slot in that channel's
// operation table. Both halves pack an index and a generation, so a handle to a
// completed operation or a closed channel compares unequal to whatever now
// occupies the slot — it is kNotFound, never somebody else's transfer. The
// generation wraps only after four billion reuses of one slot.
struct Ticket {
  std::uint64_t channel = 0;
  std::uint64_t op = 0;

  [[nodiscard]] bool valid() const noexcept { return op != 0; }
};

// One posted transfer. On Lane::kData the region is caller-owned and must not
// move, be freed, or be written until the ticket's completion event arrives —
// including when the channel dies under it. On Lane::kControl the bytes are
// copied at post time and the caller's buffer is free the moment post returns.
struct Transfer {
  Region region{};
  std::size_t offset = 0;
  std::size_t bytes = 0;
  std::uint32_t tag = 0;
  // Absolute steady-clock nanoseconds. Per operation, unlike
  // dist::TransportConfig::timeout_ms which is per connection.
  //
  // 0 means no deadline on THIS transfer, not a default one: arming every post
  // with the channel's `deadline_ms` would put one timer per post on the
  // per-token path for transfers that are almost never late. The channel's own
  // `deadline_ms` still bounds its handshake and its graceful close, and
  // `offer_ms` still bounds an unanswered offer. A transfer that must not hang
  // says so here.
  std::uint64_t deadline_ns = 0;
};

[[nodiscard]] std::uint64_t steady_now_ns() noexcept;

}  // namespace lse::comm
