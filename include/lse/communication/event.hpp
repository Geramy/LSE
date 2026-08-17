// What a poll reports. One event per thing that happened, never a callback.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "lse/communication/buffer.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/core/status.hpp"

namespace lse::comm {

#define LSE_COMM_EVENT_LIST(X) \
  X(kAccepted, "accepted")     \
  X(kConnected, "connected")   \
  X(kControl, "control")       \
  X(kDataOffered, "data-offered") \
  X(kSendComplete, "send-complete") \
  X(kRecvComplete, "recv-complete") \
  X(kWritable, "writable")     \
  X(kClosed, "closed")

LSE_DECLARE_ENUM(EventKind, std::uint8_t, LSE_COMM_EVENT_LIST)

// Trivially copyable on purpose: poll() writes these into a caller-owned span
// and allocates nothing. An error carries only its code here; the sentence that
// explains it is Reactor::last_error(channel), fetched on the rare path rather
// than built on every event.
struct Event {
  EventKind kind = EventKind::kClosed;
  StatusCode code = StatusCode::kOk;
  Lane lane = Lane::kControl;
  Direction dir = Direction::kSend;
  std::uint32_t tag = 0;
  std::uint64_t channel = 0;
  std::uint64_t listener = 0;  // kAccepted and a listener's kClosed only
  std::uint64_t op = 0;
  std::size_t bytes = 0;  // offered, or actually transferred
  // kControl only. Points into a reactor-owned arena and is INVALID once the
  // poll() that returned this event is called again. Copy it or consume it
  // before then. With lse::debug() on, released arena bytes are overwritten
  // with 0xDD so a stale read is obviously wrong instead of subtly wrong.
  const std::byte* data = nullptr;

  [[nodiscard]] Ticket ticket() const noexcept { return Ticket{channel, op}; }
};

static_assert(std::is_trivially_copyable_v<Event>);
// Pinned so adding a field is a deliberate act, not a drift.
static_assert(sizeof(Event) == 48);

}  // namespace lse::comm
