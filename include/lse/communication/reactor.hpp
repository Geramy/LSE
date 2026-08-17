// The loop the caller drives.
//
// It owns every socket, ring and queue pair in use and it creates no threads,
// ever: nothing in this module advances except inside poll(). A transport that
// would need a thread of its own to be honest declines at open instead. That is
// not asceticism — it is what makes "your region is mine from post until its
// completion event, and that event arrives inside a call you made, on your
// thread" a statement one call site can check. There is deliberately no "just
// do it in the background" shortcut, because a hidden thread is not asynchrony,
// it is the absence of a stated schedule.
//
// Every member of Reactor, Channel and Listener must be called from the thread
// that created the Reactor. The single exception is Reactor::wake(). That rule
// is what removes every lock and every atomic from the transfer path. Two
// reactors in one process are independent and share nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "lse/communication/buffer.hpp"
#include "lse/communication/capabilities.hpp"
#include "lse/communication/endpoint.hpp"
#include "lse/communication/event.hpp"
#include "lse/core/status.hpp"

namespace lse::comm {

class ReactorState;

class Channel {
 public:
  Channel() = default;

  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  [[nodiscard]] const Capabilities& capabilities() const noexcept;
  [[nodiscard]] const Endpoint& peer() const noexcept;

  // Small and framed. Copied into this channel's outbound ring, so the caller's
  // bytes are free on return and a control message is genuinely
  // fire-and-forget. Refuses with kOutOfMemory when the ring is full, and with
  // kInvalidArgument when the message exceeds the size the PEER advertised in
  // its hello; the caller waits for kWritable rather than growing an unbounded
  // queue.
  Result<Ticket> post_control(std::span<const std::byte> message,
                              std::uint32_t tag = 0);

  // Large and raw. Nothing is copied and nothing is encoded: the header goes
  // out from its own iovec and the payload bytes are never touched by this
  // module. Matched at the peer by `tag`.
  Result<Ticket> post_send(const Transfer& t);
  Result<Ticket> post_recv(const Transfer& t);

  // Answer to a kDataOffered event whose bytes are not wanted: the payload is
  // read and discarded. The peer is NOT told, because on this seam a completed
  // send means the bytes were handed to the link, not that they were wanted — a
  // caller that needs the peer to know sends a control message. The alternative
  // to reject() is abort(); there is no third option, because on a byte stream
  // "not taking these bytes" is either draining them or killing the stream.
  Status reject(std::uint32_t tag);

  // Bytes still postable on this lane before post refuses. Backpressure is a
  // number the caller can read, not an error it has to trip over.
  [[nodiscard]] std::size_t credit(Lane lane) const noexcept;
  // True once this ticket has completed. O(1), no syscall: one slot lookup and
  // a generation compare. False for a ticket whose slot has since been reused.
  [[nodiscard]] bool done(Ticket t) const noexcept;

  // Fires when a Transfer::deadline_ns passes: a transfer that has not started
  // completes kCancelled, and one already on the wire cannot be un-sent, so the
  // channel is aborted instead. Setting a tight deadline on a large transfer is
  // therefore arming a channel kill.
  //
  // Only a ticket whose bytes have not started moving can be cancelled; one
  // that is partly on the wire cannot be un-sent, and pretending otherwise
  // would leave a peer holding half a message. kNotFound for an unknown or
  // already-completed ticket; kOutOfRange for one already in flight, which the
  // caller escalates to abort() if it must.
  Status cancel(Ticket t);

  // Graceful: queued sends finish, then kClosed. Every outstanding ticket still
  // gets exactly one completion event first.
  Status close();
  // Immediate: outstanding tickets complete kCancelled, then kClosed.
  void abort() noexcept;

 private:
  friend class Reactor;
  ReactorState* state_ = nullptr;
  std::uint64_t id_ = 0;
};

class Listener {
 public:
  Listener() = default;
  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  // What was actually bound. Differs from the requested endpoint when the port
  // was 0, which is how a caller binds without racing anyone for a port.
  [[nodiscard]] const Endpoint& endpoint() const noexcept;
  Status close();

 private:
  friend class Reactor;
  ReactorState* state_ = nullptr;
  std::uint64_t id_ = 0;
};

class Reactor {
 public:
  [[nodiscard]] static Result<Reactor> create();
  ~Reactor();
  Reactor(Reactor&&) noexcept;
  Reactor& operator=(Reactor&&) noexcept;
  Reactor(const Reactor&) = delete;
  Reactor& operator=(const Reactor&) = delete;

  // Server mode. Non-blocking: bound and listening on return, and every arrival
  // afterwards is a kAccepted event naming an already-created channel.
  [[nodiscard]] Result<Listener> listen(const Endpoint& ep);
  // Client mode. Non-blocking: the channel exists on return but is not usable
  // until its kConnected event, and a post before then is kOutOfRange.
  [[nodiscard]] Result<Channel> connect(const Endpoint& ep);
  [[nodiscard]] Channel channel(std::uint64_t id) noexcept;

  // The whole engine. Writes at most out.size() events and returns how many.
  // timeout_ns == 0 is a non-blocking sweep: the mode to use between kernel
  // dispatches, and the mode in which a completion queue is drained without
  // arming an interrupt. A non-zero timeout sleeps, capped by the earliest
  // per-operation deadline so a deadline cannot be overslept. Only a poller
  // failure is an error here; every channel failure is an event.
  //
  // A call that returns a partial batch does no I/O: the queued remainder is
  // handed out first, which is what keeps every kControl payload alive for
  // exactly one poll.
  Result<std::size_t> poll(std::span<Event> out, std::uint64_t timeout_ns);

  // Registration belongs to the transport instance, not to the channel: a
  // Region outlives the channel it was registered through and is valid on every
  // channel of the same transport. `on` names which transport.
  Result<Region> register_region(const Channel& on, const RegionRequest& req);
  Status deregister_region(const Channel& on, Region r);

  // For embedding in a loop the caller already owns: wait on these descriptors
  // yourself, then call poll(out, 0). Empty when no live transport is fd-driven;
  // requires_polling() then says the caller must not sleep on fds alone and must
  // bound its wait by next_deadline_ns().
  [[nodiscard]] std::span<const int> wait_fds() const noexcept;
  [[nodiscard]] std::uint64_t next_deadline_ns() const noexcept;  // 0 = none
  [[nodiscard]] bool requires_polling() const noexcept;

  // The ONLY method callable from another thread. Writes an eventfd; a poll
  // blocked in the loop returns with zero events.
  void wake() noexcept;

  [[nodiscard]] const Status& last_error(std::uint64_t channel) const noexcept;
  [[nodiscard]] std::size_t open_channels() const noexcept;
  [[nodiscard]] std::size_t open_listeners() const noexcept;

 private:
  explicit Reactor(std::unique_ptr<ReactorState> state) noexcept;
  std::unique_ptr<ReactorState> state_;
};

}  // namespace lse::comm
