// The readiness seam: one wait for every descriptor the reactor owns.
//
// Src-only on purpose. Nothing in the public API mentions readiness, because a
// completion queue has none — poller_epoll.cpp serves the stream transports and
// poller_uring.cpp would replace that file alone, with no public change. The
// poller never blocks longer than it is told and never runs a callback outside
// wait().
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::comm {

enum class Interest : std::uint8_t {
  kNone = 0,
  kRead = 1,
  kWrite = 2,
  kReadWrite = 3,
};

[[nodiscard]] constexpr Interest operator|(Interest a, Interest b) noexcept {
  return static_cast<Interest>(static_cast<std::uint8_t>(a) |
                               static_cast<std::uint8_t>(b));
}
[[nodiscard]] constexpr bool wants_read(Interest i) noexcept {
  return (static_cast<std::uint8_t>(i) & 1u) != 0;
}
[[nodiscard]] constexpr bool wants_write(Interest i) noexcept {
  return (static_cast<std::uint8_t>(i) & 2u) != 0;
}

struct Readiness {
  bool readable = false;
  bool writable = false;
  // The peer hung up or the descriptor errored. Delivered alongside readable so
  // a final short read still lands before the teardown.
  bool hangup = false;
};

class IReady {
 public:
  virtual ~IReady() = default;
  virtual void on_ready(int fd, Readiness r) = 0;
};

class Poller {
 public:
  [[nodiscard]] static Result<Poller> create();
  ~Poller();
  Poller(Poller&&) noexcept;
  Poller& operator=(Poller&&) noexcept;
  Poller(const Poller&) = delete;
  Poller& operator=(const Poller&) = delete;

  Status add(int fd, Interest interest, IReady* owner);
  Status modify(int fd, Interest interest);
  void remove(int fd) noexcept;

  // Waits at most timeout_ns, then dispatches on_ready for every ready
  // descriptor. A callback that removes another descriptor is safe: ownership
  // is re-looked-up per event, so a removed fd is skipped rather than called.
  Status wait(std::uint64_t timeout_ns);

  [[nodiscard]] int fd() const noexcept { return epoll_fd_; }
  // Callable from any thread; every other member is owner-thread only.
  void wake() noexcept;

 private:
  Poller() = default;

  int epoll_fd_ = -1;
  int wake_fd_ = -1;
  std::map<int, IReady*> owners_;
  std::vector<unsigned char> scratch_;
};

}  // namespace lse::comm
