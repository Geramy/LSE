#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <utility>
#include <vector>

#include "poller.hpp"

namespace lse::comm {

namespace {

constexpr std::size_t kMaxEventsPerWait = 128;

// Deliberately no EPOLLRDHUP: it stays asserted once the peer has gone, so a
// level-triggered wait on a lane that has already read its EOF would spin. A
// closed peer makes the descriptor readable and recv() returns 0, which is the
// same news arriving through the door that can be shut.
std::uint32_t epoll_mask(Interest i) noexcept {
  std::uint32_t mask = 0;
  if (wants_read(i)) mask |= EPOLLIN;
  if (wants_write(i)) mask |= EPOLLOUT;
  return mask;
}

}  // namespace

Result<Poller> Poller::create() {
  Poller p;
  p.epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
  if (p.epoll_fd_ < 0) {
    return LSE_ERROR(kIoError, "epoll_create1: ", std::strerror(errno));
  }
  p.wake_fd_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (p.wake_fd_ < 0) {
    const std::string why = std::strerror(errno);
    ::close(p.epoll_fd_);
    p.epoll_fd_ = -1;
    return LSE_ERROR(kIoError, "eventfd: ", why);
  }
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.fd = p.wake_fd_;
  if (::epoll_ctl(p.epoll_fd_, EPOLL_CTL_ADD, p.wake_fd_, &ev) != 0) {
    const std::string why = std::strerror(errno);
    ::close(p.wake_fd_);
    ::close(p.epoll_fd_);
    p.wake_fd_ = -1;
    p.epoll_fd_ = -1;
    return LSE_ERROR(kIoError, "epoll_ctl(wake): ", why);
  }
  return p;
}

Poller::~Poller() {
  if (wake_fd_ >= 0) ::close(wake_fd_);
  if (epoll_fd_ >= 0) ::close(epoll_fd_);
}

Poller::Poller(Poller&& other) noexcept
    : epoll_fd_(std::exchange(other.epoll_fd_, -1)),
      wake_fd_(std::exchange(other.wake_fd_, -1)),
      owners_(std::move(other.owners_)),
      scratch_(std::move(other.scratch_)) {}

Poller& Poller::operator=(Poller&& other) noexcept {
  if (this != &other) {
    if (wake_fd_ >= 0) ::close(wake_fd_);
    if (epoll_fd_ >= 0) ::close(epoll_fd_);
    epoll_fd_ = std::exchange(other.epoll_fd_, -1);
    wake_fd_ = std::exchange(other.wake_fd_, -1);
    owners_ = std::move(other.owners_);
    scratch_ = std::move(other.scratch_);
  }
  return *this;
}

Status Poller::add(int fd, Interest interest, IReady* owner) {
  epoll_event ev{};
  ev.events = epoll_mask(interest);
  ev.data.fd = fd;
  if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) != 0) {
    return LSE_ERROR(kIoError, "epoll_ctl(add ", std::to_string(fd),
                     "): ", std::strerror(errno));
  }
  owners_[fd] = owner;
  return OkStatus();
}

Status Poller::modify(int fd, Interest interest) {
  epoll_event ev{};
  ev.events = epoll_mask(interest);
  ev.data.fd = fd;
  if (::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) != 0) {
    return LSE_ERROR(kIoError, "epoll_ctl(mod ", std::to_string(fd),
                     "): ", std::strerror(errno));
  }
  return OkStatus();
}

void Poller::remove(int fd) noexcept {
  ::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
  owners_.erase(fd);
}

Status Poller::wait(std::uint64_t timeout_ns) {
  std::array<epoll_event, kMaxEventsPerWait> events{};
  const timespec ts{static_cast<std::time_t>(timeout_ns / 1'000'000'000ull),
                    static_cast<long>(timeout_ns % 1'000'000'000ull)};
  // epoll_pwait2 takes nanoseconds; epoll_wait's millisecond argument would
  // round a sub-millisecond deadline up to a whole one, which is most of a
  // decode phase.
  const int n = ::epoll_pwait2(epoll_fd_, events.data(),
                               static_cast<int>(events.size()), &ts, nullptr);
  if (n < 0) {
    if (errno == EINTR) return OkStatus();
    return LSE_ERROR(kIoError, "epoll_pwait2: ", std::strerror(errno));
  }

  for (int i = 0; i < n; ++i) {
    const int fd = events[static_cast<std::size_t>(i)].data.fd;
    const std::uint32_t mask = events[static_cast<std::size_t>(i)].events;
    if (fd == wake_fd_) {
      std::uint64_t drained = 0;
      while (::read(wake_fd_, &drained, sizeof(drained)) == sizeof(drained)) {
      }
      continue;
    }
    // Re-looked-up per event: an earlier callback may have torn this fd down.
    const auto it = owners_.find(fd);
    if (it == owners_.end()) continue;
    Readiness r;
    r.readable = (mask & (EPOLLIN | EPOLLPRI)) != 0;
    r.writable = (mask & EPOLLOUT) != 0;
    r.hangup = (mask & (EPOLLHUP | EPOLLERR)) != 0;
    it->second->on_ready(fd, r);
  }
  return OkStatus();
}

void Poller::wake() noexcept {
  if (wake_fd_ < 0) return;
  const std::uint64_t one = 1;
  const ssize_t written = ::write(wake_fd_, &one, sizeof(one));
  (void)written;
}

}  // namespace lse::comm
