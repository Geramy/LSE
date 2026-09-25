#include "lse/communication/poller.hpp"
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/event.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>

namespace lse::comm {
namespace {
constexpr uintptr_t kWake = UINTPTR_MAX;
Status arm(int queue, int fd, Interest interest) {
  struct kevent changes[2];
  EV_SET(&changes[0], static_cast<uintptr_t>(fd), EVFILT_READ,
         EV_ADD | (wants_read(interest) ? EV_ENABLE : EV_DISABLE), 0, 0,
         nullptr);
  EV_SET(&changes[1], static_cast<uintptr_t>(fd), EVFILT_WRITE,
         EV_ADD | (wants_write(interest) ? EV_ENABLE : EV_DISABLE), 0, 0,
         nullptr);
  if (::kevent(queue, changes, 2, nullptr, 0, nullptr) < 0)
    return LSE_ERROR(kIoError, "kevent(arm): ", std::strerror(errno));
  return OkStatus();
}
} // namespace
Result<Poller> Poller::create() {
  Poller p;
  p.epoll_fd_ = ::kqueue();
  if (p.epoll_fd_ < 0)
    return LSE_ERROR(kIoError, "kqueue: ", std::strerror(errno));
  if (::fcntl(p.epoll_fd_, F_SETFD, FD_CLOEXEC) < 0)
    return LSE_ERROR(kIoError, "fcntl(kqueue): ", std::strerror(errno));
  struct kevent wake;
  EV_SET(&wake, kWake, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
  if (::kevent(p.epoll_fd_, &wake, 1, nullptr, 0, nullptr) < 0)
    return LSE_ERROR(kIoError, "kevent(wake): ", std::strerror(errno));
  return p;
}
Poller::~Poller() {
  if (epoll_fd_ >= 0)
    ::close(epoll_fd_);
}
Poller::Poller(Poller &&other) noexcept
    : epoll_fd_(std::exchange(other.epoll_fd_, -1)),
      owners_(std::move(other.owners_)), scratch_(std::move(other.scratch_)) {}
Poller &Poller::operator=(Poller &&other) noexcept {
  if (this != &other) {
    if (epoll_fd_ >= 0)
      ::close(epoll_fd_);
    epoll_fd_ = std::exchange(other.epoll_fd_, -1);
    owners_ = std::move(other.owners_);
    scratch_ = std::move(other.scratch_);
  }
  return *this;
}
Status Poller::add(int fd, Interest interest, IReady *owner) {
  if (fd < 0 || !owner)
    return LSE_ERROR(kInvalidArgument, "invalid poller descriptor/owner");
  if (owners_.contains(fd))
    return LSE_ERROR(kAlreadyExists, "descriptor already registered");
  auto status = arm(epoll_fd_, fd, interest);
  if (!status.ok()) {
    remove(fd);
    return status;
  }
  owners_[fd] = owner;
  return OkStatus();
}
Status Poller::modify(int fd, Interest interest) {
  if (!owners_.contains(fd))
    return LSE_ERROR(kNotFound, "descriptor not registered");
  return arm(epoll_fd_, fd, interest);
}
void Poller::remove(int fd) noexcept {
  struct kevent change;
  for (auto filter : {EVFILT_READ, EVFILT_WRITE}) {
    EV_SET(&change, static_cast<uintptr_t>(fd), filter, EV_DELETE, 0, 0,
           nullptr);
    ::kevent(epoll_fd_, &change, 1, nullptr, 0, nullptr);
  }
  owners_.erase(fd);
}
Status Poller::wait(std::uint64_t timeout_ns) {
  std::array<struct kevent, 128> events{};
  const timespec timeout{static_cast<time_t>(timeout_ns / 1'000'000'000ull),
                         static_cast<long>(timeout_ns % 1'000'000'000ull)};
  const int count = ::kevent(epoll_fd_, nullptr, 0, events.data(),
                             static_cast<int>(events.size()), &timeout);
  if (count < 0) {
    if (errno == EINTR)
      return OkStatus();
    return LSE_ERROR(kIoError, "kevent(wait): ", std::strerror(errno));
  }
  for (int i = 0; i < count; ++i) {
    const auto &event = events[static_cast<std::size_t>(i)];
    if (event.filter == EVFILT_USER && event.ident == kWake)
      continue;
    const int fd = static_cast<int>(event.ident);
    const auto found = owners_.find(fd);
    if (found == owners_.end())
      continue;
    Readiness ready;
    ready.readable = event.filter == EVFILT_READ;
    ready.writable = event.filter == EVFILT_WRITE;
    ready.hangup = (event.flags & (EV_EOF | EV_ERROR)) != 0;
    found->second->on_ready(fd, ready);
  }
  return OkStatus();
}
void Poller::wake() noexcept {
  if (epoll_fd_ < 0)
    return;
  struct kevent event;
  EV_SET(&event, kWake, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
  ::kevent(epoll_fd_, &event, 1, nullptr, 0, nullptr);
}
} // namespace lse::comm
