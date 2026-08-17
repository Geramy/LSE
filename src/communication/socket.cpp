// tcp:// and unix:// — two address families over one stream link.
//
// Everything that makes a byte stream behave like a channel lives in
// stream_link.cpp. What is left here is genuinely family-specific: how a name
// becomes a sockaddr, how a connection is started without blocking, and how the
// acceptor pairs the two sockets of one channel by the id in their hellos.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "frame.hpp"
#include "lse/communication/adapter.hpp"
#include "lse/communication/transport.hpp"
#include "poller.hpp"
#include "stream_link.hpp"

namespace lse::comm {

namespace {

constexpr std::array<std::string_view, 0> kNoExtraOptions{};

struct Address {
  sockaddr_storage storage{};
  socklen_t len = 0;
  int family = AF_UNSPEC;
};

bool is_unix(const Endpoint& ep) noexcept { return ep.scheme() == "unix"; }

Status split_authority(std::string_view authority, std::string& host,
                       std::string& port) {
  if (!authority.empty() && authority.front() == '[') {
    const std::size_t end = authority.find(']');
    if (end == std::string_view::npos) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(authority),
                       "' opens '[' and never closes");
    }
    host = std::string(authority.substr(1, end - 1));
    std::string_view rest = authority.substr(end + 1);
    if (rest.empty() || rest.front() != ':') {
      return LSE_ERROR(kInvalidArgument, "'", std::string(authority),
                       "' has no port");
    }
    port = std::string(rest.substr(1));
    return OkStatus();
  }
  const std::size_t colon = authority.rfind(':');
  if (colon == std::string_view::npos) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(authority),
                     "' has no port; an endpoint names host:port");
  }
  host = std::string(authority.substr(0, colon));
  port = std::string(authority.substr(colon + 1));
  return OkStatus();
}

Result<Address> inet_address(std::string_view authority) {
  std::string host;
  std::string port;
  LSE_RETURN_IF_ERROR(split_authority(authority, host, port));
  if (host.empty()) host = "0.0.0.0";

  std::uint32_t port_value = 0;
  for (const char c : port) {
    if (c < '0' || c > '9') {
      return LSE_ERROR(kInvalidArgument, "port '", port, "' is not a number");
    }
    port_value = port_value * 10u + static_cast<std::uint32_t>(c - '0');
    if (port_value > 65535u) {
      return LSE_ERROR(kOutOfRange, "port '", port, "' is above 65535");
    }
  }

  Address out;
  sockaddr_in v4{};
  if (::inet_pton(AF_INET, host.c_str(), &v4.sin_addr) == 1) {
    v4.sin_family = AF_INET;
    v4.sin_port = ::htons(static_cast<std::uint16_t>(port_value));
    std::memcpy(&out.storage, &v4, sizeof(v4));
    out.len = sizeof(v4);
    out.family = AF_INET;
    return out;
  }
  sockaddr_in6 v6{};
  if (::inet_pton(AF_INET6, host.c_str(), &v6.sin6_addr) == 1) {
    v6.sin6_family = AF_INET6;
    v6.sin6_port = ::htons(static_cast<std::uint16_t>(port_value));
    std::memcpy(&out.storage, &v6, sizeof(v6));
    out.len = sizeof(v6);
    out.family = AF_INET6;
    return out;
  }
  // resolve() is a separate, explicit call precisely because getaddrinfo
  // blocks and nothing else in this module does.
  return LSE_ERROR(kInvalidArgument, "'", host,
                   "' is not a literal address; call comm::resolve() before "
                   "handing an endpoint to listen() or connect()");
}

Result<Address> unix_address(const Endpoint& ep) {
  std::string name(ep.path());
  if (name.empty()) name = std::string(ep.authority());
  if (name.empty()) {
    return LSE_ERROR(kInvalidArgument,
                     "a unix endpoint names a socket path or an @abstract name");
  }
  sockaddr_un un{};
  un.sun_family = AF_UNIX;
  // A leading '@' is the abstract namespace: no filesystem entry, so nothing to
  // unlink and nothing left behind by a crash.
  const bool abstract = name.front() == '@';
  const std::string body = abstract ? name.substr(1) : name;
  if (body.size() + 1 > sizeof(un.sun_path)) {
    return LSE_ERROR(kOutOfRange, "unix socket name '", name, "' is longer than ",
                     std::to_string(sizeof(un.sun_path) - 1), " bytes");
  }
  Address out;
  if (abstract) {
    un.sun_path[0] = '\0';
    std::memcpy(un.sun_path + 1, body.data(), body.size());
    out.len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 +
                                     body.size());
  } else {
    std::memcpy(un.sun_path, body.data(), body.size());
    out.len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                                     body.size() + 1);
  }
  std::memcpy(&out.storage, &un, sizeof(un));
  out.family = AF_UNIX;
  return out;
}

Result<Address> address_of(const Endpoint& ep) {
  if (is_unix(ep)) return unix_address(ep);
  return inet_address(ep.authority());
}

Result<int> make_socket(int family) {
  const int fd =
      ::socket(family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    return LSE_ERROR(kIoError, "socket(): ", std::strerror(errno));
  }
  if (family != AF_UNIX) {
    const int one = 1;
    // Small control messages are the whole point of the control lane; Nagle
    // would batch them behind an ack and hand back milliseconds of latency.
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return fd;
}

std::string describe_peer(int fd, const Endpoint& fallback) {
  sockaddr_storage ss{};
  socklen_t len = sizeof(ss);
  if (::getpeername(fd, reinterpret_cast<sockaddr*>(&ss), &len) != 0) {
    return fallback.str();
  }
  if (ss.ss_family == AF_INET) {
    sockaddr_in v4{};
    std::memcpy(&v4, &ss, sizeof(v4));
    std::array<char, INET_ADDRSTRLEN> text{};
    ::inet_ntop(AF_INET, &v4.sin_addr, text.data(), text.size());
    return "tcp://" + std::string(text.data()) + ":" +
           std::to_string(::ntohs(v4.sin_port));
  }
  if (ss.ss_family == AF_INET6) {
    sockaddr_in6 v6{};
    std::memcpy(&v6, &ss, sizeof(v6));
    std::array<char, INET6_ADDRSTRLEN> text{};
    ::inet_ntop(AF_INET6, &v6.sin6_addr, text.data(), text.size());
    return "tcp://[" + std::string(text.data()) + "]:" +
           std::to_string(::ntohs(v6.sin6_port));
  }
  return fallback.str();
}

Result<Endpoint> bound_endpoint(int fd, const Endpoint& requested) {
  if (is_unix(requested)) return requested;
  sockaddr_storage ss{};
  socklen_t len = sizeof(ss);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &len) != 0) {
    return LSE_ERROR(kIoError, "getsockname: ", std::strerror(errno));
  }
  std::string authority;
  if (ss.ss_family == AF_INET) {
    sockaddr_in v4{};
    std::memcpy(&v4, &ss, sizeof(v4));
    std::array<char, INET_ADDRSTRLEN> text{};
    ::inet_ntop(AF_INET, &v4.sin_addr, text.data(), text.size());
    authority = std::string(text.data()) + ":" +
                std::to_string(::ntohs(v4.sin_port));
  } else {
    sockaddr_in6 v6{};
    std::memcpy(&v6, &ss, sizeof(v6));
    std::array<char, INET6_ADDRSTRLEN> text{};
    ::inet_ntop(AF_INET6, &v6.sin6_addr, text.data(), text.size());
    authority = "[" + std::string(text.data()) + "]:" +
                std::to_string(::ntohs(v6.sin6_port));
  }
  std::string text = std::string(requested.scheme()) + "://" + authority;
  bool first = true;
  for (const auto& kv : requested.options()) {
    text += first ? '?' : '&';
    first = false;
    text += kv.first;
    text += '=';
    text += kv.second;
  }
  return Endpoint::parse(text);
}

std::uint64_t random_u64() {
  static std::mt19937_64 gen{std::random_device{}()};
  return gen();
}

class SocketTransport;

class SocketListener final : public IListenerImpl, public IReady {
 public:
  SocketListener(SocketTransport& owner, Poller& poller, int fd, Endpoint bound,
                 std::uint64_t id, std::uint64_t handshake_ns)
      : owner_(&owner),
        poller_(&poller),
        fd_(fd),
        bound_(std::move(bound)),
        id_(id),
        handshake_ns_(handshake_ns) {}
  ~SocketListener() override { shut(); }
  SocketListener(const SocketListener&) = delete;
  SocketListener& operator=(const SocketListener&) = delete;

  [[nodiscard]] const Endpoint& endpoint() const noexcept override {
    return bound_;
  }
  Status close() override {
    shut();
    closed_ = true;
    return OkStatus();
  }
  [[nodiscard]] bool closed() const noexcept { return closed_; }
  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

  void on_ready(int fd, Readiness r) override;
  void tick(std::uint64_t now_ns);
  [[nodiscard]] std::uint64_t next_timer_ns() const noexcept;

 private:
  struct Pending {
    int fd = -1;
    std::size_t read = 0;
    std::array<std::byte, kHelloBytes> hello{};
    std::uint64_t since_ns = 0;
  };
  struct HalfPair {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
    std::array<int, 2> fds{-1, -1};
    std::uint32_t peer_ring = kDefaultControlRingBytes;
    std::uint64_t since_ns = 0;
  };

  void shut() noexcept;
  void accept_ready();
  void pending_ready(Pending& p);
  void drop(int fd) noexcept;

  SocketTransport* owner_ = nullptr;
  Poller* poller_ = nullptr;
  int fd_ = -1;
  Endpoint bound_;
  std::uint64_t id_ = 0;
  std::uint64_t handshake_ns_ = 0;
  bool closed_ = false;
  // False while the listening socket is off the poller because there were no
  // descriptors left to accept with; retry_at_ns_ is when to try again.
  bool armed_ = true;
  std::uint64_t retry_at_ns_ = 0;
  std::vector<Pending> pending_;
  std::vector<HalfPair> pairs_;
};

class SocketTransport {
 public:
  Status open_impl(const Endpoint& ep, Poller& poller, EventSink& sink) {
    LSE_RETURN_IF_ERROR(check_options(ep, kNoExtraOptions));
    poller_ = &poller;
    sink_ = &sink;
    return OkStatus();
  }

  void close_impl() noexcept {
    for (auto& link : links_) link->abort();
    links_.clear();
    listeners_.clear();
    poller_ = nullptr;
    sink_ = nullptr;
  }

  [[nodiscard]] Capabilities capabilities_impl(
      const Endpoint& ep) const noexcept {
    Capabilities caps;
    caps.device_memory_direct = false;
    caps.reliable = true;
    caps.ordered = true;
    caps.full_duplex = true;
    caps.registers_memory = false;
    // A socket carries whatever the kernel and the wire give it. This module
    // does not know that number and does not guess one: probe measures it.
    caps.bandwidth_bytes_per_s = 0;
    caps.latency_ns = 0;
    const auto inflight = ep.option_u64("max_inflight", 64);
    caps.max_inflight =
        inflight.ok() ? static_cast<std::uint32_t>(*inflight) : 64u;
    const auto frame = ep.option_u64("frame_max", 0);
    caps.max_message_bytes =
        frame.ok() ? static_cast<std::size_t>(*frame) : std::size_t{0};
    return caps;
  }

  [[nodiscard]] std::string_view declined_impl(
      const Endpoint& ep) const noexcept {
    // A socket family is present on every Linux this engine builds on, so the
    // only thing that can be missing is a usable name.
    if (is_unix(ep)) {
      if (ep.path().empty() && ep.authority().empty()) {
        return "a unix endpoint names a socket path or an @abstract name";
      }
      return {};
    }
    if (ep.authority().empty()) return "a tcp endpoint names host:port";
    if (ep.authority().find(':') == std::string_view::npos) {
      return "a tcp endpoint names host:port";
    }
    return {};
  }

  Result<ILink*> connect_impl(const Endpoint& ep, std::uint64_t channel) {
    if (poller_ == nullptr) {
      return LSE_ERROR(kInternal, "transport was not opened");
    }
    // Per endpoint, not per transport: one transport instance serves every
    // endpoint of its scheme on a reactor, so validating only the one that
    // caused it to be opened would let a typo on the second peer through.
    LSE_RETURN_IF_ERROR(check_options(ep, kNoExtraOptions));
    LSE_ASSIGN_OR(const LinkOptions opt, link_options_from(ep));
    LSE_ASSIGN_OR(const Address addr, address_of(ep));

    std::array<int, 2> fds{-1, -1};
    for (std::size_t i = 0; i < 2; ++i) {
      auto made = make_socket(addr.family);
      if (!made.ok()) {
        for (const int open_fd : fds) {
          if (open_fd >= 0) ::close(open_fd);
        }
        return made.status();
      }
      fds[i] = made.release();
      if (const std::string_view local = ep.option("bind"); !local.empty()) {
        auto bind_to = inet_address(std::string(local) + ":0");
        if (!bind_to.ok()) {
          for (const int open_fd : fds) {
            if (open_fd >= 0) ::close(open_fd);
          }
          return bind_to.status();
        }
        ::bind(fds[i], reinterpret_cast<const sockaddr*>(&bind_to->storage),
               bind_to->len);
      }
      if (::connect(fds[i], reinterpret_cast<const sockaddr*>(&addr.storage),
                    addr.len) != 0 &&
          errno != EINPROGRESS) {
        const std::string why = std::strerror(errno);
        for (const int open_fd : fds) {
          if (open_fd >= 0) ::close(open_fd);
        }
        return LSE_ERROR(kIoError, "connect to ", ep.str(), ": ", why);
      }
    }

    auto link = std::make_unique<StreamLink>(
        *poller_, *sink_, channel, ep, capabilities_impl(ep), opt, fds,
        /*client=*/true, random_u64(), random_u64());
    ILink* raw = link.get();
    links_.push_back(std::move(link));
    return raw;
  }

  Result<IListenerImpl*> listen_impl(const Endpoint& ep,
                                     std::uint64_t listener) {
    if (poller_ == nullptr) {
      return LSE_ERROR(kInternal, "transport was not opened");
    }
    LSE_RETURN_IF_ERROR(check_options(ep, kNoExtraOptions));
    LSE_ASSIGN_OR(const LinkOptions opt, link_options_from(ep));
    LSE_ASSIGN_OR(const Address addr, address_of(ep));

    if (addr.family == AF_UNIX && !ep.path().empty() &&
        ep.path().front() != '@') {
      struct ::stat st{};
      if (::stat(std::string(ep.path()).c_str(), &st) == 0) {
        return LSE_ERROR(kAlreadyExists, "'", std::string(ep.path()),
                         "' already exists; this transport never unlinks a "
                         "path it did not create");
      }
    }

    LSE_ASSIGN_OR(const int fd, make_socket(addr.family));
    if (addr.family != AF_UNIX) {
      const int one = 1;
      ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    }
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr.storage), addr.len) !=
        0) {
      const std::string why = std::strerror(errno);
      ::close(fd);
      return LSE_ERROR(kIoError, "bind ", ep.str(), ": ", why);
    }
    if (::listen(fd, 128) != 0) {
      const std::string why = std::strerror(errno);
      ::close(fd);
      return LSE_ERROR(kIoError, "listen ", ep.str(), ": ", why);
    }

    auto bound = bound_endpoint(fd, ep);
    if (!bound.ok()) {
      ::close(fd);
      return bound.status();
    }
    auto owned = std::make_unique<SocketListener>(
        *this, *poller_, fd, bound.release(), listener, opt.deadline_ns);
    const Status added = poller_->add(fd, Interest::kRead, owned.get());
    if (!added.ok()) return added;
    IListenerImpl* raw = owned.get();
    listeners_.push_back(std::move(owned));
    return raw;
  }

  Status progress_impl() {
    const std::uint64_t now = steady_now_ns();
    for (auto& link : links_) link->tick(now);
    for (auto& ln : listeners_) ln->tick(now);
    // Reaped only here, at the top of a sweep, so nothing being delivered from
    // the previous sweep still names an object about to be destroyed.
    std::erase_if(links_, [](const std::unique_ptr<StreamLink>& l) {
      return l->retired();
    });
    std::erase_if(listeners_, [](const std::unique_ptr<SocketListener>& l) {
      return l->closed();
    });
    return OkStatus();
  }

  [[nodiscard]] std::uint64_t next_timer_ns_impl() const noexcept {
    std::uint64_t soonest = 0;
    for (const auto& link : links_) {
      const std::uint64_t at = link->next_timer_ns();
      if (at != 0 && (soonest == 0 || at < soonest)) soonest = at;
    }
    for (const auto& ln : listeners_) {
      const std::uint64_t at = ln->next_timer_ns();
      if (at != 0 && (soonest == 0 || at < soonest)) soonest = at;
    }
    return soonest;
  }

  // Called by a listener once both sockets of one channel have identified
  // themselves. Ownership of the fds moves to the link.
  void adopt(std::array<int, 2> fds, const Endpoint& listen_ep,
             std::uint32_t peer_ring, std::uint64_t listener_id);

 protected:
  Poller* poller_ = nullptr;
  EventSink* sink_ = nullptr;
  std::vector<std::unique_ptr<StreamLink>> links_;
  std::vector<std::unique_ptr<SocketListener>> listeners_;
};

void SocketTransport::adopt(std::array<int, 2> fds, const Endpoint& listen_ep,
                            std::uint32_t peer_ring,
                            std::uint64_t listener_id) {
  auto peer_text = Endpoint::parse(describe_peer(fds[0], listen_ep));
  Endpoint peer = peer_text.ok() ? peer_text.release() : listen_ep;

  auto opt = link_options_from(listen_ep);
  LinkOptions options = opt.ok() ? opt.release() : LinkOptions{};
  // The peer's advertised ring is what bounds what this side may SEND, and the
  // link learns it from the hello on the control lane; the acceptor already
  // read that hello, so it is handed over rather than read twice.
  const std::uint64_t channel = sink_->reserve_channel();
  if (channel == 0) {
    for (const int fd : fds) ::close(fd);
    return;
  }
  auto link = std::make_unique<StreamLink>(
      *poller_, *sink_, channel, std::move(peer), capabilities_impl(listen_ep),
      options, fds, /*client=*/false, 0, 0);
  link->adopt_peer_ring(peer_ring);
  ILink* raw = link.get();
  links_.push_back(std::move(link));
  sink_->adopt(channel, raw, listener_id);
}

void SocketListener::shut() noexcept {
  if (fd_ >= 0) {
    poller_->remove(fd_);
    ::close(fd_);
    fd_ = -1;
  }
  for (Pending& p : pending_) {
    if (p.fd >= 0) {
      poller_->remove(p.fd);
      ::close(p.fd);
      p.fd = -1;
    }
  }
  pending_.clear();
  for (HalfPair& pair : pairs_) {
    for (const int fd : pair.fds) {
      if (fd >= 0) ::close(fd);
    }
  }
  pairs_.clear();
}

void SocketListener::drop(int fd) noexcept {
  poller_->remove(fd);
  ::close(fd);
  std::erase_if(pending_, [fd](const Pending& p) { return p.fd == fd; });
}

void SocketListener::on_ready(int fd, Readiness r) {
  if (fd == fd_) {
    accept_ready();
    return;
  }
  for (Pending& p : pending_) {
    if (p.fd != fd) continue;
    if (r.readable || r.hangup) pending_ready(p);
    return;
  }
}

// Long enough that the loop goes idle instead of hammering accept4, short
// enough that a descriptor given back is picked up within one decode step.
constexpr std::uint64_t kAcceptRetryNs = 50'000'000ull;

void SocketListener::accept_ready() {
  for (;;) {
    const int fd = ::accept4(fd_, nullptr, nullptr,
                             SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      // Out of descriptors: the connection stays queued and the listening
      // socket stays readable, so waiting on it again would report ready
      // immediately, for ever — poll() would never sleep and one core would
      // burn until something else in the process closed a file. Stop waiting on
      // it and let tick() try again.
      if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS ||
          errno == ENOMEM) {
        poller_->remove(fd_);
        armed_ = false;
        retry_at_ns_ = steady_now_ns() + kAcceptRetryNs;
      }
      return;
    }
    int family = AF_UNIX;
    sockaddr_storage ss{};
    socklen_t len = sizeof(ss);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &len) == 0) {
      family = ss.ss_family;
    }
    if (family != AF_UNIX) {
      const int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    Pending p;
    p.fd = fd;
    p.since_ns = steady_now_ns();
    pending_.push_back(p);
    if (!poller_->add(fd, Interest::kRead, this).ok()) {
      drop(fd);
    }
  }
}

void SocketListener::pending_ready(Pending& p) {
  while (p.read < kHelloBytes) {
    const ssize_t n = ::recv(p.fd, p.hello.data() + p.read, kHelloBytes - p.read,
                             0);
    if (n == 0) {
      drop(p.fd);
      return;
    }
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return;
      drop(p.fd);
      return;
    }
    p.read += static_cast<std::size_t>(n);
  }

  Hello h{};
  std::memcpy(&h, p.hello.data(), sizeof(h));
  if (h.magic != kFrameMagic || h.version != kWireVersion || h.lane > 1 ||
      (((h.flags & kHelloLittleEndian) != 0) != host_is_little_endian())) {
    drop(p.fd);
    return;
  }

  const int fd = p.fd;
  poller_->remove(fd);
  std::erase_if(pending_, [fd](const Pending& q) { return q.fd == fd; });

  const auto slot = static_cast<std::size_t>(h.lane);
  auto pair = std::find_if(pairs_.begin(), pairs_.end(), [&](const HalfPair& x) {
    return x.hi == h.channel_hi && x.lo == h.channel_lo;
  });
  if (pair == pairs_.end()) {
    HalfPair fresh;
    fresh.hi = h.channel_hi;
    fresh.lo = h.channel_lo;
    fresh.since_ns = steady_now_ns();
    pairs_.push_back(fresh);
    pair = pairs_.end() - 1;
  }
  if (pair->fds[slot] >= 0) {  // a second socket claiming the same lane
    ::close(fd);
    return;
  }
  pair->fds[slot] = fd;
  if (slot == 0) pair->peer_ring = h.control_ring_bytes;
  if (pair->fds[0] < 0 || pair->fds[1] < 0) return;

  const std::array<int, 2> fds{pair->fds[0], pair->fds[1]};
  const std::uint32_t ring = pair->peer_ring;
  pairs_.erase(pair);
  owner_->adopt(fds, bound_, ring, id_);
}

void SocketListener::tick(std::uint64_t now_ns) {
  if (!armed_ && fd_ >= 0 && now_ns >= retry_at_ns_) {
    if (poller_->add(fd_, Interest::kRead, this).ok()) {
      armed_ = true;
      retry_at_ns_ = 0;
    } else {
      retry_at_ns_ = now_ns + kAcceptRetryNs;
    }
  }
  // A socket that connected and never said who it was, or one lane of a channel
  // whose sibling never arrived, is closed rather than kept: there is no
  // channel to report it against, and the peer sees the drop.
  for (Pending& p : pending_) {
    if (p.fd >= 0 && now_ns >= p.since_ns + handshake_ns_) {
      const int fd = p.fd;
      p.fd = -1;
      poller_->remove(fd);
      ::close(fd);
    }
  }
  std::erase_if(pending_, [](const Pending& p) { return p.fd < 0; });
  for (HalfPair& pair : pairs_) {
    if (now_ns < pair.since_ns + handshake_ns_) continue;
    for (int& fd : pair.fds) {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
    }
  }
  std::erase_if(pairs_, [](const HalfPair& pair) {
    return pair.fds[0] < 0 && pair.fds[1] < 0;
  });
}

std::uint64_t SocketListener::next_timer_ns() const noexcept {
  std::uint64_t soonest = armed_ ? 0 : retry_at_ns_;
  for (const Pending& p : pending_) {
    const std::uint64_t at = p.since_ns + handshake_ns_;
    if (soonest == 0 || at < soonest) soonest = at;
  }
  for (const HalfPair& pair : pairs_) {
    const std::uint64_t at = pair.since_ns + handshake_ns_;
    if (soonest == 0 || at < soonest) soonest = at;
  }
  return soonest;
}

class TcpTransport final : public SocketTransport,
                           public Transport<TcpTransport> {
 public:
  static constexpr std::string_view kScheme = "tcp";
  [[nodiscard]] bool authority_is_host_impl() const noexcept { return true; }
};

class UnixTransport final : public SocketTransport,
                            public Transport<UnixTransport> {
 public:
  static constexpr std::string_view kScheme = "unix";
};

}  // namespace

LSE_REGISTER_COMM_TRANSPORT("tcp", TcpTransport)
LSE_REGISTER_COMM_TRANSPORT("unix", UnixTransport)

}  // namespace lse::comm
