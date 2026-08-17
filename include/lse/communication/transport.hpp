// What a transport implements, and how a scheme finds one.
//
// A transport writes the *_impl surface and one static string; the CRTP half
// supplies the rest, including honest declines for anything it did not write.
// Nothing above this header ever holds an ITransport: a caller holds a Reactor,
// a Channel and a Listener, and names a peer with an endpoint string.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/communication/buffer.hpp"
#include "lse/communication/capabilities.hpp"
#include "lse/communication/endpoint.hpp"
#include "lse/communication/event.hpp"
#include "lse/core/status.hpp"

namespace lse::comm {

// The readiness mechanism, defined in lse/communication/poller.hpp. A transport
// registers its descriptors with it and is driven from the reactor's one wait;
// replacing epoll with io_uring replaces that file alone.
class Poller;

// Reactor-owned; a transport only appends. Control payloads are copied through
// retain() rather than handed over as a pointer into the link's own buffer,
// because an event may outlive the link that produced it.
class ILink;

class EventSink {
 public:
  virtual ~EventSink() = default;
  virtual void emit(const Event& ev) = 0;
  [[nodiscard]] virtual const std::byte* retain(
      std::span<const std::byte> bytes) = 0;

  // A server-mode transport mints no identity of its own: it asks for a channel
  // id, builds the link with that id stamped into every event it will ever
  // emit, and then hands the link over. Splitting it in two is what lets the
  // link name itself from the moment it exists. 0 means the reactor refused.
  [[nodiscard]] virtual std::uint64_t reserve_channel() = 0;
  virtual void adopt(std::uint64_t channel, ILink* link,
                     std::uint64_t listener) = 0;
};

// One connection. Owned by its transport; named above the seam by a Channel.
class ILink {
 public:
  virtual ~ILink() = default;
  virtual Status post(Lane lane, Direction dir, const Transfer& t,
                      std::uint64_t op) = 0;
  virtual Status cancel(std::uint64_t op) = 0;
  [[nodiscard]] virtual std::size_t credit(Lane lane) const noexcept = 0;
  [[nodiscard]] virtual const Capabilities& capabilities() const noexcept = 0;
  [[nodiscard]] virtual const Endpoint& peer() const noexcept = 0;
  virtual Status reject(std::uint32_t tag) = 0;
  virtual Status close() = 0;  // graceful: queued sends finish first
  virtual void abort() noexcept = 0;
  // The sentence behind the code on this channel's kClosed event. Read once,
  // when the channel dies, rather than carried on every event.
  [[nodiscard]] virtual const Status& last_error() const noexcept = 0;
};

class IListenerImpl {
 public:
  virtual ~IListenerImpl() = default;
  // What was actually bound: differs from the requested endpoint when the port
  // was 0, which is how a caller binds without racing anyone for a port.
  [[nodiscard]] virtual const Endpoint& endpoint() const noexcept = 0;
  virtual Status close() = 0;
};

class ITransport {
 public:
  virtual ~ITransport() = default;

  // Local resources only — an epoll registration, a device handle. Never
  // touches the network, which is what makes select_endpoint side-effect free.
  virtual Status open(const Endpoint& ep, Poller& poller, EventSink& sink) = 0;
  virtual void close() noexcept = 0;

  // Answerable BEFORE open, so selection never has to try and fail. After open
  // a transport may refine the numbers; Channel::capabilities() returns the
  // refined instance value.
  [[nodiscard]] virtual Capabilities capabilities(
      const Endpoint& ep) const noexcept = 0;
  // Empty when this transport can serve that endpoint on this machine.
  // Otherwise it NAMES the missing piece — the file it looked for, the module
  // that is not loaded — so a decline is diagnosable rather than a silent
  // fallback.
  [[nodiscard]] virtual std::string_view declined(
      const Endpoint& ep) const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  // True when this transport's authority is a host and a port, so an endpoint
  // that names a host must pass through resolve() before it can be dialled.
  // False when the authority is a pipe name or a device node, which resolve()
  // must then hand back untouched rather than fail to look up. It is asked of
  // the transport rather than guessed from the string so that one caller flow —
  // resolve, select, connect — is correct for every endpoint.
  [[nodiscard]] virtual bool authority_is_host() const noexcept = 0;

  virtual Result<ILink*> connect(const Endpoint& ep,
                                 std::uint64_t channel) = 0;
  virtual Result<IListenerImpl*> listen(const Endpoint& ep,
                                        std::uint64_t listener) = 0;
  virtual Result<Region> register_region(const RegionRequest& req) = 0;
  virtual Status deregister_region(Region r) = 0;

  // Drains readiness and completions into the sink. The reactor has already
  // waited; this never waits itself.
  virtual Status progress() = 0;
  // True when this transport has no pollable descriptor and must be swept.
  [[nodiscard]] virtual bool requires_polling() const noexcept = 0;
  // The soonest absolute steady-clock nanosecond at which this transport has
  // something to do that no descriptor will wake it for — an unanswered data
  // offer, a handshake that never completed. 0 when it has none. The reactor
  // caps its wait by it, so such a deadline cannot be overslept.
  [[nodiscard]] virtual std::uint64_t next_timer_ns() const noexcept = 0;
};

template <typename Derived>
class Transport {
 public:
  Status open(const Endpoint& ep, Poller& poller, EventSink& sink) {
    return derived().open_impl(ep, poller, sink);
  }
  void close() noexcept { derived().close_impl(); }

  [[nodiscard]] Capabilities capabilities(const Endpoint& ep) const noexcept {
    return derived().capabilities_impl(ep);
  }
  [[nodiscard]] std::string_view declined(const Endpoint& ep) const noexcept {
    return derived().declined_impl(ep);
  }

  Result<ILink*> connect(const Endpoint& ep, std::uint64_t channel) {
    if constexpr (requires(Derived& d) { d.connect_impl(ep, channel); }) {
      return derived().connect_impl(ep, channel);
    } else {
      return LSE_ERROR(kUnimplemented, std::string(Derived::kScheme),
                       " has no client mode");
    }
  }

  Result<IListenerImpl*> listen(const Endpoint& ep, std::uint64_t listener) {
    if constexpr (requires(Derived& d) { d.listen_impl(ep, listener); }) {
      return derived().listen_impl(ep, listener);
    } else {
      return LSE_ERROR(kUnimplemented, std::string(Derived::kScheme),
                       " has no server mode");
    }
  }

  // A transport that cannot register memory still answers: an unregistered
  // host range is a legitimate region on it, and a dmabuf is refused by name
  // rather than staged behind the caller's back.
  Result<Region> register_region(const RegionRequest& req) {
    if constexpr (requires(Derived& d) { d.register_region_impl(req); }) {
      return derived().register_region_impl(req);
    } else {
      if (req.dmabuf_fd >= 0) {
        return LSE_ERROR(kUnimplemented, std::string(Derived::kScheme),
                         " has no dmabuf path; stage to host memory first");
      }
      if (req.host == nullptr) {
        return LSE_ERROR(kInvalidArgument, std::string(Derived::kScheme),
                         " was handed neither a host pointer nor a dmabuf fd");
      }
      return Region{0, static_cast<std::byte*>(req.host) + req.offset,
                    req.bytes};
    }
  }

  Status deregister_region(Region r) {
    if constexpr (requires(Derived& d) { d.deregister_region_impl(r); }) {
      return derived().deregister_region_impl(r);
    } else {
      return r.id == 0 ? OkStatus()
                       : LSE_ERROR(kInvalidArgument,
                                   std::string(Derived::kScheme),
                                   " never registered that region");
    }
  }

  Status progress() {
    if constexpr (requires(Derived& d) { d.progress_impl(); }) {
      return derived().progress_impl();
    } else {
      return OkStatus();
    }
  }

  [[nodiscard]] bool requires_polling() const noexcept {
    if constexpr (requires(const Derived& d) { d.requires_polling_impl(); }) {
      return derived().requires_polling_impl();
    } else {
      return false;
    }
  }

  [[nodiscard]] std::uint64_t next_timer_ns() const noexcept {
    if constexpr (requires(const Derived& d) { d.next_timer_ns_impl(); }) {
      return derived().next_timer_ns_impl();
    } else {
      return 0;
    }
  }

  // Defaults to false: a transport that says nothing is named by something
  // this module must not try to look up in DNS.
  [[nodiscard]] bool authority_is_host() const noexcept {
    if constexpr (requires(const Derived& d) { d.authority_is_host_impl(); }) {
      return derived().authority_is_host_impl();
    } else {
      return false;
    }
  }

 protected:
  Transport() = default;

 private:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept {
    return static_cast<const Derived&>(*this);
  }
};

template <typename Derived>
class TransportAdapter;  // defined in communication/adapter.hpp

// Registered by URI scheme: "tcp://", "unix://", "tb5://", "rdma://".
using TransportFactory = std::unique_ptr<ITransport> (*)();

void register_transport(std::string_view scheme, TransportFactory factory);
Result<std::unique_ptr<ITransport>> create_transport(std::string_view endpoint);
std::vector<std::string> available_transports();

struct TransportRegistrar {
  TransportRegistrar(std::string_view scheme, TransportFactory f) {
    register_transport(scheme, f);
  }
};

// The macro name differs from LSE_REGISTER_TRANSPORT because macros are not
// namespaced and one TU may include both headers. The registrar is named per
// line, so a TU that offers two schemes registers both.
#define LSE_REGISTER_COMM_TRANSPORT(scheme, Type)                          \
  namespace {                                                              \
  const ::lse::comm::TransportRegistrar LSE_CONCAT(                        \
      _lse_comm_transport_registrar_, __LINE__){                           \
      scheme, []() -> std::unique_ptr<::lse::comm::ITransport> {           \
        return std::make_unique<::lse::comm::TransportAdapter<Type>>();    \
      }};                                                                  \
  }  // namespace

}  // namespace lse::comm
