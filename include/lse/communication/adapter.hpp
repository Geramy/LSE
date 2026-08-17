// Erases a concrete Transport behind ITransport.
//
// The CRTP half exists so a transport writes only the surface it actually has
// and the honest declines are generated once; this adapter is what the reactor
// holds. The scheme is read off the type, so a transport cannot register under
// one name and answer to another.
#pragma once

#include <cstdint>
#include <string_view>

#include "lse/communication/transport.hpp"

namespace lse::comm {

template <typename Derived>
class TransportAdapter final : public ITransport {
 public:
  TransportAdapter() = default;

  Status open(const Endpoint& ep, Poller& poller, EventSink& sink) override {
    return impl_.open(ep, poller, sink);
  }
  void close() noexcept override { impl_.close(); }

  Capabilities capabilities(const Endpoint& ep) const noexcept override {
    return impl_.capabilities(ep);
  }
  std::string_view declined(const Endpoint& ep) const noexcept override {
    return impl_.declined(ep);
  }
  std::string_view name() const noexcept override { return Derived::kScheme; }
  bool authority_is_host() const noexcept override {
    return impl_.authority_is_host();
  }

  Result<ILink*> connect(const Endpoint& ep, std::uint64_t channel) override {
    return impl_.connect(ep, channel);
  }
  Result<IListenerImpl*> listen(const Endpoint& ep,
                                std::uint64_t listener) override {
    return impl_.listen(ep, listener);
  }
  Result<Region> register_region(const RegionRequest& req) override {
    return impl_.register_region(req);
  }
  Status deregister_region(Region r) override {
    return impl_.deregister_region(r);
  }

  Status progress() override { return impl_.progress(); }
  bool requires_polling() const noexcept override {
    return impl_.requires_polling();
  }
  std::uint64_t next_timer_ns() const noexcept override {
    return impl_.next_timer_ns();
  }

  [[nodiscard]] Derived& impl() noexcept { return impl_; }

 private:
  Derived impl_;
};

}  // namespace lse::comm
