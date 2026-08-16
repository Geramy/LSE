// Erases a concrete Transport behind ITransport.
//
// The CRTP half exists so the synthesized collectives are written once against
// send/recv with no virtual call per chunk; this adapter is what the engine
// holds. Collectives delegate to the transport's native ones when it declares
// them and fall back to the synthesized ring otherwise — that choice is the
// transport's own, and is separate from the compressed collectives in
// collective.hpp, which sit above any transport.
#pragma once

#include <string_view>
#include <utility>

#include "lse/dist/collectives.hpp"
#include "lse/dist/transport.hpp"

namespace lse::dist {

template <typename Derived>
class TransportAdapter final : public ITransport {
 public:
  TransportAdapter() = default;

  Status connect(const TransportConfig& cfg) override {
    return impl_.connect(cfg);
  }
  void disconnect() noexcept override { impl_.disconnect(); }
  Rank rank() const noexcept override { return impl_.rank(); }
  Rank world_size() const noexcept override { return impl_.world_size(); }
  const Capabilities& capabilities() const noexcept override {
    return impl_.capabilities();
  }

  Result<CommHandle> send(const CommBuffer& buf, Rank dst, int tag) override {
    return impl_.send(buf, dst, tag);
  }
  Result<CommHandle> recv(CommBuffer& buf, Rank src, int tag) override {
    return impl_.recv(buf, src, tag);
  }
  Status wait(CommHandle h) override { return impl_.wait(h); }
  Status fence() override { return impl_.fence(); }

  Status all_reduce(CommBuffer& buf, ReduceOp op) override {
    if constexpr (requires(Derived& d) { d.all_reduce_impl(buf, op); }) {
      if (impl_.capabilities().native_collectives) {
        return impl_.all_reduce_impl(buf, op);
      }
    }
    return ring_all_reduce_over(impl_, buf, op);
  }
  Status all_gather(const CommBuffer& src, CommBuffer& dst) override {
    return ring_all_gather_over(impl_, src, dst);
  }
  Status reduce_scatter(const CommBuffer& src, CommBuffer& dst,
                        ReduceOp op) override {
    return ring_reduce_scatter_over(impl_, src, dst, op);
  }
  Status broadcast(CommBuffer& buf, Rank root) override {
    return tree_broadcast_over(impl_, buf, root);
  }
  Status all_to_all(const CommBuffer& src, CommBuffer& dst) override {
    return bruck_all_to_all_over(impl_, src, dst);
  }
  Status barrier() override { return barrier_over(impl_); }

  std::string_view name() const noexcept override { return Derived::kScheme; }

  [[nodiscard]] Derived& impl() noexcept { return impl_; }

 private:
  Derived impl_;
};

}  // namespace lse::dist
