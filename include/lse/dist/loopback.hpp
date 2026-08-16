// In-process transport: N ranks in one address space, one device.
//
// This is a real transport, not test scaffolding. It is what a single-node
// multi-rank run uses before a fabric exists, it is the reference every other
// transport's collectives must agree with, and it is the only way to exercise
// a world size larger than the machine's device count. Its Capabilities are
// measured from the machine it is on, so the cost model reaches the same kind
// of decision here that it would over RDMA.
//
// Ranks rendezvous on `TransportConfig::group_id`. Each rank is expected to be
// driven by its own thread: `recv` blocks until the matching `send` lands, so
// the collectives need no separate barrier between phases.
#pragma once

#include <memory>
#include <string_view>

#include "lse/dist/transport.hpp"

namespace lse::dist {

class LoopbackTransport : public Transport<LoopbackTransport> {
 public:
  static constexpr std::string_view kScheme = "loopback";

  // The per-group rendezvous state, shared by every rank in one process.
  struct Group;

  LoopbackTransport();
  ~LoopbackTransport();
  LoopbackTransport(const LoopbackTransport&) = delete;
  LoopbackTransport& operator=(const LoopbackTransport&) = delete;

  Status connect_impl(const TransportConfig& cfg);
  void disconnect_impl() noexcept;

  [[nodiscard]] Rank rank_impl() const noexcept { return rank_; }
  [[nodiscard]] Rank world_size_impl() const noexcept { return world_; }
  [[nodiscard]] const Capabilities& capabilities_impl() const noexcept {
    return caps_;
  }

  Result<CommHandle> send_impl(const CommBuffer& buf, Rank dst, int tag);
  Result<CommHandle> recv_impl(CommBuffer& buf, Rank src, int tag);
  Status wait_impl(CommHandle h);
  Status fence_impl();

 private:
  std::shared_ptr<Group> group_;
  Rank rank_ = 0;
  Rank world_ = 1;
  Capabilities caps_{};
  std::uint32_t timeout_ms_ = 30000;
};

}  // namespace lse::dist
