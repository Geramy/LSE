// In-process transport across several devices: one rank per device, one
// address space, no fabric.
//
// The case two GPUs in one box actually are. Loopback covers N ranks on one
// device and a fabric transport covers N machines; neither describes a pool
// whose members can reach each other's memory directly, which is the shape a
// tensor-split model runs on.
//
// Ranks rendezvous on `TransportConfig::group_id` exactly as loopback's do, and
// each rank is driven by its own thread. What differs is the wire: a device
// payload moves by peer copy between the two members' buffers rather than
// through host memory, so the bytes never leave the link. `Capabilities` is
// measured on that path, not assumed from it -- a pool whose members fall back
// to host staging reports the bandwidth it actually got, and the cost model
// picks a different collective for it without anything here changing.
//
// The copy is receiver-driven: a send posts its source and returns, and the
// matching recv performs the transfer, because the receiver is the only side
// that knows both ends. `wait` on the sender's handle blocks until it lands.
#pragma once

#include <memory>
#include <vector>
#include <string_view>

#include "lse/backend/backend.hpp"
#include "lse/dist/transport.hpp"

namespace lse::dist {

class PeerTransport : public Transport<PeerTransport> {
 public:
  static constexpr std::string_view kScheme = "peer";

  // The per-group rendezvous state, shared by every rank in one process.
  struct Group;

  PeerTransport();
  ~PeerTransport();
  PeerTransport(const PeerTransport&) = delete;
  PeerTransport& operator=(const PeerTransport&) = delete;

  // The device this rank drives. Must be set before connect: a rank with no
  // backend has no memory to send from, and guessing the primary would put
  // every rank on one device and report a peer transport that never left it.
  void bind(backend::IDeviceSet& set, std::size_t member) noexcept;

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
  // One entry per posted send, indexed by the handle it returned. Cleared by
  // fence(): a collective's phases are separated by one, and a send whose recv
  // already ran is finished business.
  struct Posted;
  std::vector<std::shared_ptr<Posted>> pending_;
  std::shared_ptr<Group> group_;
  backend::IDeviceSet* set_ = nullptr;
  std::size_t member_ = 0;
  Rank rank_ = 0;
  Rank world_ = 1;
  Capabilities caps_{};
  std::uint32_t timeout_ms_ = 30000;
};

}  // namespace lse::dist
