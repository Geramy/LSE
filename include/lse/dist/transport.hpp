// Transport abstraction and collectives.
//
// The engine talks to a Transport, never to RCCL or sockets directly.
// Collectives are delegated when a transport has native ones (RCCL) and
// synthesized over point-to-point otherwise, chosen by the cost model in
// Capabilities.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::dist {

using Rank = std::int32_t;

enum class ReduceOp : std::uint8_t { kSum, kProd, kMin, kMax, kAvg };

struct Capabilities {
  bool native_collectives = false;
  // GPUDirect / XGMI / PCIe P2P. When false every transfer stages through a
  // host bounce buffer, and the cost model doubles the byte cost.
  bool device_memory_direct = false;
  // False for UDP-style transports; the engine layers sequencing/retry on top.
  bool reliable = true;
  bool ordered = true;
  bool full_duplex = true;
  bool asynchronous = true;

  std::uint64_t bandwidth_bytes_per_s = 0;
  std::uint64_t latency_ns = 0;
  std::size_t max_message_bytes = 0;   // 0 = unbounded
  // Below this, latency dominates and a tree beats a ring.
  [[nodiscard]] std::size_t latency_bound_threshold() const noexcept {
    return bandwidth_bytes_per_s
               ? static_cast<std::size_t>(latency_ns * bandwidth_bytes_per_s / 1'000'000'000ull)
               : 0;
  }
};

struct CommBuffer {
  const backend::DeviceBuffer* device = nullptr;  // null => host pointer
  void* host = nullptr;
  std::size_t offset = 0;
  std::size_t bytes = 0;
  DType dtype = DType::kF32;

  [[nodiscard]] bool on_device() const noexcept { return device != nullptr; }
};

struct CommHandle {
  std::uint64_t id = 0;
  [[nodiscard]] bool valid() const noexcept { return id != 0; }
};

struct TransportConfig {
  std::string endpoint;              // "tcp://10.0.0.1:29500", "rccl://", ...
  Rank rank = 0;
  Rank world_size = 1;
  std::string group_id;              // rendezvous key
  std::uint32_t timeout_ms = 30000;
  // Transport-specific knobs (MTU, NIC, TB5 route).
  std::vector<std::pair<std::string, std::string>> options;
};

template <typename Derived>
class Transport {
 public:
  Status connect(const TransportConfig& cfg) { return derived().connect_impl(cfg); }
  void disconnect() noexcept { derived().disconnect_impl(); }

  [[nodiscard]] Rank rank() const noexcept { return derived().rank_impl(); }
  [[nodiscard]] Rank world_size() const noexcept { return derived().world_size_impl(); }
  [[nodiscard]] const Capabilities& capabilities() const noexcept {
    return derived().capabilities_impl();
  }

  Result<CommHandle> send(const CommBuffer& buf, Rank dst, int tag = 0) {
    return derived().send_impl(buf, dst, tag);
  }
  Result<CommHandle> recv(CommBuffer& buf, Rank src, int tag = 0) {
    return derived().recv_impl(buf, src, tag);
  }
  Status wait(CommHandle h) { return derived().wait_impl(h); }
  Status fence() { return derived().fence_impl(); }

  // Deadlock-free on half-duplex transports: ordered by rank parity.
  Status sendrecv(const CommBuffer& send_buf, Rank dst, CommBuffer& recv_buf,
                  Rank src, int tag = 0);

  // Used when the transport has no native collectives.
  Status ring_all_reduce(CommBuffer& buf, ReduceOp op);
  Status tree_broadcast(CommBuffer& buf, Rank root);
  Status ring_all_gather(const CommBuffer& src, CommBuffer& dst);
  Status ring_reduce_scatter(const CommBuffer& src, CommBuffer& dst, ReduceOp op);
  Status bruck_all_to_all(const CommBuffer& src, CommBuffer& dst);

 protected:
  Transport() = default;

 private:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

class ITransport {
 public:
  virtual ~ITransport() = default;
  virtual Status connect(const TransportConfig&) = 0;
  virtual void disconnect() noexcept = 0;
  virtual Rank rank() const noexcept = 0;
  virtual Rank world_size() const noexcept = 0;
  virtual const Capabilities& capabilities() const noexcept = 0;
  virtual Result<CommHandle> send(const CommBuffer&, Rank dst, int tag) = 0;
  virtual Result<CommHandle> recv(CommBuffer&, Rank src, int tag) = 0;
  virtual Status wait(CommHandle) = 0;
  virtual Status fence() = 0;

  virtual Status all_reduce(CommBuffer&, ReduceOp) = 0;
  virtual Status all_gather(const CommBuffer&, CommBuffer&) = 0;
  virtual Status reduce_scatter(const CommBuffer&, CommBuffer&, ReduceOp) = 0;
  virtual Status broadcast(CommBuffer&, Rank root) = 0;
  virtual Status all_to_all(const CommBuffer&, CommBuffer&) = 0;
  virtual Status barrier() = 0;

  virtual std::string_view name() const noexcept = 0;
};

template <typename Derived>
class TransportAdapter;  // defined in dist/adapter.hpp

// Registered by URI scheme: "rccl://", "tb5://", "tcp://".
using TransportFactory = std::unique_ptr<ITransport> (*)();

void register_transport(std::string_view scheme, TransportFactory factory);
Result<std::unique_ptr<ITransport>> create_transport(std::string_view endpoint);
std::vector<std::string> available_transports();

struct TransportRegistrar {
  TransportRegistrar(std::string_view scheme, TransportFactory f) {
    register_transport(scheme, f);
  }
};

#define LSE_REGISTER_TRANSPORT(scheme, Type)                                \
  namespace {                                                               \
  const ::lse::dist::TransportRegistrar _lse_transport_registrar{           \
      scheme, []() -> std::unique_ptr<::lse::dist::ITransport> {            \
        return std::make_unique<::lse::dist::TransportAdapter<Type>>();     \
      }};                                                                   \
  }  // namespace

}  // namespace lse::dist
