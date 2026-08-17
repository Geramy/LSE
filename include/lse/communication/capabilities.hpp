// What a link declares about itself, and the question a caller actually asks.
//
// Declared by the transport, read by the selector and by probe's cost model,
// never inferred from the transport's name — a name is not a capability. What
// is measured lives in probe::LinkProfile and is not duplicated here; a number
// this struct does not know is ZERO, and zero means "do not choose on this",
// not "slow".
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "lse/communication/endpoint.hpp"
#include "lse/core/status.hpp"

namespace lse::comm {

struct Capabilities {
  // GPUDirect / XGMI / PCIe P2P / dmabuf. When false every device-resident
  // transfer stages through a host bounce buffer, and the cost model doubles
  // the byte cost.
  bool device_memory_direct = false;
  // False for UDP-style transports; the engine layers sequencing/retry on top.
  bool reliable = true;
  bool ordered = true;
  bool full_duplex = true;
  // A posted region must be registered before its bytes can be DMA'd, and
  // registration is expensive enough to cache (ibv_reg_mr pins pages). When
  // true, posting an unregistered region is REFUSED rather than pinned behind
  // the caller's back: a syscall hidden on the per-token path is how a decode
  // loop loses microseconds nobody can find.
  bool registers_memory = false;

  // Posts per channel per lane that may be outstanding before post() refuses.
  // 0 = bounded only by the endpoint's max_inflight option.
  std::uint32_t max_inflight = 0;
  std::uint64_t bandwidth_bytes_per_s = 0;
  std::uint64_t latency_ns = 0;
  // Largest single wire frame. 0 = unbounded. A transfer larger than this is
  // segmented by the common layer, which is what makes a 4091-byte DMA ring
  // and a socket the same seam.
  std::size_t max_message_bytes = 0;

  // Below this, latency dominates and a tree beats a ring.
  [[nodiscard]] std::size_t latency_bound_threshold() const noexcept {
    return bandwidth_bytes_per_s
               ? static_cast<std::size_t>(latency_ns * bandwidth_bytes_per_s /
                                          1'000'000'000ull)
               : 0;
  }
  // The question a placement pass actually asks: can this peer take the bytes
  // where they already live, or does moving them cost a host round trip first?
  [[nodiscard]] bool moves_device_bytes_in_place() const noexcept {
    return device_memory_direct && registers_memory;
  }
  // Same rule src/dist/collective.cpp already uses: a link that cannot DMA out
  // of device memory pays the byte cost twice. Zero bandwidth means the
  // transport declared none, so only the latency term is known.
  [[nodiscard]] std::uint64_t predicted_ns(
      std::size_t bytes, bool device_resident) const noexcept;
};

// What a caller needs, stated as facts about the payload rather than as a
// transport name. Fed to select_endpoint, never to a branch at a call site.
struct Requirements {
  bool device_resident_payload = false;
  bool needs_reliability = true;
  bool needs_ordering = true;
  std::size_t largest_message_bytes = 0;
};

struct EndpointChoice {
  std::size_t index = 0;
  // Why this one and not the others. Carried so the decision is inspectable
  // rather than inferred, exactly as CollectivePlan::reason is.
  std::string reason;
};

// Picks among endpoints the caller already knows about — a pool descriptor
// lists a peer's rails, and there may be several. It never invents an endpoint
// and never silently downgrades: when nothing satisfies the requirements it
// returns kUnimplemented naming EVERY candidate's decline, so a missing path is
// diagnosable from one message.
[[nodiscard]] Result<EndpointChoice> select_endpoint(
    std::span<const Endpoint> candidates, const Requirements& need);

}  // namespace lse::comm
