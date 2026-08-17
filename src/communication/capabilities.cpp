#include "lse/communication/capabilities.hpp"

#include <memory>
#include <string>

#include "lse/communication/transport.hpp"

namespace lse::comm {

namespace {

void append_decline(std::string& out, const Endpoint& ep,
                    std::string_view why) {
  if (!out.empty()) out += "; ";
  out += ep.str();
  out += ": ";
  out += why;
}

std::string describe(const Endpoint& winner, std::uint64_t ns,
                     const std::string& declines) {
  std::string out = winner.str();
  out += " at ~";
  out += std::to_string(ns);
  out += " ns";
  if (!declines.empty()) {
    out += "; passed over ";
    out += declines;
  }
  return out;
}

}  // namespace

std::uint64_t Capabilities::predicted_ns(std::size_t bytes,
                                         bool device_resident) const noexcept {
  // A link that cannot DMA out of device memory pays the byte cost twice: once
  // into a host bounce buffer and once onto the wire.
  const bool staged = device_resident && !moves_device_bytes_in_place();
  const auto wire = static_cast<std::uint64_t>(bytes) * (staged ? 2u : 1u);
  if (bandwidth_bytes_per_s == 0) return latency_ns;
  return latency_ns + wire * 1'000'000'000ull / bandwidth_bytes_per_s;
}

Result<EndpointChoice> select_endpoint(std::span<const Endpoint> candidates,
                                       const Requirements& need) {
  std::size_t best = candidates.size();
  std::uint64_t best_ns = 0;
  std::string declines;

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const Endpoint& ep = candidates[i];
    // Local only: creating a transport touches no network, so selection has no
    // side effect a caller has to undo.
    auto made = create_transport(ep.str());
    if (!made.ok()) {
      append_decline(declines, ep, made.status().message());
      continue;
    }
    const std::unique_ptr<ITransport> t = made.release();
    if (const std::string_view why = t->declined(ep); !why.empty()) {
      append_decline(declines, ep, why);
      continue;
    }
    const Capabilities caps = t->capabilities(ep);
    if (need.needs_reliability && !caps.reliable) {
      append_decline(declines, ep, "declares no reliable delivery");
      continue;
    }
    if (need.needs_ordering && !caps.ordered) {
      append_decline(declines, ep, "declares no ordering");
      continue;
    }
    // max_message_bytes is not a decline: the common layer segments anything
    // larger than a link's frame, so a small frame costs headers rather than
    // forbidding the transfer. Nor is a staged device path — predicted_ns
    // already doubles its byte cost, so it simply loses to a direct one.
    const std::uint64_t ns =
        caps.predicted_ns(need.largest_message_bytes, need.device_resident_payload);
    if (best == candidates.size() || ns < best_ns) {
      if (best != candidates.size()) {
        append_decline(declines, candidates[best],
                       std::to_string(best_ns) + " ns");
      }
      best = i;
      best_ns = ns;
    } else {
      append_decline(declines, ep, std::to_string(ns) + " ns");
    }
  }

  if (best == candidates.size()) {
    return LSE_ERROR(kUnimplemented, "no endpoint serves this transfer: ",
                     declines.empty() ? "(no candidates offered)" : declines);
  }
  return EndpointChoice{best, describe(candidates[best], best_ns, declines)};
}

}  // namespace lse::comm
