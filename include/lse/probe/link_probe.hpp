// Measuring what it costs to move bytes between two pool members.
//
// Over the transport seam, so one implementation covers a peer on the same PCIe
// root, a peer reached by RDMA, and N ranks in one process — the probe never
// asks which transport it is holding. What it asks is the transport's declared
// Capabilities and the topology, and from those it names the PATH; what it
// measures is the cost.
//
// The result is a matrix over ORDERED pairs. A link can be faster one way than
// the other, so (a,b) and (b,a) are separate records and each direction is
// timed by the rank that sends it, never inferred from the reverse.
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/dist/transport.hpp"
#include "lse/probe/profile.hpp"

namespace lse::probe {

struct LinkProbeConfig {
  // Sizes the fit is taken over. At least two distinct ones, or latency and
  // bandwidth cannot be separated and both are reported unknown.
  std::span<const std::size_t> sizes = default_transfer_sizes();
  // Round trips timed for the intercept.
  int latency_reps = 32;
  // Bytes one direction of one size may move. Repetition count falls out of
  // it, so a 4 MB size does not cost 16x what a 64 B size does.
  std::size_t byte_budget = 4u << 20;
};

struct LinkMember {
  DeviceId id;
  dist::Rank rank = 0;
  // Which machine this member is on. Two members sharing it can have a direct
  // peer copy; two that do not need the fabric. Asked of the topology, never of
  // the transport's name — a name is not a capability.
  std::string host;
};

// The path a pair has, from capabilities and topology alone.
[[nodiscard]] PathKind classify_path(const LinkMember& src,
                                     const LinkMember& dst,
                                     const dist::Capabilities& caps) noexcept;

// This machine's identity, for LinkMember::host.
[[nodiscard]] std::string host_identity();

// Every rank calls this with the same member list. Ranks pair off in a fixed
// schedule — at most one partner per round, so nothing shares the fabric with
// anything else and no pair can deadlock — and each rank times the directions
// it sends. The rows are then exchanged, so every rank returns the same
// complete matrix, row-major over `members`.
[[nodiscard]] Result<std::vector<LinkProfile>> probe_links(
    dist::ITransport& transport, std::span<const LinkMember> members,
    const LinkProbeConfig& cfg = {});

// One process, several devices, no transport between them.
//
// The case two GPUs in one box actually are, and the one probe_links cannot
// serve: there is no rank to pair off with and no fabric to send over, so the
// measurement has to be made by the process that drives both ends.
//
// What it times is the move the engine can actually make TODAY: out of the
// source device's memory to host, and from there into the destination's. That
// is why the path is kHostStaged and not kPeerDirect — no peer copy exists at
// this seam yet, and reporting one would put a link in the cost model that
// nothing can use. When a device-to-device path lands, this measures that
// instead and the path it reports changes with it.
//
// A member with no backend cannot be driven from here, so its pairs stay
// kUnknown rather than being credited with the mechanism that was not tried.
struct LocalMember {
  DeviceId id;
  backend::IBackend* backend = nullptr;
};

[[nodiscard]] Result<std::vector<LinkProfile>> probe_local_links(
    std::span<const LocalMember> members, const LinkProbeConfig& cfg = {});

}  // namespace lse::probe
