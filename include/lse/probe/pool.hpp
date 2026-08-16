// Qualifying a pool: the whole deliverable, in one call.
//
// A device (or a peer reached over a transport) is admitted, and the engine
// MEASURES what it can do before anything is placed on it. The result is what
// the cost model reads, and it is persisted under an identity so a normal
// start is a read rather than a re-measure.
//
// Invalidation is by fingerprint and nothing else. A profile does not expire —
// it stops describing the machine when the machine changes, and the fingerprint
// is what notices: every member's identity, every host, the transport, and the
// toolchain that would compile the kernels. Change any of them and the entry is
// a miss, whatever its age.
#pragma once

#include <span>
#include <string>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/dist/transport.hpp"
#include "lse/probe/link_probe.hpp"
#include "lse/probe/profile.hpp"

namespace lse::probe {

struct PoolMember {
  DeviceId id;
  // The backend this process can drive for this member, or null when the
  // member is a peer some other rank owns. Exactly one rank supplies a backend
  // for a given member: the one whose transport rank matches it.
  backend::IBackend* backend = nullptr;
  dist::Rank rank = 0;
  // Empty means this machine.
  std::string host;
};

struct PoolOptions {
  LinkProbeConfig links;
  // Empty means default_profile_dir().
  std::string profile_dir;
};

// Identical on every rank: the local identities are exchanged before it is
// combined, so all ranks agree on hit or miss and cannot take different paths
// through a collective probe.
[[nodiscard]] Result<std::string> pool_fingerprint(
    std::span<const PoolMember> members, dist::ITransport* transport);

// Reads the persisted profile when the fingerprint matches; measures and
// persists otherwise. Collective when the pool has more than one member —
// every rank calls it.
[[nodiscard]] Result<PoolProfile> qualify_pool(
    std::span<const PoolMember> members, dist::ITransport* transport,
    const PoolOptions& options = {});

}  // namespace lse::probe
