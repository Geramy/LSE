#include "lse/probe/pool.hpp"

#include <cstdio>
#include <set>

#include "lse/core/hash.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/probe/device_probe.hpp"
#include "lse/probe/profile_store.hpp"
#include "lse/probe/wire.hpp"

namespace lse::probe {

namespace {

constexpr int kTagFingerprint = 0x60;
constexpr int kTagProfile = 0x62;

std::string hex64(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(v));
  return buf;
}

// Everything about the local member that would change what a measurement
// means: which device it is, how much of it there is, and which toolchain would
// build the kernels that run on it.
std::string local_identity(const PoolMember& member) {
  if (member.backend == nullptr) return "absent";
  const backend::DeviceInfo& info = member.backend->device_info();
  std::string s;
  s += member.backend->name();
  s += '|';
  s += info.arch;
  s += '|';
  s += info.name;
  s += '|';
  s += std::to_string(info.total_memory);
  s += '|';
  s += std::to_string(info.compute_units);
  s += '|';
  s += std::to_string(static_cast<int>(info.ordinal));
  s += '|';
  const graph::IKernelCompiler* compiler = member.backend->compiler();
  s += compiler != nullptr ? compiler->identity() : std::string("no-compiler");
  return s;
}

// Which members this process drives.
//
// With a transport there is exactly one — the member whose rank is ours, the
// rest belonging to other ranks. Without one, every member that carries a
// backend is ours: several devices in one process is not a degenerate
// distributed pool, it is the ordinary multi-GPU box, and requiring a transport
// to describe it would mean standing up a network to measure a PCIe slot.
Result<std::vector<std::size_t>> local_indices(
    std::span<const PoolMember> members, dist::ITransport* transport) {
  std::vector<std::size_t> out;
  if (transport == nullptr) {
    for (std::size_t i = 0; i < members.size(); ++i) {
      if (members[i].backend != nullptr) out.push_back(i);
    }
    if (out.empty()) {
      return LSE_ERROR(kInvalidArgument,
                       "no member of this pool carries a backend, so nothing "
                       "here can measure any of them");
    }
    return out;
  }
  const dist::Rank me = transport->rank();
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (members[i].rank == me) {
      out.push_back(i);
      return out;
    }
  }
  return LSE_ERROR(kOutOfRange, "transport rank ", std::to_string(me),
                   " names no pool member");
}

Status validate(std::span<const PoolMember> members) {
  if (members.empty()) {
    return LSE_ERROR(kInvalidArgument, "an empty pool has nothing to qualify");
  }
  std::set<std::string> ids;
  for (const PoolMember& m : members) {
    if (m.id.backend.empty()) {
      return LSE_ERROR(kInvalidArgument,
                       "a pool member must be addressed backend-qualified, "
                       "never as a bare ordinal");
    }
    if (!ids.insert(m.id.str()).second) {
      return LSE_ERROR(kInvalidArgument, "pool member ", m.id.str(),
                       " appears twice; members are distinct load locations");
    }
  }
  return OkStatus();
}

std::vector<LinkMember> link_members(std::span<const PoolMember> members) {
  std::vector<LinkMember> out;
  out.reserve(members.size());
  for (const PoolMember& m : members) {
    out.push_back(LinkMember{m.id, m.rank,
                             m.host.empty() ? host_identity() : m.host});
  }
  return out;
}

// The profile a member has before anything is measured: who it is, and nothing
// else. Every number stays kUnknown.
DeviceProfile bare_profile(const PoolMember& member) {
  DeviceProfile p;
  p.id = member.id;
  return p;
}

}  // namespace

Result<std::string> pool_fingerprint(std::span<const PoolMember> members,
                                     dist::ITransport* transport) {
  LSE_RETURN_IF_ERROR(validate(members));
  LSE_ASSIGN_OR(const std::vector<std::size_t> locals,
                local_indices(members, transport));
  const std::size_t self = locals.front();

  const std::string mine = hex64(hash_bytes(local_identity(members[self])));
  std::vector<std::string> per_rank(members.size(), mine);
  // Every member this process drives contributes its own identity, so two local
  // devices cannot fingerprint as one. A pool of one is unchanged: `mine` is
  // already its identity and the loop rewrites it with the same value.
  if (transport == nullptr) {
    for (const std::size_t i : locals) {
      const auto r = static_cast<std::size_t>(members[i].rank);
      if (r < per_rank.size()) {
        per_rank[r] = hex64(hash_bytes(local_identity(members[i])));
      }
    }
  }
  if (transport != nullptr && transport->world_size() > 1) {
    LSE_ASSIGN_OR(per_rank,
                  wire::all_gather_strings(*transport, mine, kTagFingerprint));
    if (per_rank.size() != members.size()) {
      return LSE_ERROR(kInternal, "fingerprint gather returned ",
                       std::to_string(per_rank.size()), " of ",
                       std::to_string(members.size()), " ranks");
    }
  }

  std::uint64_t h = hash_bytes(transport != nullptr ? transport->name()
                                                    : std::string_view("none"));
  h = hash_mix(h, members.size());
  // Rank order, not member order: every rank must fold the same list in the
  // same sequence or they would key the same pool differently.
  for (dist::Rank r = 0; r < static_cast<dist::Rank>(members.size()); ++r) {
    for (const PoolMember& m : members) {
      if (m.rank != r) continue;
      h = hash_bytes(m.id.str(), h);
      h = hash_bytes(m.host.empty() ? host_identity() : m.host, h);
      h = hash_bytes(per_rank[static_cast<std::size_t>(r)], h);
      h = hash_mix(h, static_cast<std::uint64_t>(r));
    }
  }
  return hex64(h);
}

Result<PoolProfile> qualify_pool(std::span<const PoolMember> members,
                                 dist::ITransport* transport,
                                 const PoolOptions& options) {
  LSE_RETURN_IF_ERROR(validate(members));
  LSE_ASSIGN_OR(const std::vector<std::size_t> locals,
                local_indices(members, transport));
  const std::size_t self = locals.front();
  LSE_ASSIGN_OR(const std::string fingerprint,
                pool_fingerprint(members, transport));

  const std::string dir =
      options.profile_dir.empty() ? default_profile_dir() : options.profile_dir;

  // Every rank agrees on the fingerprint, so every rank agrees on hit or miss
  // and none of them can enter a collective the others skipped.
  if (auto cached = load_pool_profile(fingerprint, dir);
      cached.ok() && cached->devices.size() == members.size()) {
    bool matches = true;
    for (std::size_t i = 0; i < members.size(); ++i) {
      if (!(cached->devices[i].id == members[i].id)) matches = false;
    }
    if (matches) {
      PoolProfile pool = cached.release();
      // Every other number in a profile is a property of the hardware and
      // holds until the fingerprint changes. Free memory is a property of the
      // moment — it moves with every other process on the box — and the
      // fingerprint is deliberately blind to exactly that. Replayed, it would
      // reach capacity_for labelled kMeasured with no way to tell it is hours
      // old, and a stale pass is worse than the refusal an unknown produces.
      // So the stored figure is dropped for every member, and only the one
      // something can sample right now gets a figure back. A peer stays
      // unknown until a rank that can reach it probes again, which is the same
      // refusal an unmeasured device already gets.
      for (DeviceProfile& d : pool.devices) d.free_memory = Measured::unknown();
      for (const std::size_t i : locals) {
        if (members[i].backend == nullptr) continue;
        pool.devices[i].free_memory = sample_free_memory(*members[i].backend);
      }
      return pool;
    }
  }

  PoolProfile pool;
  pool.fingerprint = fingerprint;
  pool.devices.resize(members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    pool.devices[i] = bare_profile(members[i]);
  }
  // Every device this process drives is measured here. A member some other rank
  // owns stays bare until that rank sends its profile over.
  for (const std::size_t i : locals) {
    if (members[i].backend == nullptr) continue;
    auto probed = probe_device(*members[i].backend);
    if (!probed.ok()) continue;
    pool.devices[i] = probed.release();
    // The pool addresses this member by the name the caller gave it, which may
    // differ from the backend's own ordinal when a rank drives one of several
    // devices.
    pool.devices[i].id = members[i].id;
  }
  const DeviceProfile& local = pool.devices[self];

  if (transport != nullptr && transport->world_size() > 1) {
    LSE_ASSIGN_OR(
        const std::vector<std::string> encoded,
        wire::all_gather_strings(*transport, serialize_device_profile(local),
                                 kTagProfile));
    for (std::size_t i = 0; i < members.size(); ++i) {
      const auto r = static_cast<std::size_t>(members[i].rank);
      if (r >= encoded.size() || encoded[r].empty()) continue;
      auto parsed = parse_device_profile(encoded[r]);
      if (!parsed.ok()) continue;
      DeviceProfile d = parsed.release();
      d.id = members[i].id;
      pool.devices[i] = std::move(d);
    }
  }

  if (transport != nullptr) {
    const std::vector<LinkMember> lm = link_members(members);
    LSE_ASSIGN_OR(pool.links, probe_links(*transport, lm, options.links));
  } else {
    std::vector<LocalMember> lm;
    lm.reserve(members.size());
    for (const PoolMember& m : members) {
      lm.push_back(LocalMember{m.id, m.backend});
    }
    LSE_ASSIGN_OR(pool.links, probe_local_links(lm, options.links));
  }

  // Best effort: a pool that measured fine but cannot write its cache is still
  // a qualified pool.
  (void)save_pool_profile(pool, dir);
  return pool;
}

}  // namespace lse::probe
