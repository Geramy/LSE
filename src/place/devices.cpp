#include "lse/place/devices.hpp"

#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

#include "lse/graph/graph.hpp"

namespace lse::place {

namespace {

std::string_view trim(std::string_view s) noexcept {
  const auto space = [](char c) { return c == ' ' || c == '\t'; };
  while (!s.empty() && space(s.front())) s.remove_prefix(1);
  while (!s.empty() && space(s.back())) s.remove_suffix(1);
  return s;
}

// The descriptor enumeration published for this device, or an identity-only one
// when the backend has no enumerator. Never invented: an absent enumerator
// leaves every fact kUnknown, which is what it is.
backend::DeviceDescriptor describe(
    const std::map<std::string, std::vector<backend::DeviceDescriptor>,
                   std::less<>>& enumerated,
    const probe::DeviceId& id) {
  const auto it = enumerated.find(id.backend);
  if (it != enumerated.end()) {
    for (const backend::DeviceDescriptor& d : it->second) {
      if (d.ordinal == id.ordinal) return d;
    }
  }
  backend::DeviceDescriptor bare;
  bare.backend = id.backend;
  bare.ordinal = id.ordinal;
  return bare;
}

}  // namespace

Result<std::vector<probe::DeviceId>> parse_selector(std::string_view text) {
  std::vector<probe::DeviceId> out;
  std::string_view rest = trim(text);
  while (!rest.empty()) {
    const std::size_t comma = rest.find(',');
    const std::string_view entry =
        trim(comma == std::string_view::npos ? rest : rest.substr(0, comma));
    if (!entry.empty()) {
      LSE_ASSIGN_OR(probe::DeviceId id, probe::parse_device_id(entry));
      for (const probe::DeviceId& seen : out) {
        if (seen == id) {
          return LSE_ERROR(kInvalidArgument, "device ", id.str(),
                           " is named twice; members are distinct load "
                           "locations, not repeats");
        }
      }
      out.push_back(std::move(id));
    }
    if (comma == std::string_view::npos) break;
    rest = rest.substr(comma + 1);
  }
  return out;
}

struct Devices::Impl {
  // Owns the backends. Every buffer stamped with a member's DeviceIndex is
  // released by the instance held here, so this vector outliving those buffers
  // is the invariant the whole layer rests on.
  std::vector<std::unique_ptr<backend::IBackend>> owned;
  std::vector<Member> members;
  std::string declined;
  probe::PoolProfile profile;
};

Devices::Devices() : impl_(std::make_unique<Impl>()) {}

Devices::~Devices() = default;

Result<std::unique_ptr<Devices>> Devices::open(std::string_view selector) {
  auto set = std::unique_ptr<Devices>(new Devices());
  Impl& impl = *set->impl_;

  LSE_ASSIGN_OR(const std::vector<probe::DeviceId> named,
                parse_selector(selector));

  std::map<std::string, std::vector<backend::DeviceDescriptor>, std::less<>>
      enumerated;
  const auto enumerate_once = [&enumerated](const std::string& name) {
    if (enumerated.contains(name)) return;
    auto found = backend::enumerate_devices(name);
    enumerated.emplace(name, found.ok() ? found.release()
                                        : std::vector<backend::DeviceDescriptor>{});
  };

  const auto adopt = [&](std::unique_ptr<backend::IBackend> be,
                         const probe::DeviceId& id) {
    enumerate_once(id.backend);
    Member m;
    m.id = id;
    m.index = be->device_index();
    m.backend = be.get();
    m.descriptor = describe(enumerated, id);
    impl.owned.push_back(std::move(be));
    impl.members.push_back(std::move(m));
  };

  if (named.empty()) {
    // LSE_DEVICE picks the ordinal; on a mixed box the discrete card and the
    // integrated one are different devices with different memory models.
    int ordinal = 0;
    if (const char* env = std::getenv("LSE_DEVICE")) ordinal = std::atoi(env);

    // Preference order has to survive init, not just construction: a backend
    // builds fine and then fails to come up when the runtime on the loader path
    // is too old. Treating that as fatal would turn preferring the GPU into a
    // hard stop on any machine without one.
    for (const std::string& name : backend::default_backend_order()) {
      auto be = backend::create_backend(name);
      if (!be.ok()) continue;
      auto candidate = be.release();
      if (const Status init = candidate->init(ordinal); !init.ok()) {
        if (!impl.declined.empty()) impl.declined += "; ";
        impl.declined += name + ": " + init.to_string();
        continue;
      }
      adopt(std::move(candidate), probe::DeviceId{name, ordinal});
      return set;
    }
    return LSE_ERROR(kDeviceError, "no backend came up",
                     impl.declined.empty() ? "" : " (",
                     impl.declined, impl.declined.empty() ? "" : ")");
  }

  for (const probe::DeviceId& id : named) {
    auto be = backend::create_backend(id.backend);
    if (!be.ok()) return be.status();
    auto candidate = be.release();
    if (const Status init = candidate->init(id.ordinal); !init.ok()) {
      return LSE_ERROR(kDeviceError, "device ", id.str(),
                       " was asked for and will not come up: ",
                       init.to_string());
    }
    adopt(std::move(candidate), id);
  }
  return set;
}

std::size_t Devices::size() const noexcept { return impl_->members.size(); }

backend::IBackend& Devices::device(std::size_t i) const {
  return *impl_->members[i].backend;
}

std::size_t Devices::primary() const noexcept { return 0; }

std::size_t Devices::member_of(backend::DeviceIndex d) const noexcept {
  if (!d.bound()) return impl_->members.size();
  for (std::size_t i = 0; i < impl_->members.size(); ++i) {
    if (impl_->members[i].index == d) return i;
  }
  return impl_->members.size();
}

std::span<const Member> Devices::members() const noexcept {
  return impl_->members;
}

const Member* Devices::find(const probe::DeviceId& id) const noexcept {
  for (const Member& m : impl_->members) {
    if (m.id == id) return &m;
  }
  return nullptr;
}

std::string_view Devices::declined() const noexcept { return impl_->declined; }

Reach Devices::reach(backend::DeviceIndex held,
                     std::size_t target) const noexcept {
  if (!held.bound()) return Reach::kUnclaimed;
  const std::size_t from = member_of(held);
  if (from >= impl_->members.size() || target >= impl_->members.size()) {
    return Reach::kUnknown;
  }
  if (from == target) return Reach::kSame;

  const Member& src = impl_->members[from];
  const Member& dst = impl_->members[target];
  // The peer row belongs to the device doing the reading, and it is indexed by
  // the ordinal of the device being read. Only meaningful within one backend:
  // two backends number their devices independently.
  if (src.id.backend == dst.id.backend && src.id.ordinal >= 0) {
    const auto o = static_cast<std::size_t>(src.id.ordinal);
    if (o < dst.descriptor.peers.size()) {
      return reach_of(dst.descriptor.peers[o]);
    }
  }
  // Across backends the only founded answer is a link somebody measured.
  if (const probe::LinkProfile* l = impl_->profile.link(src.id, dst.id)) {
    return reach_of(l->path);
  }
  return Reach::kUnknown;
}

Status Devices::may_read(backend::DeviceIndex held, std::size_t target) const {
  const Reach r = reach(held, target);
  if (readable(r)) return OkStatus();
  const std::size_t from = member_of(held);
  const std::string src = from < impl_->members.size()
                              ? impl_->members[from].id.str()
                              : ("device " + std::to_string(held.value));
  const std::string dst = target < impl_->members.size()
                              ? impl_->members[target].id.str()
                              : ("member " + std::to_string(target));
  return LSE_ERROR(kInvalidArgument, "work on ", dst,
                   " cannot read bytes resident on ", src, ": reach is ",
                   std::string(to_string(r)),
                   r == Reach::kStaged
                       ? " — the bytes have to be moved first"
                       : " — nothing here says this read is legal, and a "
                         "kernel that reads the wrong device returns numbers "
                         "that look right");
}

Result<backend::DeviceBuffer> Devices::allocate(std::size_t member,
                                                std::size_t bytes,
                                                backend::MemoryClass cls) {
  if (member >= impl_->members.size()) {
    return LSE_ERROR(kOutOfRange, "this set holds ",
                     std::to_string(impl_->members.size()),
                     " devices; there is no member ", std::to_string(member));
  }
  return impl_->members[member].backend->allocate(bytes, cls);
}

Status Devices::deallocate(backend::DeviceBuffer& buf) {
  if (!buf.valid()) return OkStatus();
  const std::size_t owner = member_of(buf.residency);
  if (owner >= impl_->members.size()) {
    return LSE_ERROR(kInvalidArgument,
                     "this buffer names device ",
                     std::to_string(buf.residency.value),
                     ", which this set does not hold; releasing it through some "
                     "other member would free an allocation under whoever does");
  }
  impl_->members[owner].backend->deallocate(buf);
  return OkStatus();
}

std::vector<probe::PoolMember> Devices::pool_members() const {
  std::vector<probe::PoolMember> out;
  out.reserve(impl_->members.size());
  for (std::size_t i = 0; i < impl_->members.size(); ++i) {
    probe::PoolMember m;
    m.id = impl_->members[i].id;
    m.backend = impl_->members[i].backend;
    // Ranks are member order: with no transport every member is driven from
    // this process, and the fingerprint folds them in rank order.
    m.rank = static_cast<dist::Rank>(i);
    out.push_back(std::move(m));
  }
  return out;
}

Status Devices::qualify(const probe::PoolOptions& options) {
  const std::vector<probe::PoolMember> members = pool_members();
  LSE_ASSIGN_OR(impl_->profile,
                probe::qualify_pool(members, nullptr, options));
  return OkStatus();
}

const probe::PoolProfile& Devices::profile() const noexcept {
  return impl_->profile;
}

namespace {

struct DefaultSet {
  std::mutex mu;
  std::string selector;
  std::unique_ptr<Devices> set;
  Status opened;
  bool built = false;

  // Caller holds the lock.
  void build() {
    if (built) return;
    built = true;
    std::string want = selector;
    if (want.empty()) {
      if (const char* env = std::getenv("LSE_POOL")) want = env;
    }
    auto result = Devices::open(want);
    if (!result.ok()) {
      opened = result.status();
      return;
    }
    set = result.release();
  }
};

DefaultSet& default_set() {
  static DefaultSet d;
  return d;
}

}  // namespace

Status open_default_devices(std::string_view selector) {
  DefaultSet& d = default_set();
  std::lock_guard lock(d.mu);
  if (d.built) {
    if (!d.opened.ok()) return d.opened;
    if (selector.empty() || selector == d.selector) return OkStatus();
    return LSE_ERROR(kAlreadyExists,
                     "the device set is already open; a selector has to be "
                     "given before anything binds a device, or live buffers "
                     "would be stamped with devices the new set does not hold");
  }
  d.selector = std::string(selector);
  d.build();
  return d.opened;
}

Devices* default_devices() {
  DefaultSet& d = default_set();
  std::lock_guard lock(d.mu);
  d.build();
  return d.set.get();
}

namespace {

backend::IDeviceSet* scheduler_device_set() { return default_devices(); }

// Static init, so graph::default_scheduler() sees the set however early it is
// asked for. The factory pointer it writes to is zero-initialized before any
// dynamic initialization runs, so there is no order to get wrong.
[[maybe_unused]] const bool kFactoryRegistered = [] {
  graph::register_device_set_factory(&scheduler_device_set);
  return true;
}();

}  // namespace

}  // namespace lse::place
