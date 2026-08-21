#include "lse/place/devices.hpp"

#include <dlfcn.h>

#include <cctype>
#include <optional>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

// A pool entry that names the device by something stable instead of by
// position. Ordinals are not an identity: on an eight-GPU box this engine's
// hrx:4, the KFD's node 6 and rocm-smi's GPU[6] are three different numbers for
// three different cards, and picking the wrong one is silent -- it runs, on
// somebody else's GPU. A PCI address or a UUID says which card and cannot be
// read as any other.
//
//   hrx:pci:0000:87:00.0     hrx:uuid:eabd6af237d499cb
//
// The address keeps its own colons, so only the two prefixes are split on.
struct StableRef {
  std::string backend;
  std::string pci;   // empty unless this entry named one
  std::string uuid;  // ditto
};

std::optional<StableRef> parse_stable_ref(std::string_view entry) {
  const std::size_t first = entry.find(':');
  if (first == std::string_view::npos) return std::nullopt;
  const std::string_view rest = entry.substr(first + 1);
  StableRef out;
  out.backend = std::string(entry.substr(0, first));
  if (rest.rfind("pci:", 0) == 0) {
    out.pci = std::string(rest.substr(4));
  } else if (rest.rfind("uuid:", 0) == 0) {
    out.uuid = std::string(rest.substr(5));
  } else {
    return std::nullopt;
  }
  if (out.backend.empty() || (out.pci.empty() && out.uuid.empty())) {
    return std::nullopt;
  }
  return out;
}

bool ieq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto la = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
    const auto lb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
    if (la != lb) return false;
  }
  return true;
}

// A UUID is quoted in several shapes -- GPU-<hex>, 0x<hex>, bare hex -- so the
// comparison is on the hex, not on the decoration around it.
std::string uuid_digits(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (std::isxdigit(static_cast<unsigned char>(c)) != 0) {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  if (out.rfind("0", 0) == 0 && text.size() > 1 &&
      (text[1] == 'x' || text[1] == 'X')) {
    out.erase(0, 1);
  }
  return out;
}

Result<int> ordinal_for(const StableRef& ref) {
  auto found = backend::enumerate_devices(ref.backend);
  if (!found.ok()) return found.status();
  const std::vector<backend::DeviceDescriptor> devices = found.release();
  std::string seen;
  for (const backend::DeviceDescriptor& d : devices) {
    const std::string pci = d.pci_path.value_or("");
    const std::string uuid = d.uuid.value_or("");
    if (!seen.empty()) seen += ", ";
    seen += pci.empty() ? "?" : pci;
    if (!ref.pci.empty() && ieq(pci, ref.pci)) return d.ordinal;
    if (!ref.uuid.empty() && !uuid.empty() &&
        uuid_digits(uuid) == uuid_digits(ref.uuid)) {
      return d.ordinal;
    }
  }
  return LSE_ERROR(kNotFound, "no ", ref.backend, " device at ",
                   ref.pci.empty() ? ref.uuid : ref.pci,
                   "; this build sees ", seen.empty() ? "none" : seen);
}

Result<std::vector<probe::DeviceId>> parse_selector(std::string_view text) {
  std::vector<probe::DeviceId> out;
  std::string_view rest = trim(text);
  while (!rest.empty()) {
    const std::size_t comma = rest.find(',');
    const std::string_view entry =
        trim(comma == std::string_view::npos ? rest : rest.substr(0, comma));
    if (!entry.empty()) {
      probe::DeviceId id;
      if (const std::optional<StableRef> ref = parse_stable_ref(entry)) {
        LSE_ASSIGN_OR(const int ordinal, ordinal_for(*ref));
        id.backend = ref->backend;
        id.ordinal = ordinal;
      } else {
        LSE_ASSIGN_OR(id, probe::parse_device_id(entry));
      }
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

// The GPU runtime is dlopened by soname from inside the backend, and a soname
// lookup does not see this executable's RUNPATH -- that only covers an object's
// own direct dependencies. So the loader takes whichever libhsa-runtime64 the
// default search finds, which on a machine with a distro ROCm beside a newer
// one is the old one, and the backend then declines for a missing symbol and
// the whole model quietly runs on the host interpreter.
//
// Loading it here by absolute path first settles it: the object is in the
// process under its own soname before anything asks for that soname, so the
// backend's dlopen returns this one instead of searching. Requiring the caller
// to have exported LD_LIBRARY_PATH is not a working answer -- nobody launching
// the server knows to.
void preload_gpu_runtime() {
  namespace fs = std::filesystem;
  constexpr const char* kHsaSoname = "libhsa-runtime64.so.1";

  // Somebody already put it in the process. That somebody may be a profiler
  // that intends to intercept every HSA call, and loading a second copy beside
  // an interposer takes the calls out from under it: the backend then fails to
  // come up and the whole model runs on the host interpreter, which reads as
  // the profiler showing no GPU work at all. RTLD_NOLOAD asks without loading.
  //
  // In an ordinary run this finds nothing, because the backend dlopens the
  // runtime lazily during init and init has not happened yet.
  if (void* already = dlopen(kHsaSoname, RTLD_NOLOAD | RTLD_NOW);
      already != nullptr) {
    dlclose(already);
    return;
  }

  // Already in the process under some other name, and new enough: nothing to
  // do. A build linked directly against a good one lands here.
  if (void* self = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL); self != nullptr) {
    const bool ok = dlsym(self, "hsa_amd_vmem_address_reserve_align") != nullptr;
    dlclose(self);
    if (ok) return;
  }

  std::vector<std::string> roots;
  if (const char* env = std::getenv("ROCM_PATH"); env != nullptr && *env != 0) {
    roots.emplace_back(env);
  }
#ifdef LSE_CONFIGURED_ROCM_PATH
  // Where this build was configured to find ROCm. A machine that needed
  // -DLSE_ROCM_PATH to compile needs the same answer to run.
  roots.emplace_back(LSE_CONFIGURED_ROCM_PATH);
#endif
  roots.emplace_back("/opt/rocm");
  // A versioned or componentised install beside the unversioned link, newest
  // first so a machine carrying several picks the one most likely to have the
  // symbol rather than the first it trips over.
  std::vector<std::string> globbed;
  for (const char* base : {"/opt", "/opt/rocm"}) {
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(base, ec)) {
      const std::string name = e.path().filename().string();
      if (name.rfind("rocm-", 0) == 0 || name.rfind("core-", 0) == 0) {
        globbed.push_back(e.path().string());
      }
    }
  }
  std::sort(globbed.rbegin(), globbed.rend());
  roots.insert(roots.end(), globbed.begin(), globbed.end());

  for (const std::string& root : roots) {
    const std::string path = root + "/lib/" + kHsaSoname;
    std::error_code ec;
    if (!fs::exists(path, ec)) continue;
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) continue;
    if (dlsym(h, "hsa_amd_vmem_address_reserve_align") != nullptr) return;
    // Held open on purpose when it is wrong: closing it can unload something
    // a later candidate already pulled in. The process keeps one either way.
  }
}

}  // namespace

Status open_default_devices(std::string_view selector) {
  static const bool preloaded = [] { preload_gpu_runtime(); return true; }();
  (void)preloaded;
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
