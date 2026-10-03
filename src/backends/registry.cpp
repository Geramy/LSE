#include <cstdio>
#include <cstdlib>

#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>

namespace lse::backend {

namespace {

struct Entry {
  BackendFactory factory = nullptr;
  DeviceEnumerator enumerator = nullptr;
  StableRefResolver stable_ref_resolver = nullptr;
  RuntimePreparer runtime_preparer = nullptr;
};

struct Registry {
  std::mutex mu;
  std::map<std::string, Entry, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

// Caller holds the lock.
std::string known_names(const Registry& r) {
  std::string known;
  for (const auto& [key, _] : r.factories) {
    if (!known.empty()) known += ", ";
    known += key;
  }
  return known.empty() ? "(none)" : known;
}

}  // namespace

DeviceIndex next_device_index() noexcept {
  // Atomic because two threads may bring up two devices at once, and the whole
  // value of the token is that no two devices ever share one.
  static std::atomic<std::uint32_t> next{1};
  const std::uint32_t claimed = next.fetch_add(1, std::memory_order_relaxed);
  if (claimed > 0xffffu) return kNoDevice;
  return DeviceIndex{static_cast<std::uint16_t>(claimed)};
}

namespace {

// One ledger per memory class. Relaxed is enough for the counts; the peak is
// raised with a CAS so a concurrent pair of allocations cannot lose the larger.
struct Ledger {
  std::atomic<std::uint64_t> live{0};
  std::atomic<std::uint64_t> peak{0};
  std::atomic<std::uint64_t> count{0};

  void charge(std::uint64_t bytes) noexcept {
    const std::uint64_t now = live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    count.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t seen = peak.load(std::memory_order_relaxed);
    while (now > seen &&
           !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
    }
  }
  void credit(std::uint64_t bytes) noexcept {
    live.fetch_sub(bytes, std::memory_order_relaxed);
    count.fetch_sub(1, std::memory_order_relaxed);
  }
};

Ledger& ledger(MemoryClass cls) noexcept {
  static Ledger ledgers[2];
  return ledgers[cls == MemoryClass::kDevice ? 0 : 1];
}

constexpr auto kSites = static_cast<std::size_t>(AllocationSite::kCount);
Ledger& site_ledger(AllocationSite site) noexcept {
  static Ledger ledgers[kSites];
  const auto i = static_cast<std::size_t>(site);
  return ledgers[i < kSites ? i : 0];
}
thread_local AllocationSite t_site = AllocationSite::kOther;

// Owns the backend's own storage, so the allocation lives exactly as long as
// it did before, and returns the charge when the last view lets go.
struct Charged {
  std::shared_ptr<void> storage;
  std::uint64_t bytes;
  MemoryClass cls;
  AllocationSite site;
  ~Charged() {
    ledger(cls).credit(bytes);
    if (cls == MemoryClass::kDevice) site_ledger(site).credit(bytes);
  }
};

struct Trimmers {
  std::mutex mu;
  std::uint64_t next = 0;
  std::map<std::uint64_t, MemoryTrimmer> all;
};
Trimmers& trimmers() {
  static Trimmers t;
  return t;
}

std::string gib(std::uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.2f GiB", static_cast<double>(bytes) / double(1ull << 30));
  return text;
}

}  // namespace

AllocationTotals allocation_totals(MemoryClass cls) noexcept {
  const Ledger& l = ledger(cls);
  return AllocationTotals{l.live.load(std::memory_order_relaxed),
                          l.peak.load(std::memory_order_relaxed),
                          l.count.load(std::memory_order_relaxed)};
}

void track_allocation(DeviceBuffer& buf, std::size_t bytes, MemoryClass cls) {
  if (!buf.storage) return;
  const AllocationSite site = t_site;
  ledger(cls).charge(bytes);
  if (cls == MemoryClass::kDevice) site_ledger(site).charge(bytes);
  void* const address = buf.storage.get();
  // Built in place: a temporary Charged would return the charge as it died.
  std::shared_ptr<Charged> charged(
      new Charged{std::move(buf.storage), static_cast<std::uint64_t>(bytes), cls, site});
  // Aliased, so storage.get() still names what the backend stored there.
  buf.storage = std::shared_ptr<void>(std::move(charged), address);
}

const char* to_string(AllocationSite site) noexcept {
  switch (site) {
    case AllocationSite::kWeights: return "weights";
    case AllocationSite::kKvCache: return "kv_cache";
    case AllocationSite::kState: return "state";
    case AllocationSite::kDraft: return "draft";
    case AllocationSite::kWorkspace: return "workspace";
    case AllocationSite::kConstants: return "constants";
    case AllocationSite::kOutputs: return "outputs";
    case AllocationSite::kOther:
    case AllocationSite::kCount: break;
  }
  return "other";
}

ScopedAllocationSite::ScopedAllocationSite(AllocationSite site, bool only_if_unset) noexcept
    : previous_(t_site) {
  if (!only_if_unset || t_site == AllocationSite::kOther) t_site = site;
}

ScopedAllocationSite::~ScopedAllocationSite() { t_site = previous_; }

std::vector<SiteTotals> allocation_sites() {
  std::vector<SiteTotals> out;
  out.reserve(kSites);
  for (std::size_t i = 0; i < kSites; ++i) {
    const auto site = static_cast<AllocationSite>(i);
    const Ledger& l = site_ledger(site);
    out.push_back(SiteTotals{site, l.live.load(std::memory_order_relaxed),
                             l.peak.load(std::memory_order_relaxed),
                             l.count.load(std::memory_order_relaxed)});
  }
  return out;
}

std::string describe_device_allocations() {
  const auto all = allocation_totals(MemoryClass::kDevice);
  std::string text = gib(all.live) + " live (peak " + gib(all.peak) + ")";
  std::string parts;
  for (const SiteTotals& s : allocation_sites()) {
    if (s.live == 0 && s.peak == 0) continue;
    if (!parts.empty()) parts += ", ";
    parts += std::string(to_string(s.site)) + " " + gib(s.live) + " (peak " + gib(s.peak) + ")";
  }
  return parts.empty() ? text : text + ": " + parts;
}

std::uint64_t register_memory_trimmer(MemoryTrimmer trimmer) {
  Trimmers& t = trimmers();
  std::lock_guard lock(t.mu);
  const auto id = ++t.next;
  t.all.emplace(id, std::move(trimmer));
  return id;
}

void unregister_memory_trimmer(std::uint64_t id) noexcept {
  Trimmers& t = trimmers();
  std::lock_guard lock(t.mu);
  t.all.erase(id);
}

std::size_t trim_device_memory() {
  // Copied out: a trimmer may release buffers whose owners unregister others.
  std::vector<MemoryTrimmer> run;
  {
    Trimmers& t = trimmers();
    std::lock_guard lock(t.mu);
    for (const auto& [id, trimmer] : t.all) {
      (void)id;
      run.push_back(trimmer);
    }
  }
  std::size_t released = 0;
  for (const MemoryTrimmer& trimmer : run) released += trimmer();
  return released;
}

Status out_of_device_memory(std::size_t bytes, const Status& cause) {
  return Status(StatusCode::kOutOfMemory,
                "out of GPU memory allocating " + std::to_string(bytes) + " bytes; the engine holds " +
                    describe_device_allocations() + "; " + std::string(cause.message()));
}

void register_backend(std::string_view name, BackendFactory factory,
                      DeviceEnumerator enumerator,
                      StableRefResolver stable_ref_resolver,
                      RuntimePreparer runtime_preparer) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(name),
                      Entry{factory, enumerator, stable_ref_resolver, runtime_preparer});
}

void prepare_backend_runtimes() {
  std::vector<RuntimePreparer> callbacks;
  {
    Registry& r = registry();
    std::lock_guard lock(r.mu);
    for (const auto& [name, entry] : r.factories) {
      (void)name;
      if (entry.runtime_preparer) callbacks.push_back(entry.runtime_preparer);
    }
  }
  // dlopen may itself register backends; never hold the registry lock here.
  for (const auto callback : callbacks) callback();
}

namespace {
std::map<std::string, std::string, std::less<>>& device_groups() {
  static std::map<std::string, std::string, std::less<>> groups;
  return groups;
}
}  // namespace

void request_device_group(std::string_view name, std::string_view group) {
  device_groups()[std::string(name)] = std::string(group);
}

std::string requested_device_group(std::string_view name) {
  const auto it = device_groups().find(name);
  return it == device_groups().end() ? std::string{} : it->second;
}

Result<std::unique_ptr<IBackend>> create_backend(std::string_view name) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.factories.find(name);
  if (it == r.factories.end()) {
    return LSE_ERROR(kNotFound, "no backend named '", std::string(name),
                     "'; available: ", known_names(r));
  }
  return it->second.factory();
}

std::optional<int> resolve_stable_ref(std::string_view name,
                                      std::string_view pci,
                                      std::string_view uuid) {
  StableRefResolver resolver = nullptr;
  {
    Registry& r = registry();
    std::lock_guard lock(r.mu);
    auto it = r.factories.find(name);
    if (it != r.factories.end()) resolver = it->second.stable_ref_resolver;
  }
  if (resolver == nullptr) return std::nullopt;
  // Outside the lock, for the same reason enumerate_devices is: the resolver
  // may dlopen a vendor runtime and register things of its own while it does.
  return resolver(pci, uuid);
}

Result<std::vector<DeviceDescriptor>> enumerate_devices(std::string_view name) {
  DeviceEnumerator enumerator = nullptr;
  {
    Registry& r = registry();
    std::lock_guard lock(r.mu);
    auto it = r.factories.find(name);
    if (it == r.factories.end()) {
      return LSE_ERROR(kNotFound, "no backend named '", std::string(name),
                       "'; available: ", known_names(r));
    }
    enumerator = it->second.enumerator;
  }
  if (enumerator == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(name),
                     "' cannot say what devices exist without binding one");
  }
  // Outside the lock: an enumerator brings a driver up, and a driver is
  // entitled to register something of its own while it does.
  return enumerator();
}

std::vector<std::string> available_backends() {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::vector<std::string> out;
  out.reserve(r.factories.size());
  for (const auto& [key, _] : r.factories) out.push_back(key);
  return out;
}

std::vector<std::string> default_backend_order() {
  // An explicit choice is the only candidate: falling back from it would hide
  // the very failure the caller asked to see.
  if (const char* forced = std::getenv("LSE_BACKEND")) {
    return {std::string(forced)};
  }
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  std::vector<std::string> out;
  for (std::string_view known : {"hrx", "cpu"}) {
    if (r.factories.count(std::string(known)) != 0) out.emplace_back(known);
  }
  // Anything registered that this order does not know about still gets a turn,
  // after the ones it does.
  for (const auto& [key, _] : r.factories) {
    if (std::find(out.begin(), out.end(), key) == out.end()) out.push_back(key);
  }
  return out;
}

Result<std::unique_ptr<IBackend>> create_default_backend() {
  // LSE_BACKEND names one explicitly. Otherwise the GPU wins where there is
  // one: staging works, and the differential suite passes on device, so the
  // reason this used to prefer the CPU is gone. An op with no device kernel
  // does not force the whole graph back to the host — the scheduler falls back
  // per group — so preferring the GPU costs nothing when coverage is partial.
  // The CPU backend remains the reference the device path is diffed against.
  if (const char* forced = std::getenv("LSE_BACKEND")) {
    auto backend = create_backend(forced);
    if (backend.ok()) return backend;
    return LSE_ERROR(kNotFound, "LSE_BACKEND names '", std::string(forced),
                     "', which is not available: ",
                     backend.status().message());
  }
  for (const std::string& candidate : default_backend_order()) {
    auto backend = create_backend(candidate);
    if (backend.ok()) return backend;
  }
  return LSE_ERROR(kNotFound, "no backends are registered in this build");
}

std::string DeviceInfo::describe() const {
  std::ostringstream os;
  os << name << " [" << arch << "]\n"
     << "  memory   : " << (total_memory >> 20) << " MiB"
     << (unified_memory ? " (unified)" : "") << "\n"
     << "  occupancy: " << compute_units << " CU, "
     << max_threads_per_workgroup << " threads/wg\n";
  if (device_extension<AmdDeviceInfo>(*this) != nullptr) {
    os << "  family   : " << arch_family_name(arch_family(arch))
       << ", wave" << wavefront_size;
    if (lds_bytes_per_workgroup != 0) {
      os << ", " << (lds_bytes_per_workgroup >> 10) << " KiB LDS";
    }
    os << "\n";
  } else if (extension != nullptr && !extension_id.empty()) {
    os << "  vendor   : " << extension_id << " extension present\n";
  }
  return os.str();
}

std::string DeviceDescriptor::id() const {
  return backend + ":" + std::to_string(ordinal);
}

namespace {

// A fact renders as its value, or as the reason there is no value — never as a
// blank or a zero, either of which reads as a measurement.
template <typename T>
std::string rendered(const DeviceFact<T>& fact, std::string_view unit = {}) {
  if (!fact.known()) return std::string(to_string(fact.source));
  std::ostringstream os;
  if constexpr (std::is_same_v<T, bool>) {
    os << (fact.value ? "yes" : "no");
  } else {
    os << fact.value;
  }
  if (!unit.empty()) os << ' ' << unit;
  if (fact.source == FactSource::kDeclared) os << " (declared)";
  return os.str();
}

// An identity string carries no provenance mark: a name or a bus path is not a
// quantity a placement could act on, so the only thing worth saying about one
// is whether it was answered at all.
std::string rendered_text(const DeviceFact<std::string>& fact) {
  if (!fact.known() || fact.value.empty()) {
    return std::string(to_string(fact.source));
  }
  return fact.value;
}

std::string rendered_kib(const DeviceFact<std::uint32_t>& fact) {
  if (!fact.known()) return std::string(to_string(fact.source));
  DeviceFact<std::uint32_t> kib{fact.value >> 10, fact.source};
  return rendered(kib, "KiB");
}

// Bytes as MiB, the unit the rest of this report and DeviceInfo::describe both
// use, so two lines about the same device are comparable by eye.
std::string rendered_mib(const DeviceFact<std::size_t>& fact) {
  if (!fact.known()) return std::string(to_string(fact.source));
  DeviceFact<std::size_t> mib{fact.value >> 20, fact.source};
  return rendered(mib, "MiB");
}

}  // namespace

std::string DeviceDescriptor::describe() const {
  std::ostringstream os;
  os << rendered_text(product) << " [" << rendered_text(arch) << "]\n"
     << "  memory   : total " << rendered_mib(total_memory) << ", free "
     << rendered_mib(free_memory) << ", unified " << rendered(unified_memory)
     << "\n"
     << "  occupancy: " << rendered(compute_units, "CU") << ", "
     << rendered(max_threads_per_workgroup, "threads/wg") << ", wavefront "
     << rendered(wavefront_size) << ", LDS/wg "
     << rendered_kib(lds_bytes_per_workgroup) << "\n"
     << "  queues   : " << rendered(queue_count) << "\n"
     << "  location : uuid " << rendered_text(uuid) << ", pci "
     << rendered_text(pci_path) << "\n";
  os << "  peers    : ";
  if (peers.empty()) {
    os << "no peer query on this backend";
  } else {
    for (std::size_t i = 0; i < peers.size(); ++i) {
      if (i != 0) os << ", ";
      os << backend << ':' << i << ' ' << to_string(peers[i]);
    }
  }
  os << "\n";
  if (!declined.empty()) os << "  declined : " << declined << "\n";
  return os.str();
}

}  // namespace lse::backend
