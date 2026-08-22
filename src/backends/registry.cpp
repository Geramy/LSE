#include <cstdlib>

#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <sstream>
#include <type_traits>

namespace lse::backend {

namespace {

struct Entry {
  BackendFactory factory = nullptr;
  DeviceEnumerator enumerator = nullptr;
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

void register_backend(std::string_view name, BackendFactory factory,
                      DeviceEnumerator enumerator) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(name), Entry{factory, enumerator});
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
