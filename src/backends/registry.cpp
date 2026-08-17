#include <cstdlib>

#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>

namespace lse::backend {

namespace {

struct Registry {
  std::mutex mu;
  std::map<std::string, BackendFactory, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

}  // namespace

void register_backend(std::string_view name, BackendFactory factory) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(name), factory);
}

Result<std::unique_ptr<IBackend>> create_backend(std::string_view name) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.factories.find(name);
  if (it == r.factories.end()) {
    std::string known;
    for (const auto& [key, _] : r.factories) {
      if (!known.empty()) known += ", ";
      known += key;
    }
    return LSE_ERROR(kNotFound, "no backend named '", std::string(name),
                     "'; available: ", known.empty() ? "(none)" : known);
  }
  return it->second();
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

}  // namespace lse::backend
