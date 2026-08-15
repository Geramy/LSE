#include "lse/model/registry.hpp"

#include <algorithm>
#include <mutex>

namespace lse::model {

namespace {

struct Registry {
  std::mutex mutex;
  std::vector<ModelArch> archs;
};

Registry& registry() {
  static Registry r;
  return r;
}

}  // namespace

void register_model_arch(ModelArch arch) {
  Registry& r = registry();
  const std::lock_guard<std::mutex> lock(r.mutex);
  r.archs.push_back(std::move(arch));
}

std::vector<std::string_view> registered_architectures() {
  Registry& r = registry();
  const std::lock_guard<std::mutex> lock(r.mutex);
  std::vector<std::string_view> out;
  out.reserve(r.archs.size());
  for (const ModelArch& a : r.archs) out.push_back(a.name);
  return out;
}

Result<const ModelArch*> detect_architecture(const Config& config,
                                             const SafeTensors& weights) {
  Registry& r = registry();
  const std::lock_guard<std::mutex> lock(r.mutex);
  if (r.archs.empty()) {
    return LSE_ERROR(kNotFound,
                     "no model architectures are registered in this build");
  }

  const ModelArch* best = nullptr;
  for (const ModelArch& a : r.archs) {
    if (!a.matches || !a.matches(config, weights)) continue;
    if (best == nullptr || a.priority > best->priority) best = &a;
  }
  if (best != nullptr) return best;

  std::string tried;
  for (const ModelArch& a : r.archs) {
    if (!tried.empty()) tried += ", ";
    tried += std::string(a.name);
  }
  return LSE_ERROR(kNotFound,
                   "no registered architecture recognizes this checkpoint "
                   "(tried: ", tried, ")");
}

Result<std::unique_ptr<HybridLM>> build_model(const Config& config,
                                              const SafeTensors& weights,
                                              std::string_view override_name) {
  if (!override_name.empty()) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (const ModelArch& a : r.archs) {
      if (a.name == override_name) return a.build(config);
    }
    return LSE_ERROR(kNotFound, "no architecture named '",
                     std::string(override_name), "'");
  }

  LSE_ASSIGN_OR(const ModelArch* arch, detect_architecture(config, weights));
  return arch->build(config);
}

}  // namespace lse::model
