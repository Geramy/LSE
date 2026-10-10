#include "lse/dispatch/roofline.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace lse::dispatch {
namespace {

struct Registry {
  std::mutex mu;
  std::map<std::pair<std::string, int>, Roofline> rooflines;
};

Registry& registry() {
  static Registry r;
  return r;
}

std::pair<std::string, int> key_of(const backend::DeviceInfo& device) {
  return {device.arch, static_cast<int>(device.ordinal)};
}

}  // namespace

void publish_roofline(const backend::DeviceInfo& device, const Roofline& roofline) {
  Registry& r = registry();
  std::lock_guard held(r.mu);
  r.rooflines[key_of(device)] = roofline;
}

std::optional<Roofline> roofline(const backend::DeviceInfo* device) {
  if (device == nullptr) return std::nullopt;
  Registry& r = registry();
  std::lock_guard held(r.mu);
  const auto it = r.rooflines.find(key_of(*device));
  if (it == r.rooflines.end() || !it->second.valid()) return std::nullopt;
  return it->second;
}

double predict_ns(const Roofline& roofline, const KernelWork& work) noexcept {
  const double stream = roofline.stream_bytes_per_s > 0.0
      ? work.bytes * 1e9 / roofline.stream_bytes_per_s : 0.0;
  double compute = roofline.vector_flops > 0.0
      ? work.vector_flops * 1e9 / roofline.vector_flops : 0.0;
  if (work.matrix_flops > 0.0)
    compute += roofline.matrix_flops > 0.0 ? work.matrix_flops * 1e9 / roofline.matrix_flops
             : roofline.vector_flops > 0.0 ? work.matrix_flops * 1e9 / roofline.vector_flops : 0.0;
  return std::max(stream, compute) + work.launches * roofline.launch_ns;
}

}  // namespace lse::dispatch
