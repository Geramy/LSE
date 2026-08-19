#include "lse/graph/codegen.hpp"

#include "lse/graph/graph.hpp"
#include "lse/opt/occupancy.hpp"

#include <algorithm>


namespace lse::graph {

std::uint64_t IKernelEmitter::cache_key(const FusionGroup& group,
                                        const backend::DeviceInfo&) const {
  return group.signature();
}

const backend::KernelResources* CompiledKernel::resources_for(
    std::string_view entry) const noexcept {
  if (resources.empty()) return nullptr;
  if (entry.empty()) {
    return resources.size() == 1 ? &resources.front() : nullptr;
  }
  for (const backend::KernelResources& k : resources) {
    if (k.entry == entry) return &k;
  }
  return nullptr;
}

std::uint16_t ConstantsLayout::add(std::string name, std::uint8_t size) {
  const std::uint16_t offset = static_cast<std::uint16_t>(total_bytes);
  fields.push_back(Field{std::move(name), offset, size});
  total_bytes += size;
  return offset;
}

backend::LaunchDims choose_launch_dims(const FusionGroup& group,
                                       const backend::DeviceInfo& device,
                                       std::uint32_t lds_bytes) {
  std::size_t elements = 1;
  for (const NodePtr& out : group.outputs) {
    elements = std::max(elements, out->element_count());
  }
  if (group.outputs.empty() && !group.nodes.empty()) {
    elements = group.nodes.back()->element_count();
  }

  const std::uint32_t wave =
      device.wavefront_size != 0 ? device.wavefront_size : 64;
  const std::uint32_t cap =
      device.max_threads_per_workgroup ? device.max_threads_per_workgroup : 256;

  // Every legal workgroup size keeps the same number of threads resident on a
  // pool — residency counts *workgroups*, and workgroups x threads is constant
  // — so maximizing it just picks the smallest size every time. That is how a
  // 254M-element fill came to launch 7.9M workgroups of 32 threads. What
  // actually differs is the launch count, so take the largest size that still
  // fits, bounded by the work there is to do.
  std::uint32_t needed = wave;
  while (needed < elements && needed < cap) needed *= 2;

  const opt::DeviceCapacity capacity = opt::DeviceCapacity::of(device);
  // A device that reports nothing about its wave slots is not a device that
  // seats nothing: the residency filter drops out and the size is chosen on
  // the work alone.
  const bool countable = capacity.usable();
  std::uint32_t best = wave;
  for (std::uint32_t threads = wave; threads <= cap && threads <= needed;
       threads *= 2) {
    if (!countable) {
      best = threads;
      continue;
    }
    opt::KernelDemand demand;
    demand.threads = threads;
    demand.lds_bytes = lds_bytes;
    if (opt::occupancy(capacity, demand).seated()) best = threads;
  }

  backend::LaunchDims dims;
  dims.workgroup_size[0] = best;
  dims.workgroup_count[0] =
      static_cast<std::uint32_t>((elements + best - 1) / best);
  dims.subgroup_size = wave;
  return dims;
}

}  // namespace lse::graph
