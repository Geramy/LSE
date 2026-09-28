#pragma once

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::dispatch {

[[nodiscard]] inline std::uint32_t selector_walk_wave(
    const graph::KernelShapes& s) {
  if (!s.device || !s.intrinsics) return 0;
  const auto wave = s.device->wavefront_size;
  if ((wave != 32 && wave != 64) ||
      s.device->max_threads_per_workgroup < wave ||
      s.intrinsics->find("wave.shfl_xor").empty() ||
      s.intrinsics->find("rint").empty() ||
      s.intrinsics->find("max").empty() ||
      s.intrinsics->find("neg_inf").empty()) return 0;
  return wave;
}

}  // namespace lse::dispatch
