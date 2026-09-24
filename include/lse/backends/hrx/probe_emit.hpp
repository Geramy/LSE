#pragma once

#include <cstdint>
#include "lse/graph/codegen.hpp"

namespace lse::backend::hrx_kernels {
// Compile-only seam: records the shared calibration bodies, without allocating
// device memory, creating queues, or dispatching work.
Result<graph::EmittedKernel> emit_loom_stream_probe(
    const DeviceInfo& device, std::uint32_t elements, std::uint32_t threads,
    std::uint32_t load_bytes);
Result<graph::EmittedKernel> emit_loom_touch_probe(const DeviceInfo& device);
}  // namespace lse::backend::hrx_kernels
