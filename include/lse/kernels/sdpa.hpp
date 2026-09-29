#pragma once

#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

[[nodiscard]] const graph::KernelPrimitiveBase* flash_wmma_sdpa();

}  // namespace lse::kernels
