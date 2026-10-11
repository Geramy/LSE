#pragma once

#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

// The fused residual sum and norm graph::exact_residual_norm recognizes.
const graph::KernelPrimitiveBase* residual_norm_kernel();

}  // namespace lse::kernels
