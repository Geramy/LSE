#pragma once

#include "lse/graph/kernel_primitive.hpp"

namespace lse::backend::hrx_kernels {

// The matrix-core form of `linear` for this invocation, or null when the
// device, the shapes or the opt-in switch rule it out and the scalar loop
// should stand.
const graph::KernelPrimitiveBase* wmma_linear_for(const graph::KernelShapes& s);

}  // namespace lse::backend::hrx_kernels
