#pragma once

#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

// The matrix-shaped attention kernel for this invocation, or null when the
// pass has no query tile to share a key across.
[[nodiscard]] const graph::KernelPrimitiveBase* flash_sdpa_for(
    const graph::KernelShapes& s);

}  // namespace lse::kernels
