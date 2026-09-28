#pragma once
#include "lse/graph/kernel_primitive.hpp"
namespace lse::kernels {
[[nodiscard]] const graph::KernelPrimitiveBase* wmma_q8_linear_for(
    const graph::KernelShapes&, std::uint32_t rows);
}  // namespace lse::kernels
