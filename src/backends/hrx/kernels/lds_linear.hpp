#pragma once

#include "lse/graph/kernel_ir.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::backend::hrx_kernels {

// One wave per output column, lanes on consecutive K. `grid` maps tile/row
// onto the launch; otherwise the same body walks every tile in one workgroup.
void emit_gemv(graph::kir::KernelBody& k,
               const graph::kir::Buffer<graph::kir::f32>& x,
               const graph::kir::Buffer<graph::kir::f32>& w,
               const graph::kir::Buffer<graph::kir::f32>* idx, std::uint32_t keep,
               std::uint32_t slot, std::uint32_t N, std::uint32_t K,
               std::uint32_t M, std::uint32_t load_bytes, bool grid,
               std::uint32_t wave);

// Small-M linear through an LDS K-panel. Null when the tile does not fit or
// the device has no workgroup scratch.
const graph::KernelPrimitiveBase* lds_linear_for(const graph::KernelShapes& s);
const graph::KernelPrimitiveBase* lds_linear_indexed_for(
    const graph::KernelShapes& s);

}  // namespace lse::backend::hrx_kernels
