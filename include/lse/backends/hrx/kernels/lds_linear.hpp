#pragma once

#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::backend::hrx_kernels {

// One wave per output column, lanes on consecutive K. `grid` maps tile/row
// onto the launch; `persist` grid-strides tiles by `persist_wgs` so every
// resident block works. Otherwise the same body walks every tile in one
// workgroup. `persist_wgs` is a literal — HIP gridDim is not reliable here.
void emit_gemv(graph::env::Emit& e,
               const graph::env::In<graph::kir::f32, graph::env::Emit>& x,
               const graph::env::In<graph::kir::f32, graph::env::Emit>& w,
               const graph::env::In<graph::kir::f32, graph::env::Emit>* idx,
               std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
               std::uint32_t K, std::uint32_t M, std::uint32_t load_bytes,
               bool grid, std::uint32_t wave, bool persist = false,
               std::uint32_t persist_wgs = 1);

// Recorder-signature shim for callers that still hold raw kir buffers.
void emit_gemv(graph::kir::KernelBody& k,
               const graph::kir::Buffer<graph::kir::f32>& x,
               const graph::kir::Buffer<graph::kir::f32>& w,
               const graph::kir::Buffer<graph::kir::f32>* idx, std::uint32_t keep,
               std::uint32_t slot, std::uint32_t N, std::uint32_t K,
               std::uint32_t M, std::uint32_t load_bytes, bool grid,
               std::uint32_t wave, bool persist = false,
               std::uint32_t persist_wgs = 1);

// Small-M linear through an LDS K-panel. Null when the tile does not fit or
// the device has no workgroup scratch.
const graph::KernelPrimitiveBase* lds_linear_for(const graph::KernelShapes& s);
const graph::KernelPrimitiveBase* lds_linear_indexed_for(
    const graph::KernelShapes& s);

}  // namespace lse::backend::hrx_kernels
