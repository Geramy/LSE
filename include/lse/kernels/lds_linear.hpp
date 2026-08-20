#pragma once

#include <string>

#include "lse/core/elem.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

// One wave per output column, lanes on consecutive K. `grid` maps tile/row
// onto the launch; `persist` grid-strides tiles by `persist_wgs` so every
// resident block works. Otherwise the same body walks every tile in one
// workgroup. `persist_wgs` is a literal — HIP gridDim is not reliable here.
//
// `W` is the weight's storage element. It is read in that format and widened
// in register, so the K loop moves half the bytes for a bf16 checkpoint.
//
// `staged` is the activation row the emitter has already put in workgroup
// scratch for a whole fused run. Given one, the grid arm reads it instead of
// staging its own — no second array, no second fill, no second barrier.
template <class W>
void emit_gemv(graph::env::Emit& e,
               const graph::env::In<graph::kir::f32, graph::env::Emit>& x,
               const graph::env::In<W, graph::env::Emit>& w,
               const graph::env::In<graph::kir::f32, graph::env::Emit>* idx,
               std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
               std::uint32_t K, std::uint32_t M, std::uint32_t load_bytes,
               bool grid, std::uint32_t wave, bool persist = false,
               std::uint32_t persist_wgs = 1,
               graph::StagedPanel staged = {});

// Recorder-signature shim for callers that still hold raw kir buffers.
template <class W>
void emit_gemv(graph::kir::KernelBody& k,
               const graph::kir::Buffer<graph::kir::f32>& x,
               const graph::kir::Buffer<W>& w,
               const graph::kir::Buffer<graph::kir::f32>* idx, std::uint32_t keep,
               std::uint32_t slot, std::uint32_t N, std::uint32_t K,
               std::uint32_t M, std::uint32_t load_bytes, bool grid,
               std::uint32_t wave, bool persist = false,
               std::uint32_t persist_wgs = 1,
               graph::StagedPanel staged = {});

#define LSE_GEMV_EXTERN(W_)                                                    \
  extern template void emit_gemv<W_>(                                          \
      graph::env::Emit&,                                                       \
      const graph::env::In<graph::kir::f32, graph::env::Emit>&,                \
      const graph::env::In<W_, graph::env::Emit>&,                             \
      const graph::env::In<graph::kir::f32, graph::env::Emit>*, std::uint32_t, \
      std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,              \
      std::uint32_t, bool, std::uint32_t, bool, std::uint32_t,                 \
      graph::StagedPanel);                                                     \
  extern template void emit_gemv<W_>(                                          \
      graph::kir::KernelBody&, const graph::kir::Buffer<graph::kir::f32>&,     \
      const graph::kir::Buffer<W_>&,                                           \
      const graph::kir::Buffer<graph::kir::f32>*, std::uint32_t,               \
      std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,              \
      std::uint32_t, bool, std::uint32_t, bool, std::uint32_t,                 \
      graph::StagedPanel);
LSE_GEMV_EXTERN(graph::kir::f32)
LSE_GEMV_EXTERN(lse::bf16)
LSE_GEMV_EXTERN(lse::f16)
#undef LSE_GEMV_EXTERN

// Declare one activation panel in workgroup scratch and fill it: `count` f32
// elements of `x` holding row `workgroup_id_y`, then the barrier that publishes
// it. Records into `k`, which the caller has open.
//
// This is the emitter's half of the sharing contract in KernelShapes::staged —
// it lives here because it has to stay the same fill the GEMV above would have
// written for itself. `block` is the run's workgroup size. Returns the array's
// name, or empty when the panel does not fit `k`'s LDS budget.
[[nodiscard]] std::string emit_staged_row(
    graph::kir::KernelBody& k, const graph::kir::Buffer<graph::kir::f32>& x,
    std::uint32_t count, std::uint32_t rows, std::uint32_t block);

// The int8 staging a run could hoist for one panel: the spec every stage
// reading that row would quantize under, or bits == 0 when the integer path
// does not apply. Two stages whose plans differ must stage for themselves.
struct DotStagingPlan {
  std::uint32_t count = 0;
  std::int32_t group_size = 0;
  std::int32_t bits = 0;

  [[nodiscard]] bool valid() const noexcept { return bits != 0; }
  [[nodiscard]] bool operator==(const DotStagingPlan&) const noexcept = default;
};
[[nodiscard]] DotStagingPlan dot_staging_plan(const graph::KernelShapes& s,
                                              bool indexed);

// Declare the int8 form of an already-staged row in workgroup scratch and fill
// it: codes, one step per chunk, and one activation sum per group, then the
// barrier that publishes them. The other half of the KernelShapes::staged_quant
// contract, and the same fill quant_linear would otherwise write per stage.
// `row` is the panel emit_staged_row returned. Names are empty when the tiles
// do not fit or the spec does not divide.
struct StagedQuantNames {
  std::string codes;
  std::string scale;
  std::string sum;
};
[[nodiscard]] StagedQuantNames emit_staged_dot_acts(
    graph::kir::KernelBody& k, std::string_view row, std::uint32_t count,
    std::int32_t group_size, std::uint32_t rows, std::uint32_t block);

// Workgroup scratch emit_staged_dot_acts declares for one panel, in bytes, and
// 0 under exactly the conditions that make it decline. Beside the call that
// declares those arrays because a fusion's residency is counted from this
// figure: the two drifting apart is a run admitted against a launch that never
// happens.
[[nodiscard]] std::uint32_t staged_dot_bytes(std::uint32_t count,
                                             std::int32_t group_size,
                                             std::uint32_t block);

// Small-M linear through an LDS K-panel. Null when the tile does not fit or
// the device has no workgroup scratch.
const graph::KernelPrimitiveBase* lds_linear_for(const graph::KernelShapes& s);
const graph::KernelPrimitiveBase* lds_linear_indexed_for(
    const graph::KernelShapes& s);

}  // namespace lse::kernels
