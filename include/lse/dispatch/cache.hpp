#pragma once
#include "lse/graph/graph.hpp"
#include "lse/graph/view.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/dispatch/attention.hpp"
#include <vector>

namespace lse::dispatch {
[[nodiscard]] inline std::uint64_t specialization_cache_key(
    std::uint64_t key, const graph::FusionGroup& group,
    const backend::DeviceInfo& device, graph::kir::TypeTable types,
    const graph::DialectSourceTable& intrinsics) {
  if (graph::group_bindings_may_alias(group)) {
    key ^= implementation_id("bindings.aliases.v1");
    key *= 1099511628211ull;
  }
  key ^= implementation_id("dispatch.tables.v1");
  key *= 1099511628211ull;
  key ^= kTableRevision;
  key *= 1099511628211ull;
  std::vector<Shape> inputs;
  std::vector<DType> dtypes;
  for (const auto& node : group.nodes) {
    if (node->kind != graph::OpKind::kQuantMatMul &&
        node->kind != graph::OpKind::kRMS &&
        node->kind != graph::OpKind::kL2Norm &&
        node->kind != graph::OpKind::kAttention) continue;
    const auto* primitive = dynamic_cast<const graph::KernelPrimitiveBase*>(node->prim);
    if (!primitive) continue;
    inputs.clear(); dtypes.clear();
    for (const auto& input : node->inputs) {
      inputs.push_back(input->shape); dtypes.push_back(input->dtype);
    }
    graph::KernelShapes probe;
    probe.inputs=inputs; probe.input_dtypes=dtypes;
    probe.output=node->shape; probe.output_dtype=node->dtype;
    probe.attrs=node->attrs; probe.iattrs=node->iattrs;
    probe.device=&device; probe.types=types; probe.intrinsics=&intrinsics;
    if (node->kind == graph::OpKind::kRMS && group.is_phase) {
      key ^= implementation_id(phase_cooperative_rms_supported(probe)
          ? "rms_norm.phase-cooperative.v1" : "rms_norm.phase-scalar.v1");
      key *= 1099511628211ull;
      continue;
    }
    if (node->kind == graph::OpKind::kL2Norm && group.is_phase) {
      key ^= implementation_id("l2_normalize.phase-scalar.v1");
      key *= 1099511628211ull;
      continue;
    }
    if (node->kind == graph::OpKind::kQuantMatMul) {
      auto quant_probe = probe;
      if (inputs.size() >= 4) {
        quant_probe.inputs = probe.inputs.first(4);
        quant_probe.input_dtypes = probe.input_dtypes.first(4);
      }
      const auto plan = quant_plan(quant_probe, false);
      auto mix = [&](std::uint64_t value) { key ^= value; key *= 1099511628211ull; };
      mix(static_cast<std::uint64_t>(plan.implementation));
      mix(plan.int8_activations);
      if (inputs.size() == 5 && primitive->name() == "quant_linear.q4_gemm_f16.v1") {
        mix(implementation_id("quant.q4.gemm-f16.v1"));
      } else if (inputs.size() == 5 && primitive->name() == "quant_linear.q4_matrix_panel.v1") {
        const auto* row = q4_matrix_panel_row(quant_probe);
        mix(implementation_id("quant.q4.matrix-panel.v1"));
        mix(row ? implementation_id(row->key) : 0);
      } else if (inputs.size() == 5) {
        mix(implementation_id("quant.shared_activation_panel.v1"));
        mix(plan.shared_activation_panel);
        const auto load_chunks = q4_shared_panel_load_chunks(quant_probe);
        if (load_chunks != 1) {
          mix(implementation_id("quant.panel-adjacent-loads.v1"));
          mix(load_chunks);
        }
      }
      if (inputs.size() == 7) {
        const auto packed = q8_packed_matrix_plan(probe);
        mix(implementation_id("quant.q8.packed-weights.v1"));
        mix(packed.matrix ? implementation_id(packed.matrix->key) : 0);
        mix(packed.round_groups);
        mix(packed.lds_bytes);
      }
      mix(plan.rotate_decode_panel);
      mix(plan.decode_columns);
      mix(plan.prefill_rows);
      if (plan.row_ladder_ceiling != 0) {
        mix(implementation_id("quant.padded-row-ladder.v1"));
        mix(plan.row_ladder_ceiling);
      }
      if (plan.matrix) mix(implementation_id(plan.matrix->key));
    }
    const auto* chosen=primitive->specialize(probe);
    if ((node->kind == graph::OpKind::kRMS ||
         node->kind == graph::OpKind::kL2Norm) && group.outputs.size() != 1)
      chosen=primitive;
    key ^= implementation_id(chosen ? chosen->name() : "unavailable");
    key *= 1099511628211ull;
  }
  return key;
}
}  // namespace lse::dispatch
