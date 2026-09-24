#pragma once
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/quant_operand_policy.hpp"
#include <vector>

namespace lse::kernels {
// Selection runs before the cache lookup: a generic quant_matmul node can
// specialize to different registered implementations under one graph shape.
// Names are registered implementation identities; a new strategy/revision must
// have a distinct name. Cost samples themselves never enter this fingerprint.
[[nodiscard]] inline std::uint64_t quant_operand_specialization_key(
    std::uint64_t key, const graph::FusionGroup& group,
    const backend::DeviceInfo& device, graph::kir::TypeTable types,
    const graph::DialectSourceTable& intrinsics) {
  std::vector<Shape> inputs;
  std::vector<DType> dtypes;
  for (const auto& node : group.nodes) {
    if (node->kind != graph::OpKind::kQuantMatMul) continue;
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
    const auto* chosen=primitive->specialize(probe);
    key ^= quant_operand_implementation_id(chosen ? chosen->name() : "unavailable");
    key *= 1099511628211ull;
  }
  return key;
}
}  // namespace lse::kernels
