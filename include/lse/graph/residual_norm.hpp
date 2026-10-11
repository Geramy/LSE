#pragma once
#include <algorithm>

#include "lse/graph/graph.hpp"

namespace lse::graph {

// A layer's residual sum and the norm of it, from a K-sliced projection:
//   x = residual + slice_sum(partials)      (the next residual)
//   r = rms_norm(x, weight)                 (the next sublayer's input)
//   p = f16 activation panel of r           (optional, its GEMM's operand)
// one kernel (kernels::residual_norm_kernel) instead of a slice sum with the
// add in its epilogue, a norm, and a panel. Every output is the value the
// separate launches write: the same partial sum in slice order, the same add,
// the same reduction order (the cooperative norm's).
struct ResidualNorm {
  NodePtr sum;       // the slice sum (quant_linear.q4_gemm_f16.slice_sum.v1)
  NodePtr add;       // x
  NodePtr residual;  // the add's other operand
  NodePtr norm;      // r
  NodePtr panel;     // p, or null
  explicit operator bool() const { return sum && add && residual && norm; }
};

inline ResidualNorm exact_residual_norm(const FusionGroup& group) {
  ResidualNorm rn;
  const auto& n = group.nodes;
  if (group.is_phase || n.size() < 3 || n.size() > 4) return {};
  const auto named = [](const NodePtr& node, std::string_view name) {
    return node && node->prim && node->prim->name() == name;
  };
  if (!named(n[0], "quant_linear.q4_gemm_f16.slice_sum.v1") || !named(n[1], "add") ||
      n[2]->kind != OpKind::kRMS || n[2]->inputs.size() != 2 || n[2]->inputs[0] != n[1])
    return {};
  const NodePtr& add = n[1];
  if (add->inputs.size() != 2 || add->dtype != DType::kF32 || n[0]->dtype != DType::kF32 ||
      add->shape != n[0]->shape || n[2]->shape != add->shape || n[2]->dtype != DType::kF32)
    return {};
  const NodePtr& other = add->inputs[0] == n[0] ? add->inputs[1] : add->inputs[0];
  if ((add->inputs[0] != n[0] && add->inputs[1] != n[0]) || !other || other == n[0] ||
      other->shape != add->shape || other->dtype != DType::kF32)
    return {};
  rn.sum = n[0];
  rn.add = add;
  rn.residual = other;
  rn.norm = n[2];
  if (n.size() == 4) {
    if (!named(n[3], "quant_activation.f16_panel.v1") || n[3]->inputs.size() != 1 ||
        n[3]->inputs[0] != rn.norm)
      return {};
    rn.panel = n[3];
  }
  // Outputs: x and r, and p when it is there; the slice sum is internal.
  const auto has = [&](const NodePtr& v) {
    return std::find(group.outputs.begin(), group.outputs.end(), v) != group.outputs.end();
  };
  if (!has(rn.add) || !has(rn.norm) || (rn.panel && !has(rn.panel)) || has(rn.sum) ||
      group.outputs.size() != (rn.panel ? 3u : 2u))
    return {};
  return rn;
}

}  // namespace lse::graph
