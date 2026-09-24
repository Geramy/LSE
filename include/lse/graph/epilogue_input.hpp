#pragma once
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::graph {
// One binding can serve both a kernel pointer argument and an elementwise
// operand. The latter still needs a load at the epilogue's stored index.
inline bool needs_elementwise_input(const FusionGroup& group, const Node* input) {
  for (const auto& consumer : group.nodes) {
    if (dynamic_cast<const KernelPrimitiveBase*>(consumer->prim) != nullptr ||
        consumer->kind == OpKind::kRepeat) continue;
    for (const auto& operand : consumer->inputs) {
      if (operand.get() == input) return true;
    }
  }
  return false;
}
}  // namespace lse::graph
