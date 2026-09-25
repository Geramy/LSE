#pragma once

#include <algorithm>
#include <span>

#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::graph {

// Preserve the common partitioner's pointwise and kernel-epilogue contracts
// when an emitter has no staged-phase form. A supported kernel may be the
// first producer, but never an appended consumer: its existing output hook
// evaluates the equal-shape FP32 epilogue at the same logical element index.
// Call only for adjacent nodes within the same already ordered phase.
inline bool join_pointwise_chain(FusionGroup &previous, const NodePtr &next,
                                 std::span<const NodePtr> roots,
                                 const DialectSourceTable &sources) {
  const auto ordinary = [&](const NodePtr &n) {
    return n && !n->materialized && n->fclass == FusionClass::kElementwise &&
           n->dtype == DType::kF32 && n->prim != nullptr &&
           dynamic_cast<const KernelPrimitiveBase *>(n->prim) == nullptr &&
           !sources.find(n->prim->name()).empty();
  };
  if (!ordinary(next) || previous.is_phase || previous.nodes.empty() ||
      previous.outputs.size() != 1)
    return false;
  const NodePtr &producer = previous.outputs.front();
  if (!producer || producer != previous.nodes.back() ||
      producer->consumer_count != 1 ||
      std::find(roots.begin(), roots.end(), producer) != roots.end() ||
      std::find(next->inputs.begin(), next->inputs.end(), producer) ==
          next->inputs.end()) {
    return false;
  }
  // Reuse the common HIP/Loom partition policy, narrowed to equal-shape FP32
  // chains so eliminating an intermediate store cannot remove float narrowing
  // or change the indexing of an aliased/broadcast value.
  if (!Partitioner::can_fuse(*producer, *next))
    return false;
  bool kernel_anchor = false;
  for (std::size_t i = 0; i < previous.nodes.size(); ++i) {
    const NodePtr &n = previous.nodes[i];
    if (!n || n->materialized || n->dtype != DType::kF32 ||
        n->shape != next->shape || n->member != next->member) return false;
    if (ordinary(n)) continue;
    const auto *kernel = dynamic_cast<const KernelPrimitiveBase *>(n->prim);
    if (i != 0 || kernel == nullptr || !kernel->supports_epilogue() ||
        kernel->inplace_input() >= 0) return false;
    kernel_anchor = true;
  }

  previous.nodes.push_back(next);
  previous.outputs.assign(1, next);
  if (!kernel_anchor) {
    previous.anchor = next->kind;
    previous.anchor_class = next->fclass;
  }
  // Retain only outside bindings. Repeated operands share a binding, while
  // every internal producer remains an SSA value in the generated body.
  previous.inputs.clear();
  for (const NodePtr &n : previous.nodes) {
    for (const NodePtr &input : n->inputs) {
      if (std::find(previous.nodes.begin(), previous.nodes.end(), input) !=
          previous.nodes.end())
        continue;
      if (std::find(previous.inputs.begin(), previous.inputs.end(), input) ==
          previous.inputs.end()) {
        previous.inputs.push_back(input);
      }
    }
  }
  return true;
}

} // namespace lse::graph
