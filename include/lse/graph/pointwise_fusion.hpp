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

// A node whose result is its producer's value narrowed and stored at a mapped
// position (KernelPrimitiveBase::epilogue_output) joins the group computing
// that value as one more output: the anchor's epilogue stores the value twice,
// once for each output, instead of a second launch reading it back. Only
// where the emitter can (IKernelEmitter::stores_epilogue_output), over the
// value's own index space; the value stays an output of the group.
inline bool join_epilogue_output(FusionGroup &previous, const NodePtr &next,
                                 const IKernelEmitter &emitter,
                                 const backend::DeviceInfo &device) {
  const auto *kernel = next ? dynamic_cast<const KernelPrimitiveBase *>(next->prim) : nullptr;
  if (kernel == nullptr ||
      kernel->epilogue_output() == KernelPrimitiveBase::EpilogueLayout::kNone ||
      next->materialized || next->inputs.size() != 1 || previous.is_phase ||
      previous.nodes.empty())
    return false;
  const NodePtr &value = next->inputs.front();
  if (!value || value->dtype != DType::kF32 || value->materialized ||
      value->element_count() != next->element_count() || value->member != next->member ||
      std::find(previous.outputs.begin(), previous.outputs.end(), value) ==
          previous.outputs.end())
    return false;
  const NodePtr &anchor = previous.nodes.front();
  if (!anchor || anchor->element_count() != value->element_count()) return false;
  for (const NodePtr &out : previous.outputs)
    if (out->element_count() != value->element_count()) return false;
  if (!emitter.stores_epilogue_output(previous, *next, device)) return false;
  previous.nodes.push_back(next);
  previous.outputs.push_back(next);
  return true;
}

// A reduction reading a group's one output joins the group as one more output
// where the emitter writes both in one kernel (IKernelEmitter::joins_reduction:
// a residual sum and its norm). The group's output stays an output.
inline bool join_reduction(FusionGroup &previous, const NodePtr &next,
                           const IKernelEmitter &emitter, const backend::DeviceInfo &device) {
  if (!next || next->materialized || previous.is_phase || previous.outputs.size() != 1 ||
      next->inputs.empty() || next->inputs[0] != previous.outputs.front() ||
      next->member != previous.outputs.front()->member ||
      !emitter.joins_reduction(previous, next, device))
    return false;
  previous.nodes.push_back(next);
  previous.outputs.push_back(next);
  for (const NodePtr &input : next->inputs) {
    if (std::find(previous.nodes.begin(), previous.nodes.end(), input) != previous.nodes.end())
      continue;
    if (std::find(previous.inputs.begin(), previous.inputs.end(), input) == previous.inputs.end())
      previous.inputs.push_back(input);
  }
  return true;
}

} // namespace lse::graph
