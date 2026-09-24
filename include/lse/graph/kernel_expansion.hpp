#pragma once
#include "lse/graph/kernel_primitive.hpp"
namespace lse::graph {
// Optional device-aware graph lowering before partitioning and slot planning.
// A successful expansion preserves this node's result and external identity,
// inserts ordinary owned dependencies, and cannot request host synchronization.
// Refusal leaves the graph unchanged. No existing primitive vtable is extended.
class IKernelGraphExpansion {
public:
  virtual ~IKernelGraphExpansion() = default;
  virtual bool expand_graph(Node &node, const KernelShapes &shapes) const = 0;
};
} // namespace lse::graph
