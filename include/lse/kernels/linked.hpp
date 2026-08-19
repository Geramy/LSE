#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

// The generator walks a fusion group and, when the nodes are a linked
// contraction pipeline (SwiGLU, exclusive RMS then linear), emits one
// staged kernel instead of one launch per op.
const graph::KernelPrimitiveBase* linked_kernel_for(
    const graph::FusionGroup& group, const graph::KernelShapes& shapes);

struct LinkedBinding {
  std::vector<const graph::Node*> inputs;
  const graph::Node* sink = nullptr;
  std::array<float, 4> attrs{};
  std::array<std::int32_t, 4> iattrs{};
  bool ok = false;
};
LinkedBinding linked_bindings(const graph::FusionGroup& group);

}  // namespace lse::kernels
