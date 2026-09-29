#pragma once
#include "lse/graph/codegen.hpp"
namespace lse::graph {
// Apply before retaining an execution plan. The output node keeps its identity.
std::size_t optimize_quant_swiglu(std::span<const NodePtr> roots,
                                  const backend::DeviceInfo &device,
                                  const IKernelEmitter &emitter,
                                  std::size_t member);
} // namespace lse::graph
