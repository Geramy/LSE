#include "lse/kernels/linked.hpp"
#include "lse/kernels/gdn.hpp"

namespace lse::kernels {

LinkedBinding linked_bindings(const graph::FusionGroup& group) {
  return gdn_pair_bindings(group);
}

const graph::KernelPrimitiveBase* linked_kernel_for(
    const graph::FusionGroup& group, const graph::KernelShapes&) {
  if (!gdn_pair_bindings(group).ok) return nullptr;
  return gdn_pair_kernel();
}

}  // namespace lse::kernels
