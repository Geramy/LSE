#pragma once

#include "lse/kernels/linked.hpp"

namespace lse::kernels {

const graph::KernelPrimitiveBase* gdn_pair_kernel();
LinkedBinding gdn_pair_bindings(const graph::FusionGroup& group);

}  // namespace lse::kernels
