#pragma once

#include "lse/backends/hrx/kernels/linked.hpp"

namespace lse::backend::hrx_kernels {

const graph::KernelPrimitiveBase* gdn_pair_kernel();
LinkedBinding gdn_pair_bindings(const graph::FusionGroup& group);

}  // namespace lse::backend::hrx_kernels
