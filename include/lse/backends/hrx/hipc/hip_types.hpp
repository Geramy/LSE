// HIP's spelling of the element types a kernel can name.
#pragma once

#include "lse/graph/kernel_ir.hpp"

namespace lse::backend {

[[nodiscard]] graph::kir::TypeTable hip_types() noexcept;

}  // namespace lse::backend
