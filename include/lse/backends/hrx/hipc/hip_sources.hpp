// The HIP dialect's spellings for the built-in primitives.
#pragma once

#include "lse/graph/dialect_source.hpp"

namespace lse::backend {

[[nodiscard]] graph::DialectSourceTable hip_sources() noexcept;

}  // namespace lse::backend
