// The kernel authoring surface, under the name the kernels have always used.
//
// The surface itself is `lse::ir::env` (include/lse/ir/env.hpp): it is the
// front end of the kernel IR, not part of the tensor-level graph. `env` here
// is an alias so a kernel written against `env::In` / `env::Emit` needs no
// edit.
#pragma once

#include "lse/graph/kernel_ir.hpp"
#include "lse/ir/env.hpp"

namespace lse::graph {

namespace env = ::lse::ir::env;

}  // namespace lse::graph
