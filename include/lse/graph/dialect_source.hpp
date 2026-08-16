// How a primitive is spelled in one backend's source dialect.
//
// The table is an IR concept — it is what turns an op into text — so it lives
// in `lse::ir` with the rest of the kernel IR. The graph layer names the same
// types because a primitive and an emitter both traffic in them.
#pragma once

#include "lse/ir/dialect.hpp"

namespace lse::graph {

using ir::Dialect;
using ir::DialectExpr;
using ir::DialectSourceTable;
using ir::expr_for;
using ir::PrimitiveSource;

}  // namespace lse::graph
