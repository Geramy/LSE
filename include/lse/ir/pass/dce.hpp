// Dead code elimination.
//
// Deletes declarations nothing reads and control-flow blocks whose body is
// empty. It is the other half of both passes above it: CSE redirects uses but
// leaves the duplicate declaration standing, and the LDS fold empties a fill
// loop but does not remove the loop or the allocation it fed.
//
// A raw statement is opaque, so it is never dead and its operands are always
// live — which is what keeps the store epilogue's text from naming a value
// that has been deleted underneath it.
#pragma once

#include <memory>

#include "lse/ir/pass/pass.hpp"

namespace lse::ir {

[[nodiscard]] std::unique_ptr<Pass> make_dce();

}  // namespace lse::ir
