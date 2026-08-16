// Structural invariants of a body, checked in one place.
//
// The engine already had checks of this shape — the emitter scanned a stage's
// TEXT for "return;" before allowing it into a fused kernel — and they were
// string matching because there was nothing else to match on. A verifier is
// where those belong: it answers questions about the IR against the IR.
//
// It is not a type checker for the kernel's arithmetic. It checks the things a
// pass may break and a printer cannot survive: operands that name a value
// nothing defines, a use that its definition does not dominate, a control-flow
// op with the wrong arity, a declaration with no name.
#pragma once

#include <string>

#include "lse/core/status.hpp"
#include "lse/ir/body.hpp"

namespace lse::ir {

// Ok, or the first violation with enough context to find it.
[[nodiscard]] Status verify(const Body& body);

}  // namespace lse::ir
