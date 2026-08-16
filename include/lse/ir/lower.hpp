// IR -> source text.
//
// This is the only place that knows C's syntax. Everything a *target* spells
// differently arrives through the two tables the body already carries —
// `TypeTable` for scalars and vectors, `DialectSourceTable` for every op that
// has a name on the device — so this walk is shared by every backend that
// generates C-family kernels rather than copied per backend. What is genuinely
// HIP-specific is the translation unit around this body: the includes, the
// entry-point signature and the dispatch-constants struct, and that lives in
// `src/backends/hrx/hip_emitter.cpp` where it belongs.
#pragma once

#include <string>

#include "lse/ir/body.hpp"
#include "lse/ir/op.hpp"

namespace lse::ir {

// One expression, as the printer would inline it at a use site. A named value
// renders as its name; an unnamed one renders its whole subtree.
[[nodiscard]] std::string render(const Body& body, ValueId v);

// The whole body: the vector typedefs the kernel used, then its statements.
// Function-scope typedefs keep the result self-contained, so a kernel authored
// against the recorder needs nothing added to the emitter's preamble.
[[nodiscard]] std::string lower(const Body& body);

}  // namespace lse::ir
