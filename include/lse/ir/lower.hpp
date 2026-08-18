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

// One value, as the body's dialect names it at a use site.
//
// In a C-family dialect that is an inlined expression: a named value renders as
// its name, an unnamed one renders its whole subtree. In an SSA dialect every
// value is a name, and the subtree is a run of statements the SSA printer has
// already emitted — see ssa_name. Which of the two a caller gets is the body's
// to decide, because a store hook receives this text and cannot know.
[[nodiscard]] std::string render(const Body& body, ValueId v);

// The SSA spelling of a value: `%name` if it was declared, `%v<id>` otherwise.
// One definition, shared by the SSA printer and by `render`, so a hook's
// operand text and the statement that defined it cannot drift apart.
[[nodiscard]] std::string ssa_name(const Body& body, ValueId v);

// The whole body: the vector typedefs the kernel used, then its statements.
// Function-scope typedefs keep the result self-contained, so a kernel authored
// against the recorder needs nothing added to the emitter's preamble.
[[nodiscard]] std::string lower(const Body& body);

}  // namespace lse::ir
