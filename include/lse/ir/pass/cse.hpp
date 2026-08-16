// Common subexpression elimination over pure values.
//
// Its own wins are small — clang -O3 would find most of them — but it is what
// makes the memory passes possible: two fused stages that each computed
// `threadIdx.x / 32` have two different IR values for one number, and nothing
// downstream can see that two LDS fills are the same fill until those are one
// value. CSE is the canonicalizer the LDS fold is written against.
//
// Scoping is structural. The walk carries a stack of scopes, one per region,
// so a value found in an enclosing scope is guaranteed to dominate the use
// being rewritten; there is no separate dominance computation because the IR
// has no CFG for one to be computed over.
//
// Two things it must not touch: a subscript (a memory read, whose value can
// change between two identical-looking expressions) and any value a raw
// statement mentions by name (the text is opaque, so redirecting the *use*
// would leave the text naming a definition that no longer exists).
#pragma once

#include <memory>

#include "lse/ir/pass/pass.hpp"

namespace lse::ir {

[[nodiscard]] std::unique_ptr<Pass> make_cse();

}  // namespace lse::ir
