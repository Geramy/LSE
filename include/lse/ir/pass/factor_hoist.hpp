// Hoist a loop-invariant factor out of a summation.
//
//   acc = 0;  for (k) acc += t[k] * s;   ==>   acc = 0; for (k) acc += t[k];
//                                              acc = acc * s;
//
// Valid because multiplication distributes over addition, and `s` does not
// move: it is the same value on every iteration. What the rewrite buys is one
// multiply per iteration instead of one per reduction, and -- the reason this
// exists -- it DELAYS the need for `s` until after the loop. A scale that is
// only wanted at the end no longer has to be computed before the loop starts,
// so a reduction that produces it can share the loop that consumes it instead
// of being a separate launch that has to finish first.
//
// The motivating shape is an rms_norm feeding a linear:
//
//   out[o] = SUM_k (x[k] * gain[k] * s) * w[o][k]   with s = rsqrt(mean(x^2))
//          = s * SUM_k (x[k] * gain[k]) * w[o][k]
//
// `s` is one scalar for the whole row, so it leaves the contraction entirely
// and the norm's own sum can ride along in the linear's existing k-loop at the
// cost of one fma per element and no extra loads at all.
//
// WHY THIS IS NOT clang's JOB: this is floating-point reassociation. LLVM will
// not do it at -O3 without -ffast-math, and rightly -- the result differs in
// the last bits. We can, because we know the quantities are a norm scale and a
// contraction, and we own the tolerance.
#pragma once

#include <memory>

#include "lse/ir/pass/pass.hpp"

namespace lse::ir {

[[nodiscard]] std::unique_ptr<Pass> make_factor_hoist();

}  // namespace lse::ir
