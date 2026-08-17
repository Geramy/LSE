// Fold workgroup-scratch allocations that hold the same bytes.
//
// The motivating case was sibling GEMV fusion, where each stage independently
// staged the SAME activation row into its own `__shared__ float[K]` and the
// compiler summed them. That is now shared by construction: the emitter stages
// the row once for the whole run and hands every stage the one array, so on
// lemonseed this pass fires zero times. It stays as the net under any lowering
// that still emits duplicate stagings, not as the mechanism that shares them —
// folding after the fact leaves the redundant barriers standing and depends on
// the widest stage happening to be emitted first.
//
// This pass merges those allocations. Two are the same storage when they have
// the same element type and count, each is written from exactly one fill loop,
// and the two fill loops are the same loop over the same read-only source at
// the same addresses. The fills then both remain and both write the same bytes
// to the same array — an idempotent write, so no interleaving of the two
// changes what any thread reads.
//
// A fill is additionally deleted when it is provably redundant: its guard
// implies an earlier fill's guard, that guard is workgroup-uniform, and the
// earlier fill is followed by a barrier — so every thread that reaches here has
// already seen the array completed.
#pragma once

#include <memory>

#include "lse/ir/pass/pass.hpp"

namespace lse::ir {

[[nodiscard]] std::unique_ptr<Pass> make_lds_fold();

}  // namespace lse::ir
