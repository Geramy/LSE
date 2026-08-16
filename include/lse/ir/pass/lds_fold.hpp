// Fold workgroup-scratch allocations that hold the same bytes.
//
// The motivating case, measured: sibling GEMV fusion puts N stage bodies in one
// kernel, and each stage independently stages the SAME activation row into its
// own `__shared__ float[K]`. The compiler sums those arrays, so at K = 1024 the
// fused q/k/v/g kernel carries four identical 4 KB copies and the run cap comes
// out of `run * 4 * K <= lds_bytes / 4` — four stages, which is exactly the
// attention run and one short of the MoE group.
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
