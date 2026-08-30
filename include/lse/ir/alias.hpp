// Do two memory accesses touch the same element?
//
// This is the fact that dataflow alone cannot supply. Def-use order says what
// depends on what; it cannot say whether `buf[i]` and `buf[j]` are one
// location, and without that no store may be reordered, no load may be reused
// across a write, and no pair of loops touching memory may be fused. Every
// memory-side rewrite is gated on this analysis.
//
// TWO DISTINCT BUFFER SYMBOLS ARE NOT DISTINCT MEMORY HERE. Bindings are
// emitted `__restrict__`, but Workgroup::plan_slots recycles slots, so two
// bindings can name one allocation -- the emitter says so itself, and the
// cross-workgroup WAR that hazard produces is not something a barrier orders.
// So `alias_of` answers kMaybe for two different symbols unless the caller
// supplies a `distinct_allocations` oracle that actually knows the slot map.
// Assuming otherwise is the single most tempting unsound move available here.
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "lse/ir/body.hpp"
#include "lse/ir/index.hpp"

namespace lse::ir {

enum class Alias : std::uint8_t {
  kNo,     // provably different elements
  kMaybe,  // not proven either way -- the only safe default
  kMust,   // provably the same element
};

// One element range touched by a memory op: `buffer[index .. index+width)`.
struct Access {
  ValueId buffer = kNoValue;
  AffineExpr index;
  std::uint32_t width = 1;   // lanes, from a vector load/store
  bool is_write = false;
  // Which address space the reference lives in. Two references in DIFFERENT
  // spaces cannot be the same memory whatever the slot map says: LDS and a
  // global parameter are separate address spaces, and slot recycling happens
  // among bindings within one space, never across two. This is the one way
  // two distinct symbols are provably distinct memory without an oracle.
  Space space = Space::kNone;
};

// Reads a kLoadVec / kStoreVec. False when the op is not a memory access or
// its index is not affine, which is the analysis declining to guess.
[[nodiscard]] bool access_of(
    const Body& b, OpId id, Access* out,
    const std::unordered_map<ValueId, ValueId>* alias = nullptr);

// Answers whether two buffer symbols are KNOWN to be different allocations.
// Returning false means "unknown", not "same".
using DistinctAllocations = std::function<bool(ValueId, ValueId)>;

[[nodiscard]] Alias alias_of(const Access& a, const Access& b,
                             const DistinctAllocations* distinct = nullptr);

// Whether the two may be reordered with respect to each other: two reads
// always may; anything else needs kNo.
[[nodiscard]] bool may_reorder(const Access& a, const Access& b,
                               const DistinctAllocations* distinct = nullptr);

}  // namespace lse::ir
