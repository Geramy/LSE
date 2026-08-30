#include "lse/ir/alias.hpp"

#include <algorithm>
#include <cstdlib>

namespace lse::ir {

namespace {

// A subscript is memory only when its base is a buffer parameter or an
// allocation. `ld2[i3]` is a lane of a register vector and indexes nothing --
// treating it as memory would invent conflicts that do not exist.
bool indexes_memory(const Body& b, ValueId base) {
  if (base == kNoValue) return false;
  const OpId def = b.value(base).def;
  if (def == kNoOp) return false;
  const OpKind k = b.op(def).kind;
  return k == OpKind::kSymbol || k == OpKind::kAlloc;
}

}  // namespace

bool access_of(const Body& b, OpId id, Access* out,
               const std::unordered_map<ValueId, ValueId>* alias) {
  if (out == nullptr) return false;
  const Operation& o = b.op(id);
  if (o.erased) return false;

  ValueId base = kNoValue;
  ValueId index = kNoValue;
  bool write = false;
  std::int64_t lanes = o.imm;
  if (o.kind == OpKind::kLoadVec && o.operands.size() == 2) {
    base = o.operands[0];
    index = o.operands[1];
  } else if (o.kind == OpKind::kStoreVec && o.operands.size() == 3) {
    base = o.operands[0];
    index = o.operands[1];
    write = true;
  } else if (o.kind == OpKind::kSubscript && o.operands.size() == 2) {
    // An ordinary element read: `buf[i]` as an inline expression, which is how
    // most reads in these kernels are spelled.
    base = o.operands[0];
    index = o.operands[1];
    lanes = 1;
  } else if (o.kind == OpKind::kAssign && o.operands.size() == 2) {
    // `buf[i] = v` -- the write is the subscript on the left.
    const ValueId lhs = o.operands[0];
    const OpId lhs_def = lhs == kNoValue ? kNoOp : b.value(lhs).def;
    if (lhs_def == kNoOp || b.op(lhs_def).kind != OpKind::kSubscript ||
        b.op(lhs_def).operands.size() != 2) {
      return false;
    }
    base = b.op(lhs_def).operands[0];
    index = b.op(lhs_def).operands[1];
    write = true;
    lanes = 1;
  } else {
    return false;
  }
  if (!indexes_memory(b, base)) return false;

  out->buffer = base;
  out->space = b.value(base).type.space;
  out->index = AffineExpr::of(b, index, alias);
  // imm carries the vector width; anything below 2 touches one element.
  out->width = lanes > 1 ? static_cast<std::uint32_t>(lanes) : 1u;
  out->is_write = write;
  return true;
}

Alias alias_of(const Access& a, const Access& b,
               const DistinctAllocations* distinct) {
  if (a.buffer == kNoValue || b.buffer == kNoValue) return Alias::kMaybe;

  if (a.space != b.space && a.space != Space::kNone &&
      b.space != Space::kNone) {
    return Alias::kNo;   // different address spaces, whatever the symbols are
  }

  if (a.buffer != b.buffer) {
    // Only a caller that knows the slot map may say these are different
    // memory. Without one this is unknown -- see the header.
    if (distinct != nullptr && *distinct && (*distinct)(a.buffer, b.buffer)) {
      return Alias::kNo;
    }
    return Alias::kMaybe;
  }

  // Same buffer: the question is whether the index ranges can meet. Subtract
  // one index from the other; a difference that is not a constant leaves the
  // variable parts unaccounted for and nothing can be concluded.
  AffineExpr delta = a.index;
  delta -= b.index;
  const std::optional<std::int64_t> d = delta.as_constant();
  if (!d.has_value()) return Alias::kMaybe;

  if (*d == 0) {
    return a.width == b.width ? Alias::kMust : Alias::kMaybe;
  }
  // [ia, ia+wa) and [ib, ib+wb) are disjoint when the gap clears the width of
  // whichever one starts first.
  const std::int64_t gap = *d;
  const auto wa = static_cast<std::int64_t>(a.width);
  const auto wb = static_cast<std::int64_t>(b.width);
  if (gap > 0) return gap >= wb ? Alias::kNo : Alias::kMaybe;
  return -gap >= wa ? Alias::kNo : Alias::kMaybe;
}

bool may_reorder(const Access& a, const Access& b,
                 const DistinctAllocations* distinct) {
  if (!a.is_write && !b.is_write) return true;   // read/read never conflicts
  return alias_of(a, b, distinct) == Alias::kNo;
}

}  // namespace lse::ir
