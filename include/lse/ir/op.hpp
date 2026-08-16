// Operations, values and regions.
//
// An op has operands, at most one result, attributes, and — this is the part a
// flat statement list cannot express — zero or more nested REGIONS it owns. An
// `if` and a `for` are ops with a body region, the way scf.if / scf.for are in
// MLIR, so a pass sees structured control flow instead of re-deriving it from
// branch targets. The authoring surface is structured by construction
// (`e.when` is an RAII scope, `e.range` a scoped loop), so this IR can never be
// asked to represent irreducible flow and never has to.
//
// Values are SSA for everything pure. Mutable accumulators (`e.var`) are NOT
// in SSA: they are `kAlloc` slots written by `kAssign`. Modelling them with
// loop-carried arguments would complicate every kernel to buy something clang
// already does downstream — comgr runs -O3, and mem2reg is part of it.
//
// Only some values are *named*. A named value is declared as its own statement
// and referred to by name; an unnamed one is an expression node that the
// printer inlines at each use. That is exactly the distinction the authoring
// surface already draws between `e.let` and `a + b`, and it is what keeps the
// generated source reviewable against the C++ that produced it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lse/ir/types.hpp"

namespace lse::ir {

using OpId = std::uint32_t;
using ValueId = std::uint32_t;
using RegionId = std::uint32_t;

inline constexpr OpId kNoOp = 0xFFFFFFFFu;
inline constexpr ValueId kNoValue = 0xFFFFFFFFu;
inline constexpr RegionId kNoRegion = 0xFFFFFFFFu;

enum class OpKind : std::uint8_t {
  // -- pure expression ops. No side effects; safe to CSE and to delete when
  //    nothing uses the result.
  kConst,      // a literal. `text` is its spelling, `imm` its integer value.
  kExtent,     // a tensor dimension. `key` names it, `text` spells it, and
               // kFlagRuntimeExtent says whether it is baked in or a dispatch
               // constant. A literal extent prints exactly like a kConst; the
               // difference is that a pass and the verifier can see what it is.
  kSymbol,     // a name that already exists: a buffer parameter, the thread id.
  kBinary,     // `key` is the operator; renders (a <op> b).
  kCast,       // to `cast_to`.
  kSelect,     // (c ? a : b)
  kCall,       // a dialect row. `key` is the row, `text` its template.
  kSubscript,  // base[index] — an element of a memory reference, or a lane of
               // a register vector. Addressable, hence not CSE-able.

  // -- declarations. Each names its result.
  kBind,       // const T n = <expr>;      immutable
  kMutable,    // T n = <expr>;            an accumulator slot
  kAlloc,      // T n;  /  __shared__ T n[N];  /  vecT n;

  // -- side effects
  kAssign,     // <lvalue> = <value>;
  kLoadVec,    // const vecT n = *(const vecT*)(&buf[i]);  (or a scalar read)
  kStoreVec,   // *(vecT*)(&buf[i]) = v;                   (or a scalar write)
  kBarrier,
  kReturn,     // 0 or 1 operand
  kReturnIf,   // if (<cond>) return;
  kRawStmt,    // verbatim text. Operands are the values it mentions, so DCE
               // cannot delete something the text still names.

  // -- structured control flow. Each owns a region.
  kIf,         // operand: condition
  kFor,        // operands: lo, hi, step. `key` is the induction variable.
  kScope,      // a bare block, which is what scopes a store epilogue's locals.
};

[[nodiscard]] constexpr bool is_pure(OpKind k) noexcept {
  switch (k) {
    case OpKind::kConst:
    case OpKind::kExtent:
    case OpKind::kSymbol:
    case OpKind::kBinary:
    case OpKind::kCast:
    case OpKind::kSelect:
    case OpKind::kCall:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] constexpr bool is_declaration(OpKind k) noexcept {
  return k == OpKind::kBind || k == OpKind::kMutable || k == OpKind::kAlloc ||
         k == OpKind::kLoadVec;
}

[[nodiscard]] constexpr bool owns_region(OpKind k) noexcept {
  return k == OpKind::kIf || k == OpKind::kFor || k == OpKind::kScope;
}

// Whether the op defines a value. A `for` does: its induction variable, which
// is what lets an index expression name the loop it varies over instead of a
// string the printer happens to have chosen.
[[nodiscard]] constexpr bool produces_result(OpKind k) noexcept {
  switch (k) {
    case OpKind::kAssign:
    case OpKind::kStoreVec:
    case OpKind::kBarrier:
    case OpKind::kReturn:
    case OpKind::kReturnIf:
    case OpKind::kRawStmt:
    case OpKind::kIf:
    case OpKind::kScope:
      return false;
    default:
      return true;
  }
}

[[nodiscard]] std::string_view op_name(OpKind k) noexcept;

// Op flags. Deliberately few: each is a property the printer needs and the IR
// cannot derive.
enum OpFlags : std::uint16_t {
  kFlagNone = 0,
  // Ask the target to unroll this loop. What makes fragment indices constant.
  kFlagUnroll = 1u << 0,
  // Indent the body one level. False for the `env` surface, which emits flat
  // blocks, true for the deprecated lambda forms that indented.
  kFlagIndentBody = 1u << 1,
  // `imm` holds a meaningful integer constant (a kConst that is an integer).
  kFlagIntConst = 1u << 2,
  // A kAlloc that is an array declaration rather than a scalar or vector.
  kFlagArray = 1u << 3,
  // A kExtent that is bound at dispatch instead of baked into the source.
  kFlagRuntimeExtent = 1u << 4,
  // A kExtent that is the base index of an iteration-space window rather than
  // a size. Legal in an address with a constant coefficient; a size is not.
  kFlagWindowBase = 1u << 5,
};

struct Operation {
  OpKind kind = OpKind::kRawStmt;
  Type type{};
  ValueId result = kNoValue;
  std::vector<ValueId> operands;
  std::vector<RegionId> regions;

  // Literal spelling, raw statement text, or a dialect template.
  std::string text;
  // Binary operator, dialect row key, or loop induction variable.
  std::string key;

  std::int64_t imm = 0;
  Scalar cast_to = Scalar::kF32;
  std::uint16_t flags = kFlagNone;
  // Erased ops stay in the arena so ids never dangle; the region lists drop
  // them and the printer skips them.
  bool erased = false;

  [[nodiscard]] bool has(OpFlags f) const noexcept { return (flags & f) != 0; }
};

struct ValueDef {
  OpId def = kNoOp;
  Type type{};
  // Empty means this value is an expression the printer inlines at each use.
  std::string name;
};

// A region is one block of ops in order. There is no CFG and no block
// argument: structured control flow is the only control flow this IR has.
struct Region {
  OpId parent = kNoOp;
  std::vector<OpId> ops;
};

}  // namespace lse::ir
