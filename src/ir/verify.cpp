#include "lse/ir/verify.hpp"

#include <vector>

#include "lse/ir/index.hpp"

namespace lse::ir {

namespace {

struct Checker {
  const Body& b;
  // A value "carries a runtime size" when a runtime EXTENT flows into it
  // through pure arithmetic. A loop's induction variable deliberately does NOT
  // inherit it: `for (row = 0; row < k.rows; ++row) x[row * K + t]` is the
  // legal shape — the address stride stays constant, only the trip count
  // moves. A window BASE is tracked separately: it is allowed in an address,
  // because it moves the origin rather than the stride.
  std::vector<bool> tainted;
  std::vector<bool> windowed;
  // A value is available once its defining op has been walked in an enclosing
  // region. Popped on the way out, which is what makes this a dominance check
  // and not merely a definedness check.
  std::vector<bool> live;
  Status status;

  explicit Checker(const Body& body)
      : b(body), tainted(body.value_count(), false),
        windowed(body.value_count(), false),
        live(body.value_count(), false) {}

  [[nodiscard]] bool fail(std::string what) {
    if (status.ok()) status = Status(StatusCode::kInternal, std::move(what));
    return false;
  }

  [[nodiscard]] std::size_t arity(OpKind k) const {
    switch (k) {
      case OpKind::kBinary: return 2;
      case OpKind::kCast: return 1;
      case OpKind::kSelect: return 3;
      case OpKind::kSubscript: return 2;
      case OpKind::kBind:
      case OpKind::kMutable: return 1;
      case OpKind::kAssign: return 2;
      case OpKind::kLoadVec: return 2;
      case OpKind::kStoreVec: return 3;
      case OpKind::kReturnIf: return 1;
      case OpKind::kIf: return 1;
      case OpKind::kFor: return 3;
      default: return 0;  // variadic or none
    }
  }

  [[nodiscard]] bool fixed_arity(OpKind k) const {
    return k != OpKind::kCall && k != OpKind::kRawStmt &&
           k != OpKind::kReturn && k != OpKind::kConst &&
           k != OpKind::kSymbol && k != OpKind::kAlloc &&
           k != OpKind::kBarrier && k != OpKind::kScope;
  }

  // A runtime extent is legal in a guard and in an outermost loop bound; it is
  // refused in inner address arithmetic, in a nested (reduction) trip count,
  // and — structurally impossible, since a width is a compile-time immediate —
  // in a vector width. Those are the three places where losing the constant
  // costs the inner loop its unroll, its constant address offsets and its
  // proven alignment, on a kernel already running at the memory roof.
  [[nodiscard]] bool carries_runtime_extent(ValueId v) const {
    return v < tainted.size() && tainted[v];
  }
  [[nodiscard]] bool carries_window_base(ValueId v) const {
    return v < windowed.size() && windowed[v];
  }

  void propagate(const Operation& o) {
    if (!produces_result(o.kind) || o.result == kNoValue) return;
    if (o.kind == OpKind::kExtent) {
      const bool runtime = o.has(kFlagRuntimeExtent);
      const bool base = o.has(kFlagWindowBase);
      tainted[o.result] = runtime && !base;
      windowed[o.result] = runtime && base;
      return;
    }
    // The induction variable is a fresh runtime value, not the extent.
    if (o.kind == OpKind::kFor) return;
    for (ValueId v : o.operands) {
      if (carries_runtime_extent(v)) tainted[o.result] = true;
      if (carries_window_base(v)) windowed[o.result] = true;
    }
  }

  // A window base may enter an address, but only as an offset: linearized, it
  // must appear as a term with a constant coefficient. `base * stride + i` is
  // that; `i * base` is not affine at all, so it does not linearize and the
  // base survives as an opaque symbol — which is how the check tells them
  // apart without special-casing a shape.
  [[nodiscard]] bool base_is_an_offset(ValueId index) const {
    const AffineExpr e = AffineExpr::of(b, index);
    for (const AffineExpr::Term& t : e.terms()) {
      if (carries_window_base(t.symbol) && !windowed_leaf(t.symbol)) {
        return false;
      }
    }
    return true;
  }

  // True when the symbol IS the base rather than some non-affine expression
  // the base got buried inside.
  [[nodiscard]] bool windowed_leaf(ValueId v) const {
    if (v >= b.value_count()) return false;
    const ValueDef& d = b.value(v);
    if (d.def == kNoOp) return false;
    const Operation& o = b.op(d.def);
    if (o.kind == OpKind::kExtent) return o.has(kFlagWindowBase);
    if (o.kind == OpKind::kBind) return windowed_leaf(o.operands[0]);
    return false;
  }

  bool check_extent_use(const Operation& o, int loop_depth) {
    switch (o.kind) {
      case OpKind::kSubscript:
      case OpKind::kLoadVec:
      case OpKind::kStoreVec:
        if (carries_runtime_extent(o.operands[1])) {
          return fail(std::string(op_name(o.kind)) +
                      ": a runtime extent reached address arithmetic");
        }
        if (carries_window_base(o.operands[1]) &&
            !base_is_an_offset(o.operands[1])) {
          return fail(std::string(op_name(o.kind)) +
                      ": a window base reached address arithmetic as more "
                      "than an offset");
        }
        return true;
      case OpKind::kFor:
        // A window base is never a trip count, at any depth: the number of
        // iterations is the window's extent, not where it starts.
        for (ValueId v : o.operands) {
          if (carries_window_base(v)) {
            return fail("for: a window base is not a trip count");
          }
        }
        if (loop_depth == 0) return true;  // an outermost bound is the legal use
        for (ValueId v : o.operands) {
          if (carries_runtime_extent(v)) {
            return fail("for: a runtime extent reached an inner trip count");
          }
        }
        return true;
      default:
        return true;
    }
  }

  bool walk(RegionId r, int loop_depth = 0) {
    std::vector<ValueId> introduced;
    for (OpId id : b.region(r).ops) {
      const Operation& o = b.op(id);
      if (o.erased) return fail("erased op still listed in a region");
      if (fixed_arity(o.kind) && o.operands.size() != arity(o.kind)) {
        return fail(std::string(op_name(o.kind)) + ": wrong operand count");
      }
      for (ValueId v : o.operands) {
        if (v == kNoValue || v >= b.value_count()) {
          return fail(std::string(op_name(o.kind)) + ": operand is not a value");
        }
        if (!live[v]) {
          // Name the value and what produced it. "operand is not in scope"
          // alone does not say which of an op's operands, nor where it came
          // from, and finding that by hand costs hours.
          std::string d = std::string(op_name(o.kind));
          if (!o.key.empty()) d += " '" + o.key + "'";
          d += ": operand v" + std::to_string(v) +
               " is not in scope at its use";
          if (v < b.value_count() && !b.value(v).name.empty()) {
            d += " (" + b.value(v).name + ")";
          }
          for (std::size_t oi = 0; oi < b.op_count(); ++oi) {
            const Operation& po = b.op(static_cast<OpId>(oi));
            if (!po.erased && po.result == v) {
              d += ", produced by " + std::string(op_name(po.kind)) +
                   " in a region that does not enclose this use";
              break;
            }
          }
          return fail(d);
        }
      }
      if (!check_extent_use(o, loop_depth)) return false;
      propagate(o);
      if (produces_result(o.kind)) {
        if (o.result == kNoValue || o.result >= b.value_count()) {
          return fail(std::string(op_name(o.kind)) + ": missing result value");
        }
        if (b.value(o.result).def != id) {
          return fail(std::string(op_name(o.kind)) + ": result def mismatch");
        }
        if (is_declaration(o.kind) && b.value(o.result).name.empty()) {
          return fail(std::string(op_name(o.kind)) +
                      ": a declaration must name its result");
        }
        if (o.kind == OpKind::kSymbol && b.value(o.result).name.empty()) {
          return fail("symbol: a symbol must have a name");
        }
        live[o.result] = true;
        introduced.push_back(o.result);
      }
      if (owns_region(o.kind) && o.regions.size() != 1) {
        return fail(std::string(op_name(o.kind)) + ": must own exactly one region");
      }
      if (!owns_region(o.kind) && !o.regions.empty()) {
        return fail(std::string(op_name(o.kind)) + ": owns a region it cannot");
      }
      for (RegionId sub : o.regions) {
        if (b.region(sub).parent != id) return fail("region parent mismatch");
        if (!walk(sub, loop_depth + (o.kind == OpKind::kFor ? 1 : 0))) {
          return false;
        }
      }
      // A memory reference must say where it lives, or a memory pass has to
      // guess and the wrong guess is a wrong answer.
      if (o.kind == OpKind::kSubscript) {
        const ValueDef& base = b.value(o.operands[0]);
        if (base.type.space == Space::kNone && !base.type.is_vector()) {
          return fail("subscript: base is neither memory nor a vector");
        }
      }
    }
    // Leaving the region ends the scope of everything it introduced.
    for (ValueId v : introduced) live[v] = false;
    return true;
  }
};

}  // namespace

namespace {

// The space's own invariants. A dimension that lies about its granularity or
// names a window base nothing declares would produce a split nobody can honour.
Status verify_space(const Body& b) {
  for (const Dim& d : b.space().dims()) {
    if (d.name.empty()) {
      return LSE_ERROR(kInternal, "iteration space: a dimension needs a name");
    }
    if (d.extent <= 0) {
      return LSE_ERROR(kInternal, "iteration space: dimension '", d.name,
                       "' has no extent");
    }
    if (d.granularity < 0 || d.granularity > d.extent) {
      return LSE_ERROR(kInternal, "iteration space: dimension '", d.name,
                       "' has a granularity its extent cannot hold");
    }
    if (!d.window_base.empty() && !d.splittable()) {
      return LSE_ERROR(kInternal, "iteration space: dimension '", d.name,
                       "' takes a window base but declares no granularity");
    }
  }
  return OkStatus();
}

}  // namespace

Status verify(const Body& body) {
  Checker c(body);
  if (!c.walk(body.entry())) return c.status;
  return verify_space(body);
}

}  // namespace lse::ir
