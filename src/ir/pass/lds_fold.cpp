#include "lse/ir/pass/lds_fold.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/ir/index.hpp"

namespace lse::ir {

namespace {

// One `__shared__` array and the single loop that fills it.
struct Staging {
  OpId alloc = kNoOp;
  ValueId array = kNoValue;
  OpId loop = kNoOp;        // the fill loop
  ValueId induction = kNoValue;
  ValueId index = kNoValue;  // the subscript index the fill writes
  ValueId source = kNoValue;  // the value the fill writes
  // The `if` conditions the fill sits under, outermost first.
  std::vector<ValueId> guards;
  // Whether a barrier follows the fill in the same region.
  bool barrier_after = false;
};

// True when `name` occurs in `text` as a whole identifier rather than as part of
// a longer one.
//
// The raw scan used a plain substring search, so `b1` and `b2` were both marked
// written by a store epilogue that mentions only `b10`, `b11` and `b12` — real
// and reproducible on a 13-binding body. That direction only ever *adds* marks,
// so it declined folds rather than approving wrong ones, but a predicate that
// silently disables the pass on any body with ten or more bindings is not a
// predicate. `b0` inside `b100` is the same bug at the next order of magnitude.
bool mentions_identifier(const std::string& text, const std::string& name) {
  if (name.empty()) return false;
  const auto part = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
  };
  for (std::size_t at = 0;
       (at = text.find(name, at)) != std::string::npos; at += 1) {
    const bool left = at > 0 && part(text[at - 1]);
    const std::size_t end = at + name.size();
    const bool right = end < text.size() && part(text[end]);
    if (!left && !right) return true;
  }
  return false;
}

// Every memory reference this body writes. A staging whose source reads one of
// them is not reproducible, so the two fills would not be idempotent.
//
// A raw statement is opaque text — the store epilogue is one — so any buffer
// whose name appears in one as an identifier counts as written. Conservative by
// design: a false positive costs a fold, a false negative costs an answer.
void collect_written(const Body& b, std::vector<bool>& written) {
  for (OpId id = 0; id < b.op_count(); ++id) {
    const Operation& o = b.op(id);
    if (o.erased) continue;
    ValueId target = kNoValue;
    if (o.kind == OpKind::kAssign) {
      target = o.operands[0];
    } else if (o.kind == OpKind::kStoreVec) {
      target = o.operands[0];
    }
    if (target == kNoValue || target >= b.value_count()) continue;
    // An assignment through a subscript writes the subscript's base.
    const ValueDef& d = b.value(target);
    if (d.def != kNoOp && b.op(d.def).kind == OpKind::kSubscript) {
      target = b.op(d.def).operands[0];
    }
    if (target < written.size()) written[target] = true;
  }
  std::string raw;
  for (OpId id = 0; id < b.op_count(); ++id) {
    const Operation& o = b.op(id);
    if (!o.erased && o.kind == OpKind::kRawStmt) raw += o.text + "\n";
  }
  if (raw.empty()) return;
  for (ValueId v = 0; v < b.value_count(); ++v) {
    const ValueDef& d = b.value(v);
    // A nameless memory value cannot be reached by name from raw text, so this
    // scan has nothing to say about it. What writes it structurally is already
    // marked by the loop above, which does not look at names at all.
    if (d.type.space == Space::kNone || d.name.empty()) continue;
    if (mentions_identifier(raw, d.name)) written[v] = true;
  }
}

// Whether the op is a statement. A region's *shape* is its statements; the
// expression ops that computed their operands sit in the same list and are
// inlined at their uses, so counting raw ops would say a one-line fill loop
// has five things in it.
bool is_statement(OpKind k) {
  switch (k) {
    case OpKind::kConst:
    case OpKind::kExtent:
    case OpKind::kSymbol:
    case OpKind::kBinary:
    case OpKind::kCast:
    case OpKind::kSelect:
    case OpKind::kCall:
    case OpKind::kSubscript:
      return false;
    default:
      return true;
  }
}

// The memory references an expression reads.
void reads_of(const Body& b, ValueId v, std::vector<ValueId>& out, int depth) {
  if (depth > 64 || v == kNoValue || v >= b.value_count()) return;
  const ValueDef& d = b.value(v);
  if (d.def == kNoOp) return;
  const Operation& o = b.op(d.def);
  if (o.kind == OpKind::kSubscript || o.kind == OpKind::kLoadVec) {
    out.push_back(o.operands[0]);
  }
  for (ValueId a : o.operands) reads_of(b, a, out, depth + 1);
}

// Structural equality of two expression trees under a symbol substitution.
// Used for the value a fill writes; the addresses go through the affine form,
// which is stronger.
bool same_expr(const Body& b, ValueId a, ValueId c,
               const std::unordered_map<ValueId, ValueId>& alias, int depth) {
  if (depth > 64) return false;
  if (a == c) return true;
  if (const auto it = alias.find(a); it != alias.end() && it->second == c) {
    return true;
  }
  if (a == kNoValue || c == kNoValue || a >= b.value_count() ||
      c >= b.value_count()) {
    return false;
  }
  const ValueDef& da = b.value(a);
  const ValueDef& dc = b.value(c);
  if (da.def == kNoOp || dc.def == kNoOp) return false;
  const Operation& oa = b.op(da.def);
  const Operation& oc = b.op(dc.def);
  // A binding is a name for its operand; look through it on either side.
  if (oa.kind == OpKind::kBind) return same_expr(b, oa.operands[0], c, alias, depth + 1);
  if (oc.kind == OpKind::kBind) return same_expr(b, a, oc.operands[0], alias, depth + 1);
  if (oa.kind != oc.kind || oa.key != oc.key || oa.text != oc.text ||
      oa.cast_to != oc.cast_to || oa.imm != oc.imm || oa.type != oc.type ||
      oa.operands.size() != oc.operands.size()) {
    return false;
  }
  if (oa.kind == OpKind::kSymbol || oa.kind == OpKind::kAlloc ||
      oa.kind == OpKind::kMutable || oa.kind == OpKind::kFor) {
    // Distinct declarations are distinct storage; only the alias map above
    // may equate them.
    return false;
  }
  for (std::size_t i = 0; i < oa.operands.size(); ++i) {
    if (!same_expr(b, oa.operands[i], oc.operands[i], alias, depth + 1)) {
      return false;
    }
  }
  return true;
}

// `b`'s guard chain implies `a`'s: every condition a is under appears among
// b's with an equal or tighter bound, over the same affine expression.
bool guards_imply(const Body& body, const std::vector<ValueId>& outer,
                  const std::vector<ValueId>& inner) {
  std::vector<GuardAtom> want;
  for (ValueId g : outer) {
    if (!guard_atoms(body, g, &want)) return false;
  }
  std::vector<GuardAtom> have;
  for (ValueId g : inner) {
    if (!guard_atoms(body, g, &have)) return false;
  }
  for (const GuardAtom& w : want) {
    bool covered = false;
    for (const GuardAtom& h : have) {
      if (h.expr == w.expr && h.bound <= w.bound) covered = true;
    }
    if (!covered) return false;
  }
  return true;
}

class LdsFold final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "lds_fold";
  }

  std::size_t run(Body& body) const override {
    std::vector<Staging> staged;
    std::vector<ValueId> guards;
    scan(body, body.entry(), guards, &staged);
    if (staged.size() < 2) return 0;

    std::vector<bool> written(body.value_count(), false);
    collect_written(body, written);

    std::size_t fired = 0;
    std::vector<bool> gone(staged.size(), false);
    for (std::size_t i = 0; i < staged.size(); ++i) {
      if (gone[i]) continue;
      for (std::size_t j = i + 1; j < staged.size(); ++j) {
        if (gone[j] || !equivalent(body, staged[i], staged[j], written)) {
          continue;
        }
        // One array, both fills. The two fills write identical bytes to
        // identical addresses, so a thread reading the array while another is
        // re-filling it reads the value it would have read anyway.
        //
        // The now-unread allocation is left standing: deleting a declaration
        // nothing reads is what DCE is for, and doing it here would mean two
        // passes owning the same rule.
        body.replace_uses(staged[j].array, staged[i].array);
        gone[j] = true;
        ++fired;
        // The later fill is pure repetition once every thread that reaches it
        // has already passed a barrier behind the earlier one.
        if (staged[i].barrier_after &&
            workgroup_uniform_chain(body, staged[i].guards) &&
            guards_imply(body, staged[i].guards, staged[j].guards)) {
          body.erase(staged[j].loop);
          ++fired;
        }
      }
    }
    return fired;
  }

 private:
  static bool workgroup_uniform_chain(const Body& b,
                                      const std::vector<ValueId>& guards) {
    for (ValueId g : guards) {
      if (!workgroup_uniform(b, g)) return false;
    }
    return true;
  }

  // Walks the region tree collecting `__shared__` arrays whose only writer is
  // one loop whose body is exactly that one assignment.
  static void scan(const Body& b, RegionId r, std::vector<ValueId>& guards,
                   std::vector<Staging>* out) {
    for (OpId id : b.region(r).ops) {
      const Operation& o = b.op(id);
      if (o.erased) continue;
      if (o.kind == OpKind::kAlloc && o.type.space == Space::kWorkgroup &&
          o.has(kFlagArray)) {
        Staging s;
        s.alloc = id;
        s.array = o.result;
        out->push_back(s);
      }
      if (o.kind == OpKind::kFor) {
        match_fill(b, id, guards, out);
      }
      if (o.kind == OpKind::kBarrier) {
        for (Staging& s : *out) {
          if (s.loop != kNoOp && s.guards == guards) s.barrier_after = true;
        }
      }
      for (RegionId sub : o.regions) {
        if (o.kind == OpKind::kIf) guards.push_back(o.operands[0]);
        scan(b, sub, guards, out);
        if (o.kind == OpKind::kIf) guards.pop_back();
      }
    }
  }

  static void match_fill(const Body& b, OpId loop,
                         const std::vector<ValueId>& guards,
                         std::vector<Staging>* out) {
    const Operation& o = b.op(loop);
    if (o.regions.empty()) return;
    const Region& body_region = b.region(o.regions[0]);
    OpId only = kNoOp;
    for (OpId sid : body_region.ops) {
      if (b.op(sid).erased || !is_statement(b.op(sid).kind)) continue;
      if (only != kNoOp) return;  // more than a plain fill
      only = sid;
    }
    if (only == kNoOp) return;
    const Operation& stmt = b.op(only);
    if (stmt.kind != OpKind::kAssign) return;
    const ValueId target = stmt.operands[0];
    if (target >= b.value_count()) return;
    const ValueDef& td = b.value(target);
    if (td.def == kNoOp || b.op(td.def).kind != OpKind::kSubscript) return;
    const ValueId array = b.op(td.def).operands[0];
    for (Staging& s : *out) {
      if (s.array != array || s.loop != kNoOp) continue;
      s.loop = loop;
      s.induction = o.result;
      s.index = b.op(td.def).operands[1];
      s.source = stmt.operands[1];
      s.guards = guards;
      return;
    }
  }

  static bool equivalent(const Body& b, const Staging& a, const Staging& c,
                         const std::vector<bool>& written) {
    if (a.loop == kNoOp || c.loop == kNoOp) return false;
    if (a.array == kNoValue || c.array == kNoValue) return false;
    const Operation& aa = b.op(a.alloc);
    const Operation& ca = b.op(c.alloc);
    if (aa.type != ca.type || aa.imm != ca.imm) return false;

    // The two loops' induction variables stand for the same thing, and so do
    // the two arrays.
    std::unordered_map<ValueId, ValueId> alias;
    alias.emplace(a.induction, c.induction);
    alias.emplace(a.array, c.array);
    std::unordered_map<ValueId, ValueId> to_a;
    to_a.emplace(c.induction, a.induction);

    const Operation& al = b.op(a.loop);
    const Operation& cl = b.op(c.loop);
    for (int i = 0; i < 3; ++i) {
      if (!(AffineExpr::of(b, al.operands[static_cast<std::size_t>(i)]) ==
            AffineExpr::of(b, cl.operands[static_cast<std::size_t>(i)], &to_a))) {
        return false;
      }
    }
    if (!(AffineExpr::of(b, a.index) == AffineExpr::of(b, c.index, &to_a))) {
      return false;
    }
    if (!same_expr(b, a.source, c.source, alias, 0)) return false;

    // Reproducible only if nothing this kernel writes feeds the fill.
    std::vector<ValueId> reads;
    reads_of(b, a.source, reads, 0);
    reads_of(b, c.source, reads, 0);
    for (ValueId m : reads) {
      if (m < written.size() && written[m]) return false;
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> make_lds_fold() { return std::make_unique<LdsFold>(); }

}  // namespace lse::ir
