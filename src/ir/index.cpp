#include "lse/ir/index.hpp"

#include <algorithm>

namespace lse::ir {

namespace {

ValueId resolve(ValueId v, const std::unordered_map<ValueId, ValueId>* alias) {
  if (alias == nullptr) return v;
  const auto it = alias->find(v);
  return it == alias->end() ? v : it->second;
}

// A binding is a name for its operand, so linearizing looks straight through
// it. Without this every `e.let` would become an opaque symbol and nothing
// past the first one would compare equal.
ValueId through_binds(const Body& b, ValueId v) {
  for (int guard = 0; guard < 64; ++guard) {
    if (v == kNoValue || v >= b.value_count()) return v;
    const ValueDef& d = b.value(v);
    if (d.def == kNoOp) return v;
    const Operation& o = b.op(d.def);
    if (o.kind != OpKind::kBind) return v;
    v = o.operands[0];
  }
  return v;
}

}  // namespace

AffineExpr AffineExpr::constant(std::int64_t c) {
  AffineExpr e;
  e.c_ = c;
  return e;
}

AffineExpr AffineExpr::symbol(ValueId v) {
  AffineExpr e;
  e.terms_.push_back(Term{v, 1});
  return e;
}

void AffineExpr::add_term(ValueId v, std::int64_t k) {
  if (k == 0) return;
  const auto it = std::lower_bound(
      terms_.begin(), terms_.end(), v,
      [](const Term& t, ValueId s) { return t.symbol < s; });
  if (it != terms_.end() && it->symbol == v) {
    it->coeff += k;
    if (it->coeff == 0) terms_.erase(it);
    return;
  }
  terms_.insert(it, Term{v, k});
}

AffineExpr& AffineExpr::operator+=(const AffineExpr& o) {
  c_ += o.c_;
  for (const Term& t : o.terms_) add_term(t.symbol, t.coeff);
  return *this;
}

AffineExpr& AffineExpr::operator-=(const AffineExpr& o) {
  c_ -= o.c_;
  for (const Term& t : o.terms_) add_term(t.symbol, -t.coeff);
  return *this;
}

AffineExpr& AffineExpr::scale(std::int64_t k) {
  c_ *= k;
  if (k == 0) {
    terms_.clear();
    return *this;
  }
  for (Term& t : terms_) t.coeff *= k;
  return *this;
}

bool AffineExpr::depends_on(ValueId v) const noexcept {
  for (const Term& t : terms_) {
    if (t.symbol == v) return true;
  }
  return false;
}

bool operator==(const AffineExpr& a, const AffineExpr& b) {
  if (a.c_ != b.c_ || a.terms_.size() != b.terms_.size()) return false;
  for (std::size_t i = 0; i < a.terms_.size(); ++i) {
    if (a.terms_[i].symbol != b.terms_[i].symbol ||
        a.terms_[i].coeff != b.terms_[i].coeff) {
      return false;
    }
  }
  return true;
}

AffineExpr AffineExpr::of(const Body& b, ValueId v,
                          const std::unordered_map<ValueId, ValueId>* alias) {
  const ValueId aliased = resolve(v, alias);
  if (aliased != v) return symbol(aliased);
  const ValueId root = through_binds(b, v);
  if (root != v) {
    const ValueId r = resolve(root, alias);
    if (r != root) return symbol(r);
  }
  if (root == kNoValue || root >= b.value_count()) return symbol(root);
  const ValueDef& d = b.value(root);
  if (d.def == kNoOp) return symbol(root);
  const Operation& o = b.op(d.def);

  if (o.kind == OpKind::kConst && o.has(kFlagIntConst)) return constant(o.imm);
  // A baked extent is a number and linearizes; a runtime one is a value the
  // kernel reads and stays a symbol, which is exactly the distinction that
  // keeps it out of address arithmetic.
  if (o.kind == OpKind::kExtent && !o.has(kFlagRuntimeExtent)) {
    return constant(o.imm);
  }
  if (o.kind == OpKind::kBinary) {
    AffineExpr lhs = of(b, o.operands[0], alias);
    AffineExpr rhs = of(b, o.operands[1], alias);
    if (o.key == "+") {
      lhs += rhs;
      return lhs;
    }
    if (o.key == "-") {
      lhs -= rhs;
      return lhs;
    }
    if (o.key == "*") {
      if (const auto k = rhs.as_constant()) return lhs.scale(*k);
      if (const auto k = lhs.as_constant()) return rhs.scale(*k);
    }
  }
  return symbol(root);
}

bool guard_atoms(const Body& b, ValueId cond, std::vector<GuardAtom>* out,
                 const std::unordered_map<ValueId, ValueId>* alias) {
  if (cond == kNoValue || cond >= b.value_count()) return false;
  const ValueDef& d = b.value(cond);
  if (d.def == kNoOp) return false;
  const Operation& o = b.op(d.def);
  if (o.kind != OpKind::kBinary) return false;
  if (o.key == "&&") {
    return guard_atoms(b, o.operands[0], out, alias) &&
           guard_atoms(b, o.operands[1], out, alias);
  }
  if (o.key != "<") return false;
  const AffineExpr bound = AffineExpr::of(b, o.operands[1], alias);
  const auto c = bound.as_constant();
  if (!c.has_value()) return false;
  out->push_back(GuardAtom{AffineExpr::of(b, o.operands[0], alias), *c});
  return true;
}

bool workgroup_uniform(const Body& b, ValueId v) {
  if (v == kNoValue || v >= b.value_count()) return false;
  const ValueDef& d = b.value(v);
  if (d.def == kNoOp) return false;
  const Operation& o = b.op(d.def);
  switch (o.kind) {
    case OpKind::kConst:
    case OpKind::kExtent:
      return true;
    case OpKind::kCall:
      // Workgroup and grid coordinates are the same for every thread of a
      // workgroup; a thread id or a lane is the whole point of not being.
      return o.key == "thread.workgroup_id.x" ||
             o.key == "thread.workgroup_id.y" ||
             o.key == "thread.workgroup_size" || o.key == "thread.grid_dim.x";
    case OpKind::kBind:
    case OpKind::kBinary:
    case OpKind::kCast:
    case OpKind::kSelect:
      for (ValueId a : o.operands) {
        if (!workgroup_uniform(b, a)) return false;
      }
      return true;
    default:
      return false;
  }
}

}  // namespace lse::ir
