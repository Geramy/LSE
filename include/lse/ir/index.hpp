// Index expressions, symbolically.
//
// Every extent in this engine is baked in as a literal, so every address a
// kernel computes is AFFINE over loop variables, thread ids and constants:
// `(w_base + col) * 1024u + k0 + lane * 8u`. That is the property that lets a
// pass decide two loads read the same address without matching text, and it is
// the difference between folding the duplicate LDS staging and pattern-matching
// strings with extra steps.
//
// The form is the usual one: a sum of (coefficient * symbol) terms plus a
// constant. A subexpression the analysis cannot linearize — a division, a
// modulus, a call — becomes an opaque symbol, so the representation is total
// and never lies about what it knows.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/ir/body.hpp"

namespace lse::ir {

class AffineExpr {
 public:
  struct Term {
    ValueId symbol;
    std::int64_t coeff;
  };

  AffineExpr() = default;

  [[nodiscard]] static AffineExpr constant(std::int64_t c);
  [[nodiscard]] static AffineExpr symbol(ValueId v);

  // Linearize `v`. `alias` renames symbols before they are recorded, which is
  // how two loops' induction variables are compared as if they were one.
  [[nodiscard]] static AffineExpr of(
      const Body& b, ValueId v,
      const std::unordered_map<ValueId, ValueId>* alias = nullptr);

  [[nodiscard]] std::int64_t constant_term() const noexcept { return c_; }
  [[nodiscard]] const std::vector<Term>& terms() const noexcept { return terms_; }
  [[nodiscard]] std::optional<std::int64_t> as_constant() const noexcept {
    if (terms_.empty()) return c_;
    return std::nullopt;
  }
  [[nodiscard]] bool depends_on(ValueId v) const noexcept;

  AffineExpr& operator+=(const AffineExpr& o);
  AffineExpr& operator-=(const AffineExpr& o);
  AffineExpr& scale(std::int64_t k);

  friend bool operator==(const AffineExpr& a, const AffineExpr& b);

 private:
  void add_term(ValueId v, std::int64_t k);

  // Sorted by symbol id, so equality is a straight comparison.
  std::vector<Term> terms_;
  std::int64_t c_ = 0;
};

// One conjunct of a guard: `expr < bound`, which is the only comparison shape
// the kernels here build guards from.
struct GuardAtom {
  AffineExpr expr;
  std::int64_t bound = 0;
};

// Split a guard condition into its `&&` conjuncts. Returns false when the
// condition contains anything other than `&&` of `expr < constant` — the
// analysis then knows nothing about it rather than guessing.
[[nodiscard]] bool guard_atoms(
    const Body& b, ValueId cond, std::vector<GuardAtom>* out,
    const std::unordered_map<ValueId, ValueId>* alias = nullptr);

// Whether every thread of a workgroup agrees on this value. True for a guard
// built from workgroup ids and constants, false as soon as a thread id or a
// memory read is involved — which is what decides whether a barrier inside the
// guard is reached by the whole workgroup.
[[nodiscard]] bool workgroup_uniform(const Body& b, ValueId v);

}  // namespace lse::ir
