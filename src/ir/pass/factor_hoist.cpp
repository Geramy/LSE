#include "lse/ir/pass/factor_hoist.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lse::ir {

namespace {

// Every op inside a loop, its nested regions included.
void collect_inside(const Body& b, RegionId r, std::unordered_set<OpId>& out) {
  for (OpId id : b.region(r).ops) {
    if (b.op(id).erased) continue;
    out.insert(id);
    for (RegionId sub : b.op(id).regions) collect_inside(b, sub, out);
  }
}

// INVARIANCE IS A PROPERTY OF THE DEPENDENCY CHAIN, NOT OF POSITION.
//
// A value does not move with the loop if it is defined outside it, OR if it is
// computed inside from nothing that moves. Asking only where a value was
// written misses the second case entirely and refuses hoists that are plainly
// valid -- a scale derived inside the loop body from two constants moves with
// nothing, whatever line it sits on.
class Invariance {
 public:
  Invariance(const Body& b, const std::unordered_set<OpId>& inside)
      : b_(b), inside_(inside) {}

  bool operator()(ValueId v) {
    if (v == kNoValue) return false;
    if (const auto it = memo_.find(v); it != memo_.end()) return it->second;
    // Cycles cannot occur in this IR -- a value's operands are defined before
    // it -- but seeding false keeps a malformed body from recursing forever.
    memo_[v] = false;
    const OpId def = b_.value(v).def;
    bool answer = false;
    if (def == kNoOp || inside_.count(def) == 0) {
      answer = true;                     // defined outside: cannot move
    } else if (is_pure(b_.op(def).kind)) {
      answer = true;                     // pure and built only from invariants
      for (ValueId in : b_.op(def).operands) {
        if (!(*this)(in)) {
          answer = false;
          break;
        }
      }
    }
    memo_[v] = answer;
    return answer;
  }

 private:
  const Body& b_;
  const std::unordered_set<OpId>& inside_;
  std::unordered_map<ValueId, bool> memo_;
};

// A term is a product of factors; an expression is a sum of terms. Both `+`
// and the dialect's fused multiply-add flatten into this one form, so nothing
// downstream needs to know which spelling produced it. THIS is what replaces
// matching on "fma": the row contributes {a, b} as a term and recurses into
// its addend like any other sum.
using Term = std::vector<ValueId>;

bool is_add(const Operation& o) {
  return o.kind == OpKind::kBinary && o.key == "+";
}
bool is_mul(const Operation& o) {
  return o.kind == OpKind::kBinary && o.key == "*";
}
// a * b + c, however the dialect spells it.
bool is_fused_mul_add(const Operation& o) {
  return o.kind == OpKind::kCall && o.operands.size() == 3 &&
         (o.key == "fma" || o.key == "mad");
}

void flatten_product(const Body& b, ValueId v, Term& out,
                     std::unordered_set<OpId>& walked) {
  const OpId def = v == kNoValue ? kNoOp : b.value(v).def;
  if (def != kNoOp && is_mul(b.op(def)) && b.op(def).operands.size() == 2) {
    walked.insert(def);
    flatten_product(b, b.op(def).operands[0], out, walked);
    flatten_product(b, b.op(def).operands[1], out, walked);
    return;
  }
  out.push_back(v);
}

// `walked` collects the ops this expression is BUILT FROM. They are part of
// the update being rewritten, so a later scan for "who else reads the
// accumulator" must not count them -- the `+` node naturally names acc, and
// counting it as an outside reader refuses every hoist there is.
void flatten_sum(const Body& b, ValueId v, std::vector<Term>& out,
                 std::unordered_set<OpId>& walked) {
  const OpId def = v == kNoValue ? kNoOp : b.value(v).def;
  if (def != kNoOp) {
    const Operation& o = b.op(def);
    if (is_add(o) && o.operands.size() == 2) {
      walked.insert(def);
      flatten_sum(b, o.operands[0], out, walked);
      flatten_sum(b, o.operands[1], out, walked);
      return;
    }
    if (is_fused_mul_add(o)) {
      walked.insert(def);
      Term product;
      flatten_product(b, o.operands[0], product, walked);
      flatten_product(b, o.operands[1], product, walked);
      out.push_back(std::move(product));
      flatten_sum(b, o.operands[2], out, walked);
      return;
    }
  }
  Term single;
  flatten_product(b, v, single, walked);
  out.push_back(std::move(single));
}

bool is_zero_literal(const Body& b, ValueId v) {
  if (v == kNoValue) return false;
  const OpId def = b.value(v).def;
  if (def == kNoOp || b.op(def).kind != OpKind::kConst) return false;
  if (b.op(def).imm != 0) return false;
  for (char c : b.op(def).text) {
    if (c != '0' && c != '.' && c != 'f' && c != '-' && c != '+') return false;
  }
  return true;
}

struct Site {
  OpId assign = kNoOp;
  RegionId region = 0;
  ValueId factor = kNoValue;
  Term rest;              // the term with `factor` removed
  std::unordered_set<OpId> walked;   // ops the old update was built from
};

// Walk regions in order so every assignment is found WITH the region holding
// it -- a rewrite has to insert next to the statement, not at whatever the
// body's insertion point happens to be.
void walk_assigns(const Body& b, RegionId r,
                  std::vector<std::pair<RegionId, OpId>>& out) {
  for (OpId id : b.region(r).ops) {
    if (b.op(id).erased) continue;
    if (b.op(id).kind == OpKind::kAssign) out.emplace_back(r, id);
    for (RegionId sub : b.op(id).regions) walk_assigns(b, sub, out);
  }
}

class FactorHoist final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "factor_hoist";
  }

  std::size_t run(Body& body) const override {
    std::size_t fired = 0;
    for (RegionId r = 0; r < body.region_count(); ++r) {
      const std::vector<OpId> ops = body.region(r).ops;
      for (OpId loop : ops) {
        if (body.op(loop).erased) continue;
        if (body.op(loop).kind != OpKind::kFor) continue;
        if (body.op(loop).regions.empty()) continue;
        if (try_loop(body, r, loop)) ++fired;
      }
    }
    return fired;
  }

 private:
  static bool try_loop(Body& body, RegionId parent, OpId loop) {
    std::unordered_set<OpId> inside;
    for (RegionId sub : body.op(loop).regions) collect_inside(body, sub, inside);
    Invariance invariant(body, inside);

    std::vector<std::pair<RegionId, OpId>> assigns;
    for (RegionId sub : body.op(loop).regions) walk_assigns(body, sub, assigns);

    std::unordered_map<ValueId, std::vector<Site>> found;
    std::unordered_set<ValueId> disqualified;

    for (const auto& [region, id] : assigns) {
      const Operation& o = body.op(id);
      if (o.operands.size() != 2) continue;
      const ValueId acc = o.operands[0];
      const OpId decl = acc == kNoValue ? kNoOp : body.value(acc).def;
      if (decl == kNoOp || body.op(decl).kind != OpKind::kMutable) continue;
      if (inside.count(decl) != 0) continue;
      // The slot must be declared in the region that HOLDS this loop, not
      // merely somewhere outside it. If an enclosing loop carries the
      // accumulator, its total spans outer iterations while the factor would
      // be applied once per inner loop -- s*(s*a1 + a2) instead of
      // s*(a1 + a2). A tiled linear is exactly that shape, and it returned 6
      // where it owed 1.0 and 2.0.
      const std::vector<OpId>& holds = body.region(parent).ops;
      if (std::find(holds.begin(), holds.end(), decl) == holds.end()) {
        disqualified.insert(acc);
        continue;
      }
      if (body.op(decl).operands.empty() ||
          !is_zero_literal(body, body.op(decl).operands[0])) {
        disqualified.insert(acc);   // s*(init+SUM) != init + s*SUM
        continue;
      }

      // The update, as a sum of products, with no reference to how it was
      // written. Valid when it is exactly `acc` plus one other term, and that
      // term carries a factor that does not move with the loop.
      std::vector<Term> terms;
      std::unordered_set<OpId> walked;
      flatten_sum(body, o.operands[1], terms, walked);
      if (terms.size() != 2) {
        disqualified.insert(acc);
        continue;
      }
      const bool first_is_acc = terms[0].size() == 1 && terms[0][0] == acc;
      const bool second_is_acc = terms[1].size() == 1 && terms[1][0] == acc;
      if (first_is_acc == second_is_acc) {
        disqualified.insert(acc);
        continue;
      }
      Term& product = first_is_acc ? terms[1] : terms[0];

      const auto at = std::find_if(product.begin(), product.end(),
                                   [&](ValueId f) { return invariant(f); });
      if (at == product.end()) {
        disqualified.insert(acc);
        continue;
      }
      Site site;
      site.assign = id;
      site.region = region;
      site.walked = std::move(walked);
      site.factor = *at;
      site.rest.assign(product.begin(), product.end());
      site.rest.erase(site.rest.begin() + (at - product.begin()));
      if (site.rest.empty()) {   // the whole term was invariant: not a sum over k
        disqualified.insert(acc);
        continue;
      }
      found[acc].push_back(std::move(site));
    }

    for (auto& [acc, sites] : found) {
      if (disqualified.count(acc) != 0) continue;
      const ValueId s = sites.front().factor;
      bool uniform = true;
      for (const Site& site : sites) uniform = uniform && site.factor == s;
      if (!uniform) continue;
      if (read_elsewhere(body, acc, inside, sites)) continue;

      for (const Site& site : sites) rewrite_site(body, acc, site);

      // Pay the factor once, after the loop. Inserted, not appended: appending
      // puts the definition after the use and it stops dominating it.
      const std::size_t at = index_after(body, parent, loop);
      Operation mul;
      mul.kind = OpKind::kBinary;
      mul.key = "*";
      mul.type = body.value(acc).type;
      mul.operands = {acc, s};
      const OpId mul_id = body.insert_at(parent, at, mul);

      Operation store;
      store.kind = OpKind::kAssign;
      store.operands = {acc, body.op(mul_id).result};
      body.insert_at(parent, at + 1, store);
      return true;
    }
    return false;
  }

  // Rebuild the update as `acc + (remaining factors)`, whatever it used to be.
  static void rewrite_site(Body& body, ValueId acc, const Site& site) {
    const std::size_t pos = index_of(body, site.region, site.assign);
    ValueId product = site.rest.front();
    for (std::size_t i = 1; i < site.rest.size(); ++i) {
      Operation mul;
      mul.kind = OpKind::kBinary;
      mul.key = "*";
      mul.type = body.value(product).type;
      mul.operands = {product, site.rest[i]};
      product = body.op(body.insert_at(site.region, pos, mul)).result;
    }
    Operation add;
    add.kind = OpKind::kBinary;
    add.key = "+";
    add.type = body.value(acc).type;
    add.operands = {acc, product};
    const OpId add_id =
        body.insert_at(site.region, index_of(body, site.region, site.assign),
                       add);
    body.op(site.assign).operands[1] = body.op(add_id).result;
  }

  static bool read_elsewhere(const Body& b, ValueId acc,
                             const std::unordered_set<OpId>& inside,
                             const std::vector<Site>& sites) {
    std::unordered_set<OpId> allowed;
    for (const Site& s : sites) {
      allowed.insert(s.assign);
      allowed.insert(s.walked.begin(), s.walked.end());
    }
    for (OpId id : inside) {
      const Operation& o = b.op(id);
      if (o.erased) continue;
      if (allowed.count(id) != 0) continue;
      for (ValueId v : o.operands) {
        if (v == acc) return true;
      }
    }
    return false;
  }

  static std::size_t index_of(const Body& b, RegionId r, OpId op) {
    const std::vector<OpId>& ops = b.region(r).ops;
    for (std::size_t i = 0; i < ops.size(); ++i) {
      if (ops[i] == op) return i;
    }
    return ops.size();
  }
  static std::size_t index_after(const Body& b, RegionId r, OpId op) {
    return index_of(b, r, op) + 1;
  }
};

}  // namespace

std::unique_ptr<Pass> make_factor_hoist() {
  return std::make_unique<FactorHoist>();
}

}  // namespace lse::ir
