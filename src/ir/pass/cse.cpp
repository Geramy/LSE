#include "lse/ir/pass/cse.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lse::ir {

namespace {

// A value a raw statement mentions by name. The statement's text is opaque, so
// redirecting a use would leave it naming a definition that is about to be
// deleted. Mirrors the printer exactly: a named value stops the walk because
// that is all the text contains of it; an unnamed one was inlined, so whatever
// it inlined is mentioned too.
void pin_from(const Body& b, ValueId v, std::vector<bool>& pinned) {
  if (v == kNoValue || v >= b.value_count()) return;
  const ValueDef& d = b.value(v);
  if (!d.name.empty()) {
    pinned[v] = true;
    return;
  }
  if (d.def == kNoOp) return;
  for (ValueId a : b.op(d.def).operands) pin_from(b, a, pinned);
}

// Whether a value is the same number wherever it is evaluated. A memory read is
// not (the memory may be written between); a mutable slot is not; a cross-lane
// op is not, because its result depends on values other lanes are holding.
bool pure_value(const Body& b, ValueId v, std::vector<signed char>& memo) {
  if (v == kNoValue || v >= b.value_count()) return false;
  if (memo[v] != -1) return memo[v] != 0;
  memo[v] = 0;  // cycles are impossible here, but do not loop if one appears
  const ValueDef& d = b.value(v);
  if (d.def == kNoOp) return false;
  const Operation& o = b.op(d.def);
  bool ok = false;
  switch (o.kind) {
    case OpKind::kConst:
    case OpKind::kExtent:
    case OpKind::kSymbol:
    case OpKind::kFor:  // an induction variable is a leaf, scoped to its loop
      ok = true;
      break;
    case OpKind::kCall:
      ok = o.key.rfind("wave.", 0) != 0;
      break;
    case OpKind::kBinary:
    case OpKind::kCast:
    case OpKind::kSelect:
    case OpKind::kBind:
      ok = true;
      for (ValueId a : o.operands) {
        if (!pure_value(b, a, memo)) ok = false;
      }
      break;
    default:
      ok = false;
      break;
  }
  memo[v] = ok ? 1 : 0;
  return ok;
}

std::string key_of(const Body& b, const Operation& o) {
  std::string k;
  k += static_cast<char>(static_cast<int>(o.kind) + 1);
  // A symbol IS its name — `in0` and `out` are different memory with the same
  // type — while a declaration's name is generated and is exactly what CSE
  // exists to collapse. So the name is part of identity for one and not the
  // other.
  if (o.kind == OpKind::kSymbol && o.result != kNoValue) {
    k += '\x01';
    k += b.value(o.result).name;
  }
  k += '\x01';
  k += o.key;
  k += '\x01';
  k += o.text;
  k += '\x01';
  k += std::to_string(static_cast<int>(o.cast_to));
  k += '\x01';
  k += std::to_string(static_cast<int>(o.type.elem));
  k += ':';
  k += std::to_string(o.type.lanes);
  for (ValueId v : o.operands) {
    k += '\x01';
    k += std::to_string(v);
  }
  return k;
}

class Cse final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const noexcept override { return "cse"; }

  std::size_t run(Body& body) const override {
    std::vector<bool> pinned(body.value_count(), false);
    for (std::size_t i = 0; i < body.op_count(); ++i) {
      const Operation& o = body.op(static_cast<OpId>(i));
      if (o.erased || o.kind != OpKind::kRawStmt) continue;
      for (ValueId v : o.operands) pin_from(body, v, pinned);
    }

    std::vector<signed char> memo(body.value_count(), -1);
    std::vector<std::unordered_map<std::string, ValueId>> scopes;
    std::size_t fired = 0;
    walk(body, body.entry(), scopes, memo, pinned, &fired);
    return fired;
  }

 private:
  static void walk(Body& body, RegionId r,
                   std::vector<std::unordered_map<std::string, ValueId>>& scopes,
                   std::vector<signed char>& memo,
                   const std::vector<bool>& pinned, std::size_t* fired) {
    scopes.emplace_back();
    // Copy: the loop erases from the region while iterating.
    const std::vector<OpId> ops = body.region(r).ops;
    for (OpId id : ops) {
      const OpKind kind = body.op(id).kind;
      const ValueId result = body.op(id).result;
      bool folded = false;
      if (result != kNoValue && kind != OpKind::kFor &&
          pure_value(body, result, memo) &&
          !(result < pinned.size() && pinned[result])) {
        const std::string k = key_of(body, body.op(id));
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
          const auto found = it->find(k);
          if (found == it->end()) continue;
          body.replace_uses(result, found->second);
          body.erase(id);
          // Only a declaration was a line of source; folding an inline
          // expression changes nothing anybody could read.
          if (is_declaration(kind)) ++*fired;
          folded = true;
          break;
        }
        if (!folded) scopes.back().emplace(k, result);
      }
      if (folded) continue;
      for (RegionId sub : body.op(id).regions) walk(body, sub, scopes, memo, pinned, fired);
    }
    scopes.pop_back();
  }
};

}  // namespace

std::unique_ptr<Pass> make_cse() { return std::make_unique<Cse>(); }

}  // namespace lse::ir
