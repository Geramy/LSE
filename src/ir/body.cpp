#include "lse/ir/body.hpp"

#include <array>

namespace lse::ir {

namespace {

// Ordered by Scalar, like the backend type tables: a new element type without
// a suffix fails to compile rather than silently becoming "x".
constexpr std::array<std::string_view, 12> kSuffixes{{
    "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "bf16",
    "f32", "b",
}};
static_assert(kSuffixes.size() == static_cast<std::size_t>(Scalar::kBool) + 1,
              "kSuffixes must have one entry per Scalar, in enumerator order");

constexpr std::array<std::uint32_t, 12> kBytes{{
    1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 1,
}};

}  // namespace

std::string_view scalar_suffix(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kSuffixes.size() ? kSuffixes[i] : std::string_view{};
}

std::uint32_t scalar_bytes(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kBytes.size() ? kBytes[i] : 4u;
}

std::string_view to_string(DimKind k) noexcept {
  switch (k) {
    case DimKind::kParallel: return "parallel";
    case DimKind::kReduction: return "reduction";
    case DimKind::kSequential: return "sequential";
  }
  return "?";
}

std::string_view op_name(OpKind k) noexcept {
  switch (k) {
    case OpKind::kConst: return "const";
    case OpKind::kExtent: return "extent";
    case OpKind::kSymbol: return "symbol";
    case OpKind::kBinary: return "binary";
    case OpKind::kCast: return "cast";
    case OpKind::kSelect: return "select";
    case OpKind::kCall: return "call";
    case OpKind::kSubscript: return "subscript";
    case OpKind::kBind: return "bind";
    case OpKind::kMutable: return "mutable";
    case OpKind::kAlloc: return "alloc";
    case OpKind::kAssign: return "assign";
    case OpKind::kLoadVec: return "load";
    case OpKind::kStoreVec: return "store";
    case OpKind::kBarrier: return "barrier";
    case OpKind::kReturn: return "return";
    case OpKind::kReturnIf: return "return_if";
    case OpKind::kRawStmt: return "raw";
    case OpKind::kIf: return "if";
    case OpKind::kFor: return "for";
    case OpKind::kScope: return "scope";
  }
  return "?";
}

Body::Body(const TypeTable& types, const DialectSourceTable& intrinsics)
    : types_(&types), intrinsics_(&intrinsics) {
  regions_.push_back(Region{kNoOp, {}});
  stack_.push_back(0);
}

OpId Body::add(Operation o, std::string name) {
  return add_to(stack_.back(), std::move(o), std::move(name));
}

OpId Body::add_to(RegionId r, Operation o, std::string name) {
  return insert_at(r, regions_[r].ops.size(), std::move(o), std::move(name));
}

OpId Body::insert_at(RegionId r, std::size_t pos, Operation o,
                     std::string name) {
  const auto id = static_cast<OpId>(ops_.size());
  ops_.push_back(std::move(o));
  regions_[r].ops.insert(
      regions_[r].ops.begin() + static_cast<std::ptrdiff_t>(pos), id);
  Operation& stored = ops_[id];
  if (produces_result(stored.kind)) {
    stored.result = static_cast<ValueId>(values_.size());
    values_.push_back(ValueDef{id, stored.type, std::move(name)});
  } else {
    stored.result = kNoValue;
  }
  return id;
}

ValueId Body::add_value(Operation o, std::string name) {
  return ops_[add(std::move(o), std::move(name))].result;
}

RegionId Body::open_region(OpId parent) {
  const auto id = static_cast<RegionId>(regions_.size());
  regions_.push_back(Region{parent, {}});
  ops_[parent].regions.push_back(id);
  return id;
}

// A cached value whose op a pass has since deleted must not be handed out
// again: nothing would put the declaration back and the verifier would see a
// use with no definition in scope.
ValueId Body::interned(const std::string& key) {
  const auto it = interned_.find(key);
  if (it == interned_.end()) return kNoValue;
  if (!ops_[values_[it->second].def].erased) return it->second;
  interned_.erase(it);
  return kNoValue;
}

ValueId Body::symbol(std::string_view name, Type t) {
  std::string key = "sym\x01";
  key += name;
  key += '\x01';
  key += std::to_string(static_cast<int>(t.elem));
  key += '\x01';
  key += std::to_string(t.lanes);
  key += '\x01';
  key += std::to_string(static_cast<int>(t.space));
  if (const ValueId v = interned(key); v != kNoValue) return v;
  Operation o;
  o.kind = OpKind::kSymbol;
  o.type = t;
  const ValueId v = ops_[insert_at(entry(), 0, std::move(o), std::string(name))].result;
  interned_.emplace(std::move(key), v);
  return v;
}

ValueId Body::constant(std::string text, Type t, std::int64_t imm,
                       bool is_int) {
  std::string key = "con\x01";
  key += text;
  key += '\x01';
  key += std::to_string(static_cast<int>(t.elem));
  if (const ValueId v = interned(key); v != kNoValue) return v;
  Operation o;
  o.kind = OpKind::kConst;
  o.type = t;
  o.text = std::move(text);
  o.imm = imm;
  if (is_int) o.flags |= kFlagIntConst;
  const ValueId v = ops_[insert_at(entry(), 0, std::move(o), {})].result;
  interned_.emplace(std::move(key), v);
  return v;
}

ValueId Body::extent(std::string_view name, ExtentBinding binding,
                     ExtentRole role, std::string spelling,
                     std::int64_t value) {
  std::string key = "ext\x01";
  key += name;
  key += '\x01';
  key += spelling;
  if (const ValueId v = interned(key); v != kNoValue) return v;
  Operation o;
  o.kind = OpKind::kExtent;
  o.type = scalar_type(Scalar::kU32);
  o.text = std::move(spelling);
  o.key = std::string(name);
  o.imm = value;
  if (binding == ExtentBinding::kRuntime) o.flags |= kFlagRuntimeExtent;
  if (role == ExtentRole::kWindowBase) o.flags |= kFlagWindowBase;
  const ValueId v = ops_[insert_at(entry(), 0, std::move(o), {})].result;
  interned_.emplace(std::move(key), v);
  if (binding == ExtentBinding::kRuntime) {
    runtime_extents_.emplace_back(name);
  }
  return v;
}

void Body::erase(OpId id) {
  Operation& o = ops_[id];
  if (o.erased) return;
  o.erased = true;
  // Erasing an `if` or a `for` erases what it owned. Leaving the nested ops
  // marked live would keep counting their operands as used, so the values only
  // that dead block read would never be collected.
  const std::vector<RegionId> owned = o.regions;
  for (RegionId r : owned) {
    const std::vector<OpId> inner = regions_[r].ops;
    for (OpId sub : inner) erase(sub);
  }
  for (Region& r : regions_) {
    for (std::size_t i = 0; i < r.ops.size(); ++i) {
      if (r.ops[i] == id) {
        r.ops.erase(r.ops.begin() + static_cast<std::ptrdiff_t>(i));
        return;
      }
    }
  }
}

std::size_t Body::replace_uses(ValueId from, ValueId to) {
  std::size_t n = 0;
  for (Operation& o : ops_) {
    if (o.erased) continue;
    for (ValueId& v : o.operands) {
      if (v == from) {
        v = to;
        ++n;
      }
    }
  }
  return n;
}

std::vector<std::uint32_t> Body::use_counts() const {
  std::vector<std::uint32_t> counts(values_.size(), 0);
  for (const Operation& o : ops_) {
    if (o.erased) continue;
    for (ValueId v : o.operands) {
      if (v < counts.size()) ++counts[v];
    }
  }
  return counts;
}

std::string Body::fresh_name(std::string_view stem) {
  std::string out = prefix_;
  out += stem;
  out += std::to_string(next_id_++);
  return out;
}

std::string Body::vector_typedef(Scalar s, int n) {
  const std::string name =
      "lse_v" + std::to_string(n) + "_" + std::string(scalar_suffix(s));
  const std::string decl = types_->vector_typedef(s, n, name);
  for (const std::string& t : typedefs_) {
    if (t == decl) return name;
  }
  typedefs_.push_back(decl);
  return name;
}

bool Body::contains(OpKind k) const {
  for (const Operation& o : ops_) {
    if (!o.erased && o.kind == k) return true;
  }
  return false;
}

void Body::splice(const Body& other, RegionId into) {
  std::vector<ValueId> vmap(other.values_.size(), kNoValue);

  // Depth-first over `other`'s region tree, appending as we go. `stack_` is
  // the insertion point, so a nested region is filled while its parent op is
  // already in place.
  auto copy_region = [&](auto&& self, RegionId src, RegionId dst) -> void {
    push(dst);
    for (OpId sid : other.regions_[src].ops) {
      const Operation& s = other.ops_[sid];
      if (s.erased) continue;
      // Symbols and constants intern, so two stages naming the same buffer or
      // the same extent become one value here rather than after a rewrite.
      if (s.kind == OpKind::kSymbol) {
        vmap[s.result] = symbol(other.values_[s.result].name, s.type);
        continue;
      }
      if (s.kind == OpKind::kConst) {
        vmap[s.result] =
            constant(s.text, s.type, s.imm, (s.flags & kFlagIntConst) != 0);
        continue;
      }
      if (s.kind == OpKind::kExtent) {
        vmap[s.result] =
            extent(s.key,
                   (s.flags & kFlagRuntimeExtent) != 0 ? ExtentBinding::kRuntime
                                                       : ExtentBinding::kLiteral,
                   (s.flags & kFlagWindowBase) != 0 ? ExtentRole::kWindowBase
                                                    : ExtentRole::kSize,
                   s.text, s.imm);
        continue;
      }
      Operation o = s;
      o.regions.clear();
      for (ValueId& v : o.operands) {
        if (v < vmap.size() && vmap[v] != kNoValue) v = vmap[v];
      }
      std::string name;
      if (s.result != kNoValue) name = other.values_[s.result].name;
      const OpId nid = add(std::move(o), std::move(name));
      if (s.result != kNoValue) vmap[s.result] = ops_[nid].result;
      for (RegionId sub : s.regions) self(self, sub, open_region(nid));
    }
    pop();
  };
  copy_region(copy_region, other.entry(), into);

  for (const std::string& t : other.typedefs_) {
    bool seen = false;
    for (const std::string& have : typedefs_) {
      if (have == t) seen = true;
    }
    if (!seen) typedefs_.push_back(t);
  }
}

}  // namespace lse::ir
