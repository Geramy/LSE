// One kernel body: the op arena, the region tree, and the builder that fills
// them.
//
// Ops, values and regions live in flat arenas addressed by index. Nothing
// holds a pointer into them, so a pass may append while it walks and an id
// stays valid for the life of the body. Erasing an op drops it from its
// region's list and marks it erased; the arena slot stays so no id dangles.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lse/ir/dialect.hpp"
#include "lse/ir/op.hpp"
#include "lse/ir/space.hpp"
#include "lse/ir/types.hpp"

namespace lse::ir {

class Body {
 public:
  Body(const TypeTable& types, const DialectSourceTable& intrinsics);

  Body(const Body&) = delete;
  Body& operator=(const Body&) = delete;
  Body(Body&&) noexcept = default;
  Body& operator=(Body&&) noexcept = default;

  // -- arenas ---------------------------------------------------------------
  [[nodiscard]] std::size_t op_count() const noexcept { return ops_.size(); }
  [[nodiscard]] Operation& op(OpId id) { return ops_[id]; }
  [[nodiscard]] const Operation& op(OpId id) const { return ops_[id]; }
  [[nodiscard]] std::size_t value_count() const noexcept {
    return values_.size();
  }
  [[nodiscard]] const ValueDef& value(ValueId id) const { return values_[id]; }
  [[nodiscard]] ValueDef& value(ValueId id) { return values_[id]; }
  [[nodiscard]] std::size_t region_count() const noexcept {
    return regions_.size();
  }
  [[nodiscard]] Region& region(RegionId id) { return regions_[id]; }
  [[nodiscard]] const Region& region(RegionId id) const { return regions_[id]; }
  [[nodiscard]] RegionId entry() const noexcept { return 0; }

  [[nodiscard]] const TypeTable& types() const noexcept { return *types_; }
  [[nodiscard]] const DialectSourceTable& intrinsics() const noexcept {
    return *intrinsics_;
  }
  [[nodiscard]] const std::vector<std::string>& typedefs() const noexcept {
    return typedefs_;
  }

  // -- building -------------------------------------------------------------
  // Append `op` to the current insertion region. Mints the result value when
  // the op declares one; pass `name` empty for an inline expression.
  OpId add(Operation op, std::string name = {});
  // Same, but the caller wants the result rather than the op.
  ValueId add_value(Operation op, std::string name = {});

  RegionId open_region(OpId parent);
  // Append to a named region rather than the insertion point.
  OpId add_to(RegionId r, Operation op, std::string name = {});
  // Interned pure ops go to the FRONT of the entry region. They print nothing,
  // so their position is not observable, and only there do they dominate every
  // use: a constant first mentioned inside a loop would otherwise be defined
  // after the `if` whose body already referred to it.
  OpId insert_at(RegionId r, std::size_t pos, Operation op,
                 std::string name = {});
  void push(RegionId r) { stack_.push_back(r); }
  void pop() { stack_.pop_back(); }
  [[nodiscard]] RegionId here() const noexcept { return stack_.back(); }

  // Interned: the same (name, type) is one value, which is what lets two
  // stages that read the same buffer compare equal without a rewrite.
  ValueId symbol(std::string_view name, Type t);
  // A literal. Interned on (text, type) for the same reason.
  ValueId constant(std::string text, Type t, std::int64_t imm, bool is_int);
  // A tensor dimension. `spelling` is the literal for a baked extent and the
  // dispatch-constant expression for a runtime one.
  ValueId extent(std::string_view name, ExtentBinding binding, ExtentRole role,
                 std::string spelling, std::int64_t value);
  // The runtime extents this body reads, in first-use order. An emitter turns
  // each into a dispatch-constants field; a body with none needs no field.
  [[nodiscard]] const std::vector<std::string>& runtime_extents() const noexcept {
    return runtime_extents_;
  }

  // What this kernel covers. Empty means it covers a grid and nothing may be
  // split off it — which is every kernel today.
  [[nodiscard]] IterationSpace& space() noexcept { return space_; }
  [[nodiscard]] const IterationSpace& space() const noexcept { return space_; }

  // -- editing --------------------------------------------------------------
  void erase(OpId id);
  // Replace every *operand* use of `from` with `to`. Raw statement text is not
  // rewritten, which is why a kRawStmt lists what it mentions: the original
  // definition stays live rather than being silently renamed underneath it.
  std::size_t replace_uses(ValueId from, ValueId to);
  // How many live ops name each value as an operand.
  [[nodiscard]] std::vector<std::uint32_t> use_counts() const;

  // -- names ----------------------------------------------------------------
  // One id sequence per body, shared by every generator layered on it, so two
  // recorders on the same body cannot mint the same name.
  [[nodiscard]] std::uint32_t fresh_id() noexcept { return next_id_++; }
  [[nodiscard]] std::string fresh_name(std::string_view stem);
  void set_name_prefix(std::string p) { prefix_ = std::move(p); }
  [[nodiscard]] const std::string& name_prefix() const noexcept {
    return prefix_;
  }

  // Declares the typedef once and returns the name to use for it.
  std::string vector_typedef(Scalar s, int n);

  // -- queries --------------------------------------------------------------
  // Walks every region. `fn(OpId)` on each live op, parents before children.
  template <class F>
  void walk(F&& fn) const {
    walk_region(entry(), fn);
  }
  [[nodiscard]] bool contains(OpKind k) const;

  // Splice every op of `other`'s entry region onto the end of this body's
  // region `into`, remapping its ids. Used to merge fused sibling stages into
  // one body so a pass can see across what used to be separate kernels.
  void splice(const Body& other, RegionId into);

 private:
  template <class F>
  void walk_region(RegionId r, F& fn) const {
    for (OpId id : regions_[r].ops) {
      if (ops_[id].erased) continue;
      fn(id);
      for (RegionId sub : ops_[id].regions) walk_region(sub, fn);
    }
  }

  const TypeTable* types_;
  const DialectSourceTable* intrinsics_;
  std::vector<Operation> ops_;
  std::vector<ValueDef> values_;
  std::vector<Region> regions_;
  std::vector<RegionId> stack_;
  std::vector<std::string> typedefs_;
  std::unordered_map<std::string, ValueId> interned_;
  std::vector<std::string> runtime_extents_;
  IterationSpace space_;
  ValueId interned(const std::string& key);

  std::string prefix_;
  std::uint32_t next_id_ = 0;
};

}  // namespace lse::ir
