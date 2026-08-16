#include "lse/ir/pass/dce.hpp"

#include <vector>

namespace lse::ir {

namespace {

// A declaration nothing reads, or a block with nothing left in it. Assignments,
// stores, barriers, returns and raw statements are effects and are never dead;
// a mutable slot is never dead either, because the assignments that write it
// name it and therefore use it.
bool deletable(const Body& b, OpId id) {
  const Operation& o = b.op(id);
  switch (o.kind) {
    case OpKind::kIf:
    case OpKind::kFor:
    case OpKind::kScope:
      return o.regions.empty() || b.region(o.regions[0]).ops.empty();
    case OpKind::kConst:
    case OpKind::kExtent:
    case OpKind::kSymbol:
    case OpKind::kBinary:
    case OpKind::kCast:
    case OpKind::kSelect:
    case OpKind::kCall:
    case OpKind::kSubscript:
    case OpKind::kBind:
    case OpKind::kAlloc:
    case OpKind::kLoadVec:
      return true;
    default:
      return false;
  }
}

// Whether removing it removes a line of generated source. An inline expression
// was never printed, so deleting it is bookkeeping, not an optimization, and
// counting it would inflate what the pass claims to have done.
bool prints(const Body& b, OpId id) {
  const Operation& o = b.op(id);
  switch (o.kind) {
    case OpKind::kBind:
    case OpKind::kAlloc:
    case OpKind::kLoadVec:
    case OpKind::kIf:
    case OpKind::kFor:
    case OpKind::kScope:
      return true;
    default:
      (void)b;
      return false;
  }
}

class Dce final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const noexcept override { return "dce"; }

  std::size_t run(Body& body) const override {
    std::size_t fired = 0;
    // To a fixpoint: deleting a declaration frees whatever only it read.
    for (bool changed = true; changed;) {
      changed = false;
      const std::vector<std::uint32_t> uses = body.use_counts();
      for (OpId id = 0; id < body.op_count(); ++id) {
        const Operation& o = body.op(id);
        if (o.erased) continue;
        if (!deletable(body, id)) continue;
        const ValueId r = o.result;
        const bool unused = r == kNoValue || r >= uses.size() || uses[r] == 0;
        if (!unused) continue;
        const bool counted = prints(body, id);
        body.erase(id);
        if (counted) ++fired;
        changed = true;
      }
    }
    return fired;
  }
};

}  // namespace

std::unique_ptr<Pass> make_dce() { return std::make_unique<Dce>(); }

}  // namespace lse::ir
