#include "lse/backends/hrx/loomc/loom_print.hpp"

#include <cctype>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/ir/lower.hpp"

namespace lse::backend {

namespace {

using namespace lse::ir;

// The four operand classes Loom draws a line between. Which one a value is in
// decides the whole op name, not a suffix on it: `+` is index.add, scalar.addf
// or scalar.addi and there is no shared spelling to fall back to.
enum class Cls { kIndex, kFloat, kSignedInt, kUnsignedInt, kBool, kOther };

bool is_signed(Scalar s) noexcept {
  return s == Scalar::kI8 || s == Scalar::kI16 || s == Scalar::kI32 ||
         s == Scalar::kI64;
}

bool is_float(Scalar s) noexcept {
  return s == Scalar::kF16 || s == Scalar::kBF16 || s == Scalar::kF32;
}

std::uint32_t width_bits(Scalar s) noexcept { return scalar_bytes(s) * 8u; }

// One value, as the printer knows it after emitting its definition.
struct Typed {
  std::string type;  // the Loom type string, e.g. "index", "f32", "view<..>"
  Scalar elem = Scalar::kF32;
  Cls cls = Cls::kOther;
  bool defined = false;
};

class Printer {
 public:
  Printer(const Body& b, const LoomPrintOptions& o) : b_(b), o_(o) {}

  Result<LoomBody> run() {
    types_.assign(b_.value_count(), Typed{});
    classify_words();
    mark_store_targets();
    collect_assigns(b_.entry());
    collect_subs();
    LSE_RETURN_IF_ERROR(emit_region(b_.entry(), o_.indent, true));
    LoomBody out;
    out.text = std::move(text_);
    out.result = result_;
    out.result_type = result_type_;
    return out;
  }

 private:
  // ---- naming -------------------------------------------------------------

  // Through the bind chain, the way ssa_name does, to the value that actually
  // has a definition.
  ValueId resolve(ValueId v) const {
    for (int hops = 0; v != kNoValue && v < b_.value_count() && hops < 64;
         ++hops) {
      const ValueDef& d = b_.value(v);
      if (d.def == kNoOp) break;
      const Operation& o = b_.op(d.def);
      if (o.kind != OpKind::kBind || o.operands.empty()) break;
      v = o.operands[0];
    }
    return v;
  }

  std::string name(ValueId v) const {
    // The value's OWN binding first: a bind of an accumulator holds the value
    // the slot had when the bind ran, and resolving past it would hand back
    // whatever the slot holds now.
    const auto self = cur_.find(v);
    if (self != cur_.end()) return self->second;
    const auto acc = cur_.find(resolve(v));
    if (acc != cur_.end()) return acc->second;
    const std::string s = ssa_name(b_, v);
    if (s.empty() || o_.value_prefix.empty() || unprefixed_.count(v) != 0) {
      return s;
    }
    return "%" + o_.value_prefix + s.substr(1);
  }
  std::string fresh(std::string_view stem) {
    return "%" + o_.name_prefix + std::string(stem) + std::to_string(seq_++);
  }

  void line(int depth, std::string s) {
    text_.append(static_cast<std::size_t>(2 * depth), ' ');
    text_ += s;
    text_ += '\n';
  }

  Status unsupported(std::string_view what) const {
    return LSE_ERROR(kUnimplemented, "loom printer: ", std::string(what));
  }

  // ---- the u32 split ------------------------------------------------------
  //
  // LSE types every index AND every packed weight word as Scalar::kU32; Loom
  // has `index` for the first and `i32` for the second and no unsigned type
  // for either. A word is born in a memory read of an unsigned buffer or in a
  // bit-manipulation row, and it stays a word through the arithmetic it feeds.
  // Everything else is an address. Nothing is guessed: a value that would have
  // to be both is a decline, not a silent choice.
  void classify_words() {
    word_.assign(b_.value_count(), 0);
    auto seed = [&](OpId id) {
      const Operation& o = b_.op(id);
      if (o.result == kNoValue) return;
      if (o.type.elem != Scalar::kU32) return;
      if (o.kind == OpKind::kLoadVec || o.kind == OpKind::kSubscript) {
        word_[o.result] = 1;
      } else if (o.kind == OpKind::kCall &&
                 loom_result_type(o.key) == "i32") {
        word_[o.result] = 1;
      }
    };
    b_.walk(seed);
    for (bool changed = true; changed;) {
      changed = false;
      b_.walk([&](OpId id) {
        const Operation& o = b_.op(id);
        if (o.kind != OpKind::kBinary && o.kind != OpKind::kSelect) return;
        bool any = o.result != kNoValue && o.type.elem == Scalar::kU32 &&
                   word_[o.result] != 0;
        for (ValueId v : o.operands) {
          if (v < word_.size() && b_.value(v).type.elem == Scalar::kU32 &&
              word_[v] != 0) {
            any = true;
          }
        }
        if (!any) return;
        if (o.result != kNoValue && o.type.elem == Scalar::kU32 &&
            word_[o.result] == 0) {
          word_[o.result] = 1;
          changed = true;
        }
        for (ValueId v : o.operands) {
          if (v < word_.size() && b_.value(v).type.elem == Scalar::kU32 &&
              word_[v] == 0) {
            word_[v] = 1;
            changed = true;
          }
        }
      });
    }
  }

  // Which accumulators each region assigns, transitively.
  //
  // `e.var` is not in SSA: it is a slot the C printer declares and assigns.
  // Loom has no slot, so a region that writes one has to CARRY it — a loop as
  // an iter-argument it yields back, a conditional as a result with the
  // unchanged value on the other arm. Knowing which regions write which slots
  // is the whole of the analysis; the rest is bookkeeping while printing.
  void collect_assigns(RegionId r) {
    std::set<ValueId>& here = region_assigns_[r];
    for (OpId id : b_.region(r).ops) {
      const Operation& o = b_.op(id);
      if (o.erased) continue;
      if (o.kind == OpKind::kAssign && !o.operands.empty()) {
        const ValueId t = resolve(o.operands[0]);
        if (t < b_.value_count()) {
          const ValueDef& d = b_.value(t);
          if (d.def != kNoOp && b_.op(d.def).kind == OpKind::kMutable) {
            here.insert(t);
          }
        }
      }
      for (RegionId sub : o.regions) {
        collect_assigns(sub);
        const auto it = region_assigns_.find(sub);
        if (it != region_assigns_.end()) {
          here.insert(it->second.begin(), it->second.end());
        }
      }
    }
  }

  // The accumulators a region carries, in a fixed order so the argument list,
  // the yield and the result list cannot disagree.
  std::vector<ValueId> carried(RegionId r) const {
    const auto it = region_assigns_.find(r);
    if (it == region_assigns_.end()) return {};
    std::vector<ValueId> out;
    for (ValueId v : it->second) {
      // A slot the region also DECLARES is local to it and needs no carrying:
      // it is born and dies inside, so there is nothing outside to thread.
      if (cur_.count(v) != 0) out.push_back(v);
    }
    return out;
  }

  // A subtraction on the index domain whose result can be negative.
  //
  // LSE types an index as u32, so `a - b` in C wraps to 2^32 + (a - b) and the
  // huge value is discarded by a guard on `a` — that is how every windowed
  // kernel is written. Loom's `index` is a SIGNED carrier and this target
  // additionally constrains every index value to 32 bits
  // (`amdgpu.address.u32`), so a negative intermediate makes the whole address
  // dataflow unrepresentable — measured: the concat body is rejected at its
  // `index.madd`, the root of the chain, not at the subtraction. Reproducing
  // the wrap through i32 and back is rejected too, by
  // `memory_access.dynamic_index_source`.
  //
  // So a subtraction is clamped at zero, and ONLY where clamping cannot be
  // observed: an address, a loop bound, or more index arithmetic, all of which
  // are unusable for a wrapped value in the C kernel as well. If the result
  // reaches a comparison, a cast or an intrinsic — where the wrapped value is
  // the answer C computes — the body declines instead.
  void collect_subs() {
    users_.assign(b_.value_count(), {});
    b_.walk([&](OpId id) {
      const Operation& o = b_.op(id);
      for (ValueId v : o.operands) {
        const ValueId r = resolve(v);
        if (r < users_.size()) users_[r].push_back(id);
      }
    });
  }

  // Walked, not recursed: a use graph this wide reconverges, and the recursive
  // form visited every path through it. topk's index chain has enough of them
  // that the printer exhausted memory before it emitted a line.
  bool clamp_is_observable(ValueId v) const {
    std::vector<ValueId> work{v};
    std::unordered_set<ValueId> seen{v};
    while (!work.empty()) {
      const ValueId at = work.back();
      work.pop_back();
      if (at >= users_.size()) return true;
      for (OpId id : users_[at]) {
        const Operation& u = b_.op(id);
        switch (u.kind) {
          case OpKind::kSubscript:
          case OpKind::kLoadVec:
          case OpKind::kStoreVec:
          case OpKind::kFor:
          case OpKind::kAssign:
            break;
          case OpKind::kBinary: {
            const std::string& k = u.key;
            const bool arith = k == "+" || k == "-" || k == "*" || k == "/" ||
                               k == "%";
            if (!arith) return true;
            if (seen.insert(u.result).second) work.push_back(u.result);
            break;
          }
          default:
            return true;
        }
      }
    }
    return false;
  }

  // A subscript that an assignment writes through is a store address, not a
  // load. Both are the same op in the IR because C spells them the same way.
  void mark_store_targets() {
    store_target_.assign(b_.value_count(), 0);
    b_.walk([&](OpId id) {
      const Operation& o = b_.op(id);
      if (o.kind != OpKind::kAssign || o.operands.empty()) return;
      const ValueId t = o.operands[0];
      if (t < store_target_.size()) store_target_[t] = 1;
    });
  }

  // ---- types --------------------------------------------------------------

  Cls class_of(Scalar s, bool word) const {
    if (s == Scalar::kBool) return Cls::kBool;
    if (is_float(s)) return Cls::kFloat;
    if (s == Scalar::kU32 && !word) return Cls::kIndex;
    return is_signed(s) ? Cls::kSignedInt : Cls::kUnsignedInt;
  }

  // The Loom type of a value the recorder typed as `t`, in the role `word`
  // says it plays.
  std::string value_type(Type t, bool word) const {
    if (t.is_vector()) return loom_vector_type(t.elem, t.lanes);
    if (t.elem == Scalar::kU32 && !word) return "index";
    return std::string(loom_storage_type(t.elem));
  }

  void define(ValueId v, std::string type, Scalar elem, Cls cls) {
    if (v == kNoValue || v >= types_.size()) return;
    types_[v] = Typed{std::move(type), elem, cls, true};
  }

  void define_from(ValueId v, Type t) {
    const bool word = v < word_.size() && word_[v] != 0;
    define(v, value_type(t, word), t.elem, class_of(t.elem, word));
  }

  const Typed* typed(ValueId v) const {
    if (v == kNoValue || v >= types_.size() || !types_[v].defined) {
      return nullptr;
    }
    return &types_[v];
  }

  // An operand's record, following the bind chain the same way ssa_name does.
  const Typed* operand(ValueId v) const { return typed(resolve(v)); }

  // ---- ops ----------------------------------------------------------------

  // An early return retires this thread and lets the rest of the body run for
  // the others. Loom has kernel.exit for exactly that, but the RDNA3.5 export
  // config refuses the branch it lowers to ('masked_region_exits_by_cfg'), so
  // the guard becomes what it means: everything after it is predicated on the
  // negation. Measured — kernel.exit parses and verifies and is rejected at the
  // target, so this is not a spelling choice.
  OpId last_live_op(RegionId r) const {
    OpId last = kNoOp;
    for (OpId id : b_.region(r).ops) {
      if (!b_.op(id).erased) last = id;
    }
    return last;
  }

  Status emit_region(RegionId r, int depth, bool top = false) {
    int d = depth;
    int opened = 0;
    for (OpId id : b_.region(r).ops) {
      const Operation& o = b_.op(id);
      if (o.erased) continue;
      if (o.kind == OpKind::kReturnIf) {
        if (!top) {
          return unsupported("an early return nested inside a region");
        }
        const Typed* c = operand(o.operands[0]);
        if (c == nullptr || c->cls != Cls::kBool) {
          return unsupported("an early return whose condition is not i1");
        }
        const std::string one = fresh("true");
        const std::string neg = fresh("live");
        line(d, one + " = scalar.constant 1 : i1");
        line(d, neg + " = scalar.xori " + name(o.operands[0]) + ", " + one +
                    " : i1");
        line(d, "scf.if " + neg + " {");
        ++d;
        ++opened;
        continue;
      }
      // A bare `return;` as the last thing a body does is falling off the
      // end, which is what the emitter's own kernel.return already is.
      if (top && o.kind == OpKind::kReturn && o.operands.empty() &&
          id == last_live_op(r)) {
        continue;
      }
      LSE_RETURN_IF_ERROR(emit_op(o, d, top));
    }
    while (opened-- > 0) {
      --d;
      line(d, "}");
    }
    return Status{};
  }

  Status emit_op(const Operation& o, int depth, bool top) {
    switch (o.kind) {
      case OpKind::kConst: return emit_const(o, depth);
      case OpKind::kExtent: return emit_extent(o, depth);
      case OpKind::kSymbol: return emit_symbol(o, depth);
      case OpKind::kBinary: return emit_binary(o, depth);
      case OpKind::kCast: return emit_cast(o, depth);
      case OpKind::kSelect: return emit_select(o, depth);
      case OpKind::kCall: return emit_call(o, depth);
      case OpKind::kSubscript: return emit_subscript(o, depth);
      case OpKind::kBind: {
        // `const T n = e;` prints nothing, but it does FIX what `n` means. When
        // `e` is an accumulator the bind is a snapshot, not an alias: the C
        // printer copies the slot's value here and a later assignment to the
        // slot must not reach back and change `n`. Recording the operand's
        // name now is what keeps `t = a; a = b; b = t;` a swap instead of
        // collapsing both halves onto b.
        if (o.operands.empty()) return Status{};
        const Typed* src = operand(o.operands[0]);
        if (src == nullptr) return unsupported("a bind of an undefined value");
        define(o.result, src->type, src->elem, src->cls);
        cur_[o.result] = name(o.operands[0]);
        return Status{};
      }
      case OpKind::kAlloc: return emit_alloc(o, depth);
      case OpKind::kAssign: return emit_assign(o, depth);
      case OpKind::kLoadVec: return emit_load_vec(o, depth);
      case OpKind::kStoreVec: return emit_store_vec(o, depth);
      case OpKind::kBarrier:
        line(depth, o.text);
        return Status{};
      case OpKind::kReturn: return emit_return(o, top);
      case OpKind::kReturnIf:
        return unsupported("an early return nested inside a region");
      case OpKind::kRawStmt: return emit_raw(o, depth);
      case OpKind::kIf: return emit_if(o, depth);
      case OpKind::kFor: return emit_for(o, depth);
      case OpKind::kScope:
        // Loom has no block scope for names, and no name it prints can
        // collide: the braces simply go away.
        return o.regions.empty() ? Status{}
                                 : emit_region(o.regions[0], depth);
      case OpKind::kMutable: {
        const Typed* init = operand(o.operands.empty() ? kNoValue
                                                       : o.operands[0]);
        if (init == nullptr) {
          return unsupported("an accumulator with no initial value");
        }
        define(o.result, init->type, init->elem, init->cls);
        cur_[o.result] = name(o.operands[0]);
        return Status{};
      }
    }
    return unsupported("unhandled op");
  }

  Status emit_int_literal(const Operation& o, int depth, std::int64_t v) {
    const bool word = o.result < word_.size() && word_[o.result] != 0;
    if (o.type.elem == Scalar::kU32 && !word) {
      line(depth, name(o.result) + " = index.constant " + std::to_string(v) +
                      " : index");
    } else {
      const std::string t(loom_storage_type(o.type.elem));
      if (t.empty()) return unsupported("constant of an unspellable type");
      line(depth, name(o.result) + " = scalar.constant " + std::to_string(v) +
                      " : " + t);
    }
    define_from(o.result, o.type);
    return Status{};
  }

  Status emit_const(const Operation& o, int depth) {
    if (o.has(kFlagIntConst)) return emit_int_literal(o, depth, o.imm);
    if (o.type.elem == Scalar::kBool) {
      return unsupported("a boolean literal has no Loom spelling here");
    }
    if (!is_float(o.type.elem)) {
      return unsupported("a non-integer constant that is not a float");
    }
    // The recorder spelled this for C: `1.00000000f`. Loom takes the digits
    // and refuses the suffix.
    std::string digits = o.text;
    if (!digits.empty() && digits.back() == 'f') digits.pop_back();
    if (digits.find('.') == std::string::npos &&
        digits.find('e') == std::string::npos &&
        digits.find("inf") == std::string::npos &&
        digits.find("nan") == std::string::npos) {
      digits += ".0";
    }
    line(depth, name(o.result) + " = scalar.constant " + digits + " : " +
                    std::string(loom_storage_type(o.type.elem)));
    define_from(o.result, o.type);
    return Status{};
  }

  Status emit_extent(const Operation& o, int depth) {
    if (o.has(kFlagRuntimeExtent)) {
      // A dispatch-bound extent is a launch parameter plus an index.assume
      // stating the bound Loom will not take on faith. Both belong to the
      // emitter's parameter list, which does not carry them yet.
      return unsupported("extent '" + o.key +
                         "' is bound at dispatch, and this dialect has no "
                         "launch parameter for it yet");
    }
    return emit_int_literal(o, depth, o.imm);
  }

  Status emit_symbol(const Operation& o, int depth) {
    const std::string& sym = b_.value(o.result).name;
    if (o.type.is_memory()) {
      if (o.type.space != Space::kGlobal) {
        return unsupported("symbol '" + sym +
                           "' names storage the recorder did not allocate");
      }
      const auto it = o_.buffers.find(sym);
      if (it == o_.buffers.end()) {
        return unsupported("buffer '" + sym + "' has no view in this kernel");
      }
      view_of_[o.result] = it->second.view;
      extent_of_[o.result] = it->second.elements;
      define(o.result, loom_view_type(it->second.elem, it->second.elements),
             it->second.elem, Cls::kOther);
      return Status{};
    }
    if (sym != o_.thread_id) {
      return unsupported("symbol '" + sym +
                         "' is not a value this dialect can produce");
    }
    // Defined by the emitter, which needs the same value for the launch guard.
    // It keeps its bare name across every body spliced into one kernel.
    (void)depth;
    unprefixed_.insert(o.result);
    define(o.result, "index", Scalar::kU32, Cls::kIndex);
    return Status{};
  }

  Status emit_binary(const Operation& o, int depth) {
    if (o.operands.size() != 2) return unsupported("binary with 2 operands");
    const Typed* a = operand(o.operands[0]);
    const Typed* b = operand(o.operands[1]);
    if (a == nullptr || b == nullptr) {
      return unsupported("operand of '" + o.key + "' was never defined");
    }
    // C has one u32 for an address and a packed word; Loom has `index` and
    // `i32`. Converting the odd one out here is not enough: the quantized
    // codecs carry the same value through both roles for a dozen lines, and a
    // per-operator conversion leaves every OTHER use of it in the role the
    // classifier picked. Measured — coercing at this site turns the decline
    // into `TYPE/001` on a value the store hook then writes as f32. The role
    // has to be decided for the whole body, which is what classify_words does
    // not yet do for a codec, so this stays a decline.
    if (a->type != b->type) {
      return unsupported("'" + o.key + "' mixes " + a->type + " and " +
                         b->type +
                         "; the u32 index and word roles cannot both be one "
                         "value");
    }
    const std::string lhs = name(o.operands[0]);
    const std::string rhs = name(o.operands[1]);
    const std::string res = name(o.result);
    const std::string& k = o.key;

    const bool cmp = k == "<" || k == ">" || k == "<=" || k == ">=" ||
                     k == "==" || k == "!=";
    if (cmp) {
      std::string pred;
      if (a->cls == Cls::kFloat) {
        pred = k == "<"    ? "olt"
               : k == ">"  ? "ogt"
               : k == "<=" ? "ole"
               : k == ">=" ? "oge"
               : k == "==" ? "oeq"
                           : "one";
        line(depth, res + " = scalar.cmpf " + pred + ", " + lhs + ", " + rhs +
                        " : " + a->type);
      } else if (a->cls == Cls::kIndex || a->cls == Cls::kUnsignedInt ||
                 a->cls == Cls::kSignedInt) {
        const bool sgn = a->cls == Cls::kSignedInt;
        pred = k == "<"    ? (sgn ? "slt" : "ult")
               : k == ">"  ? (sgn ? "sgt" : "ugt")
               : k == "<=" ? (sgn ? "sle" : "ule")
               : k == ">=" ? (sgn ? "sge" : "uge")
               : k == "==" ? "eq"
                           : "ne";
        const std::string op =
            a->cls == Cls::kIndex ? "index.cmp " : "scalar.cmpi ";
        line(depth, res + " = " + op + pred + ", " + lhs + ", " + rhs + " : " +
                        a->type);
      } else {
        return unsupported("comparison of " + a->type);
      }
      define(o.result, "i1", Scalar::kBool, Cls::kBool);
      return Status{};
    }

    std::string op;
    switch (a->cls) {
      case Cls::kIndex:
        op = k == "+"   ? "index.add"
             : k == "-" ? "index.sub"
             : k == "*" ? "index.mul"
             : k == "/" ? "index.div"
             : k == "%" ? "index.rem"
                        : "";
        break;
      case Cls::kFloat:
        op = k == "+"   ? "scalar.addf"
             : k == "-" ? "scalar.subf"
             : k == "*" ? "scalar.mulf"
             : k == "/" ? "scalar.divf"
             : k == "%" ? "scalar.remf"
                        : "";
        break;
      case Cls::kSignedInt:
      case Cls::kUnsignedInt: {
        const bool sgn = a->cls == Cls::kSignedInt;
        // `/` and `%` by a power-of-two constant canonicalize to shrui/andi in
        // Loom, which is the strength reduction the codecs count on and the
        // reason they are written as division at all. A runtime divisor emits
        // a real divide, here as in HIP.
        op = k == "+"   ? "scalar.addi"
             : k == "-" ? "scalar.subi"
             : k == "*" ? "scalar.muli"
             : k == "/" ? (sgn ? "scalar.divsi" : "scalar.divui")
             : k == "%" ? (sgn ? "scalar.remsi" : "scalar.remui")
                        : "";
        break;
      }
      case Cls::kBool:
        // Non-short-circuiting, and that is safe: every operand of a recorded
        // `&&` is a pure expression or a read the printer has already emitted,
        // so there is no second operand whose evaluation C would have skipped.
        op = k == "&&" ? "scalar.andi" : k == "||" ? "scalar.ori" : "";
        break;
      default:
        break;
    }
    if (op.empty()) {
      return unsupported("operator '" + k + "' on " + a->type);
    }
    if (a->cls == Cls::kIndex && k == "-") {
      if (clamp_is_observable(o.result)) {
        return unsupported(
            "an index subtraction whose wrapped value is observable; C's u32 "
            "wrap has no Loom form under this target's 32-bit address "
            "constraint");
      }
      const std::string raw = fresh("sub");
      const std::string zero = fresh("zero");
      line(depth, raw + " = index.sub " + lhs + ", " + rhs + " : index");
      line(depth, zero + " = index.constant 0 : index");
      line(depth, res + " = index.max " + raw + ", " + zero + " : index");
      define(o.result, a->type, o.type.elem, a->cls);
      return Status{};
    }
    line(depth, res + " = " + op + " " + lhs + ", " + rhs + " : " + a->type);
    define(o.result, a->type, o.type.elem, a->cls);
    return Status{};
  }

  // An `index` the printer just minted from a value that is not one.
  //
  // LSE types every index as u32, but a Loom `index` is a signed carrier with
  // no width until a fact gives it one, and this target's integer contract
  // guards every index.add/madd/mul with `amdgpu.address.u32` — the operand
  // must be PROVEN to fit in 32 unsigned bits. A value that entered the index
  // domain through a cast carries no such proof, so the first address it feeds
  // is rejected at the target rather than at the cast. The range restated here
  // is the source type's own: kU32, exactly. Measured: without it the rope and
  // gdn address chains fail as `rejected 'index.add' address-width 'u32'`.
  void assume_u32(ValueId v, int depth) {
    const std::string in = name(v);
    const std::string r = fresh("u32");
    line(depth, r + " = index.assume " + in + " [range(" + in +
                    ", 0, 4294967295)] : index");
    cur_[v] = r;
  }

  Status emit_cast(const Operation& o, int depth) {
    if (o.operands.empty()) return unsupported("cast with an operand");
    const Typed* src = operand(o.operands[0]);
    if (src == nullptr) return unsupported("cast operand was never defined");
    const bool word = o.result < word_.size() && word_[o.result] != 0;
    const std::string dst_type = value_type(scalar_type(o.cast_to), word);
    const Cls dst_cls = class_of(o.cast_to, word);
    const std::string in = name(o.operands[0]);
    const std::string res = name(o.result);

    auto done = [&] {
      define(o.result, dst_type, o.cast_to, dst_cls);
      return Status{};
    };
    if (src->type == dst_type) {
      // A conversion between one Loom type and itself. C spells it and Loom
      // has no op for it, so there is nothing to name the result with; the
      // recorder has never produced one and a decline says so rather than
      // inventing an instruction to carry the name.
      return unsupported("a cast from " + src->type + " to itself");
    }

    const bool src_float = src->cls == Cls::kFloat;
    const bool dst_float = dst_cls == Cls::kFloat;
    if (src_float && dst_float) {
      const char* op = width_bits(o.cast_to) > width_bits(src->elem)
                           ? "scalar.extf"
                           : "scalar.fptrunc";
      line(depth, res + " = " + op + " " + in + " : " + src->type + " to " +
                      dst_type);
      return done();
    }
    if (src_float && dst_cls == Cls::kIndex) {
      // `(unsigned)(x)` in C, then into the address domain. fptoui, not
      // fptosi: a negative float is what the two disagree about and the HIP
      // path takes the unsigned answer.
      const std::string t = fresh("cast");
      line(depth, t + " = scalar.fptoui " + in + " : " + src->type + " to i32");
      line(depth, res + " = index.cast " + t + " : i32 to index");
      auto st = done();
      assume_u32(o.result, depth);
      return st;
    }
    if (src_float) {
      const char* op = dst_cls == Cls::kSignedInt ? "scalar.fptosi"
                                                  : "scalar.fptoui";
      line(depth, res + " = " + op + " " + in + " : " + src->type + " to " +
                      dst_type);
      return done();
    }
    if (dst_float) {
      std::string from = in;
      std::string from_type = src->type;
      if (src->cls == Cls::kIndex) {
        const std::string t = fresh("cast");
        line(depth, t + " = index.cast " + in + " : index to i32");
        from = t;
        from_type = "i32";
      }
      const char* op = src->cls == Cls::kSignedInt ? "scalar.sitofp"
                                                   : "scalar.uitofp";
      line(depth, res + " = " + op + " " + from + " : " + from_type + " to " +
                      dst_type);
      return done();
    }
    if (src->cls == Cls::kIndex || dst_cls == Cls::kIndex) {
      line(depth, res + " = index.cast " + in + " : " + src->type + " to " +
                      dst_type);
      auto st = done();
      if (dst_cls == Cls::kIndex) assume_u32(o.result, depth);
      return st;
    }
    if (src->cls == Cls::kBool || dst_cls == Cls::kBool) {
      return unsupported("a cast to or from i1");
    }
    const std::uint32_t from_bits = width_bits(src->elem);
    const std::uint32_t to_bits = width_bits(o.cast_to);
    const char* op = to_bits == from_bits ? nullptr
                     : to_bits > from_bits
                         ? (src->cls == Cls::kSignedInt ? "scalar.extsi"
                                                        : "scalar.extui")
                         : "scalar.trunci";
    if (op == nullptr) {
      return unsupported("a same-width integer reinterpretation");
    }
    line(depth, res + " = " + op + " " + in + " : " + src->type + " to " +
                    dst_type);
    return done();
  }

  Status emit_select(const Operation& o, int depth) {
    if (o.operands.size() != 3) return unsupported("select with 3 operands");
    const Typed* a = operand(o.operands[1]);
    if (a == nullptr) return unsupported("select operand was never defined");
    line(depth, name(o.result) + " = scf.select " + name(o.operands[0]) + ", " +
                    name(o.operands[1]) + ", " + name(o.operands[2]) + " : " +
                    a->type);
    define(o.result, a->type, a->elem, a->cls);
    return Status{};
  }

  Status emit_call(const Operation& o, int depth) {
    const std::string_view tmpl = b_.intrinsics().find(o.key);
    if (tmpl.empty()) {
      return unsupported("primitive '" + o.key + "' has no Loom spelling");
    }
    std::vector<std::string> args;
    args.reserve(o.operands.size());
    for (ValueId v : o.operands) {
      if (operand(v) == nullptr) {
        return unsupported("operand of '" + o.key + "' was never defined");
      }
      args.push_back(name(v));
    }
    const std::string prefix =
        "%" + o_.name_prefix + "c" + std::to_string(seq_++) + "_t";
    const std::string body =
        loom_splice(tmpl, args, {}, name(o.result), prefix);
    for (std::size_t at = 0; at <= body.size();) {
      const std::size_t nl = body.find('\n', at);
      const std::string one = body.substr(
          at, nl == std::string::npos ? body.size() - at : nl - at);
      if (!one.empty()) line(depth, one);
      if (nl == std::string::npos) break;
      at = nl + 1;
    }
    const std::string_view rt = loom_result_type(o.key);
    if (o.result != kNoValue) {
      if (rt.empty()) return unsupported("row '" + o.key + "' has no result");
      define(o.result, std::string(rt), o.type.elem,
             rt == "index" ? Cls::kIndex
             : rt == "f32" ? Cls::kFloat
                           : Cls::kUnsignedInt);
    }
    return Status{};
  }

  // Storage a subscript or a vector access indexes, as `<view name, view
  // type, element>` — the three things every Loom memory op needs.
  struct Access {
    std::string view;
    std::string view_type;
    Scalar elem = Scalar::kF32;
    std::uint64_t extent = 0;
  };

  Result<Access> access_of(ValueId base) const {
    const Typed* t = typed(base);
    if (t == nullptr || t->type.rfind("view<", 0) != 0) {
      return LSE_ERROR(kUnimplemented,
                       "loom printer: a subscript whose base is not a view");
    }
    const auto it = view_of_.find(base);
    if (it == view_of_.end()) {
      return LSE_ERROR(kUnimplemented,
                       "loom printer: a view with no name in this kernel");
    }
    return Access{it->second, t->type, t->elem, extent_of_.at(base)};
  }

  // The address a Loom memory op reads, with the range the caller is required
  // to have honoured stated where Loom can use it.
  //
  // A view is bounded and Loom proves every access against that bound; our
  // index arithmetic runs on a SIGNED index carrier, so nothing in
  // `(a * 2048 + b) * 512 + c` proves even non-negativity. The C kernel has
  // exactly the same precondition and simply does not say it — an index past
  // the extent there is a wild store, not a diagnostic. Saying it here changes
  // no contract; it only moves where the contract is written down.
  Result<std::string> bounded(ValueId v, std::uint64_t extent,
                              std::uint32_t lanes, int depth) {
    const Typed* t = operand(v);
    if (t == nullptr) {
      return LSE_ERROR(kUnimplemented,
                       "loom printer: an index that was never defined");
    }
    const std::uint32_t n = lanes == 0 ? 1u : lanes;
    if (extent < n) {
      return LSE_ERROR(kInternal, "loom printer: a ", std::to_string(n),
                       "-wide access into a view of ", std::to_string(extent));
    }
    std::string in = name(v);
    if (t->cls != Cls::kIndex) {
      // A 32-bit word used as an address: an index that was stored and read
      // back, which is what argmax's index accumulator and every gather row
      // is. Entering the address domain is explicit in Loom and free here.
      if (t->cls != Cls::kUnsignedInt && t->cls != Cls::kSignedInt) {
        return LSE_ERROR(kUnimplemented,
                         "loom printer: a memory index of type ", t->type,
                         ", where Loom needs an index");
      }
      const std::string c = fresh("idx");
      line(depth, c + " = index.cast " + in + " : " + t->type + " to index");
      in = c;
    }
    const std::string r = fresh("at");
    line(depth, r + " = index.assume " + in + " [range(" + in + ", 0, " +
                    std::to_string(extent - n) + ")] : index");
    return r;
  }

  Status emit_subscript(const Operation& o, int depth) {
    if (o.operands.size() != 2) return unsupported("subscript with 2 operands");
    if (o.result < store_target_.size() && store_target_[o.result] != 0) {
      // The address half of `buf[i] = v`. kAssign prints the store; nothing is
      // loaded here.
      auto acc = access_of(o.operands[0]);
      if (!acc.ok()) return acc.status();
      pending_store_[o.result] = *acc;
      return Status{};
    }
    const Typed* base = operand(o.operands[0]);
    if (base != nullptr && base->type.rfind("vector<", 0) == 0) {
      // A lane of a register vector, not an element of memory. Loom takes a
      // literal lane, which is what every Pack subscript in this engine is:
      // the widths come from a device byte budget and the indices are unrolled
      // constants.
      const ValueId lane = resolve(o.operands[1]);
      const ValueDef& d = b_.value(lane);
      const bool literal = d.def != kNoOp && b_.op(d.def).has(kFlagIntConst);
      // A lane that is not already a literal is legal only when the loop that
      // varies it is one the target's `unroll-scf-for` will peel: the pass runs
      // before the cleanup that folds an exact-valued dynamic index back into
      // `static_indices`, so the extract is static by the time the AMDGPU
      // lowering sees it. A lane from any other loop would survive to codegen
      // as a dynamic lane select, which this target has no plan for, so it
      // declines instead of emitting something that fails downstream.
      if (!literal && !unrolled_iv_.count(lane)) {
        return unsupported("a register-vector lane that is not a literal");
      }
      const std::string elem(loom_storage_type(base->elem));
      const std::string at =
          literal ? std::to_string(b_.op(d.def).imm) : name(o.operands[1]);
      line(depth, name(o.result) + " = vector.extract " +
                      name(o.operands[0]) + "[" + at + "] : " + base->type +
                      " -> " + elem);
      define(o.result, elem, base->elem,
             class_of(base->elem, base->elem == Scalar::kU32));
      return Status{};
    }
    auto acc = access_of(o.operands[0]);
    if (!acc.ok()) return acc.status();
    auto at = bounded(o.operands[1], acc->extent, 1, depth);
    if (!at.ok()) return at.status();
    const std::string elem(loom_storage_type(acc->elem));
    line(depth, name(o.result) + " = view.load " + acc->view + "[" + *at +
                    "] : " + acc->view_type + " -> " + elem);
    define(o.result, elem, acc->elem,
           class_of(acc->elem, acc->elem == Scalar::kU32));
    return Status{};
  }

  Status emit_alloc(const Operation& o, int depth) {
    if (!o.has(kFlagArray) || o.type.space != Space::kWorkgroup) {
      return unsupported(
          "a private allocation; Loom keeps register state in SSA and this "
          "printer does not lower one to scratch");
    }
    const auto count = static_cast<std::uint64_t>(o.imm);
    const std::uint64_t bytes = count * scalar_bytes(o.type.elem);
    const std::string n = name(o.result);
    const std::string b = fresh("lds");
    const std::string nbytes = fresh("ldsbytes");
    const std::string off = fresh("ldsbase");
    // 16-byte alignment, the same reservation kir::Lds prices and the same one
    // Body::workgroup_bytes charges for.
    line(depth, nbytes + " = index.constant " + std::to_string(bytes) +
                    " : offset");
    line(depth, b + " = buffer.alloca " + nbytes +
                    " {base_alignment = 16, memory_space = workgroup} : buffer");
    line(depth, off + " = index.constant 0 : offset");
    const std::string vt = loom_view_type(o.type.elem, count);
    line(depth, n + " = buffer.view " + b + "[" + off + "] : buffer -> " + vt);
    view_of_[o.result] = n;
    extent_of_[o.result] = count;
    define(o.result, vt, o.type.elem, Cls::kOther);
    return Status{};
  }

  Status emit_assign(const Operation& o, int depth) {
    if (o.operands.size() != 2) return unsupported("assign with 2 operands");
    const ValueId target = resolve(o.operands[0]);
    if (cur_.count(target) != 0) {
      const Typed* v = operand(o.operands[1]);
      if (v == nullptr) return unsupported("assigned value was never defined");
      const Typed* slot = typed(target);
      if (slot != nullptr && slot->type != v->type) {
        return unsupported("an accumulator of type " + slot->type +
                           " assigned a " + v->type);
      }
      (void)depth;
      cur_[target] = name(o.operands[1]);
      return Status{};
    }
    const auto it = pending_store_.find(target);
    if (it == pending_store_.end()) {
      return unsupported(
          "an assignment to something that is not a memory element");
    }
    const ValueDef& td = b_.value(target);
    const Operation& sub = b_.op(td.def);
    const Typed* v = operand(o.operands[1]);
    if (v == nullptr) return unsupported("assigned value was never defined");
    auto at = bounded(sub.operands[1], it->second.extent, 1, depth);
    if (!at.ok()) return at.status();
    line(depth, "view.store " + name(o.operands[1]) + ", " + it->second.view +
                    "[" + *at + "] : " + v->type + ", " +
                    it->second.view_type);
    return Status{};
  }

  Status emit_load_vec(const Operation& o, int depth) {
    if (o.operands.size() != 2) return unsupported("a vector load");
    auto acc = access_of(o.operands[0]);
    if (!acc.ok()) return acc.status();
    const auto n = static_cast<std::uint32_t>(o.imm);
    auto at = bounded(o.operands[1], acc->extent, n, depth);
    if (!at.ok()) return at.status();
    const std::string res = name(o.result);
    if (n <= 1) {
      const std::string elem(loom_storage_type(acc->elem));
      line(depth, res + " = view.load " + acc->view + "[" + *at + "] : " +
                      acc->view_type + " -> " + elem);
      define(o.result, elem, acc->elem,
             class_of(acc->elem, acc->elem == Scalar::kU32));
      return Status{};
    }
    const std::string vt = loom_vector_type(acc->elem, n);
    line(depth, res + " = vector.load " + acc->view + "[" + *at + "] : " +
                    acc->view_type + " -> " + vt);
    define(o.result, vt, acc->elem, Cls::kOther);
    return Status{};
  }

  Status emit_store_vec(const Operation& o, int depth) {
    if (o.operands.size() != 3) return unsupported("a vector store");
    auto acc = access_of(o.operands[0]);
    if (!acc.ok()) return acc.status();
    const Typed* v = operand(o.operands[2]);
    if (v == nullptr) return unsupported("stored value was never defined");
    const auto n = static_cast<std::uint32_t>(o.imm);
    auto at = bounded(o.operands[1], acc->extent, n, depth);
    if (!at.ok()) return at.status();
    const std::string op = n <= 1 ? "view.store " : "vector.store ";
    line(depth, op + name(o.operands[2]) + ", " + acc->view + "[" + *at +
                    "] : " + v->type + ", " + acc->view_type);
    return Status{};
  }

  Status emit_return(const Operation& o, bool top) {
    if (o.operands.empty()) {
      // The emitter closes the launch region; a bare return in the middle of a
      // body is the same early exit kReturnIf is, without the predicate.
      return unsupported("an unconditional return inside the body");
    }
    // The per-element form: the body computes one output element and hands it
    // back. Loom has func.def for that, but a helper whose parameters are
    // views buys nothing here — the caller splices the value.
    const Typed* v = operand(o.operands[0]);
    if (v == nullptr) return unsupported("returned value was never defined");
    if (!top) {
      return unsupported("a return that is not at the top of the body");
    }
    result_ = name(o.operands[0]);
    result_type_ = v->type;
    return Status{};
  }

  // The name a raw statement spelled, retargeted to what that value is called
  // here and now.
  //
  // A store hook runs at RECORD time and names its operands with ssa_name, so
  // an accumulator comes out as the slot's own `%vN`. In SSA the slot is not a
  // value: it is whatever name currently holds it, which the printer only
  // knows once it has walked the assignments. Emitting the recorded text
  // verbatim referenced a `%vN` that no statement defines — the epilogue of
  // every `owns its indexing` kernel parsed as an undefined SSA value.
  const std::string& raw_name_map_entry(std::string_view tok) {
    static const std::string kNone;
    if (raw_names_.empty()) {
      for (ValueId v = 0; v < b_.value_count(); ++v) {
        std::string n = ssa_name(b_, v);
        if (!n.empty()) raw_names_.emplace(std::move(n), v);
      }
    }
    const auto it = raw_names_.find(std::string(tok));
    if (it == raw_names_.end()) return kNone;
    retarget_ = name(it->second);
    return retarget_;
  }

  Status emit_raw(const Operation& o, int depth) {
    std::string out;
    out.reserve(o.text.size());
    for (std::size_t i = 0; i < o.text.size();) {
      if (o.text[i] != '%') {
        out += o.text[i++];
        continue;
      }
      std::size_t j = i + 1;
      while (j < o.text.size() &&
             (std::isalnum(static_cast<unsigned char>(o.text[j])) != 0 ||
              o.text[j] == '_')) {
        ++j;
      }
      const std::string_view tok(o.text.data() + i, j - i);
      const std::string& to = raw_name_map_entry(tok);
      if (to.empty()) {
        out.append(tok);
      } else {
        out += to;
      }
      i = j;
    }
    line(depth, out);
    return Status{};
  }

  // The accumulator list a carrying region opens with, and the pieces the
  // three sites of it must agree on.
  struct Carry {
    std::vector<ValueId> slots;
    std::string args;     // `%a0 = %init : f32, ...`
    std::string types;    // `f32, ...`
    std::string results;  // `%r0, %r1`
    std::vector<std::string> inner;
  };

  Result<Carry> open_carry(RegionId r) {
    Carry c;
    c.slots = carried(r);
    for (std::size_t i = 0; i < c.slots.size(); ++i) {
      const Typed* t = typed(c.slots[i]);
      if (t == nullptr) {
        return LSE_ERROR(kInternal,
                         "loom printer: a carried accumulator has no type");
      }
      const std::string in = fresh("acc");
      const std::string res = fresh("out");
      if (i != 0) {
        c.args += ", ";
        c.types += ", ";
        c.results += ", ";
      }
      c.args += in + " = " + cur_.at(c.slots[i]) + " : " + t->type;
      c.types += t->type;
      c.results += res;
      c.inner.push_back(in);
      results_.push_back(res);
    }
    return c;
  }

  void yield_carry(const Carry& c, int depth) {
    if (c.slots.empty()) return;
    std::string vals;
    std::string types;
    for (std::size_t i = 0; i < c.slots.size(); ++i) {
      if (i != 0) {
        vals += ", ";
        types += ", ";
      }
      vals += cur_.at(c.slots[i]);
      const Typed* t = typed(c.slots[i]);
      types += t != nullptr ? t->type : std::string("f32");
    }
    line(depth, "scf.yield " + vals + " : " + types);
  }

  Status emit_if(const Operation& o, int depth) {
    const Typed* c = operand(o.operands[0]);
    if (c == nullptr || c->cls != Cls::kBool) {
      return unsupported("an `if` whose condition is not i1");
    }
    if (o.regions.empty()) {
      line(depth, "scf.if " + name(o.operands[0]) + " {");
      line(depth, "}");
      return Status{};
    }
    auto carry = open_carry(o.regions[0]);
    if (!carry.ok()) return carry.status();
    if (carry->slots.empty()) {
      line(depth, "scf.if " + name(o.operands[0]) + " {");
      LSE_RETURN_IF_ERROR(emit_region(o.regions[0], depth + 1));
      line(depth, "}");
      return Status{};
    }
    // A conditional that writes an accumulator produces its new value, with
    // the untaken arm yielding the value it already had.
    const std::size_t base = results_.size() - carry->slots.size();
    std::vector<std::string> before;
    for (ValueId m : carry->slots) before.push_back(cur_.at(m));
    line(depth, carry->results + " = scf.if " + name(o.operands[0]) + " -> (" +
                    carry->types + ") {");
    for (std::size_t i = 0; i < carry->slots.size(); ++i) {
      cur_[carry->slots[i]] = before[i];
    }
    LSE_RETURN_IF_ERROR(emit_region(o.regions[0], depth + 1));
    yield_carry(*carry, depth + 1);
    line(depth, "} else {");
    for (std::size_t i = 0; i < carry->slots.size(); ++i) {
      cur_[carry->slots[i]] = before[i];
    }
    yield_carry(*carry, depth + 1);
    line(depth, "}");
    for (std::size_t i = 0; i < carry->slots.size(); ++i) {
      cur_[carry->slots[i]] = results_[base + i];
    }
    return Status{};
  }

  // The literal an index value is, when it is one.
  bool int_const(ValueId v, std::int64_t* out) const {
    const ValueId r = resolve(v);
    if (r >= b_.value_count()) return false;
    const ValueDef& d = b_.value(r);
    if (d.def == kNoOp) return false;
    const Operation& def = b_.op(d.def);
    if (!def.has(kFlagIntConst)) return false;
    *out = def.imm;
    return true;
  }

  Status emit_for(const Operation& o, int depth) {
    if (o.operands.size() != 3) return unsupported("a loop with 3 bounds");
    for (ValueId v : o.operands) {
      const Typed* t = operand(v);
      if (t == nullptr || t->cls != Cls::kIndex) {
        return unsupported("a loop bound that is not an index");
      }
    }
    define(o.result, "index", Scalar::kU32, Cls::kIndex);
    std::int64_t lo = 0, hi = 0, st = 0;
    if (o.has(kFlagUnroll) && int_const(o.operands[0], &lo) &&
        int_const(o.operands[1], &hi) && int_const(o.operands[2], &st) &&
        st > 0 && hi > lo) {
      unrolled_iv_.insert(o.result);
    }
    const std::string span = "[" + name(o.operands[0]) + " to " +
                             name(o.operands[1]) + " step " +
                             name(o.operands[2]) + "]";
    const std::string unroll = o.has(kFlagUnroll) ? " unroll" : "";
    if (o.regions.empty()) {
      line(depth, "scf.for " + name(o.result) + " = " + span + unroll + " {");
      line(depth, "}");
      return Status{};
    }
    auto carry = open_carry(o.regions[0]);
    if (!carry.ok()) return carry.status();
    if (carry->slots.empty()) {
      line(depth, "scf.for " + name(o.result) + " = " + span + unroll + " {");
      LSE_RETURN_IF_ERROR(emit_region(o.regions[0], depth + 1));
      line(depth, "}");
      return Status{};
    }
    // Every accumulator the body writes becomes an iter-argument the loop
    // yields back. This is what the HIP path gets from clang's mem2reg on an
    // ordinary local; Loom has no slot to promote, so the loop carries it.
    const std::size_t base = results_.size() - carry->slots.size();
    line(depth, carry->results + " = scf.for " + name(o.result) + " = " + span +
                    "(" + carry->args + ") -> (" + carry->types + ")" +
                    unroll + " {");
    for (std::size_t i = 0; i < carry->slots.size(); ++i) {
      cur_[carry->slots[i]] = carry->inner[i];
    }
    LSE_RETURN_IF_ERROR(emit_region(o.regions[0], depth + 1));
    yield_carry(*carry, depth + 1);
    line(depth, "}");
    for (std::size_t i = 0; i < carry->slots.size(); ++i) {
      cur_[carry->slots[i]] = results_[base + i];
    }
    return Status{};
  }

  const Body& b_;
  const LoomPrintOptions& o_;
  std::string text_;
  std::string result_;
  std::string result_type_;
  std::vector<Typed> types_;
  std::vector<char> word_;
  std::vector<char> store_target_;
  std::unordered_map<ValueId, std::string> view_of_;
  std::unordered_map<ValueId, std::uint64_t> extent_of_;
  // The SSA name each accumulator currently holds. Reading one reads this.
  std::unordered_map<ValueId, std::string> cur_;
  std::map<RegionId, std::set<ValueId>> region_assigns_;
  // Result names handed out by open_carry, kept alive because a carrying
  // region names them twice: once in its result list and once after it closes.
  std::vector<std::string> results_;
  std::vector<std::vector<OpId>> users_;
  std::unordered_map<ValueId, Access> pending_store_;
  std::unordered_set<ValueId> unprefixed_;
  std::unordered_map<std::string, ValueId> raw_names_;
  std::string retarget_;
  // Induction variables of `unroll`-marked loops with a constant trip
  // count: the values the target's unroller turns into literals.
  std::unordered_set<ValueId> unrolled_iv_;
  std::uint32_t seq_ = 0;
};

}  // namespace

Result<LoomBody> loom_print(const graph::kir::Body& body,
                            const LoomPrintOptions& opts) {
  return Printer(body, opts).run();
}

}  // namespace lse::backend
