#include "lse/ir/lower.hpp"

#include <vector>

#include "lse/ir/spell.hpp"

namespace lse::ir {

namespace {

std::string scalar_name(const Body& b, Scalar s) {
  return std::string(b.types().scalar(s));
}

std::string indent_of(int depth) {
  return std::string(static_cast<std::size_t>(2 * depth), ' ');
}

class Printer {
 public:
  explicit Printer(const Body& b) : b_(b) {}

  std::string run() {
    for (const std::string& t : b_.typedefs()) out_ += "  " + t + "\n";
    emit_region(b_.entry(), 1);
    if (!out_.empty()) out_.pop_back();
    return std::move(out_);
  }

  std::string render(ValueId v) const {
    if (v == kNoValue || v >= b_.value_count()) return {};
    const ValueDef& def = b_.value(v);
    if (!def.name.empty()) return def.name;
    if (def.def == kNoOp) return {};
    const Operation& o = b_.op(def.def);
    switch (o.kind) {
      case OpKind::kConst:
      case OpKind::kExtent:
        // A literal extent is its number; a runtime one is the dispatch
        // constant that carries it. One spelling field, both bindings.
        return o.text;
      case OpKind::kBinary:
        return "(" + render(o.operands[0]) + " " + o.key + " " +
               render(o.operands[1]) + ")";
      case OpKind::kCast:
        return "(" + scalar_name(b_, o.cast_to) + ")(" +
               render(o.operands[0]) + ")";
      case OpKind::kSelect:
        return "(" + render(o.operands[0]) + " ? " + render(o.operands[1]) +
               " : " + render(o.operands[2]) + ")";
      case OpKind::kCall: {
        std::vector<std::string> argv;
        argv.reserve(o.operands.size());
        for (ValueId a : o.operands) argv.push_back(render(a));
        return substitute(o.text, argv);
      }
      case OpKind::kSubscript:
        return render(o.operands[0]) + "[" + render(o.operands[1]) + "]";
      default:
        // A declaration whose name was dropped would print as nothing, which
        // is a bug in whatever built it rather than something to paper over.
        return {};
    }
  }

 private:
  void line(int depth, std::string text) {
    out_ += indent_of(depth);
    out_ += text;
    out_ += '\n';
  }

  void emit_region(RegionId r, int depth) {
    for (OpId id : b_.region(r).ops) {
      const Operation& o = b_.op(id);
      if (o.erased) continue;
      emit_op(id, o, depth);
    }
  }

  void emit_op(OpId id, const Operation& o, int depth) {
    (void)id;
    switch (o.kind) {
      // Pure expressions are inlined at their uses and emit no statement.
      case OpKind::kConst:
      case OpKind::kExtent:
      case OpKind::kSymbol:
      case OpKind::kBinary:
      case OpKind::kCast:
      case OpKind::kSelect:
      case OpKind::kCall:
      case OpKind::kSubscript:
        return;

      case OpKind::kBind:
        line(depth, "const " + scalar_name(b_, o.type.elem) + " " +
                        b_.value(o.result).name + " = " +
                        render(o.operands[0]) + ";");
        return;
      case OpKind::kMutable:
        line(depth, scalar_name(b_, o.type.elem) + " " +
                        b_.value(o.result).name + " = " +
                        render(o.operands[0]) + ";");
        return;
      case OpKind::kAlloc: {
        const std::string& name = b_.value(o.result).name;
        if (o.has(kFlagArray)) {
          const std::string_view storage =
              o.type.space == Space::kWorkgroup ? b_.intrinsics().find("shared")
                                                : std::string_view{};
          std::string decl;
          if (!storage.empty()) decl = std::string(storage) + " ";
          decl += scalar_name(b_, o.type.elem) + " " + name + "[" +
                  std::to_string(o.imm) + "];";
          line(depth, decl);
        } else {
          line(depth, o.text + " " + name + ";");
        }
        return;
      }
      case OpKind::kAssign:
        line(depth, render(o.operands[0]) + " = " + render(o.operands[1]) + ";");
        return;
      case OpKind::kLoadVec: {
        const std::string& name = b_.value(o.result).name;
        const std::string addr =
            render(o.operands[0]) + "[" + render(o.operands[1]) + "]";
        if (o.imm <= 1) {
          line(depth, "const " + scalar_name(b_, o.type.elem) + " " + name +
                          " = " + addr + ";");
        } else {
          line(depth, "const " + o.text + " " + name + " = *(const " + o.text +
                          "*)(&" + addr + ");");
        }
        return;
      }
      case OpKind::kStoreVec: {
        const std::string addr =
            render(o.operands[0]) + "[" + render(o.operands[1]) + "]";
        if (o.imm <= 1) {
          line(depth, addr + " = " + render(o.operands[2]) + ";");
        } else {
          line(depth, "*(" + o.text + "*)(&" + addr + ") = " +
                          render(o.operands[2]) + ";");
        }
        return;
      }
      case OpKind::kBarrier:
        line(depth, o.text + ";");
        return;
      case OpKind::kReturn:
        if (o.operands.empty()) {
          line(depth, "return;");
        } else {
          line(depth, "return " + render(o.operands[0]) + ";");
        }
        return;
      case OpKind::kReturnIf:
        line(depth, "if (" + render(o.operands[0]) + ") return;");
        return;
      case OpKind::kRawStmt:
        line(depth, o.text);
        return;

      case OpKind::kIf: {
        line(depth, "if (" + render(o.operands[0]) + ") {");
        const int inner = depth + (o.has(kFlagIndentBody) ? 1 : 0);
        if (!o.regions.empty()) emit_region(o.regions[0], inner);
        line(depth, "}");
        return;
      }
      case OpKind::kFor: {
        if (o.has(kFlagUnroll)) line(depth, "#pragma unroll");
        const std::string& var = b_.value(o.result).name;
        const std::string ty = scalar_name(b_, Scalar::kU32);
        line(depth, "for (" + ty + " " + var + " = " + render(o.operands[0]) +
                        "; " + var + " < " + render(o.operands[1]) + "; " +
                        var + " += " + render(o.operands[2]) + ") {");
        const int inner = depth + (o.has(kFlagIndentBody) ? 1 : 0);
        if (!o.regions.empty()) emit_region(o.regions[0], inner);
        line(depth, "}");
        return;
      }
      case OpKind::kScope:
        line(depth, "{");
        if (!o.regions.empty()) emit_region(o.regions[0], depth + 1);
        line(depth, "}");
        return;
    }
  }

  const Body& b_;
  std::string out_;
};

}  // namespace

std::string ssa_name(const Body& body, ValueId v) {
  // `const T n = expr;` introduces a second name for one value. C needs the
  // declaration; SSA does not have one — the value already exists and `n` IS
  // it — so a bind resolves to what it bound. Doing that here rather than in
  // the printer is what keeps a store hook's operand text and the statement
  // that defined it the same string.
  for (int hops = 0; v != kNoValue && v < body.value_count() && hops < 64;
       ++hops) {
    const ValueDef& def = body.value(v);
    if (def.def == kNoOp) break;
    const Operation& o = body.op(def.def);
    if (o.kind != OpKind::kBind || o.operands.empty()) break;
    v = o.operands[0];
  }
  if (v == kNoValue || v >= body.value_count()) return {};
  const ValueDef& def = body.value(v);
  return def.name.empty() ? "%v" + std::to_string(v) : "%" + def.name;
}

std::string render(const Body& body, ValueId v) {
  // An SSA dialect has no inline expression to render: every op is a statement
  // that names its result, so a use is that name. The whole of the dialect
  // split at a use site is this line; the statement walk itself is a separate
  // printer, in the generator that owns the dialect.
  if (body.intrinsics().dialect() == Dialect::kLoom) return ssa_name(body, v);
  return Printer(body).render(v);
}

std::string lower(const Body& body) { return Printer(body).run(); }

}  // namespace lse::ir
