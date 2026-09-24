// Loom spellings for the built-in primitives — the mirror of hipc/hip_sources.
//
// A HIP row is an INFIX EXPRESSION spliced into a larger one. Loom is SSA, so a
// row here is one or more complete STATEMENTS: `$r` names the result, `$0`..
// name the operands, `$a0`.. the node's float attrs (already materialized as
// constants by the caller, because Loom takes no inline literal operand), and
// `$t0`.. are fresh temporaries so a row that needs several instructions can
// say so without inventing a name that collides.
//
// Rows that are NOT here are the point of the file. A primitive whose row is
// absent declines for this dialect and the group falls back visibly, which is
// the same contract a device missing a matrix-core capability already has:
//
//   * `shared` — a storage-class keyword in HIP; in Loom workgroup scratch is
//     a `buffer.alloca` + `buffer.view` pair the printer synthesizes from the
//     allocation's own type. A row would let something text-scan for a keyword
//     that has no Loom form.
//   * Unqualified matrix rows remain absent. The measured RDNA4 wave32
//     rows below join the shared matrix contract to Loom fragment schemas;
//     the register payload and lane mapping remain owned by that contract.
//
// Where Loom offers a NAMED op for something HIP spells as a formula —
// `siluf`, `logisticf`, `softplusf`, `geluf` — the formula is spelled out here
// anyway. The named ops are a different function until measured against the
// HIP row, and this table's job is to reproduce the engine's arithmetic, not
// to improve it.
#include "lse/backends/hrx/loomc/loom_sources.hpp"

#include <array>
#include <vector>
#include "lse/math.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"

namespace lse::backend {

namespace {

constexpr std::array<graph::PrimitiveSource, 37> kLoomSources{{
    {"add", "$r = scalar.addf $0, $1 : f32"},
    {"sub", "$r = scalar.subf $0, $1 : f32"},
    {"mul", "$r = scalar.mulf $0, $1 : f32"},
    {"div", "$r = scalar.divf $0, $1 : f32"},

    // A predicate that reaches the graph as a float. `oeq`/`oge`, the ordered
    // forms, are what C's `==` and `>=` are on floats.
    {"eq",
     "$t0 = scalar.cmpf oeq, $0, $1 : f32\n"
     "$t1 = scalar.constant 1.0 : f32\n"
     "$t2 = scalar.constant 0.0 : f32\n"
     "$r = scf.select $t0, $t1, $t2 : f32"},
    {"ge",
     "$t0 = scalar.cmpf oge, $0, $1 : f32\n"
     "$t1 = scalar.constant 1.0 : f32\n"
     "$t2 = scalar.constant 0.0 : f32\n"
     "$r = scf.select $t0, $t1, $t2 : f32"},

    {"neg", "$r = scalar.negf $0 : f32"},
    // `<afn>` — approximate-function — is not a relaxation chosen here: the
    // AMDGPU math policy REJECTS the exact f32 forms of exp, log and tanh
    // outright (LOWERING/036, `math.exp.exact_f32`), because the target has no
    // instruction that meets them. `<afn>` is the hardware transcendental,
    // which is also exactly what HIP's `__expf` / `__logf` / `tanhf` compile
    // to. Measured on gfx1151: the plain forms do not lower at all.
    {"exp", "$r = scalar.expf<afn> $0 : f32"},
    {"log", "$r = scalar.logf<afn> $0 : f32"},
    {"sqrt", "$r = scalar.sqrtf $0 : f32"},
    {"rsqrt", "$r = scalar.rsqrtf $0 : f32"},
    {"sigmoid",
     "$t0 = scalar.negf $0 : f32\n"
     "$t1 = scalar.expf<afn> $t0 : f32\n"
     "$t2 = scalar.constant 1.0 : f32\n"
     "$t3 = scalar.addf $t2, $t1 : f32\n"
     "$r = scalar.divf $t2, $t3 : f32"},
    {"silu",
     "$t0 = scalar.negf $0 : f32\n"
     "$t1 = scalar.expf<afn> $t0 : f32\n"
     "$t2 = scalar.constant 1.0 : f32\n"
     "$t3 = scalar.addf $t2, $t1 : f32\n"
     "$r = scalar.divf $0, $t3 : f32"},
    {"tanh", "$r = scalar.tanhf<afn> $0 : f32"},
    // maxnumf is IEEE-754 maxNum, which is what fmaxf is. maximumf propagates
    // NaN and would disagree with the HIP row on exactly the inputs a softmax
    // path can produce.
    {"relu",
     "$t0 = scalar.constant 0.0 : f32\n"
     "$r = scalar.maxnumf $0, $t0 : f32"},

    // log1p(exp(x)); the branch keeps exp from overflowing past ~88.
    {"softplus",
     "$t0 = scalar.constant 20.0 : f32\n"
     "$t1 = scalar.cmpf ogt, $0, $t0 : f32\n"
     "$t2 = scalar.expf<afn> $0 : f32\n"
     "$t3 = scalar.constant 1.0 : f32\n"
     "$t4 = scalar.addf $t3, $t2 : f32\n"
     "$t5 = scalar.logf<afn> $t4 : f32\n"
     "$r = scf.select $t1, $0, $t5 : f32"},

    // The same tanh approximation the HIP row spells, associated the same way:
    // 0.5 * x * (1 + tanh(0.7978845608 * (x + ((0.044715 * x) * x) * x))).
    // `scalar.geluf<tanh>` exists and is one instruction, but its polynomial is
    // Loom's rather than the training stack's until that is measured.
    {"gelu",
     "$t0 = scalar.constant 0.044715 : f32\n"
     "$t1 = scalar.mulf $t0, $0 : f32\n"
     "$t2 = scalar.mulf $t1, $0 : f32\n"
     "$t3 = scalar.mulf $t2, $0 : f32\n"
     "$t4 = scalar.addf $0, $t3 : f32\n"
     "$t5 = scalar.constant 0.7978845608 : f32\n"
     "$t6 = scalar.mulf $t5, $t4 : f32\n"
     "$t7 = scalar.tanhf<afn> $t6 : f32\n"
     "$t8 = scalar.constant 1.0 : f32\n"
     "$t9 = scalar.addf $t8, $t7 : f32\n"
     "$t10 = scalar.constant 0.5 : f32\n"
     "$t11 = scalar.mulf $t10, $0 : f32\n"
     "$r = scalar.mulf $t11, $t9 : f32"},

    // fminf(fmaxf(x, lo), hi). Not scalar.clampf: that op carries its own NaN
    // mode and the pair above is what the HIP row does, exactly.
    {"clamp",
     "$t0 = scalar.maxnumf $0, $a0 : f32\n"
     "$r = scalar.minnumf $t0, $a1 : f32"},

    {"fma", "$r = scalar.fmaf $0, $1, $2 : f32"},
    {"max", "$r = scalar.maxnumf $0, $1 : f32"},
    // Unsigned min/max may be workgroup-uniform. The current target lowers
    // scalar.minui/maxui to VGPR-only instructions without converting uniform
    // SGPR operands/results. Compare/select preserves unsigned semantics and
    // lets the target choose scalar or vector register classes consistently.
    {"min.u32", "$t0 = scalar.cmpi ult, $0, $1 : i32\n"
                "$r = scf.select $t0, $0, $1 : i32"},
    {"max.u32", "$t0 = scalar.cmpi ugt, $0, $1 : i32\n"
                "$r = scf.select $t0, $0, $1 : i32"},
    {"min", "$r = scalar.minnumf $0, $1 : f32"},
    {"neg_inf", "$r = scalar.constant -inf : f32"},
    {"abs", "$r = scalar.absf $0 : f32"},

    // roundf, not roundevenf: the block codecs must reproduce
    // quant::QuantScheme's host pack bit-for-bit, and that uses std::round
    // (half away from zero). See the same note in hip_sources.
    {"round", "$r = scalar.roundf $0 : f32"},

    // A narrow float seen as its bits, and back. Three ops each because Loom
    // separates the value conversion from the reinterpretation from the width
    // change; the conversion itself is still the hardware's round-to-nearest-
    // even. The i32 result is a WORD, not an index — see loom_types.
    {"bits.f16",
     "$t0 = scalar.fptrunc $0 : f32 to f16\n"
     "$t1 = scalar.bitcast $t0 : f16 to i16\n"
     "$r = scalar.extui $t1 : i16 to i32"},
    {"value.f16",
     "$t0 = scalar.trunci $0 : i32 to i16\n"
     "$t1 = scalar.bitcast $t0 : i16 to f16\n"
     "$r = scalar.extf $t1 : f16 to f32"},

    // Preserve the HIP mixed-sign dot: signed activation bytes, unsigned
    // affine codes, and a wrapping i32 accumulator (no saturation).
    {"dot4.i32.iu8",
     "$t0 = vector.splat $0 : vector<1xi32>\n"
     "$t1 = vector.splat $1 : vector<1xi32>\n"
     "$t2 = vector.splat $2 : vector<1xi32>\n"
     "$t3 = vector.bitcast $t0 : vector<1xi32> to vector<4xi8>\n"
     "$t4 = vector.bitcast $t1 : vector<1xi32> to vector<4xi8>\n"
     "$t5 = vector.dot4i<s8u8> $t3, $t4, $t2 : vector<4xi8>, vector<4xi8>, vector<1xi32>\n"
     "$r = vector.extract $t5[0] : vector<1xi32> -> i32"},
    {"rint", "$r = scalar.roundevenf $0 : f32"},

    {"thread.local_id", "$r = kernel.workitem.id<x> : index"},
    {"thread.workgroup_id.x", "$r = kernel.workgroup.id<x> : index"},
    {"thread.workgroup_id.y", "$r = kernel.workgroup.id<y> : index"},
    {"thread.workgroup_size", "$r = kernel.workgroup.size<x> : index"},
    {"thread.grid_dim.x", "$r = kernel.workgroup.count<x> : index"},

    // The shuffle returns the value and a validity bit; HIP's `__shfl_xor`
    // returns only the value, so the bit is dropped into a temporary. The
    // width comes from the device rather than being baked, which is what keeps
    // this row wave-size agnostic.
    {"wave.shfl_xor",
     "$t0 = index.cast $1 : index to i32\n"
     "$t1 = kernel.subgroup.size : index\n"
     "$t2 = index.cast $t1 : index to i32\n"
     "$r, $t3 = kernel.subgroup.shuffle<xor> $0, $t0, $t2 : f32, i32, i32"},

    {"barrier", "kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}"},
}};

struct ResultType {
  std::string_view primitive;
  std::string_view type;
};

// Everything not listed produces f32, which is what every arithmetic row does.
constexpr std::array<ResultType, 9> kNonFloatResults{{
    {"bits.f16", "i32"},
    {"dot4.i32.iu8", "i32"},
    {"min.u32", "i32"},
    {"max.u32", "i32"},
    {"thread.local_id", "index"},
    {"thread.workgroup_id.x", "index"},
    {"thread.workgroup_id.y", "index"},
    {"thread.workgroup_size", "index"},
    {"thread.grid_dim.x", "index"},
}};

// Only these shared measured layouts match Loom's RDNA4 single-tile catalog.
// Chained/unmeasured rows are deliberately absent even when a key is shared.
struct MatrixSpelling {
  const math::MatrixCoreRow* row = nullptr;
  std::string text;
  std::string result;
};
const std::vector<MatrixSpelling>& matrix_spellings() {
  static const std::vector<MatrixSpelling> rows = [] {
    std::vector<MatrixSpelling> out;
    for (const auto& r : math::matrix_core_table()) {
      if (r.target != math::MatrixTarget::kRdna4 || !r.emittable() ||
          r.chained != 1 || r.m != 16 || r.n != 16 || r.k != 16 ||
          r.wave != 32 || r.operands != math::OperandLayout::kLaneRowSplitK ||
          r.acc_layout != math::AccLayout::kRowBlockHalfWave) continue;
      std::string a_format, b_format;
      switch (r.operand) {
        case math::MatrixElem::kF16: a_format = b_format = "f16"; break;
        case math::MatrixElem::kBF16: a_format = b_format = "bf16"; break;
        case math::MatrixElem::kI8: a_format = b_format = "i8"; break;
        case math::MatrixElem::kSU8: a_format = "i8"; b_format = "u8"; break;
        case math::MatrixElem::kFp8: a_format = b_format = "f8e4m3"; break;
        case math::MatrixElem::kBf8: a_format = b_format = "f8e5m2"; break;
        default: continue;
      }
      const std::string a = loom_vector_type(r.a_elem, static_cast<std::uint32_t>(r.a_len));
      const std::string b = loom_vector_type(r.b_elem, static_cast<std::uint32_t>(r.b_len));
      const std::string c = loom_vector_type(r.c_elem, static_cast<std::uint32_t>(r.c_len));
      auto schema = [&](int temporary, const std::string& format, int length,
                        ir::Scalar elem) {
        return "$t" + std::to_string(temporary) +
            " = encoding.define #matrix_operand<element_format=" + format +
            ", payload_elements=" + std::to_string(length * r.pack) +
            ", payload_registers=" + std::to_string(static_cast<unsigned>(length) * ir::scalar_bytes(elem) / 4) +
            "> : encoding<schema>\n";
      };
      std::string text = "$t0 = index.constant " + std::to_string(r.m) + " : index\n";
      text += "$t1 = index.constant " + std::to_string(r.n) + " : index\n";
      text += "$t2 = index.constant " + std::to_string(r.k) + " : index\n";
      text += schema(3, a_format, r.a_len, r.a_elem);
      text += schema(4, b_format, r.b_len, r.b_elem);
      text += "$t5 = vector.fragment<lhs> $0 shape [$t0, $t2] using {schema = $t3 : encoding<schema>} : " + a + "\n";
      text += "$t6 = vector.fragment<rhs> $1 shape [$t2, $t1] using {schema = $t4 : encoding<schema>} : " + b + "\n";
      text += "$t7 = vector.fragment<init> $2 shape [$t0, $t1] : " + c + "\n";
      text += "$r = vector.mma $t5, $t6, $t7 : " + a + ", " + b + ", " + c;
      out.push_back({&r, std::move(text), c});
    }
    return out;
  }();
  return rows;
}
}  // namespace

const math::MatrixCoreRow* loom_matrix_row(std::string_view key) noexcept {
  for (const auto& r : matrix_spellings()) if (r.row->key == key) return r.row;
  return nullptr;
}

graph::DialectSourceTable loom_sources() noexcept {
  static const std::vector<graph::PrimitiveSource> rows = [] {
    std::vector<graph::PrimitiveSource> all(kLoomSources.begin(), kLoomSources.end());
    for (const auto& r : matrix_spellings()) all.push_back({r.row->key, r.text});
    return all;
  }();
  return graph::DialectSourceTable(rows, graph::Dialect::kLoom);
}

std::string_view loom_result_type(std::string_view primitive) noexcept {
  for (const auto& r : matrix_spellings()) {
    if (r.row->key == primitive) return r.result;
  }
  if (graph::DialectSourceTable(kLoomSources).find(primitive).empty()) {
    return {};
  }
  for (const ResultType& r : kNonFloatResults) {
    if (r.primitive == primitive) return r.type;
  }
  return "f32";
}

std::string loom_splice(std::string_view tmpl,
                        std::span<const std::string> args,
                        std::span<const std::string> attrs,
                        std::string_view result, std::string_view temp_prefix) {
  std::string out;
  out.reserve(tmpl.size() + args.size() * 8);
  for (std::size_t i = 0; i < tmpl.size(); ++i) {
    if (tmpl[i] != '$') {
      out += tmpl[i];
      continue;
    }
    if (i + 1 < tmpl.size() && tmpl[i + 1] == 'r') {
      out += result;
      ++i;
      continue;
    }
    const bool is_attr = i + 1 < tmpl.size() && tmpl[i + 1] == 'a';
    const bool is_temp = i + 1 < tmpl.size() && tmpl[i + 1] == 't';
    std::size_t j = i + 1 + ((is_attr || is_temp) ? 1 : 0);
    std::size_t index = 0;
    bool digits = false;
    while (j < tmpl.size() && tmpl[j] >= '0' && tmpl[j] <= '9') {
      index = index * 10 + static_cast<std::size_t>(tmpl[j] - '0');
      ++j;
      digits = true;
    }
    if (!digits) {
      out += tmpl[i];
      continue;
    }
    if (is_temp) {
      out += temp_prefix;
      out += std::to_string(index);
    } else if (is_attr) {
      if (index < attrs.size()) out += attrs[index];
    } else if (index < args.size()) {
      out += args[index];
    }
    i = j - 1;
  }
  return out;
}

}  // namespace lse::backend
