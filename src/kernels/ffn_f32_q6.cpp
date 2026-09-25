// cand-ffn-f32: the M=512 FFN GEMM candidate (f32, no mma).
//
// Replaces the slow scalar body (4.7 TFLOPS) for the whitelisted M=512
// prefill FFN shapes. Keeps the scalar body's EXACT per-output-element
// arithmetic (STEP 0 measured the exact-f32 arm at 1.275e-6 rel-L2 vs the
// bf16-staged floor 1.655e-3 — the win must be schedule, not rounding):
//
//   * Q6 unpack: the same exact integer ops (/, %, cross-word spillover);
//   * dequant: one f32 fma(code, widen(bf16 scale), widen(bf16 bias));
//   * accumulate: acc = f32 fma(x, w, acc) in K-ascending order —
//     no reassociation, so expected accuracy vs the scalar body is 0.0.
//
// Schedule (the 4.7 TFLOPS fix; no vector.mma, f32 FMAs + LDS only):
//   * 128-thread WG = 4 waves; the WG covers a 64(M) x 64(N) output tile,
//     each wave owns a 16-row x 64-col quadrant (16 f32 accs/lane);
//   * one 64x64 f32 weight panel in LDS (16 KiB, single-stage) staged by
//     the 128 lanes from the packed plane (one dwordx3 per 16-code run);
//   * the scalar body's K schedule: within each 64-K block lane l decodes
//     k = block + l*4 + j*16 (j=0..3, ascending) — the SAME ascending
//     per-element order the scalar prefill body uses (lane -> chunk
//     base + lane*16 over 16-code chunks) — so each output element's
//     fma chain is identical term-for-term to the scalar body;
//   * x rows staged in LDS (64x64 f32, 16 KiB) for coalesced 16 B loads
//     and bank-conflict-free per-lane 4-f32 reads;
//   * total LDS 32 KiB <= 64 KiB/WG.
//
// Why this beats 4.7: the scalar body is 256-thread WGs of 8 waves each
// owning ONE column, re-reading the packed plane N/8 times and stalling on
// scalar dwordx1 loads + per-chunk scale reads. Here 4 waves share one
// 64-wide panel (one decode pass feeds 16 rows x 16 cols at once), loads
// are 16 B vector ops, and each FMA retires into one of 16 live
// accumulators. The K loop is barrier-light (2 per 64-K) and the fill's
// global loads are independent of the consume's LDS reads.
//
// Selection (new env LSE_FFN_F32; emission identity bump 6 -> 7):
//   LSE_FFN_F32=0        decline (kill-switch)
//   LSE_FFN_F32_SCALAR=1 decline the FFN 17408 shapes only (attribution)
//   default (or =1)      select the f32 kernel on the whitelisted M=512
//                        prefill shapes; everything else declines.
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/ffn_f32_q6.hpp"
#include "lse/math.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
namespace lse::kernels {
namespace {
namespace env = graph::env;
namespace kir = graph::kir;
using namespace graph;
namespace math = lse::math;
using backend::workgroup_lds_bytes;
struct Dims {
  std::uint32_t m = 0, n = 0, k = 0, words = 0, groups = 0, group = 0;
  bool valid = false;
};
// Field-for-field the quant_linear dims contract (re-implemented here; the
// namespace-internal copy is not exported).
Dims dims_of(const KernelShapes &s) {
  Dims d;
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.iattrs.size() < 2 || s.iattrs[0] != 6 || s.iattrs[1] <= 0 ||
      s.iattrs[1] % 32 || s.inputs[1].rank() != 2 || s.inputs[0].rank() == 0)
    return d;
  const auto n = s.inputs[1].dim(0), words = s.inputs[1].dim(1);
  const std::int64_t group = s.iattrs[1];
  if (n <= 0 || words <= 0 || words % 3 || n > UINT32_MAX ||
      words > UINT32_MAX / 32)
    return d;
  const auto k = words * 32 / 6;
  if (k % group || s.inputs[0].dim(s.inputs[0].rank() - 1) != k ||
      s.output.elem_count() % n)
    return d;
  const auto m = s.output.elem_count() / n;
  if (m < 16 || m > UINT32_MAX || m * k > UINT32_MAX ||
      n * words > UINT32_MAX || m * n > UINT32_MAX)
    return d;
  if (s.inputs[2].elem_count() != n * (k / group) ||
      s.inputs[3].elem_count() != n * (k / group) ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16)
    return d;
  d = {static_cast<std::uint32_t>(m),
       static_cast<std::uint32_t>(n),
       static_cast<std::uint32_t>(k),
       static_cast<std::uint32_t>(words),
       static_cast<std::uint32_t>(k / group),
       static_cast<std::uint32_t>(group),
       true};
  return d;
}
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales, biases;
  env::Out<kir::f32, env::Emit> out;
};
// The task's M=512 prefill whitelist: the two FFN 17408 shapes + the
// mid-shapes 10240/6144/12288 at K=5120 + the 1024-K shapes.
bool ffn_shape(const Dims &d) {
  if (d.k == 5120)
    return d.n == 17408 || d.n == 5120 || d.n == 10240 || d.n == 6144 ||
           d.n == 12288;
  if (d.k == 17408)
    return d.n == 5120;  // the FFN down shape
  return d.k == 1024;
}
// LDS: 64x64 f32 activation panel + 64x64 f32 weight panel = 32 KiB.
constexpr std::uint32_t kLdsBytes = 2u * (64u * 64u * 4u);
std::string emit_ffn_f32(const KernelShapes &s, const Dims &d) {
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<kir::f32>(64u * 64u);
  const auto ws = e.lds<kir::f32>(64u * 64u);
  // Bank swizzle for the 64-wide f32 panels: rotate eight-float spans by
  // row (the production wmma_q6_linear idiom, in f32) so neighboring
  // lanes read different bank groups rather than stride64.
  const auto address = [&](const kir::Val<kir::u32> &r,
                           const kir::Val<kir::u32> &k) {
    return r * 64u + ((k / 8u + r % 8u) % 8u) * 8u + k % 8u;
  };
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u);
  const auto nbase = e.let((wg % nblocks) * 64u);
  const auto rbase = e.let(wave * 16u);  // the wave's 16-row quadrant
  // 16 accumulators: rows 0..15 of the quadrant (each lane owns the full
  // 16-row x 64-col quadrant, column l + 32*q for q=0..1... see consume).
  std::vector<kir::LValue<kir::f32>> acc;
  for (unsigned r = 0; r < 16; ++r) acc.push_back(e.var(0.0f));
  for (auto kb : e.range(0u, d.k, 64u)) {
    // ---- Fill: all 128 lanes stage both panels of this 64-K block.
    // 64x64 elements per panel / 128 lanes = 32 f32 per lane. Each 64-wide
    // row needs 16 runs of 4 (16 lanes x 4 f32 = 64 k); 64 rows x 16 lanes
    // = 1024 lane-rows, /128 lanes = 8 lane-rows per lane. Each lane-row is
    // one 4-f32 run (16-B load, never crosses a row boundary). t in
    // [lid, 1024, 128): lane-row = t (1024 of them), row = t/16 (0..63),
    // k-run = (t%16)*4 (16 runs -> 64 k).
    for (auto t : e.range(lid, e.u32(1024u), 128u)) {
      const auto rr = e.let(t / 16u);
      const auto kc0 = e.let((t % 16u) * 4u);
      const auto ar = e.let(mbase + rr);
      const auto loaded =
          e.load(a.x, e.let(ar * d.k + kb + kc0), 16u);
      const auto index = e.let(address(rr, kc0));
      for (unsigned j = 0; j < 4; ++j)
        xs[index + e.u32(j)] = e.let(loaded[j]);
    }
    for (unsigned block = 0; block < 2; ++block) {
      // Lane L stages weight row r = L/4 + block*32, k-run (L%4)*16 —
      // exactly one 16-code run = 3 packed words per lane.
      const auto r = e.let(lid / 4u + e.u32(block * 32u));
      const auto kc = e.let((lid % 4u) * 16u);
      const auto bc = e.let(nbase + r);
      const auto w = e.local<kir::f32, 16>();
      for (unsigned j = 0; j < 16; ++j) w[j] = e.f32(0.0f);
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi]));
        const auto bias = e.let(math::widen(a.biases[gi]));
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
        // Each of the 3 packed words is a scalar subscript (the production's
        // idiom); a 12-byte vector load + [wi+1] overran its width-3 range.
        const std::array<kir::Val<kir::u32>, 3> packed{
            e.let(a.packed[wb]), e.let(a.packed[wb + 1u]),
            e.let(a.packed[wb + 2u])};
        for (unsigned j = 0; j < 16; ++j) {
          const unsigned off = (j * 6) % 32, wi = j * 6 / 32;
          auto code = packed[wi] / (1u << off);
          if (off > 26)
            code = code +
                   (packed[wi + 1] % (1u << (off - 26))) * (1u << (32 - off));
          else
            code = code % 64u;
          w[j] = math::fma(math::cast<kir::f32>(e.let(code)), scale, bias);
        }
      }
      for (unsigned j = 0; j < 16; ++j)
        ws[e.let(address(r, kc + e.u32(j)))] = w[j].read();
    }
    e.barrier();
    // ---- Consume: each output element (row t, col c) accumulates
    // sum_{k=0}^{63} x[t][k] * w[c][k] over this 64-K block. The 64-row tile
    // is split 4 ways across the waves: wave w owns rows rbase..rbase+15
    // (rbase = wave*16). Within a wave, lane l owns cols l and l+32 (2 cols)
    // x all 16 rows of the quadrant = 16 accs each. So the wave's 16 rows x
    // 64 cols = 16x64 outputs are covered by 32 lanes x 2 cols x 16 rows.
    // The 64 k are split into 4 slices of 16 (j=0..3, ascending k => the
    // scalar body's per-element fma order).
    for (auto j : e.range(e.u32(0), e.u32(4u), 1u)) {
      const auto k0 = e.let(j * 16u);
      for (unsigned kt = 0; kt < 16; ++kt) {
        const auto k = e.let(k0 + e.u32(kt));
        // 16 x-values: x[rbase+t][k] for t=0..15 (the wave's 16-row column).
        const auto xv = e.local<kir::f32, 16>();
        for (unsigned t = 0; t < 16; ++t)
          xv[t] = xs[e.let(address(rbase + e.u32(t), k))].read();
        for (unsigned q = 0; q < 2; ++q) {
          const auto c = e.let(lane + 32u * q);
          const auto wv = e.let(ws[e.let(address(c, k))].read());
          for (unsigned t = 0; t < 16; ++t)
            acc[t] = math::fma(xv[t].read(), wv, acc[t].read());
        }
      }
    }
    e.barrier();
  }
  // Epilogue: lane l stores rows 0..15 of its two columns.
  for (unsigned q = 0; q < 2; ++q) {
    const auto col = e.let(nbase + lane + 32u * q);
    for (unsigned t = 0; t < 16; ++t) {
      const auto rr = e.let(mbase + rbase + e.u32(t));
      if (auto active = e.when(rr < d.m && col < d.n))
        e.store(e.let(rr * d.n + col), acc[t].read());
    }
  }
  if (!body.lds().ok())
    return {};
  return body.str();
}
struct FfnF32Kernel final : KernelPrimitive<FfnF32Kernel> {
  static constexpr std::string_view kName = "quant_linear.q6_f32_prefill";
  static constexpr std::string_view kEntry = "lse_q6_f32_prefill";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  const KernelPrimitiveBase *specialize(const KernelShapes &) const override {
    return this;
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    const auto d = dims_of(s);
    if (!d.valid || !s.store || !s.types.scalar || !s.intrinsics)
      return {};
    return emit_ffn_f32(s, d);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "Q6 matrix requires four operands");
    Shape o;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)
      o.push_back(in[0].dim(i));
    o.push_back(in[1].dim(0));
    return o;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan p;
    const auto d = dims_of(s);
    if (!d.valid)
      return p;
    p.workgroup_size[0] = 128;
    p.workgroup_count[0] = ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.workgroup_count[1] = p.workgroup_count[2] = 1;
    p.lds_bytes = kLdsBytes;
    return p;
  }
};
} // namespace
namespace {
const FfnF32Kernel _lse_prim_instance_FfnF32{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnF32{
    &_lse_prim_instance_FfnF32};
} // namespace
const graph::KernelPrimitiveBase *ffn_f32_q6_for(
    const graph::KernelShapes &s) {
  const auto d = dims_of(s);
  if (!s.device || !s.intrinsics || !d.valid ||
      s.device->max_threads_per_workgroup < 128 ||
      s.device->wavefront_size != 32 || d.m != 512 || !ffn_shape(d))
    return nullptr;
  // Per-shape decline for the gate attribution arm: LSE_FFN_F32_SCALAR=1
  // declines ONLY the 17408 shapes (mid-shapes keep the f32 kernel).
  if (const auto *sc = std::getenv("LSE_FFN_F32_SCALAR")) {
    if (sc[0] != '0' && sc[0] != '\0' &&
        (d.n == 17408 || d.n == 5120))
      return nullptr;
  }
  if (const auto *kill = std::getenv("LSE_FFN_F32")) {
    if (kill[0] == '0')
      return nullptr;
  }
  static const FfnF32Kernel kernel;
  return &kernel;
}
} // namespace lse::kernels
