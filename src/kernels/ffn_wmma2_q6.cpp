// cand-ffn-wmma2: the M=512 FFN GEMM candidate (plan cand-ffn-wmma2/plan.md).
//
// Build 198 re-shape: the kernel now matches the PRODUCTION live-range profile
// exactly, which is the structure the pinned loomc demonstrably compiles and
// the spill-materialization iteration limit demonstrably accepts. One workgroup
// covers a 64x64 output tile; four waves each retain a 32x32 quadrant (four
// 16x16x16 accumulators per wave) and one K-slice of operands is in flight at
// a time in single-stage LDS (xs/ws, +wl for residual-2). The B operand is
// dequantized in-kernel from the packed Q6 storage (six-bit codes with
// cross-word spillover), verbatim the production staged-bf16 extraction. The
// single-product variant runs one product per fragment pair; the residual-2
// variant stages bf16(w) + bf16(w - bf16(w)) and runs two products per
// fragment pair, with the production mark_unsafe / flags exceptional-block
// fallback.
//
// The fill and consume are the production's, line for line (128-thread WG,
// wave = lid/32, per-wave 32x32 quadrant at am=(wave%2)*32 / bn=(wave/2)*32,
// 16-element LDS swizzle, two-barrier K step). The Build-197 non-finite output
// came from the earlier single-wave 32x64 geometry (custom addr_a/addr_b,
// 8-acc row layout, K/32 blocks); matching production's 64-wide swizzle and
// 4-acc quadrant mapping removes it (verified by the constant-operand smoke
// case in ffn-micro, Build 198 step 3).
//
// Build 199 selection: residual-2 is the DEFAULT for the M=512 FFN 17408
// shapes (the 3-context gate arbiter); LSE_FFN_WMMA2=single opts back into
// single-product for the Build 198 A/B only; LSE_FFN_WMMA2=0 and
// LSE_FFN_WMMA2_SCALAR=1 decline to the production / scalar paths.
#include "lse/graph/kernel_args.hpp"
#include "lse/kernels/ffn_wmma2_q6.hpp"
#include "lse/kernels/quant_operand_policy.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math/fp8.hpp"
#include "lse/quant/group_affine_codec.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
namespace lse::kernels {
namespace {
namespace env = graph::env;
namespace kir = graph::kir;
using namespace graph;
namespace math = lse::math;
struct Dims {
  std::uint32_t m = 0, n = 0, k = 0, words = 0, groups = 0, group = 0;
  bool valid = false;
};
// Field-for-field the wmma_q6_linear.cpp contract (re-implemented here
// because that copy is namespace-internal).
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
template <math::MatrixElem T>
using Base = MatrixTile<struct FfnWmma2Tile, math::MatrixTarget::kRdna4,
                        math::MatrixElem::kF32, T, 16, 16, 16>;
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales, biases;
  env::Out<kir::f32, env::Emit> out;
};
// LDS budget (plan 3.3), the production's: single 2 x (64*64*2) = 16384;
// residual-2 3 x (64*64*2) + 4 flags = 24592.
constexpr std::uint32_t kSingleLdsBytes = 16384;
constexpr std::uint32_t kResidual2LdsBytes = 24592;
// The plan's whitelist: the two M=512 FFN shapes.
bool ffn_shape(const Dims &d) {
  return (d.n == 17408 && d.k == 5120) || (d.n == 5120 && d.k == 17408);
}
// Production geometry, verbatim: 128-thread WG, 4 waves, 64x64 tile, each
// wave owns a 32x32 quadrant (4 accs), single-stage operand LDS, one K-slice
// in flight, two barriers per 64-K block.
template <bool Residual2>
std::string emit_ffn_wmma2(const KernelShapes &s, const Dims &d) {
  using Tile = Base<math::MatrixElem::kBF16>;
  using F = typename Tile::AFrag;
  using Op = typename Tile::Op;
  constexpr auto geo = geometry_of(Tile::kRow);
  static_assert(geo.wave == 32 && geo.split_k && geo.lane_k == 8);
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<F>(64u * 64u), ws = e.lds<F>(64u * 64u);
  kir::Tile<F> wl;
  kir::Tile<kir::f32> flags;
  if constexpr (Residual2) {
    wl = e.lds<F>(64u * 64u);
    flags = e.lds<kir::f32>(4u);
  }
  constexpr float min_normal = std::numeric_limits<float>::min();
  constexpr float max_bf16 = 3.3895313892515355e38f;
  const auto lid = e.let(math::local_id()), lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u), lo = e.let(lane % 16u),
             hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u),
             nbase = e.let((wg % nblocks) * 64u);
  const auto am = e.let((wave % 2u) * 32u), bn = e.let((wave / 2u) * 32u);
  // Rotate eight-half spans by row. Every span remains contiguous, while
  // neighboring lanes read different LDS bank groups rather than stride64.
  const auto address = [&](const kir::Val<kir::u32> &r,
                           const kir::Val<kir::u32> &k) {
    return r * 64u + ((k / 8u + r % 8u) % 8u) * 8u + k % 8u;
  };
  std::vector<kir::Local<kir::f32, 8>> acc;
  for (unsigned i = 0; i < 4; ++i) {
    acc.push_back(e.local<kir::f32, 8>());
    for (auto j : e.unroll(8u))
      acc.back()[j] = e.f32(0);
  }
  for (auto kb : e.range(0u, d.k, 64u)) {
    auto bad = e.var(0.0f);
    struct Marker {
      env::Emit &e;
      kir::LValue<kir::f32> bad;
      void operator()(const kir::Val<kir::f32> &value) {
        if constexpr (!Residual2)
          return;
        auto invalid = e.var(1.0f);
        if (auto safe = e.when(math::abs(value) <= max_bf16 &&
                               (value == 0.0f ||
                                math::abs(value) >= min_normal)))
          invalid = e.f32(0);
        bad = math::max(bad.read(), invalid.read());
      }
    } mark_unsafe{e, bad};
    // Each lane stages two 16-value row fragments. All 128 lanes execute both
    // barriers, including lanes covering padded M/N output edges.
    for (auto block : e.unroll(2u)) {
      const auto r = e.let(lid / 4u + block * 32u),
                 kc = e.let((lid % 4u) * 16u);
      const auto ar = e.let(mbase + r), bc = e.let(nbase + r);
      const auto av = e.local<F, 16>(), bv = e.local<F, 16>();
      const auto bl = e.local<F, 16>();
      for (auto j : e.unroll(16u)) {
        av[j] = math::narrow<F>(e.f32(0));
        bv[j] = math::narrow<F>(e.f32(0));
        if constexpr (Residual2)
          bl[j] = math::narrow<F>(e.f32(0));
      }
      if (auto active = e.when(ar < d.m)) {
        for (unsigned v = 0; v < 4; ++v) {
          const auto loaded = e.load(a.x, ar * d.k + kb + kc + v * 4u, 16u);
          for (unsigned j = 0; j < 4; ++j) {
            const auto original = e.let(loaded[j]);
            if constexpr (Residual2)
              mark_unsafe(original);
            av[v * 4 + j] = math::narrow<F>(original);
          }
        }
      }
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
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
          const auto original = e.let(
              math::fma(math::cast<kir::f32>(e.let(code)), scale, bias));
          if constexpr (Residual2)
            mark_unsafe(original);
          bv[j] = math::narrow<F>(original);
          if constexpr (Residual2) {
            // Preserve the original FP32 dequantization and subtract the
            // widened rounded high part. The low part is a second native BF16
            // operand.
            const auto residual =
                e.let(original - math::widen(bv[j].read()));
            mark_unsafe(residual);
            bl[j] = math::narrow<F>(residual);
          }
        }
      }
      for (auto j : e.unroll(16u)) {
        const auto index = e.let(address(r, kc + j));
        xs[index] = av[j].read();
        ws[index] = bv[j].read();
        if constexpr (Residual2)
          wl[index] = bl[j].read();
      }
    }
    if constexpr (Residual2) {
      for (unsigned bit : {1u, 2u, 4u, 8u, 16u})
        bad = math::max(bad.read(), math::shfl_xor(bad.read(), e.u32(bit)));
      if (auto leader = e.when(lane == 0u))
        flags[wave] = bad.read();
      e.barrier();
      const auto block_bad = e.let(flags[0u].read() + flags[1u].read() +
                                   flags[2u].read() + flags[3u].read());
      if (auto regular = e.when(block_bad == 0.0f)) {
        for (auto slice : e.unroll(4u)) {
          std::vector<kir::Local<F, 8>> af, bf, blf;
          for (unsigned i = 0; i < 2; ++i) {
            af.push_back(e.local<F, 8>());
            bf.push_back(e.local<F, 8>());
            blf.push_back(e.local<F, 8>());
            const auto ar = e.let(am + i * 16u + lo),
                       bc = e.let(bn + i * 16u + lo);
            const auto k = e.let(slice * 16u + hi * 8u);
            const auto ap = xs.load(e.let(address(ar, k)), 16u);
            const auto bp = ws.load(e.let(address(bc, k)), 16u);
            const auto blp = wl.load(e.let(address(bc, k)), 16u);
            for (auto j : e.unroll(8u)) {
              af.back()[j] = ap[j];
              bf.back()[j] = bp[j];
              blf.back()[j] = blp[j];
            }
          }
          for (unsigned m = 0; m < 2; ++m)
            for (unsigned n = 0; n < 2; ++n) {
              acc[m * 2 + n] =
                  math::mma<Op>(af[m].value(), bf[n].value(),
                                acc[m * 2 + n].value());
              acc[m * 2 + n] =
                  math::mma<Op>(af[m].value(), blf[n].value(),
                                acc[m * 2 + n].value());
            }
        }
      }
      if (auto exceptional = e.when(block_bad != 0.0f)) {
        for (unsigned m = 0; m < 2; ++m)
          for (unsigned n = 0; n < 2; ++n) {
            const auto col = e.let(nbase + bn + n * 16u + lo);
            for (auto j : e.unroll(8u)) {
              const auto rr =
                  e.let(mbase + am + m * 16u + j * geo.slot_step +
                        hi * geo.half_rows);
              if (auto active = e.when(rr < d.m && col < d.n)) {
                for (auto chunk :
                     e.range(kb / 16u, kb / 16u + 4u, 1u)) {
                  const auto gi =
                      e.let(col * d.groups + (chunk * 16u) / d.group);
                  const auto scale = e.let(math::widen(a.scales[gi])),
                             bias = e.let(math::widen(a.biases[gi]));
                  quant::dequant_chunk(
                      e, a.packed, quant::GroupAffine{6, 64},
                      e.let(col * d.words + chunk * 3u), scale, bias,
                      [&](int q, const kir::Val<kir::f32> &weight) {
                        const auto x =
                            a.x[rr * d.k + chunk * 16u +
                                static_cast<unsigned>(q)];
                        acc[m * 2 + n][j] =
                            math::fma(x, weight,
                                      acc[m * 2 + n][j].read());
                      });
                }
              }
            }
          }
      }
      e.barrier();
    } else {
      e.barrier();
      for (auto slice : e.unroll(4u)) {
        std::vector<kir::Local<F, 8>> af, bf;
        for (unsigned i = 0; i < 2; ++i) {
          af.push_back(e.local<F, 8>());
          bf.push_back(e.local<F, 8>());
          const auto ar = e.let(am + i * 16u + lo),
                     bc = e.let(bn + i * 16u + lo);
          for (auto j : e.unroll(8u)) {
            const auto k = e.let(slice * 16u + hi * 8u + j);
            af.back()[j] = xs[e.let(address(ar, k))];
            bf.back()[j] = ws[e.let(address(bc, k))];
          }
        }
        for (unsigned m = 0; m < 2; ++m)
          for (unsigned n = 0; n < 2; ++n)
            acc[m * 2 + n] =
                math::mma<Op>(af[m].value(), bf[n].value(),
                              acc[m * 2 + n].value());
      }
      e.barrier();
    }
  }
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned n = 0; n < 2; ++n) {
      const auto col = e.let(nbase + bn + n * 16u + lo);
      for (auto j : e.unroll(8u)) {
        const auto rr =
            e.let(mbase + am + m * 16u + j * geo.slot_step +
                  hi * geo.half_rows);
        if (auto active = e.when(rr < d.m && col < d.n))
          e.store(rr * d.n + col, acc[m * 2 + n][j].read());
      }
    }
  if (!body.lds().ok())
    return {};
  return body.str();
}
template <bool Residual2>
struct FfnWmma2Kernel final : KernelPrimitive<FfnWmma2Kernel<Residual2>> {
  static constexpr std::string_view kName = Residual2
      ? "quant_linear.q6_wmma2_bf16_residual2"
      : "quant_linear.q6_wmma2_bf16_single";
  static constexpr std::string_view kEntry = Residual2
      ? "lse_q6_wmma2_bf16_residual2"
      : "lse_q6_wmma2_bf16_single";
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
    return emit_ffn_wmma2<Residual2>(s, d);
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
    p.lds_bytes = Residual2 ? kResidual2LdsBytes : kSingleLdsBytes;
    return p;
  }
};
const KernelPrimitiveBase *select_ffn_wmma2_single(const KernelShapes &s) {
  constexpr auto r = Base<math::MatrixElem::kBF16>::kRow;
  if (!r.emittable() || !math::has_cap(device_matrix_caps(*s.device), r.cap) ||
      s.intrinsics->find(r.key).empty())
    return nullptr;
  static const FfnWmma2Kernel<false> kernel;
  return &kernel;
}
const KernelPrimitiveBase *select_ffn_wmma2_residual2(const KernelShapes &s) {
  constexpr auto r = Base<math::MatrixElem::kBF16>::kRow;
  if (!r.emittable() || !math::has_cap(device_matrix_caps(*s.device), r.cap) ||
      s.intrinsics->find(r.key).empty())
    return nullptr;
  static const FfnWmma2Kernel<true> kernel;
  return &kernel;
}
} // namespace
// Static registration: the micro fixture and the diagnostic server find the
// variants through find_primitive by name, which reads this registry.
namespace {
const FfnWmma2Kernel<false> _lse_prim_instance_FfnWmma2Single{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnWmma2Single{
    &_lse_prim_instance_FfnWmma2Single};

const FfnWmma2Kernel<true> _lse_prim_instance_FfnWmma2Residual2{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnWmma2Residual2{
    &_lse_prim_instance_FfnWmma2Residual2};
} // namespace
const graph::KernelPrimitiveBase *ffn_wmma2_q6_for(
    const graph::KernelShapes &s) {
  const auto d = dims_of(s);
  const auto *global = std::getenv("LSE_FFN_WMMA2");
  if (!s.device || !s.intrinsics || !d.valid ||
      s.device->max_threads_per_workgroup < 128 ||
      s.device->wavefront_size != 32 || d.m != 512 || !ffn_shape(d))
    return nullptr;
  // Env gate (Build 199): residual-2 is the DEFAULT for the M=512 FFN 17408
  // shapes; the single-product variant is reachable ONLY by explicit opt-in
  // (LSE_FFN_WMMA2=single, retained for the Build 198 A/B only). "0" declines
  // to the production path. LSE_FFN_WMMA2_SCALAR (any value other than "0")
  // forces the scalar path for the gated 17408 shapes only — the mid-shape
  // M=512 tile path (wmma_q6_linear_for, other shapes) is untouched. This is
  // the per-shape decline the gate attribution arm needs. The Build 198
  // micro numbers (cand-single 13.06-13.61 vs prod 13.08-13.52, flat;
  // cand-r2 10.7/10.0) are the A/B result store these shapes draw from: r2 is
  // the accuracy-qualified default, single is performance-record-only.
  // LSE_FFN_WMMA2_SCALAR=1 declines to the PRODUCTION wmma_q6_linear_for path
  // (single-product staged-bf16, the Build 199 attribution arm). When the
  // companion LSE_WMMA_FORCE_SCALAR_FFNS is set for those shapes, the
  // production selector also declines (wmma_q6_linear.cpp), so the 17408
  // shapes fall through to the true scalar quant_linear body while the
  // mid-shape M=512 tile path stays on the production tile. Either value
  // leaves the mid-shape path untouched.
  if (const auto *sc = std::getenv("LSE_FFN_WMMA2_SCALAR")) {
    if (sc[0] != '0' && sc[0] != '\0')
      return nullptr;
  }
  if (global && std::strcmp(global, "0") == 0)
    return nullptr;
  if (global && std::strcmp(global, "single") == 0)
    return select_ffn_wmma2_single(s);
  return select_ffn_wmma2_residual2(s);
}
} // namespace lse::kernels
