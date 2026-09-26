// cand-fp8-mma: quant_linear.q6_fp8_mma_prefill.
//
// The fp8 packed-WMMA version of the 13.7 TFLOPS bf16 record
// (cand-ffn-wmma2). Same geometry line for line: 128-thread workgroup,
// 4 waves, 64x64 output tile, each wave a 32x32 quadrant (four 16x16
// accumulators), single-stage LDS (xs/ws, 2 x 64*64*4 = 32768 B), one
// 64-K slice in flight, two barriers per K block. What changes is only the
// operand format: both operands ride the packed8 fragment
// (wmma12.f32.16x16x16.fp8_fp8, two i32 registers of four e4m3 per lane,
// split-K across the half-waves, f32 row-block-half-wave accumulator) and
// the mma is `vector.mma` on the f8e4m3 encoding.
//
// Weight path: the verbatim Q6 extraction of the production staged-bf16
// kernel (six-bit codes with cross-word spillover), dequantized in-kernel
// to f32 at the group scale/bias, then RNE-saturated to e4m3 and packed
// four-to-i32 (math::pack_fp8<kFp8> -> the `pack4.fp8.ocp` builtin, which
// is SATFINITE(448) + RNE -- the documented rounding). Per-64-block,
// because the group IS the 64-block: every weight in a (row, group) pair
// is converted with the same scale, so the e4m3 value stream is exactly the
// per-64-block RN-even value.
//
// Activation path: f32 x -> e4m3 (same clamp/RNE), four-to-i32, staged.
// No rescale (fp32 scale = 1.0); the clamp bounds the range.
//
// Error attribution (measured in ffn_check.cpp, not asserted here):
//   weight error  = |e4m3(f32_w) - f32_w|   per-64-block, max abs, vs the
//                   exact scalar body (which never rounds the weight);
//   activation error = |e4m3(x) - x|         per element, max abs / rel-L2.
// The end-to-end rel-L2 vs the scalar body is MEASURED and reported, not
// gated at 0.005 -- fp8 rounds BOTH operands, so it is expected around/
// above the measured bf16-staging floor (0.0229 r2, 0.0091 single-product).
// The PPL system is the quality arbiter, not this gate.
//
// Selection: LSE_FFN_FP8_MMA=1 AND the M=512 FFN whitelist. Any other value
// (unset, "0") declines, so the default is bit-exact the wmma2 path.
#include "lse/graph/kernel_args.hpp"
#include "lse/kernels/ffn_fp8_q6.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math/fp8.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
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
// Field-for-field the wmma2 contract (re-implemented here because that copy
// is namespace-internal there).
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
// The fp8 packed row: f32 acc, four e4m3 per i32 register per lane,
// k_step 16 (chained 1), the measured RDNA4 layouts.
using RowFp8 =
    MatrixTile<struct FfnFp8Tile, math::MatrixTarget::kRdna4,
               math::MatrixElem::kF32, math::MatrixElem::kFp8, 16, 16, 16>;
constexpr auto kRowFp8 = RowFp8::kRow;
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales, biases;
  env::Out<kir::f32, env::Emit> out;
};
// LDS budget: two 64x64 i32 panels. The fp8 operand is 1 byte wide, so a
// panel is 4x the bf16 record's 2x64x64 footprint: 32768 B.
constexpr std::uint32_t kFp8LdsBytes = 2u * 64u * 64u * 4u;
// The whitelist: the two M=512 FFN shapes (the 13.7-record shapes).
bool ffn_shape(const Dims &d) {
  return (d.n == 17408 && d.k == 5120) || (d.n == 5120 && d.k == 17408);
}
std::string emit_ffn_fp8(const KernelShapes &s, const Dims &d) {
  using Tile = RowFp8;
  using F = std::int32_t;  // packed fp8 fragment register: 4 e4m3 in an i32
  using Op = typename Tile::Op;
  constexpr auto geo = geometry_of(Tile::kRow);
  static_assert(geo.wave == 32 && geo.split_k && geo.lane_k == 8);
  static_assert(kRowFp8.pack == 4 && kRowFp8.a_len == 2 &&
                   kRowFp8.chained == 1,
                "the fp8 row is two i32 registers of four e4m3 per lane");
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<F>(64u * 64u), ws = e.lds<F>(64u * 64u);
  const auto lid = e.let(math::local_id()), lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u), lo = e.let(lane % 16u),
             hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u),
             nbase = e.let((wg % nblocks) * 64u);
  const auto am = e.let((wave % 2u) * 32u), bn = e.let((wave / 2u) * 32u);
  // The same swizzle as the bf16 record, in i32 elements (four e4m3 per
  // element). Eight-e4m3 runs stay contiguous per lane; neighboring lanes
  // read different bank groups.
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
    // Each lane stages two 16-value row fragments; the weight fragment is
    // dequantized then RNE-saturated to e4m3, the activation converted
    // element-wise (scale 1.0).
    for (auto block : e.unroll(2u)) {
      const auto r = e.let(lid / 4u + block * 32u),
                 kc = e.let((lid % 4u) * 16u);
      const auto ar = e.let(mbase + r), bc = e.let(nbase + r);
      const auto av = e.local<kir::i32, 16>(), bv = e.local<kir::i32, 16>();
      for (auto j : e.unroll(16u)) {
        av[j] = e.let(math::cast<kir::i32>(e.u32(0)));
        bv[j] = e.let(math::cast<kir::i32>(e.u32(0)));
      }
      if (auto active = e.when(ar < d.m)) {
        for (unsigned v = 0; v < 4; ++v) {
          const auto loaded = e.load(a.x, ar * d.k + kb + kc + v * 4u, 16u);
          for (unsigned j = 0; j < 4; ++j) {
            // One e4m3 word per element: pack the same value four times into
            // the four byte lanes of the i32 fragment register.
            const auto packed1 =
                math::pack_fp8<math::MatrixElem::kFp8>(loaded[j], loaded[j],
                                                       loaded[j], loaded[j]);
            av[v * 4 + j] = e.let(math::cast<kir::i32>(packed1));
          }
        }
      }
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        const auto wwords = e.let(((kb + kc) / 16u) * 3u);
        const auto wb = e.let(bc * d.words + wwords);
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
          // Four consecutive 6-bit codes share the group's scale/bias (the
          // group is a multiple of 32, a fragment is 16 wide), so they pack
          // into one i32 as one RNE-saturated e4m3 word.
          std::array<kir::Val<kir::f32>, 4> q4{};
          for (unsigned g = 0; g < 4; ++g) {
            const unsigned jj = j + g;
            const unsigned offg = (jj * 6) % 32, wig = (jj * 6) / 32;
            auto codeg = packed[wig] / (1u << offg);
            if (offg > 26)
              codeg = codeg + (packed[wig + 1] % (1u << (offg - 26))) *
                             (1u << (32 - offg));
            else
              codeg = codeg % 64u;
            q4[g] = e.let(math::fma(math::cast<kir::f32>(e.let(codeg)),
                                     scale, bias));
          }
          bv[j] = e.let(math::cast<kir::i32>(
              math::pack_fp8<math::MatrixElem::kFp8>(q4[0], q4[1], q4[2],
                                                     q4[3])));
        }
      }
      for (auto j : e.unroll(16u)) {
        const auto index = e.let(address(r, kc + j));
        xs[index] = av[j].read();
        ws[index] = bv[j].read();
      }
    }
    e.barrier();
    // Consume: 4 slices of 16 K, each split across the half-waves (hi*8),
    // two i32 fragment registers per operand per fragment, four 16x16 mma.
    for (auto slice : e.unroll(4u)) {
      // The mma row takes two i32 fragment registers per operand (a_len ==
      // 2): one per 16x16 tile of the 16-row window. Lanes 0..7 of the row's
      // 16-e4m3 window feed the first register, 8..15 the second, so each
      // register is its own 2-lane local — the shared fragment contract of
      // the spelling, not one 8-lane vector of both tiles.
      std::vector<kir::Local<F, 2>> af, bf;
      for (unsigned i = 0; i < 2; ++i) {
        af.push_back(e.local<F, 2>());
        bf.push_back(e.local<F, 2>());
        const auto ar = e.let(am + i * 16u + lo),
                   bc = e.let(bn + i * 16u + lo);
        for (auto j : e.unroll(2u)) {
          const auto k = e.let(slice * 16u + hi * 8u + j * 8u);
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
struct FfnFp8Kernel final : KernelPrimitive<FfnFp8Kernel> {
  static constexpr std::string_view kName = "quant_linear.q6_fp8_mma_prefill";
  static constexpr std::string_view kEntry = "lse_q6_fp8_mma_prefill";
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
    return emit_ffn_fp8(s, d);
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
    p.lds_bytes = kFp8LdsBytes;
    return p;
  }
};
} // namespace
// Static registration: find_primitive by name locates it (the check suite
// and the diagnostic server use this).
namespace {
const FfnFp8Kernel _lse_prim_instance_FfnFp8{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnFp8{
    &_lse_prim_instance_FfnFp8};
} // namespace
const graph::KernelPrimitiveBase *ffn_fp8_q6_for(
    const graph::KernelShapes &s) {
  const auto d = dims_of(s);
  const auto *global = std::getenv("LSE_FFN_FP8_MMA");
  if (!global || std::strcmp(global, "1") != 0)
    return nullptr;
  if (!s.device || !s.intrinsics || !d.valid ||
      s.device->max_threads_per_workgroup < 128 ||
      s.device->wavefront_size != 32 || d.m != 512 || !ffn_shape(d))
    return nullptr;
  // The fp8 row must be emittable and the device must carry the capability;
  // the row's key must be spellable by this dialect's source table (the
  // 0.4.1 Loom matrix spelling or the HIP builtin).
  if (!kRowFp8.emittable() ||
      !math::has_cap(device_matrix_caps(*s.device), kRowFp8.cap) ||
      s.intrinsics->find(kRowFp8.key).empty())
    return nullptr;
  static const FfnFp8Kernel kernel;
  return &kernel;
}
} // namespace lse::kernels
