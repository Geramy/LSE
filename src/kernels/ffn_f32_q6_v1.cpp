// cand-ffn-f32 v1: the two schedule-sweep candidates for the M=512 FFN GEMM
// (f32, no vector.mma). v0 (ffn_f32_q6.cpp: 128t/4-wave, 64x64, single-stage
// 32 KiB LDS, 16 accs/lane) measured 2.16/1.31/1.16 TFLOPS vs the scalar
// body's 4.7. The sweep (v1-probe-report.md, section 3) ranks these two
// distinct schedule hypotheses above everything else:
//
// v1a  "DRAM-streamed weights + 2-stage x, 32x64 tile, 128t"
//   * NO weight staging at all: each lane decodes its TWO output columns'
//     16 Q6 codes (2 x 3 scalar u32 packed loads + 2 scale/bias pairs, the
//     scalar body's exact per-column path) IN REGISTER per 16-K sub-run,
//     while the 4 waves share the x panel through 2 LDS stages
//     (2 x 32x64x4 = 16 KiB). The K step is a 32-K half-ping-pong: the
//     fill of the next 32-K half is issued DURING the consume of the
//     current one, one barrier per half-step, depth 2. Fixes the v0
//     suspects: the 32 KiB single-stage weight staging + two 64-K
//     barriers per block, and the two-panel LDS read fan-out. Weights
//     stream from DRAM/L3 the way the scalar body does; x stays in LDS so
//     the broadcast reads are local.
//   * 32x64 output tile per WG: each wave owns an 8-row strip x all 64
//     cols = 512 outputs / 32 lanes = 16 accs/lane (8 rows x 2 cols, the
//     v0 consume shape at 8 rows). Each decoded weight feeds 8 FMAs.
//
// v1b  "LDS-both, 64x64, 128t, 2-stage x"
//   * v0's geometry verbatim (64x64 tile, 4 waves, 16-row x 64-col strip
//     per wave, lane owns cols l and l+32, 16 accs/lane -- the shape v0
//     compiled on the pinned loomc) with the weight panel double-buffered
//     (two 64x64 f32 panels) so the fill of K-block i+1 runs AHEAD of the
//     consume of block i, with the x refill in the same window. v0's
//     serial fill->barrier->consume->barrier becomes
//     fill-ahead -> barrier -> consume: the weight decode (the expensive
//     part) overlaps the consume, and only the x refill (16 KiB of 16-B
//     loads) is serial. One fill window per block instead of two.
//
// Both keep the exact Q6 unpack algebra (/, %, cross-word spillover, one
// f32 fma) and the ascending per-element fma order, so bit-exactness vs the
// scalar body is expected (rel-L2 0.0), as for v0.
//
// Selection: LSE_FFN_F32_V1=v1a|v1b (see ffn_f32_q6.hpp). v0 stays intact
// and is the default when the env names anything else.
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
// Field-for-field the quant_linear dims contract (same as ffn_f32_q6.cpp).
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
// The same M=512 prefill whitelist v0 uses.
bool ffn_shape(const Dims &d) {
  if (d.k == 5120)
    return d.n == 17408 || d.n == 5120 || d.n == 10240 || d.n == 6144 ||
           d.n == 12288;
  if (d.k == 17408)
    return d.n == 5120;
  return d.k == 1024;
}
// The exact Q6 dequant of a 16-code run at packed-word base `wb` (the v0 /
// scalar-body algebra: '/', '%', cross-word spillover, one f32 fma per code).
void dequant_16(env::Emit &e, const Args &a, const kir::Val<kir::u32> &wb,
                const kir::Val<kir::f32> &scale,
                const kir::Val<kir::f32> &bias,
                std::array<kir::Val<kir::f32>, 16> &w) {
  const std::array<kir::Val<kir::u32>, 3> packed{
      e.let(a.packed[wb]), e.let(a.packed[wb + 1u]), e.let(a.packed[wb + 2u])};
  for (unsigned j = 0; j < 16; ++j) {
    const unsigned off = (j * 6) % 32, wi = j * 6 / 32;
    auto code = packed[wi] / (1u << off);
    if (off > 26)
      code = code +
             (packed[wi + 1] % (1u << (off - 26))) * (1u << (32 - off));
    else
      code = code % 64u;
    w[j] = e.let(math::fma(math::cast<kir::f32>(e.let(code)), scale, bias));
  }
}
// Bank swizzle for the 64-wide f32 panels (the v0/production 8-float
// rotate). A 32-wide swizzle would collide lanes within one 4-float group
// (lane l%16 and (l%16)+16 hit the same bank quad); the 64-wide map stays.
inline kir::Val<kir::u32>
x16_address(env::Emit &e, const kir::Val<kir::u32> &r,
            const kir::Val<kir::u32> &k) {
  return e.let(r * 64u + ((k / 8u + r % 8u) % 8u) * 8u + k % 8u);
}
// ---------------------------------------------------------------------------
// v1a: DRAM-streamed weights + 2-stage x, 32x64 tile, 128t, 16 accs/lane.
// LDS: 2 x (32 x 64 x 4) = 16384 B. Grid: (m/32) * (n/64).
//
// Layout: wave w owns rows [8w, 8w+8) x all 64 cols; lane l owns columns
// nbase+l and nbase+l+32 (16 accs = 8 rows x 2 cols). Per 16-K sub-run the
// lane decodes its two columns in registers (dequant_16 x2) and runs
// 16 k-steps of (8 LDS row reads x 2 cols).
// ---------------------------------------------------------------------------
constexpr std::uint32_t kV1aLdsBytes = 2u * (32u * 64u * 4u);
std::string emit_ffn_f32_v1a(const KernelShapes &s, const Dims &d) {
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs0 = e.lds<kir::f32>(32u * 64u);
  const auto xs1 = e.lds<kir::f32>(32u * 64u);
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 32u);
  const auto nbase = e.let((wg % nblocks) * 64u);
  const auto rbase = e.let(wave * 8u);
  const auto col0 = e.let(nbase + lane);
  const auto col1 = e.let(nbase + lane + e.u32(32u));
  // 16 accumulators: 8 rows x 2 columns.
  std::vector<kir::LValue<kir::f32>> acc;
  for (unsigned r = 0; r < 16; ++r) acc.push_back(e.var(0.0f));
  // Prefill BOTH stages: stage h holds k [32h, 32h+32) of block 0. A
  // 32-row x 32-k stage = 1024 4-f32 runs = 128 lanes x 8: run = t,
  // row = run/8 (8 rows per run-window, 2 windows per 16-row span),
  // k-run = (run%8)*4 + (lid%16)*4.
  for (unsigned r16 = 0; r16 < 2; ++r16) {
    for (unsigned half = 0; half < 2; ++half) {
      auto& dst = half == 0 ? xs0 : xs1;
      // rows 16*r16..16*r16+15: 16 runs x 16 lanes = 256 4-f32 runs =
      // 1024 / 128 lanes = 8 per lane: run = t, k-run = (run%8)*4 + (lid%16)*4,
      // row = 16*r16 + run/8.
      for (unsigned t = 0; t < 8; ++t) {
        const auto rr = e.let(e.u32(r16 * 16 + t) / 8u);
        const auto kc0 = e.let(e.u32((r16 * 16 + t) % 8u) * 4u +
                               (lid % 16u) * 4u);
        const auto ar = e.let(mbase + rr);
        const auto loaded = e.load(
            a.x, e.let(ar * d.k + e.u32(half * 32) + kc0), 16u);
        for (unsigned j = 0; j < 4; ++j)
          dst[x16_address(e, rr, kc0 + e.u32(j))] = e.let(loaded[j]);
      }
    }
  }
  e.barrier();
  for (auto kb : e.range(0u, d.k, 64u)) {
    for (unsigned half = 0; half < 2; ++half) {
      const auto cur = half == 0 ? xs0 : xs1;
      const auto nxt = half == 0 ? xs1 : xs0;
      const auto hbase = e.let(kb + e.u32(32u * half));
      for (int sub = 0; sub < 2; ++sub) {
        const auto kbase = e.let(hbase + e.u32(sub * 16u));
        // -- Fill the NEXT 32-K half (k = kb + 64 + (half^1)*32 + sub*16)
        //    while consuming this one. 512 runs / 128 lanes = 4 per lane.
        //    In-bounds: on the whitelist K is a multiple of 64, so the
        //    last fill lands exactly at the end of row K (index < ar*K +
        //    (K-1) since koff+kc0+j <= 64+32+16+28+3 < 144 <= K).
        for (unsigned t = 0; t < 4; ++t) {
          const std::uint32_t thalf = (half + 1u) % 2u;
          const std::uint32_t run = sub * 16 + t;  // 16-run row window
          const std::uint32_t rr_c = run / 8u;
          const std::uint32_t kc0_c = (run % 8u) * 4u;
          const std::uint32_t nko_c = 64u + thalf * 32u + sub * 16u;
          const auto rr = e.let(e.u32(rr_c));
          const auto kc0 =
              e.let(e.u32(kc0_c) + (lid % 16u) * 4u);
          const auto ar = e.let(mbase + rr);
          const auto nko = e.let(kb + e.u32(nko_c));
          const auto loaded = e.load(a.x, e.let(ar * d.k + nko + kc0), 16u);
          for (unsigned j = 0; j < 4; ++j)
            nxt[x16_address(e, rr, kc0 + e.u32(j))] = e.let(loaded[j]);
        }
        // -- Consume the 16 K values. The lane decodes its TWO output
        //    columns (the scalar body's per-column Q6 path), then 16
        //    k-steps x 8 rows x 2 cols of FMAs, k ascending.
        const auto gi0 = e.let(col0 * d.groups + kbase / d.group);
        const auto gi1 = e.let(col1 * d.groups + kbase / d.group);
        const auto scale0 = e.let(math::widen(a.scales[gi0]));
        const auto bias0 = e.let(math::widen(a.biases[gi0]));
        const auto scale1 = e.let(math::widen(a.scales[gi1]));
        const auto bias1 = e.let(math::widen(a.biases[gi1]));
        const auto wb0 = e.let(col0 * d.words + kbase / 16u * 3u);
        const auto wb1 = e.let(col1 * d.words + kbase / 16u * 3u);
        std::array<kir::Val<kir::f32>, 16> w0{}, w1{};
        if (auto active = e.when(col0 < d.n))
          dequant_16(e, a, wb0, scale0, bias0, w0);
        if (auto active = e.when(col1 < d.n))
          dequant_16(e, a, wb1, scale1, bias1, w1);
        for (unsigned kk = 0; kk < 16; ++kk) {
          const auto k = e.let(kbase + e.u32(kk));
          for (unsigned r = 0; r < 8; ++r) {
            const auto x =
                cur[x16_address(e, rbase + e.u32(r), k)].read();
            acc[2 * r] = math::fma(x, w0[kk], acc[2 * r].read());
            acc[2 * r + 1] = math::fma(x, w1[kk], acc[2 * r + 1].read());
          }
        }
      }
      e.barrier();
    }
  }
  // Epilogue: the lane stores its 2 columns x 8 rows.
  for (unsigned r = 0; r < 8; ++r) {
    const auto rr0 = e.let(mbase + rbase + e.u32(r));
    if (auto active = e.when(rr0 < d.m && col0 < d.n))
      e.store(e.let(rr0 * d.n + col0), acc[2 * r].read());
    if (auto active = e.when(rr0 < d.m && col1 < d.n))
      e.store(e.let(rr0 * d.n + col1), acc[2 * r + 1].read());
  }
  return body.lds().ok() ? body.str() : std::string{};
}
// ---------------------------------------------------------------------------
// v1b: LDS-both, 64x64 tile, 128t, 2-stage weight, 16 accs/lane.
// LDS: xs 16384 + ws0 16384 + ws1 16384 = 49152 B. Grid: (m/64)*(n/64).
//
// Layout: v0's verbatim (wave w owns rows [16w, 16w+16), lane l owns cols
// l and l+32, 16 accs = 16 rows x 2 cols). Staging: fill-ahead -- the
// fill of block i+1 (weights into ws^(i+1)%2, x into xs) happens BEFORE
// the consume of block i, after one barrier. v0's per-block sequence is
// fill -> BARRIER -> consume -> BARRIER (the consume cannot start until
// the whole WG has filled); here the consume of block i starts one block
// sooner in the pipeline and the only serial part is the next x refill
// (16 KiB of 16-B loads); the expensive weight decode overlaps the consume
// of the previous block.
//
// Wait -- that is still 2 barriers per block (the fill writes BOTH xs and
// ws while the previous consume reads both). The true 1-barrier form needs
// x double-buffered too (64 KiB, the cap, no headroom -- the sweep table
// rows the 49152 B / 2-barrier form and notes the 64 KiB full form as
// out of budget). What v1b buys over v0 with the same 2 barriers: the
// weight DECODE (not just its memory loads) is prefetched one block ahead,
// so the consume window sees no integer-division tail and no scale/bias
// loads; v0's fill window carried them all serially before the consume.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kV1bLdsBytes = 3u * (64u * 64u * 4u);  // 49152
std::string emit_ffn_f32_v1b(const KernelShapes &s, const Dims &d) {
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto xs = e.lds<kir::f32>(64u * 64u);
  const auto ws0 = e.lds<kir::f32>(64u * 64u);
  const auto ws1 = e.lds<kir::f32>(64u * 64u);
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u);
  const auto nbase = e.let((wg % nblocks) * 64u);
  const auto rbase = e.let(wave * 16u);
  // Fill one 64-K block at offset kb: xs (v0's 1024-run pattern) + the
  // weight panel (v0's 2 x 16-code row-run pattern).
  auto fill_block = [&](kir::Tile<kir::f32> ws,
                        const kir::Val<kir::u32> &kb) {
    for (auto t : e.range(lid, e.u32(1024u), 128u)) {
      const auto rr = e.let(t / 16u);
      const auto kc0 = e.let((t % 16u) * 4u);
      const auto ar = e.let(mbase + rr);
      const auto loaded = e.load(a.x, e.let(ar * d.k + kb + kc0), 16u);
      const auto index = e.let(x16_address(e, rr, kc0));
      for (unsigned j = 0; j < 4; ++j)
        xs[index + e.u32(j)] = e.let(loaded[j]);
    }
    for (unsigned block = 0; block < 2; ++block) {
      const auto r = e.let(lid / 4u + e.u32(block * 32u));
      const auto kc = e.let((lid % 4u) * 16u);
      const auto bc = e.let(nbase + r);
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi]));
        const auto bias = e.let(math::widen(a.biases[gi]));
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
        std::array<kir::Val<kir::f32>, 16> w{};
        dequant_16(e, a, wb, scale, bias, w);
        for (unsigned j = 0; j < 16; ++j)
          ws[e.let(x16_address(e, r, kc + e.u32(j)))] = w[j];
      }
    }
  };
  // v0's consume of one 64-K block (16 accs, 64 k-steps x 16 rows x 2 cols),
  // with k ascending so the per-element fma order matches the scalar body.
  auto consume_block = [&](kir::Tile<kir::f32> ws,
                           std::vector<kir::LValue<kir::f32>> &acc) {
    for (unsigned kt = 0; kt < 64; ++kt) {
      const auto k = e.u32(kt);
      for (unsigned q = 0; q < 2; ++q) {
        const auto c = e.let(lane + e.u32(32u * q));
        const auto wv = ws[e.let(x16_address(e, c, k))].read();
        for (unsigned t = 0; t < 16; ++t)
          acc[t] = math::fma(xs[e.let(x16_address(e, rbase + e.u32(t), k))]
                                 .read(),
                             wv, acc[t].read());
      }
    }
  };
  std::vector<kir::LValue<kir::f32>> acc;
  for (unsigned r = 0; r < 16; ++r) acc.push_back(e.var(0.0f));
  // Prefill: block 0 into (xs, ws0).
  fill_block(ws0, e.u32(0));
  e.barrier();
  for (auto kb : e.range(0u, d.k, 64u)) {
    // Fill-ahead (the v0 suspect's fix): block i+1's fill runs BEFORE the
    // barrier, so its expensive WEIGHT decode (integer unpack + scale/bias
    // loads) completes while this block's consume is about to run; the only
    // serial remainder is the x refill (16 KiB of 16-B loads), staged in
    // the same window as the weights. v0's serial fill->BARRIER->consume
    // ->BARRIER pair becomes fill-ahead->BARRIER->consume->BARRIER with
    // the next block's weights already in LDS when this consume ends.
    {
      // Guard the fill-ahead on the DEVICE value: kb is a Val, so the
      // comparison must stay device-side (the host cannot see it).
      if (auto not_last = e.when(kb < e.let(e.u32(d.k) - 64u)))
        fill_block(ws1, e.let(kb + e.u32(64u)));
    }
    e.barrier();
    consume_block(ws0, acc);
    e.barrier();
  }
  // Epilogue: the lane stores its 2 columns x 16 rows (v0's).
  for (unsigned q = 0; q < 2; ++q) {
    const auto col = e.let(nbase + lane + e.u32(32u * q));
    for (unsigned t = 0; t < 16; ++t) {
      const auto rr = e.let(mbase + rbase + e.u32(t));
      if (auto active = e.when(rr < d.m && col < d.n))
        e.store(e.let(rr * d.n + col), acc[t].read());
    }
  }
  return body.lds().ok() ? body.str() : std::string{};
}
// ---------------------------------------------------------------------------
// Primitives + selection.
// ---------------------------------------------------------------------------
template <bool IsA>
struct FfnF32V1Kernel final : KernelPrimitive<FfnF32V1Kernel<IsA>> {
  static constexpr std::string_view kName = IsA
      ? "quant_linear.q6_f32_prefill_v1a"
      : "quant_linear.q6_f32_prefill_v1b";
  static constexpr std::string_view kEntry =
      IsA ? "lse_q6_f32_prefill_v1a" : "lse_q6_f32_prefill_v1b";
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
    return IsA ? emit_ffn_f32_v1a(s, d) : emit_ffn_f32_v1b(s, d);
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
    p.workgroup_count[0] = IsA
        ? ((d.m + 31) / 32) * ((d.n + 63) / 64)
        : ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.workgroup_count[1] = p.workgroup_count[2] = 1;
    p.lds_bytes = IsA ? kV1aLdsBytes : kV1bLdsBytes;
    return p;
  }
};
const KernelPrimitiveBase *ffn_f32_q6_v1_select(bool is_a,
                                                const KernelShapes &s) {
  const auto d = dims_of(s);
  if (!s.device || !s.intrinsics || !d.valid ||
      s.device->max_threads_per_workgroup < 128 ||
      s.device->wavefront_size != 32 || d.m != 512 || !ffn_shape(d))
    return nullptr;
  static FfnF32V1Kernel<true> a;
  static FfnF32V1Kernel<false> b;
  if (is_a)
    return &a;
  return &b;
}
} // namespace
const graph::KernelPrimitiveBase *ffn_f32_q6_v1a_for(
    const graph::KernelShapes &s) {
  const char *sel = std::getenv("LSE_FFN_F32_V1");
  if (sel == nullptr || std::strcmp(sel, "v1a") != 0)
    return nullptr;
  return ffn_f32_q6_v1_select(true, s);
}
const graph::KernelPrimitiveBase *ffn_f32_q6_v1b_for(
    const graph::KernelShapes &s) {
  const char *sel = std::getenv("LSE_FFN_F32_V1");
  if (sel == nullptr || std::strcmp(sel, "v1b") != 0)
    return nullptr;
  return ffn_f32_q6_v1_select(false, s);
}
namespace {
const FfnF32V1Kernel<true> _lse_prim_instance_FfnF32V1a{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnF32V1a{
    &_lse_prim_instance_FfnF32V1a};
const FfnF32V1Kernel<false> _lse_prim_instance_FfnF32V1b{};
const graph::PrimitiveRegistrar _lse_prim_reg_FfnF32V1b{
    &_lse_prim_instance_FfnF32V1b};
} // namespace
} // namespace lse::kernels
