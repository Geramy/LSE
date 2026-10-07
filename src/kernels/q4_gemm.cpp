// Group-affine 4-bit weights against f32 activations, as a tiled matrix-core
// GEMM for prefill-shaped M.
//
// Two launches. The first narrows the activation to an f16 panel once
// (quant_activation.f16_panel.v1); every GEMM that reads the same activation
// shares it. The second (quant_linear.q4_gemm_f16.v1) owns a BM x BN output
// tile per workgroup and walks K one step at a time: it copies the step's
// activation tile from the panel into workgroup scratch, decodes the step's
// weight tile there (codes to f16 with the group's scale and bias applied),
// and every wave reads its matrix operands from scratch. The weight tile is
// decoded once per workgroup rather than once per wave, and no activation is
// converted more than once, which is what the per-wave forms spend their
// time on at M >= 16. The next step's global reads are issued before the
// current step's matrix instructions so their latency hides behind the math.
//
// Within every aligned run of eight K positions the panel and the decoded
// weights both order their lanes k0,k4,k1,k5,k2,k6,k3,k7. That is the order
// the weight decode produces for free (each 32-bit lane takes code j and code
// j+4 of a word); since a lane's matrix operand is a whole aligned run, the
// contraction over K is unchanged.
//
// Accumulation is FP32. Nothing here names a device: the matrix row comes from
// the table, the load width from the device's load budget, and the tile from
// the shape. A device without the row runs the original quant_linear on the
// first four operands.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <cstdlib>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

namespace env = graph::env;
namespace kir = graph::kir;
namespace math = lse::math;

using graph::KernelShapes;
using graph::ThreadPlan;

namespace {

constexpr std::uint32_t kThreads = 256;
// One K step. A whole number of them must make one quantization group so a
// column's scale and bias are constant across the step.
constexpr std::uint32_t kBK = dispatch::kQ4GemmStepK;
// Halves of padding per scratch row: a row stride of 144 bytes puts the
// sixteen rows a fragment load touches on sixteen distinct 16-byte bank
// groups.
constexpr std::uint32_t kPad = 8;
constexpr std::uint32_t kRowHalves = kBK + kPad;
// Panel lanes per piece: one 16-byte move, and one aligned run of K.
constexpr std::uint32_t kRun = 8;

constexpr std::string_view kPanelName = "quant_activation.f16_panel.v1";
constexpr std::string_view kGemmName = "quant_linear.q4_gemm_f16.v1";
constexpr std::string_view kGemmSlicesName = "quant_linear.q4_gemm_f16.slices.v1";
constexpr std::string_view kSliceSumName = "quant_linear.q4_gemm_f16.slice_sum.v1";

// The panel's position for element k of a row: within each aligned run of
// eight, k0,k4,k1,k5,k2,k6,k3,k7.
constexpr std::size_t panel_slot(std::size_t k) noexcept {
  const std::size_t base = k & ~std::size_t{7};
  const std::size_t r = k & 7;
  return base + (r < 4 ? 2 * r : 2 * (r - 4) + 1);
}

// ---------------------------------------------------------------------------
// The f16 activation panel
// ---------------------------------------------------------------------------

Result<Shape> panel_shape(std::span<const Shape> in) {
  if (in.size() != 1 || !in[0].rank() || !in[0].elem_count() ||
      in[0].elem_count() > UINT32_MAX)
    return LSE_ERROR(kInvalidArgument, "f16 panel takes one activation");
  const auto k = in[0].dim(in[0].rank() - 1);
  if (k <= 0 || k % static_cast<std::int64_t>(kRun) != 0)
    return LSE_ERROR(kInvalidArgument, "f16 panel rows must be whole runs");
  return Shape{static_cast<std::int64_t>(in[0].elem_count() /
                                         static_cast<std::uint64_t>(k)),
               k};
}

struct PanelArgs {
  env::In<kir::f32, env::Emit> x;
  env::Out<lse::f16, env::Emit> out;
};

struct F16PanelKernel final : graph::KernelPrimitive<F16PanelKernel> {
  static constexpr std::string_view kName = kPanelName;
  static constexpr std::string_view kEntry = "lse_quant_activation_f16_panel_v1";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF16;
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return panel_shape(in);
  }

  static bool supported(const KernelShapes& s) {
    const auto out = panel_shape(s.inputs);
    return out.ok() && s.output == *out && s.input_dtypes.size() == 1 &&
           s.input_dtypes[0] == DType::kF32 && s.output_dtype == DType::kF16 &&
           s.device && s.intrinsics && device_load_bytes(s.device) >= 16u &&
           device_store_bytes(s.device) >= 16u &&
           !s.intrinsics->find("cvt8i.f16.f32").empty();
  }

  Status eval_cpu_typed(std::span<const graph::HostTensorView> in,
                        graph::HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 ||
        out.dtype != DType::kF16)
      return LSE_ERROR(kInvalidArgument, "invalid f16 panel storage");
    const std::array shapes{in[0].shape};
    LSE_ASSIGN_OR(const auto expected, panel_shape(shapes));
    const std::size_t count = expected.elem_count();
    if (out.shape != expected || in[0].bytes.size() != count * 4 ||
        out.bytes.size() != count * 2)
      return LSE_ERROR(kInvalidArgument, "invalid f16 panel byte extent");
    for (std::size_t i = 0; i < count; ++i) {
      float v;
      std::memcpy(&v, in[0].bytes.data() + i * 4, 4);
      const float16_t h = static_cast<float16_t>(v);
      std::memcpy(out.bytes.data() + panel_slot(i) * 2, &h, 2);
    }
    return OkStatus();
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!supported(s)) return {};
    const auto count = static_cast<std::uint32_t>(s.output.elem_count());
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    PanelArgs a;
    if (!env::bind(kb, a, s)) return {};
    env::Emit e{&kb};
    const auto piece = e.let(math::workgroup_id_x() * kThreads + math::local_id());
    if (auto live = e.when(piece < count / kRun)) {
      const auto at = e.let(piece * kRun);
      const auto lo = e.load(a.x, at, 16u);
      const auto hi = e.load(a.x, e.let(at + 4u), 16u);
      using F32x4 = lse::vec<kir::f32, 4>;
      const auto v = kb.call<lse::vec<lse::f16, 8>>(
          "cvt8i.f16.f32", kir::Val<F32x4>(&kb.types(), &kb.ir(), lo.id()),
          kir::Val<F32x4>(&kb.types(), &kb.ir(), hi.id()));
      a.out.b.store(at, kir::Pack<lse::f16>(v.types(), v.body(), v.id(), 8),
                    16u);
    }
    return kb.str();
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    if (!supported(s)) return tp;
    const auto pieces = static_cast<std::uint32_t>(s.output.elem_count()) / kRun;
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = (pieces + kThreads - 1u) / kThreads;
    return tp;
  }
};

// ---------------------------------------------------------------------------
// The GEMM
// ---------------------------------------------------------------------------

struct Tile {
  std::uint32_t bm = 0, bn = 0, wm = 0, wn = 0;
};

struct Dims {
  std::uint32_t m = 0, n = 0, k = 0, groups = 0, gsize = 0, lanes = 0;
  std::uint32_t bits = 4;
  // Slices of K, each written to its own plane of a [slices, ...] output.
  std::uint32_t slices = 1;
  bool valid = false;
};

KernelShapes original_shapes(const KernelShapes& s) {
  KernelShapes original = s;
  original.inputs = s.inputs.first(std::min<std::size_t>(4, s.inputs.size()));
  original.input_dtypes =
      s.input_dtypes.first(std::min<std::size_t>(4, s.input_dtypes.size()));
  return original;
}

Dims dims_of(const KernelShapes& s, bool sliced) {
  Dims d;
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.input_dtypes[4] != DType::kF16)
    return d;
  KernelShapes original = original_shapes(s);
  if (sliced) {
    // [slices, ...output] -> output, for the shape contract.
    if (s.output.rank() < 2 || s.iattrs[2] < 2 ||
        s.output.dim(0) != s.iattrs[2])
      return d;
    Shape plane;
    for (std::size_t i = 1; i < s.output.rank(); ++i) plane.push_back(s.output.dim(i));
    original.output = plane;
    d.slices = static_cast<std::uint32_t>(s.iattrs[2]);
  }
  if (!dispatch::q4_gemm_shape(original)) return d;
  d.k = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  d.n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  d.lanes = static_cast<std::uint32_t>(s.inputs[1].dim(1));
  d.gsize = static_cast<std::uint32_t>(s.iattrs[1]);
  d.bits = static_cast<std::uint32_t>(s.iattrs[0]);
  d.groups = d.k / d.gsize;
  d.m = static_cast<std::uint32_t>(original.output.elem_count() / d.n);
  if ((d.k / kBK) % d.slices != 0 ||
      s.inputs[4] != Shape{static_cast<std::int64_t>(d.m),
                           static_cast<std::int64_t>(d.k)})
    return d;
  d.valid = true;
  return d;
}

std::uint32_t burst_steps(const Dims& d, const struct Tile& t);
// The part's tile (its tuning header picks the wide-pass one).
Tile tile_for(const KernelShapes& s, const Dims& d) {
  const auto t = dispatch::q4_gemm_tile(
      d.m, dispatch::arch::tuning(s.device ? std::string_view(s.device->arch)
                                           : std::string_view{}));
  return {t.bm, t.bn, t.wm, t.wn};
}

template <class S>
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::In<lse::f16, env::Emit> panel;
  env::Out<kir::f32, env::Emit> out;
};

const math::MatrixCoreRow* f16_row(const KernelShapes& s) {
  if (!s.device || !s.intrinsics) return nullptr;
  const auto target = matrix_target(*s.device);
  if (!target) return nullptr;
  const auto caps = device_matrix_caps(*s.device);
  for (const auto& row : math::matrix_core_table()) {
    if (row.target == *target && row.wave == s.device->wavefront_size &&
        row.wave == 32 && row.acc == math::MatrixElem::kF32 &&
        row.operand == math::MatrixElem::kF16 && row.m == 16 && row.n == 16 &&
        row.k_step == 16 && row.chained == 1 && row.emittable() &&
        math::has_cap(caps, row.cap) && !s.intrinsics->find(row.key).empty())
      return &row;
  }
  return nullptr;
}

// Steps per K-loop iteration. A short tile does little matrix work per
// step, so the loop is bound by memory latency and wants several steps'
// reads in flight; a tall tile's step bodies are large enough that the
// target cannot compile more than one per iteration. gfx1201, N 17408 x
// K 5120: M 9 300 -> 179 us at four steps, M 64 324 -> 263 us at two.
std::uint32_t burst_steps(const Dims& d, const Tile& t) {
  // The burst is capped by the matrix instructions it puts in one block:
  // 32 for most tiles, which the target's operand staging once imposed, and
  // 48 for tiles of 48-row waves, which then take two steps (M 144 gate:
  // 0.48 -> 0.44 ms against one). Four steps of those (96 instructions)
  // compile to wrong code at some widths.
  const std::uint32_t per_step = (t.bm / t.wm / 16u) * (t.bn / t.wn / 16u) * (kBK / 16u);
  const std::uint32_t steps = d.k / kBK / d.slices;
  const std::uint32_t cap =
      dispatch::q4_gemm_wave48({t.bm, t.bn, t.wm, t.wn}) ? 48u : 32u;
  for (std::uint32_t b = 4u; b > 1u; b /= 2u)
    if (b * per_step <= cap && steps % b == 0u) return b;
  return 1u;
}

std::uint32_t lds_bytes(const Tile& t) {
  return (t.bm + t.bn) * kRowHalves * 2u;
}

// What a variant that reads A from the panel keeps of the A tile: a token
// allocation, so the B tile sits where it always does.
constexpr std::uint32_t kDirectAHalves = 8u;

// Two B buffers take the K loop two steps at a time, one step per iteration.
bool b_double_fits(const Dims& d, const Tile& t) {
  return burst_steps(d, t) <= 1u && (d.k / kBK / d.slices) % 2u == 0u;
}

std::uint32_t variant_lds_bytes(const Tile& t, std::uint32_t variant) {
  if (variant == 0u) return lds_bytes(t);
  return (kDirectAHalves + t.bn * kRowHalves * (variant == 2u ? 2u : 1u)) * 2u;
}

bool device_fits(const KernelShapes& s, const Dims& d) {
  if (!d.valid || !s.device || !s.intrinsics) return false;
  const Tile t = tile_for(s, d);
  // Every thread stages the same number of activation pieces.
  if (s.device->wavefront_size != 32u ||
      s.device->max_threads_per_workgroup < t.wm * t.wn * 32u ||
      t.bm * (kBK / kRun) % (t.wm * t.wn * 32u) != 0u)
    return false;
  if (device_load_bytes(s.device) < 16u) return false;
  for (const auto symbol : {"barrier", d.bits == 8 ? "q8x8.f16" : "q4x8.f16"})
    if (s.intrinsics->find(symbol).empty()) return false;
  return backend::workgroup_lds_bytes(s.device) >= lds_bytes(tile_for(s, d)) &&
         f16_row(s) != nullptr;
}

template <class S, math::MatrixTarget G>
std::string emit_body(const KernelShapes& s, const Dims& d) {
  using Op = math::op::Mma<G, math::MatrixElem::kF32, math::MatrixElem::kF16,
                           16, 16, 16>;
  constexpr math::MatrixCoreRow kRow = Op::kRow;
  constexpr TileGeometry kGeo = geometry_of(kRow);
  constexpr int kFrag = kRow.a_len;
  constexpr int kSlots = kRow.c_len;
  // A lane's operand slice in halves; it is loaded in whole aligned runs.
  constexpr std::uint32_t kLaneK = kGeo.lane_k;
  static_assert(kLaneK % kRun == 0u, "fragments load in whole runs");

  const Tile t = tile_for(s, d);
  const std::uint32_t M = d.m, N = d.n, K = d.k;
  const std::uint32_t KT = K / kBK;
  const std::uint32_t tiles_n = (N + t.bn - 1u) / t.bn;
  const std::uint32_t TM = t.bm / t.wm, TN = t.bn / t.wn;
  const std::uint32_t FM = TM / 16u, FN = TN / 16u;

  kir::KernelBody kb(s.types, *s.intrinsics,
                     backend::workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  Args<S> a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  // Variants (Q4GemmKernel::variants): 1 reads each A fragment straight
  // from the panel instead of through workgroup memory; 2 does that and
  // double-buffers B, so a K step waits at one barrier instead of two. Each
  // feeds the matrix instructions the same halves in the same order.
  const bool a_direct = kFrag == 16 && s.variant >= 1u;
  const bool b_double = a_direct && s.variant == 2u;
  if (s.variant > 2u || (s.variant != 0u && !a_direct) ||
      (b_double && !b_double_fits(d, t)))
    return {};
  const auto As = e.lds<lse::f16>(a_direct ? kDirectAHalves : t.bm * kRowHalves);
  const auto Bs0 = e.lds<lse::f16>(t.bn * kRowHalves);
  const auto Bs1 = b_double ? e.lds<lse::f16>(t.bn * kRowHalves) : Bs0;
  kir::Tile<lse::f16> Bs = Bs0;
  if (!As || !Bs0 || !Bs1) return {};

  const auto lid = e.let(math::local_id());
  const auto wave = e.let(lid / 32u);
  const auto lane = e.let(lid % 32u);
  const auto lane_lo = e.let(lane % 16u);
  const auto lane_hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  // Column tiles fastest: the workgroups resident together share one
  // activation tile, which then stays in cache while the weights stream.
  const std::uint32_t tiles_m = (M + t.bm - 1u) / t.bm;
  const auto tn = e.let(wg % tiles_n);
  const auto tm = e.let((wg / tiles_n) % tiles_m);
  // This workgroup's slice of K, and the output plane it writes.
  const std::uint32_t slice_steps = KT / d.slices;
  const auto ks = e.let(wg / (tiles_n * tiles_m));
  const auto k_begin = e.let(ks * slice_steps);
  const auto k_end = e.let(k_begin + slice_steps);
  const auto plane = e.let(ks * (M * N));
  const auto m0 = e.let(tm * t.bm);
  const auto n0 = e.let(tn * t.bn);
  const auto wave_m = e.let(wave / t.wn);
  const auto wave_n = e.let(wave % t.wn);

  // Activation staging: one aligned run (16 bytes) per piece, kBK/kRun per
  // row, row-major over the tile so consecutive threads read one contiguous
  // row segment of the panel.
  constexpr std::uint32_t kAPerRow = kBK / kRun;
  const std::uint32_t threads = t.wm * t.wn * 32u;
  const std::uint32_t a_pieces = t.bm * kAPerRow / threads;
  std::vector<kir::Val<kir::u32>> a_src, a_dst;
  for (std::uint32_t c = 0; c < a_pieces; ++c) {
    const auto f = e.let(lid + c * threads);
    const auto r = e.let(f / kAPerRow);
    const auto run = e.let(f % kAPerRow);
    const auto grow = e.let(m0 + r);
    const auto safe = e.let(select(grow < M, grow, e.u32(M - 1u)));
    a_src.push_back(e.let(safe * K + run * kRun));
    a_dst.push_back(e.let(r * kRowHalves + run * kRun));
  }
  // Weight staging: four packed words (one 16-byte load) per piece: 32
  // 4-bit codes or 16 8-bit ones.
  const std::uint32_t kCodesPerWord = 32u / d.bits;
  const std::uint32_t kWordsPerStep = kBK / kCodesPerWord;
  const std::uint32_t kBPerCol = kWordsPerStep / 4u;
  // A tile with fewer weight pieces than threads lets the surplus threads
  // repeat a piece: they write the same values to the same place.
  const std::uint32_t b_total = t.bn * kBPerCol;
  const std::uint32_t b_pieces = (b_total + threads - 1u) / threads;
  std::vector<kir::Val<kir::u32>> b_src, b_dst, b_aff;
  for (std::uint32_t c = 0; c < b_pieces; ++c) {
    const auto f = b_total % threads == 0u
                       ? e.let(lid + c * threads)
                       : e.let((lid + c * threads) % b_total);
    const auto col = e.let(f / kBPerCol);
    const auto part = e.let(f % kBPerCol);
    const auto gcol = e.let(n0 + col);
    const auto safe = e.let(select(gcol < N, gcol, e.u32(N - 1u)));
    b_src.push_back(e.let(safe * d.lanes + part * 4u));
    b_dst.push_back(e.let(col * kRowHalves + part * 4u * kCodesPerWord));
    b_aff.push_back(e.let(safe * d.groups));
  }

  using F16x8 = lse::vec<lse::f16, 8>;
  struct Staged {
    std::vector<kir::Pack<lse::f16>> av;
    std::vector<kir::Pack<std::uint32_t>> bv;
    std::vector<kir::Val<S>> scale, bias;  // widened in commit
  };
  const auto issue = [&](const kir::Val<kir::u32>& kt) {
    Staged st;
    const auto koff = e.let(kt * kBK);
    if (!a_direct) {
      for (std::uint32_t c = 0; c < a_pieces; ++c)
        st.av.push_back(e.load(a.panel, e.let(a_src[c] + koff), 16u));
    }
    const auto woff = e.let(kt * kWordsPerStep);
    const auto g = e.let(koff / d.gsize);
    for (std::uint32_t c = 0; c < b_pieces; ++c) {
      st.bv.push_back(e.load(a.packed, e.let(b_src[c] + woff), 16u));
      const auto at = e.let(b_aff[c] + g);
      st.scale.push_back(e.let(a.scales[at]));
      st.bias.push_back(e.let(a.biases[at]));
    }
    return st;
  };
  const auto store8 = [&](const kir::Tile<lse::f16>& tile,
                          const kir::Val<kir::u32>& at,
                          const kir::Pack<lse::f16>& v) {
    kb.store_pack<lse::f16>(tile.id(), at, v, 16u);
  };
  const auto commit = [&](const Staged& st) {
    for (std::uint32_t c = 0; c < st.av.size(); ++c) store8(As, a_dst[c], st.av[c]);
    for (std::uint32_t c = 0; c < b_pieces; ++c) {
      const auto scale = e.let(math::widen(st.scale[c]));
      const auto bias = e.let(math::widen(st.bias[c]));
      if (d.bits == 8) {
        // Two aligned runs of eight, each from a pair of words.
        for (std::uint32_t r = 0; r < 2u; ++r) {
          const auto lo = e.let(st.bv[c][static_cast<int>(2u * r)]);
          const auto hi = e.let(st.bv[c][static_cast<int>(2u * r + 1u)]);
          const auto v = kb.call<F16x8>("q8x8.f16", lo, hi, scale, bias);
          store8(Bs, e.let(b_dst[c] + r * kRun),
                 kir::Pack<lse::f16>(v.types(), v.body(), v.id(), 8));
        }
        continue;
      }
      const auto decode = [&] {
        for (std::uint32_t w = 0; w < 4u; ++w) {
          const auto word = e.let(st.bv[c][static_cast<int>(w)]);
          const auto v =
              kb.call<F16x8>("q4x8.f16", word, scale, bias);
          store8(Bs, e.let(b_dst[c] + w * kCodesPerWord),
                 kir::Pack<lse::f16>(v.types(), v.body(), v.id(), 8));
        }
      };
      // A tile with fewer weight pieces than threads: only each piece's
      // first thread decodes it; the repeats would write the same values
      // and spend the step's VALU on it. Taken on the gfx11 generation,
      // where it was measured (gfx1151, test_q4_gemm --gpu, 256 x 128 on
      // 4 x 4 waves: M1024 N17408 K5120 5.48 -> 5.28 ms, N5120 K17408
      // 6.05 -> 5.49 ms); RDNA4 keeps its form pending its own measurement.
      if (kFrag == 16 && b_total % threads != 0u && c + 1u == b_pieces) {
        if (auto first = e.when(lid + c * threads < b_total)) decode();
      } else {
        decode();
      }
    }
  };

  std::vector<kir::Local<lse::f32, kSlots>> acc;
  acc.reserve(FM * FN);
  for (std::uint32_t i = 0; i < FM * FN; ++i) {
    acc.push_back(e.local<lse::f32, kSlots>());
    for (int z = 0; z < kSlots; ++z) acc[i][z] = e.f32(0.0f);
  }

  // This lane's k origin within one instruction's step: the upper half-wave
  // reads the second half on a split-K layout, nothing on a contiguous one.
  const auto lane_k0 = e.let(lane_hi * (kGeo.split_k ? kLaneK : 0u));
  std::vector<kir::Val<kir::u32>> a_frag_base, b_frag_base;
  for (std::uint32_t fm = 0; fm < FM; ++fm)
    a_frag_base.push_back(
        e.let((wave_m * TM + fm * 16u + lane_lo) * kRowHalves + lane_k0));
  for (std::uint32_t fn = 0; fn < FN; ++fn)
    b_frag_base.push_back(
        e.let((wave_n * TN + fn * 16u + lane_lo) * kRowHalves + lane_k0));

  using FragVec = lse::vec<lse::f16, kFrag>;
  const auto fragment = [&](const kir::Tile<lse::f16>& tile,
                            const kir::Val<kir::u32>& at) -> kir::Val<FragVec> {
    // One vector of the whole fragment, which the target splits into its
    // 16-byte loads straight into the operand registers. A gfx11 fragment is
    // two loads wide; assembling it a half at a time from two 8-half loads
    // cost ~1500 VALU bit moves per K step per wave on gfx1151, more than
    // the step's 32 matrix instructions (test_q4_gemm --gpu, M1024 N17408
    // K5120: 10.16 -> 5.89 ms, 18.0 -> 31.0 TFLOPS).
    const auto p = kFrag == 8 ? tile.load(at, 16u)
                              : tile.load_elems(at, static_cast<std::uint32_t>(kFrag));
    return kir::Val<FragVec>(&kb.types(), &kb.ir(), p.id());
  };

  std::vector<kir::Val<kir::u32>> a_glob_base;
  if (a_direct)
    for (std::uint32_t fm = 0; fm < FM; ++fm) {
      const auto grow = e.let(m0 + wave_m * TM + fm * 16u + lane_lo);
      a_glob_base.push_back(e.let(select(grow < M, grow, e.u32(M - 1u)) * K));
    }
  std::optional<kir::Val<kir::u32>> cur_k;
  const auto compute = [&]() {
    for (std::uint32_t kk = 0; kk < kBK / 16u; ++kk) {
      std::vector<kir::Val<FragVec>> af, bf;
      for (std::uint32_t fm = 0; fm < FM; ++fm) {
        if (a_direct) {
          const auto p = kb.load_elems<lse::f16>(
              a.panel.b.id(), e.let(a_glob_base[fm] + *cur_k * kBK + kk * 16u), 16u);
          af.push_back(kir::Val<FragVec>(&kb.types(), &kb.ir(), p.id()));
        } else {
          af.push_back(fragment(As, e.let(a_frag_base[fm] + kk * 16u)));
        }
      }
      for (std::uint32_t fn = 0; fn < FN; ++fn)
        bf.push_back(fragment(Bs, e.let(b_frag_base[fn] + kk * 16u)));
      for (std::uint32_t fm = 0; fm < FM; ++fm)
        for (std::uint32_t fn = 0; fn < FN; ++fn) {
          auto& c = acc[fm * FN + fn];
          c = math::mma<Op>(af[fm], bf[fn], c.value());
        }
    }
  };
  const std::uint32_t burst = burst_steps(d, t);
  if (b_double) {
    // B in two buffers, A straight from the panel: one barrier a step, the
    // next step's B written while this one's is read.
    Bs = Bs0;
    commit(issue(k_begin));
    e.barrier();
    for (auto kt : e.range(k_begin, k_end, 2u)) {
      {
        const Staged st = issue(e.let(kt + 1u));
        Bs = Bs0; cur_k = kt; compute();
        Bs = Bs1; commit(st);
        e.barrier();
      }
      {
        const auto next = e.let(select(kt + 2u < k_end, kt + 2u, kt + 1u));
        const Staged st = issue(next);
        Bs = Bs1; cur_k = e.let(kt + 1u); compute();
        Bs = Bs0; commit(st);
        e.barrier();
      }
    }
  } else if (burst <= 1u) {
    commit(issue(k_begin));
    e.barrier();
    for (auto kt : e.range(k_begin, k_end, 1u)) {
      const auto next = e.let(select(kt + 1u < k_end, kt + 1u, kt));
      const Staged st = issue(next);
      if (a_direct) cur_k = kt;
      compute();
      e.barrier();
      commit(st);
      e.barrier();
    }
  } else {
    // A burst of steps per iteration, every step's global reads issued at
    // its top: the target waits for outstanding reads at each loop header,
    // so reads cannot be carried across iterations, and one step ahead hides
    // only one step's matrix work -- next to nothing at a few rows, where
    // the loop then waits a full memory latency per step.
    for (auto kt : e.range(k_begin, k_end, burst)) {
      std::vector<Staged> sts;
      for (std::uint32_t u = 0; u < burst; ++u) sts.push_back(issue(e.let(kt + u)));
      for (std::uint32_t u = 0; u < burst; ++u) {
        commit(sts[u]);
        e.barrier();
        if (a_direct) cur_k = e.let(kt + u);
        compute();
        e.barrier();
      }
    }
  }

  for (std::uint32_t fm = 0; fm < FM; ++fm) {
    for (std::uint32_t fn = 0; fn < FN; ++fn) {
      const auto col = e.let(n0 + wave_n * TN + fn * 16u + lane_lo);
      for (int z = 0; z < kSlots; ++z) {
        const auto row = e.let(m0 + wave_m * TM + fm * 16u +
                               static_cast<std::uint32_t>(z) * kGeo.slot_step +
                               lane_hi * kGeo.half_rows);
        // A guard costs a region per store, and the region exit a store
        // wait: emit one only for the extents a tile can overrun.
        const auto value = acc[fm * FN + fn][z].read();
        if (M % t.bm == 0 && N % t.bn == 0) {
          e.store(plane + row * N + col, value);
        } else if (N % t.bn == 0) {
          if (auto in = e.when(row < M)) e.store(plane + row * N + col, value);
        } else {
          if (auto in = e.when(row < M && col < N))
            e.store(plane + row * N + col, value);
        }
      }
    }
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}

// The original contraction on the first four operands, for a device the GEMM
// does not fit. The panel is then simply not read.
const graph::KernelPrimitiveBase* legacy(const KernelShapes& original) {
  const auto* base = dynamic_cast<const graph::KernelPrimitiveBase*>(
      graph::find_primitive("quant_linear"));
  return base ? base->specialize(original) : nullptr;
}

// The plain form writes the product; the sliced form writes one partial
// product per slice of K into a leading [slices] axis, which
// Q4SliceSumKernel then sums.
template <bool Sliced>
struct Q4GemmKernel final : graph::KernelPrimitive<Q4GemmKernel<Sliced>> {
  static constexpr std::string_view kName =
      Sliced ? kGemmSlicesName : kGemmName;
  static constexpr std::string_view kEntry =
      Sliced ? "lse_quant_linear_q4_gemm_f16_slices_v1"
             : "lse_quant_linear_q4_gemm_f16_v1";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  // Partial products take no epilogue; the sum does.
  bool supports_epilogue() const noexcept override { return !Sliced; }
  bool has_typed_host_impl() const noexcept override { return true; }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (Sliced)
      return LSE_ERROR(kInvalidArgument,
                       "a sliced q4 gemm's shape carries its slice count; it "
                       "is set when the node is built");
    if (in.size() != 5 || !in[0].rank() || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument,
                       "q4 gemm takes x, packed[N, lanes], scales, biases, "
                       "f16 panel");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  // The same function as quant_linear on the first four operands, computed
  // from the f32 activation; the panel is a device staging artifact. A
  // sliced output carries the whole product in slice 0 and zeros elsewhere,
  // which sums to the same thing.
  Status eval_cpu_typed(std::span<const graph::HostTensorView> in,
                        graph::HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>& iattrs) const override {
    if (in.size() != 5 || in[0].dtype != DType::kF32 ||
        in[1].dtype != DType::kU32 || out.dtype != DType::kF32 ||
        (in[2].dtype != DType::kBF16 && in[2].dtype != DType::kF16) ||
        in[3].dtype != in[2].dtype || (iattrs[0] != 4 && iattrs[0] != 8) ||
        iattrs[1] <= 0)
      return LSE_ERROR(kInvalidArgument, "invalid q4 gemm storage");
    const auto k = static_cast<std::size_t>(in[0].shape.dim(in[0].shape.rank() - 1));
    const auto n = static_cast<std::size_t>(in[1].shape.dim(0));
    const auto gs = static_cast<std::size_t>(iattrs[1]);
    const auto m = in[0].shape.elem_count() / k;
    const std::size_t slices = Sliced ? static_cast<std::size_t>(iattrs[2]) : 1;
    const auto bits = static_cast<std::size_t>(iattrs[0]);
    const auto per_word = 32 / bits;
    if (k % gs != 0 || slices == 0 || in[1].bytes.size() != n * k * bits / 8 ||
        in[2].bytes.size() != n * (k / gs) * 2 ||
        out.bytes.size() != slices * m * n * 4)
      return LSE_ERROR(kInvalidArgument, "invalid q4 gemm byte extent");
    std::memset(out.bytes.data(), 0, out.bytes.size());
    const auto affine = [&](const graph::HostTensorView& v, std::size_t at) {
      if (v.dtype == DType::kBF16) {
        bfloat16_t h;
        std::memcpy(&h, v.bytes.data() + at * 2, 2);
        return static_cast<float>(h);
      }
      float16_t h;
      std::memcpy(&h, v.bytes.data() + at * 2, 2);
      return static_cast<float>(h);
    };
    for (std::size_t col = 0; col < n; ++col) {
      for (std::size_t row = 0; row < m; ++row) {
        float acc = 0.0f;
        for (std::size_t kk = 0; kk < k; ++kk) {
          std::uint32_t word;
          std::memcpy(&word, in[1].bytes.data() + (col * (k / per_word) + kk / per_word) * 4, 4);
          const float code = static_cast<float>(
              (word >> (bits * (kk % per_word))) & ((1u << bits) - 1u));
          const auto g = col * (k / gs) + kk / gs;
          float x;
          std::memcpy(&x, in[0].bytes.data() + (row * k + kk) * 4, 4);
          acc = std::fma(x, std::fma(affine(in[2], g), code, affine(in[3], g)), acc);
        }
        std::memcpy(out.bytes.data() + (row * n + col) * 4, &acc, 4);
      }
    }
    return OkStatus();
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const Dims d = dims_of(s, Sliced);
    if (!d.valid) return {};
    if (!device_fits(s, d)) {
      // The graph slices K only for a device that runs the GEMM.
      if (Sliced) return {};
      const KernelShapes original = original_shapes(s);
      const auto* kernel = legacy(original);
      return kernel ? kernel->emit_kernel(original) : std::string{};
    }
    if (!s.store || s.types.scalar == nullptr) return {};
    const auto* row = f16_row(s);
    return with_matrix_target<std::string>(
        row->target, [&]<math::MatrixTarget G>() -> std::string {
          if constexpr (!math::has_matrix_core_row(G, math::MatrixElem::kF32,
                                                   math::MatrixElem::kF16, 16,
                                                   16, 16)) {
            return {};
          } else if constexpr (math::matrix_core_row(
                                   G, math::MatrixElem::kF32,
                                   math::MatrixElem::kF16, 16, 16, 16)
                                   .wave != 32) {
            return {};
          } else {
            return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
              return emit_body<S, G>(s, d);
            });
          }
        });
  }

  // A from workgroup memory (0), A from the panel (1), and that with B
  // double-buffered (2), on a part whose fragments are whole 16-half loads.
  std::uint32_t variants(const KernelShapes& s) const override {
    const Dims d = dims_of(s, Sliced);
    if (!d.valid || !device_fits(s, d)) return 1;
    const auto* row = f16_row(s);
    if (row == nullptr || row->a_len != 16) return 1;
    const Tile t = tile_for(s, d);
    const bool pairs = b_double_fits(d, t) &&
        backend::workgroup_lds_bytes(s.device) >= variant_lds_bytes(t, 2u);
    return pairs ? 3u : 2u;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const Dims d = dims_of(s, Sliced);
    if (!d.valid) return tp;
    if (!device_fits(s, d)) {
      if (Sliced) return tp;
      const KernelShapes original = original_shapes(s);
      const auto* kernel = legacy(original);
      return kernel ? kernel->plan(original) : tp;
    }
    const Tile t = tile_for(s, d);
    tp.workgroup_size[0] = t.wm * t.wn * 32u;
    tp.workgroup_count[0] =
        ((d.m + t.bm - 1u) / t.bm) * ((d.n + t.bn - 1u) / t.bn) * d.slices;
    tp.lds_bytes = variant_lds_bytes(t, s.variant);
    return tp;
  }
};
using Q4GemmPlain = Q4GemmKernel<false>;
using Q4GemmSlices = Q4GemmKernel<true>;

// Sums a [slices, ...] partial product over its leading axis. Four outputs
// per thread, read as one 16-byte load per slice; trailing elementwise work
// fuses into its stores.
struct SliceSumArgs {
  env::In<kir::f32, env::Emit> partial;
  env::Out<kir::f32, env::Emit> out;
};
struct Q4SliceSumKernel final : graph::KernelPrimitive<Q4SliceSumKernel> {
  static constexpr std::string_view kName = kSliceSumName;
  static constexpr std::string_view kEntry = "lse_quant_linear_q4_gemm_f16_slice_sum_v1";
  static constexpr std::string_view kSource = {};
  static constexpr std::uint32_t kLanes = 4;

  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool has_typed_host_impl() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() < 2)
      return LSE_ERROR(kInvalidArgument, "slice sum takes [slices, ...]");
    Shape out;
    for (std::size_t i = 1; i < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    return out;
  }
  static bool supported(const KernelShapes& s) {
    if (s.inputs.size() != 1 || s.input_dtypes.size() != 1 ||
        s.input_dtypes[0] != DType::kF32 || s.output_dtype != DType::kF32 ||
        s.inputs[0].rank() < 2 || !s.device || !s.intrinsics ||
        device_load_bytes(s.device) < 16u)
      return false;
    const auto count = s.output.elem_count();
    return count != 0 && count % kLanes == 0 &&
           s.inputs[0].elem_count() ==
               static_cast<std::uint64_t>(s.inputs[0].dim(0)) * count &&
           s.inputs[0].elem_count() <= UINT32_MAX;
  }
  Status eval_cpu_typed(std::span<const graph::HostTensorView> in,
                        graph::HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 || out.dtype != DType::kF32 ||
        in[0].shape.rank() < 2)
      return LSE_ERROR(kInvalidArgument, "invalid slice sum storage");
    const std::size_t slices = static_cast<std::size_t>(in[0].shape.dim(0));
    const std::size_t count = out.bytes.size() / 4;
    if (in[0].bytes.size() != slices * count * 4)
      return LSE_ERROR(kInvalidArgument, "invalid slice sum byte extent");
    for (std::size_t i = 0; i < count; ++i) {
      float acc = 0.0f;
      for (std::size_t sl = 0; sl < slices; ++sl) {
        float v;
        std::memcpy(&v, in[0].bytes.data() + (sl * count + i) * 4, 4);
        acc += v;
      }
      std::memcpy(out.bytes.data() + i * 4, &acc, 4);
    }
    return OkStatus();
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!supported(s) || !s.store || s.types.scalar == nullptr) return {};
    const auto count = static_cast<std::uint32_t>(s.output.elem_count());
    const auto slices = static_cast<std::uint32_t>(s.inputs[0].dim(0));
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    kb.set_store(s.store);
    SliceSumArgs a;
    if (!env::bind(kb, a, s)) return {};
    env::Emit e{&kb};
    const auto piece = e.let(math::workgroup_id_x() * kThreads + math::local_id());
    if (auto live = e.when(piece < count / kLanes)) {
      const auto at = e.let(piece * kLanes);
      std::vector<kir::Pack<kir::f32>> parts;
      for (std::uint32_t sl = 0; sl < slices; ++sl)
        parts.push_back(e.load(a.partial, e.let(at + sl * count), 16u));
      for (std::uint32_t j = 0; j < kLanes; ++j) {
        auto sum = e.let(parts[0][static_cast<int>(j)]);
        for (std::uint32_t sl = 1; sl < slices; ++sl)
          sum = e.let(sum + parts[sl][static_cast<int>(j)]);
        e.store(e.let(at + j), sum);
      }
    }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    if (!supported(s)) return tp;
    const auto pieces = static_cast<std::uint32_t>(s.output.elem_count()) / kLanes;
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = (pieces + kThreads - 1u) / kThreads;
    return tp;
  }
};

}  // namespace

LSE_REGISTER_PRIMITIVE(F16PanelKernel);
LSE_REGISTER_PRIMITIVE(Q4GemmPlain);
LSE_REGISTER_PRIMITIVE(Q4GemmSlices);
LSE_REGISTER_PRIMITIVE(Q4SliceSumKernel);

}  // namespace lse::kernels
