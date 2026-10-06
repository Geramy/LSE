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
constexpr std::uint32_t kWaves = 8;
// One K step. A whole number of them must make one quantization group so a
// column's scale and bias are constant across the step.
constexpr std::uint32_t kBK = dispatch::kQ4GemmStepK;
// Halves of padding per scratch row: a row stride of 144 bytes puts the
// sixteen rows a fragment load touches on sixteen distinct 16-byte bank
// groups.
constexpr std::uint32_t kPad = 8;
constexpr std::uint32_t kRowHalves = kBK + kPad;
constexpr std::uint32_t kCodesPerWord = 8;
// Panel lanes per piece: one 16-byte move, and one aligned run of K.
constexpr std::uint32_t kRun = 8;

constexpr std::string_view kPanelName = "quant_activation.f16_panel.v1";
constexpr std::string_view kGemmName = "quant_linear.q4_gemm_f16.v1";

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
  bool valid = false;
};

KernelShapes original_shapes(const KernelShapes& s) {
  KernelShapes original = s;
  original.inputs = s.inputs.first(std::min<std::size_t>(4, s.inputs.size()));
  original.input_dtypes =
      s.input_dtypes.first(std::min<std::size_t>(4, s.input_dtypes.size()));
  return original;
}

Dims dims_of(const KernelShapes& s) {
  Dims d;
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.input_dtypes[4] != DType::kF16)
    return d;
  const KernelShapes original = original_shapes(s);
  if (!dispatch::q4_gemm_shape(original)) return d;
  d.k = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  d.n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  d.lanes = static_cast<std::uint32_t>(s.inputs[1].dim(1));
  d.gsize = static_cast<std::uint32_t>(s.iattrs[1]);
  d.groups = d.k / d.gsize;
  d.m = static_cast<std::uint32_t>(s.output.elem_count() / d.n);
  if (s.inputs[4] != Shape{static_cast<std::int64_t>(d.m),
                           static_cast<std::int64_t>(d.k)})
    return d;
  d.valid = true;
  return d;
}

// The tile follows the grid: a tall tile when M fills it, a short one when it
// does not, so a short prompt still puts a workgroup on most compute units.
Tile tile_for(const Dims& d) {
  if (d.m <= 64) return {64, 128, 2, 4};
  return {128, 128, 2, 4};
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

std::uint32_t lds_bytes(const Tile& t) {
  return (t.bm + t.bn) * kRowHalves * 2u;
}

bool device_fits(const KernelShapes& s, const Dims& d) {
  if (!d.valid || !s.device || !s.intrinsics) return false;
  if (s.device->wavefront_size * kWaves != kThreads ||
      s.device->max_threads_per_workgroup < kThreads)
    return false;
  if (device_load_bytes(s.device) < 16u) return false;
  for (const auto symbol : {"barrier", "q4x8.f16"})
    if (s.intrinsics->find(symbol).empty()) return false;
  return backend::workgroup_lds_bytes(s.device) >= lds_bytes(tile_for(d)) &&
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

  const Tile t = tile_for(d);
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
  const auto As = e.lds<lse::f16>(t.bm * kRowHalves);
  const auto Bs = e.lds<lse::f16>(t.bn * kRowHalves);
  if (!As || !Bs) return {};

  const auto lid = e.let(math::local_id());
  const auto wave = e.let(lid / 32u);
  const auto lane = e.let(lid % 32u);
  const auto lane_lo = e.let(lane % 16u);
  const auto lane_hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  // Column tiles fastest: the workgroups resident together share one
  // activation tile, which then stays in cache while the weights stream.
  const auto tn = e.let(wg % tiles_n);
  const auto tm = e.let(wg / tiles_n);
  const auto m0 = e.let(tm * t.bm);
  const auto n0 = e.let(tn * t.bn);
  const auto wave_m = e.let(wave / t.wn);
  const auto wave_n = e.let(wave % t.wn);

  // Activation staging: one aligned run (16 bytes) per piece, kBK/kRun per
  // row, row-major over the tile so consecutive threads read one contiguous
  // row segment of the panel.
  constexpr std::uint32_t kAPerRow = kBK / kRun;
  const std::uint32_t a_pieces = t.bm * kAPerRow / kThreads;
  std::vector<kir::Val<kir::u32>> a_src, a_dst;
  for (std::uint32_t c = 0; c < a_pieces; ++c) {
    const auto f = e.let(lid + c * kThreads);
    const auto r = e.let(f / kAPerRow);
    const auto run = e.let(f % kAPerRow);
    const auto grow = e.let(m0 + r);
    const auto safe = e.let(select(grow < M, grow, e.u32(M - 1u)));
    a_src.push_back(e.let(safe * K + run * kRun));
    a_dst.push_back(e.let(r * kRowHalves + run * kRun));
  }
  // Weight staging: four packed words (32 codes, one 16-byte load) per
  // piece.
  constexpr std::uint32_t kWordsPerStep = kBK / kCodesPerWord;
  constexpr std::uint32_t kBPerCol = kWordsPerStep / 4u;
  const std::uint32_t b_pieces = t.bn * kBPerCol / kThreads;
  std::vector<kir::Val<kir::u32>> b_src, b_dst, b_aff;
  for (std::uint32_t c = 0; c < b_pieces; ++c) {
    const auto f = e.let(lid + c * kThreads);
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
    std::vector<kir::Val<kir::f32>> scale, bias;
  };
  const auto issue = [&](const kir::Val<kir::u32>& kt) {
    Staged st;
    const auto koff = e.let(kt * kBK);
    for (std::uint32_t c = 0; c < a_pieces; ++c)
      st.av.push_back(e.load(a.panel, e.let(a_src[c] + koff), 16u));
    const auto woff = e.let(kt * kWordsPerStep);
    const auto g = e.let(koff / d.gsize);
    for (std::uint32_t c = 0; c < b_pieces; ++c) {
      st.bv.push_back(e.load(a.packed, e.let(b_src[c] + woff), 16u));
      const auto at = e.let(b_aff[c] + g);
      st.scale.push_back(e.let(math::widen(a.scales[at])));
      st.bias.push_back(e.let(math::widen(a.biases[at])));
    }
    return st;
  };
  const auto store8 = [&](const kir::Tile<lse::f16>& tile,
                          const kir::Val<kir::u32>& at,
                          const kir::Pack<lse::f16>& v) {
    kb.store_pack<lse::f16>(tile.id(), at, v, 16u);
  };
  const auto commit = [&](const Staged& st) {
    for (std::uint32_t c = 0; c < a_pieces; ++c) store8(As, a_dst[c], st.av[c]);
    for (std::uint32_t c = 0; c < b_pieces; ++c) {
      for (std::uint32_t w = 0; w < 4u; ++w) {
        const auto word = e.let(st.bv[c][static_cast<int>(w)]);
        const auto v =
            kb.call<F16x8>("q4x8.f16", word, st.scale[c], st.bias[c]);
        store8(Bs, e.let(b_dst[c] + w * kCodesPerWord),
               kir::Pack<lse::f16>(v.types(), v.body(), v.id(), 8));
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
    if constexpr (kFrag == 8) {
      const auto p = tile.load(at, 16u);
      return kir::Val<FragVec>(&kb.types(), &kb.ir(), p.id());
    } else {
      const auto f = e.local<lse::f16, kFrag>();
      for (int piece = 0; piece < kFrag / 8; ++piece) {
        const auto p =
            tile.load(e.let(at + static_cast<std::uint32_t>(piece * 8)), 16u);
        for (int j = 0; j < 8; ++j) f[piece * 8 + j] = p[j];
      }
      return f.value();
    }
  };

  const auto compute = [&]() {
    for (std::uint32_t kk = 0; kk < kBK / 16u; ++kk) {
      std::vector<kir::Val<FragVec>> af, bf;
      for (std::uint32_t fm = 0; fm < FM; ++fm)
        af.push_back(fragment(As, e.let(a_frag_base[fm] + kk * 16u)));
      for (std::uint32_t fn = 0; fn < FN; ++fn)
        bf.push_back(fragment(Bs, e.let(b_frag_base[fn] + kk * 16u)));
      for (std::uint32_t fm = 0; fm < FM; ++fm)
        for (std::uint32_t fn = 0; fn < FN; ++fn) {
          auto& c = acc[fm * FN + fn];
          c = math::mma<Op>(af[fm], bf[fn], c.value());
        }
    }
  };
  commit(issue(e.u32(0)));
  e.barrier();
  for (auto kt : e.range(0u, KT, 1u)) {
    const auto next = e.let(select(kt + 1u < KT, kt + 1u, kt));
    const Staged st = issue(next);
    compute();
    e.barrier();
    commit(st);
    e.barrier();
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
          e.store(row * N + col, value);
        } else if (N % t.bn == 0) {
          if (auto in = e.when(row < M)) e.store(row * N + col, value);
        } else {
          if (auto in = e.when(row < M && col < N))
            e.store(row * N + col, value);
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

struct Q4GemmKernel final : graph::KernelPrimitive<Q4GemmKernel> {
  static constexpr std::string_view kName = kGemmName;
  static constexpr std::string_view kEntry = "lse_quant_linear_q4_gemm_f16_v1";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  // A fused epilogue runs once per accumulator element, 64 per lane here;
  // Loom's scheduler interleaves them and exhausts scalar registers on an
  // exact divide (silu). Trailing elementwise work runs in its own launch.
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
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
  // from the f32 activation; the panel is a device staging artifact.
  Status eval_cpu_typed(std::span<const graph::HostTensorView> in,
                        graph::HostOutputView out,
                        const std::array<float, 4>&,
                        const std::array<std::int32_t, 4>& iattrs) const override {
    if (in.size() != 5 || in[0].dtype != DType::kF32 ||
        in[1].dtype != DType::kU32 || out.dtype != DType::kF32 ||
        (in[2].dtype != DType::kBF16 && in[2].dtype != DType::kF16) ||
        in[3].dtype != in[2].dtype || iattrs[0] != 4 || iattrs[1] <= 0)
      return LSE_ERROR(kInvalidArgument, "invalid q4 gemm storage");
    const auto k = static_cast<std::size_t>(in[0].shape.dim(in[0].shape.rank() - 1));
    const auto n = static_cast<std::size_t>(in[1].shape.dim(0));
    const auto gs = static_cast<std::size_t>(iattrs[1]);
    const auto m = in[0].shape.elem_count() / k;
    if (k % gs != 0 || in[1].bytes.size() != n * k / 2 ||
        in[2].bytes.size() != n * (k / gs) * 2 || out.bytes.size() != m * n * 4)
      return LSE_ERROR(kInvalidArgument, "invalid q4 gemm byte extent");
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
          std::memcpy(&word, in[1].bytes.data() + (col * (k / 8) + kk / 8) * 4, 4);
          const float code = static_cast<float>((word >> (4 * (kk % 8))) & 15u);
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
    const Dims d = dims_of(s);
    if (!d.valid) return {};
    if (!device_fits(s, d)) {
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

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const Dims d = dims_of(s);
    if (!d.valid) return tp;
    if (!device_fits(s, d)) {
      const KernelShapes original = original_shapes(s);
      const auto* kernel = legacy(original);
      return kernel ? kernel->plan(original) : tp;
    }
    const Tile t = tile_for(d);
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] =
        ((d.m + t.bm - 1u) / t.bm) * ((d.n + t.bn - 1u) / t.bn);
    tp.lds_bytes = lds_bytes(t);
    return tp;
  }
};

}  // namespace

LSE_REGISTER_PRIMITIVE(F16PanelKernel);
LSE_REGISTER_PRIMITIVE(Q4GemmKernel);

}  // namespace lse::kernels
