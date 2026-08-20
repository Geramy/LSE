// The group-affine contraction on the matrix core.
//
// The scalar path spends most of its vector instructions unpacking nibbles:
// measured on a 27B projection, 98 shifts and masks feed 48 dot products. One
// wmma.i32.16x16x16.iu8 replaces sixteen of those dots, so the unpack is paid
// once per fragment instead of once per dot.
//
// The algebra is the one the scalar path already uses. A weight is
// `code * scale + bias` with scale and bias per group of K, so
//
//   sum_k x_k (c_k s + b) = s * sum_k x_k c_k + b * sum_k x_k
//
// and the matrix core computes the left sum in integers. The activation is
// quantized per WMMA K slice rather than per group: the step has to be
// constant across the sixteen values one instruction consumes or it cannot be
// factored out of the integer accumulator at all.
#include <array>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::kernels {

namespace env = graph::env;
namespace kir = graph::kir;
namespace math = lse::math;
namespace quant = lse::quant;

using graph::KernelShapes;
using graph::ThreadPlan;

namespace {

// One wave owns one 16x16 output tile.
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
constexpr std::uint32_t kBlock = 256;

using MmaRdna3 = math::op::Mma<math::MatrixTarget::kRdna3, math::MatrixElem::kI32,
                               math::MatrixElem::kI8, kTileM, kTileN, kTileK>;

// 127 and not 128 so -x and x quantize to the same magnitude, which is what
// keeps the bias term unbiased. The floor stops an all-zero slice dividing.
constexpr float kAmaxFloor = 1e-30f;

struct Dims {
  quant::GroupAffine spec{};
  std::int64_t m = 0, n = 0, k = 0, lanes = 0, groups = 0;
  bool valid = false;
};

Dims dims_of(const KernelShapes& s) {
  Dims d;
  if (s.inputs.size() < 4) return d;
  d.spec.bits = s.iattrs[0];
  d.spec.group_size = s.iattrs[1];
  if (d.spec.bits != 4 && d.spec.bits != 8) return d;
  if (d.spec.group_size <= 0) return d;
  const Shape& w = s.inputs[1];
  if (w.rank() != 2) return d;
  d.n = w.dim(0);
  d.lanes = w.dim(1);
  d.k = d.lanes * 32 / d.spec.bits;
  if (d.k % d.spec.group_size != 0) return d;
  // The K slice one instruction consumes has to sit inside one group, or the
  // group's scale is not constant across it.
  if (d.spec.group_size % kTileK != 0) return d;
  d.groups = d.k / d.spec.group_size;
  const std::int64_t elems = s.output.elem_count();
  if (d.n == 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

template <class S>
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};


// The A fragment for one lane: sixteen activations of one row, quantized to
// int8 and packed four to a word. `step` comes back because the accumulator
// element that needs it lives in another lane and has to read it from scratch.
template <class A>
void fill_acts(env::Emit& e, const A& a, const kir::Val<kir::u32>& row,
               const kir::Val<kir::u32>& k0, std::uint32_t k,
               const kir::Local<kir::u32, 4>& frag,
               const kir::LValue<kir::f32>& step,
               const kir::LValue<kir::f32>& sum, std::uint32_t load_bytes) {
  const std::uint32_t wide = row_pack(kTileK, load_bytes, 4);
  std::array<kir::Val<kir::f32>, kTileK> v;
  const auto base = e.let(row * k + k0);
  for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(kTileK); j += wide) {
    const auto pack = e.load(a.x, e.let(base + j), load_bytes);
    for (std::uint32_t u = 0; u < wide; ++u) v[j + u] = e.let(pack[static_cast<int>(u)]);
  }
  auto amax = e.let(math::abs(v[0]));
  auto total = v[0];
  for (int j = 1; j < kTileK; ++j) {
    amax = e.let(math::max(amax, math::abs(v[j])));
    total = e.let(total + v[j]);
  }
  step = amax * (1.0f / 127.0f);
  sum = total;
  const auto inv = e.let(127.0f / math::max(amax, e.f32(kAmaxFloor)));
  for (int w = 0; w < 4; ++w) {
    auto word = e.let(e.u32(0));
    for (int b = 0; b < 4; ++b) {
      const auto code = e.let(math::rint(v[w * 4 + b] * inv));
      const auto byte = e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
      word = e.let(word + byte * (1u << (8 * b)));
    }
    frag[w] = word;
  }
}

// The weight fragment comes from the operand layer, which owns the stored
// width. This kernel names a column and a K slice and nothing about nibbles.
template <class A>
void fill_weights(env::Emit& e, const A& a, const kir::Val<kir::u32>& col,
                  const kir::Val<kir::u32>& k0, std::uint32_t lanes, int bits,
                  const kir::Local<kir::u32, 4>& frag) {
  const std::uint32_t per_word = 32u / static_cast<std::uint32_t>(bits);
  const auto base = e.let(col * lanes + k0 / per_word);
  for (std::uint32_t f = 0; f < 4u; ++f) {
    frag[f] = bits == 4 ? PackedCodes<4>::word(e, a.packed, base, f)
                        : PackedCodes<8>::word(e, a.packed, base, f);
  }
}

template <class A>
std::string emit_body(const KernelShapes& s, const Dims& d) {
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto gsize = static_cast<std::uint32_t>(d.spec.group_size);
  const std::uint32_t slices = gsize / kTileK;   // instructions per group
  const std::uint32_t waves = kBlock / 32u;
  const std::uint32_t tiles_n = (n + kTileN - 1u) / kTileN;
  const std::uint32_t tiles = ((m + kTileM - 1u) / kTileM) * tiles_n;
  const std::uint32_t load_bytes = device_load_bytes(s.device);

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  // The activation step and row sum are produced by the lane that owns that
  // row of the A fragment and consumed by the lanes that own that row of the
  // accumulator, which are different lanes. Scratch, not a shuffle: the
  // mapping is z-dependent and a shuffle would need one per accumulator slot.
  const auto steps = e.lds<kir::f32>(waves * kTileM * 4u);
  const auto sums = e.lds<kir::f32>(waves * kTileM);

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / 32u);
  const auto lane = e.let(lid % 32u);
  const auto lane_lo = e.let(lane % static_cast<std::uint32_t>(kTileN));
  const auto lane_hi = e.let(lane / static_cast<std::uint32_t>(kTileN));
  const auto tile0 = e.let(math::workgroup_id_x() * waves + wave_id);

  // Not guarded on the tile: the barriers below order scratch that lanes of
  // one wave write and lanes of another read, and `tile0 < tiles` is false for
  // some waves of the last workgroup and true for others. A barrier inside
  // that guard is taken by part of the workgroup, which is undefined. The
  // guard moves down to the accesses that actually need it.
  const auto live = e.let(tile0 < tiles);
  {
    const auto m0 = e.let((tile0 / tiles_n) * static_cast<std::uint32_t>(kTileM));
    const auto n0 = e.let((tile0 % tiles_n) * static_cast<std::uint32_t>(kTileN));
    const auto arow = e.let(m0 + lane_lo);
    const auto bcol = e.let(n0 + lane_lo);
    const auto sbase = e.let(wave_id * static_cast<std::uint32_t>(kTileM));

    std::vector<kir::LValue<kir::f32>> out;
    out.reserve(8);
    for (int z = 0; z < 8; ++z) out.push_back(e.var(0.0f));

    for (auto g : e.range(0u, groups, 1u)) {
      // One integer accumulator per K slice: each carries its own activation
      // step, so they cannot be summed before that step is applied.
      std::vector<kir::Local<kir::i32, 8>> acc;
      acc.reserve(slices);
      for (std::uint32_t t = 0; t < slices; ++t) {
        acc.push_back(e.local<kir::i32, 8>());
        for (auto z : e.unroll(8u)) acc[t][z] = kir::cast<kir::i32>(e.u32(0));
      }

      // The bias term is bias_g times the sum of the group's activations, and
      // a group spans every slice. Writing one slice's sum was three quarters
      // of the term missing.
      auto rowsum = e.var(0.0f);
      for (std::uint32_t t = 0; t < slices; ++t) {
        const auto k0 = e.let(g * gsize + t * static_cast<std::uint32_t>(kTileK));
        const auto af = e.local<kir::u32, 4>();
        const auto bf = e.local<kir::u32, 4>();
        auto st = e.var(0.0f);
        auto sm = e.var(0.0f);
        if (auto g = e.when(live && arow < m)) {
          fill_acts(e, a, arow, k0, k, af, st, sm, load_bytes);
        }
        steps[e.let(sbase * 4u + lane_lo * 4u + t)] = st.read();
        rowsum = rowsum.read() + sm.read();
        if (auto g = e.when(live && bcol < n)) {
          fill_weights(e, a, bcol, k0, lanes, d.spec.bits, bf);
        }
        // The row is chained: one issue consumes a single register, four
        // codes, and it takes four of them to walk the sixteen this slice
        // covers. Handing the whole fragment to one issue drops three
        // quarters of the K it was built from.
        for (std::uint32_t c = 0; c < 4u; ++c) {
          const auto a1 = e.local<kir::u32, 1>();
          const auto b1 = e.local<kir::u32, 1>();
          a1[0] = af[c].read();
          b1[0] = bf[c].read();
          acc[t] = math::mma<MmaRdna3>(a1.value(), b1.value(), acc[t].value());
        }
      }
      sums[e.let(sbase + lane_lo)] = rowsum.read();
      e.barrier();

      // Element z of this lane holds row m0 + z*2 + lane_hi and column
      // n0 + lane_lo, which is the measured mapping the shared tile uses. The
      // weight scale is per column and so uniform down the lane; the
      // activation step is per row and so changes with z.
      const auto gscale = e.let(math::widen(a.scales[e.let(bcol * groups + g)]));
      const auto gbias = e.let(math::widen(a.biases[e.let(bcol * groups + g)]));
      for (std::uint32_t z = 0; z < 8; ++z) {
        auto part = e.var(0.0f);
        for (std::uint32_t t = 0; t < slices; ++t) {
          const auto st = e.let(
              steps[e.let(sbase * 4u + (z * 2u + lane_hi) * 4u + t)].read());
          part = math::fma(st, kir::cast<kir::f32>(acc[t][z].read()),
                           part.read());
        }
        const auto rs = e.let(sums[e.let(sbase + z * 2u + lane_hi)].read());
        // Accumulated, not stored: every group contributes to the same output
        // element and the store happens once, after the last of them.
        out[z] = out[z].read() + gscale * part.read() + gbias * rs;
      }
      e.barrier();
    }

    for (std::uint32_t z = 0; z < 8; ++z) {
      const auto row = e.let(m0 + z * 2u + lane_hi);
      if (auto gr = e.when(live && row < m && bcol < n)) {
        e.store(row * n + bcol, out[z].read());
      }
    }
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}


struct QuantWmmaKernel final : graph::KernelPrimitive<QuantWmmaKernel> {
  static constexpr std::string_view kName = "quant_linear.wmma";
  static constexpr std::string_view kEntry = "lse_quant_linear_wmma";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }

  // The same contract as the scalar form: this is a specialization of it, not
  // a different operation, so it must answer identically.
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2) {
      return LSE_ERROR(kInvalidArgument,
                       "quant_linear takes x, packed[N, lanes], scales, biases");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }

  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const Dims d = dims_of(s);
    if (!d.valid || !s.store || s.types.scalar == nullptr ||
        s.intrinsics == nullptr) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<Args<S>>(s, d);
    });
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    const Dims d = dims_of(s);
    ThreadPlan tp;
    if (!d.valid) return tp;
    const auto n = static_cast<std::uint32_t>(d.n);
    const auto m = static_cast<std::uint32_t>(d.m);
    const std::uint32_t waves = kBlock / 32u;
    const std::uint32_t tiles_n = (n + kTileN - 1u) / kTileN;
    const std::uint32_t tiles = ((m + kTileM - 1u) / kTileM) * tiles_n;
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = (tiles + waves - 1u) / waves;
    tp.workgroup_count[1] = 1;
    tp.workgroup_count[2] = 1;
    tp.lds_bytes = waves * kTileM * 5u * kir::pack_elem_bytes<kir::f32>();
    return tp;
  }
};

const QuantWmmaKernel kQuantWmma;

}  // namespace

// The matrix-core form of the group-affine contraction, or null when the
// device has no int8 row, the shape has no full tile to fill, or the rows
// disagree on their expert -- a routed row picks its own matrix and there is
// no shared operand for a tile to hold.
const graph::KernelPrimitiveBase* wmma_quant_linear_for(const KernelShapes& s) {
  const Dims d = dims_of(s);
  if (!d.valid || s.device == nullptr || s.intrinsics == nullptr) return nullptr;

  // One row is decode, and decode is bandwidth bound: a 16x16 tile would mask
  // fifteen of its rows to move the same weights. The scalar path keeps it.
  if (d.m < kTileM) return nullptr;
  // Correct but not yet faster: every wave re-reads and re-quantizes the same
  // activation rows for its own column block, so the work is multiplied by the
  // number of column tiles. Off until the staging below removes that.
  if (true) return nullptr;

  const std::optional<math::MatrixTarget> target = matrix_target(*s.device);
  if (!target.has_value() || *target != math::MatrixTarget::kRdna3) return nullptr;
  if (!math::has_cap(device_matrix_caps(*s.device), math::MatrixCap::kWmmaInt8)) {
    return nullptr;
  }
  if (s.intrinsics->find(MmaRdna3::kRow.key).empty()) return nullptr;
  return &kQuantWmma;
}

}  // namespace lse::kernels
