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

// Row blocks a workgroup owns. More of them would let one unpacked weight
// fragment feed several row blocks, which is the reuse this tile still wants.
// Measured on gfx1151 it does not pay yet: at 401 prompt tokens 1 block is
// 1.65 s, 2 is 1.65 s and 4 is 1.81 s, because the extra accumulators push the
// kernel to 256 VGPRs and it spills. Raise this only together with whatever
// buys the registers back.
constexpr std::uint32_t kRowBlocks = 1u;
constexpr std::uint32_t kRowsPerGroup = kTileM * kRowBlocks;

using MmaRdna3 = math::op::Mma<math::MatrixTarget::kRdna3, math::MatrixElem::kI32,
                               math::MatrixElem::kSU8, kTileM, kTileN, kTileK>;

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
  // 4-bit only, and not because 8-bit cannot be spelled -- PackedCodes has the
  // arm and the tile runs it correctly. It is what the integer path costs: the
  // activation has to become int8 for the matrix core to take it, and against
  // an 8-bit weight that rounding is several times the weight's own error, so
  // the contraction comes out worse than the scalar float decode it replaced.
  // At 4 bits the weight error already dominates and the activation is free.
  // An 8-bit weight belongs on a float operand class (bf16/f16), which is a
  // different row of the same table, not this one.
  if (d.spec.bits != 4) return d;
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
  const std::uint32_t slices = gsize / kTileK;
  // Groups a staging round covers, which is also how many K slices the body
  // unrolls. Widening it trades barriers for live fragments and the fragments
  // cost more: measured on gfx1151, one group holds the tile at 140 VGPRs with
  // no spill, two reaches 239, and four hits the 256 ceiling and spills 85
  // times, which is 1.88 s against 1.64 s at 401 prompt tokens.
  const std::uint32_t kGroupsPerRound = 1u;
  const std::uint32_t round_groups =
      groups < kGroupsPerRound ? groups : kGroupsPerRound;
  const std::uint32_t round_slices = slices * round_groups;
  const std::uint32_t words = gsize * round_groups / 4u;
  const std::uint32_t waves = kBlock / 32u;
  const std::uint32_t tiles_n = (n + kTileN - 1u) / kTileN;
  const std::uint32_t nblocks = (tiles_n + waves - 1u) / waves;
  const std::uint32_t load_bytes = device_load_bytes(s.device);

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  // A workgroup owns one row block and `waves` column blocks of it, so the
  // sixteen rows are quantized once here and every wave reads them. Doing it
  // per wave repeated the amax, the rounding and the loads once per column
  // tile, which is what made the correct kernel three hundred times slower
  // than the loop it replaces.
  const auto xq = e.lds<kir::u32>(kRowsPerGroup * words);
  const auto xs = e.lds<kir::f32>(kRowsPerGroup * round_slices);
  const auto xsum = e.lds<kir::f32>(kRowsPerGroup * round_slices);

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / 32u);
  const auto lane = e.let(lid % 32u);
  const auto lane_lo = e.let(lane % static_cast<std::uint32_t>(kTileN));
  const auto lane_hi = e.let(lane / static_cast<std::uint32_t>(kTileN));

  const auto wg = e.let(math::workgroup_id_x());
  const auto m0 = e.let((wg / nblocks) * kRowsPerGroup);
  const auto ntile = e.let((wg % nblocks) * waves + wave_id);
  const auto n0 = e.let(ntile * static_cast<std::uint32_t>(kTileN));
  const auto bcol = e.let(n0 + lane_lo);
  const auto live = e.let(ntile < tiles_n);

  std::vector<kir::LValue<kir::f32>> out;
  out.reserve(kRowBlocks * 8u);
  for (std::uint32_t i = 0; i < kRowBlocks * 8u; ++i) out.push_back(e.var(0.0f));

  // Where each row block's accumulator slot reads its step and lands its
  // result. Neither depends on the group, so both are formed once here: left
  // inside the loop they are loop invariant, and cse hoists them above the
  // definitions they are built from, which is the body the verifier rejects.
  // One base per row block, not one per accumulator slot: the slot's own
  // offset is a constant, so folding it in at the use costs an add the
  // scheduler hides, where hoisting all of them costs a live register each and
  // spills the whole tile to scratch.
  std::vector<kir::Val<kir::u32>> slot_step;
  std::vector<kir::Val<kir::u32>> slot_out;
  std::vector<kir::Val<kir::u32>> lane_words;
  slot_step.reserve(kRowBlocks);
  slot_out.reserve(kRowBlocks);
  lane_words.reserve(kRowBlocks);
  for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
    const std::uint32_t rb = i * static_cast<std::uint32_t>(kTileM);
    lane_words.push_back(e.let((lane_lo + rb) * words));
    slot_step.push_back(e.let((rb + lane_hi) * round_slices));
    slot_out.push_back(e.let(m0 + rb + lane_hi));
  }
  const auto col_scales = e.let(bcol * groups);

  // Staging is one thread per (row, slice). There are more of those than there
  // are threads once a workgroup owns several row blocks, so each thread takes
  // a fixed stride of them.
  const std::uint32_t items = kRowsPerGroup * round_slices;
  const std::uint32_t chunks = (items + kBlock - 1u) / kBlock;
  std::vector<kir::Val<kir::u32>> st_t, st_row, st_xbase, st_qbase, st_sbase;
  std::vector<kir::Val<kir::boolean>> st_in;
  for (std::uint32_t c = 0; c < chunks; ++c) {
    const auto si = e.let(lid + c * kBlock);
    const auto sr = e.let(si / round_slices);
    st_t.push_back(e.let(si % round_slices));
    st_row.push_back(e.let(m0 + sr));
    st_xbase.push_back(e.let(st_row[c] * k));
    st_qbase.push_back(e.let(sr * words + st_t[c] * 4u));
    st_sbase.push_back(e.let(sr * round_slices + st_t[c]));
    st_in.push_back(e.let(si < items));
  }

  for (auto rnd : e.range(0u, groups / round_groups, 1u)) {
    for (std::uint32_t c = 0; c < chunks; ++c) {
      if (auto stager = e.when(st_in[c])) {
        // Everything that reads the loaded values lives where they are
        // defined. A value produced inside this guard and used after it is a
        // body the verifier rejects, and the first pass to run is the one that
        // reports it, which is why this looked like a cse bug.
        if (auto in_rows = e.when(st_row[c] < m)) {
          const auto base = e.let(st_xbase[c] +
                                  (rnd * (gsize * round_groups) +
                                   st_t[c] * static_cast<std::uint32_t>(kTileK)));
          const std::uint32_t wide = row_pack(kTileK, load_bytes, 4);
          std::array<kir::Val<kir::f32>, kTileK> v;
          for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(kTileK);
               j += wide) {
            const auto pack = e.load(a.x, e.let(base + j), load_bytes);
            for (std::uint32_t u = 0; u < wide; ++u) {
              v[j + u] = e.let(pack[static_cast<int>(u)]);
            }
          }
          auto amax = e.let(math::abs(v[0]));
          auto total = v[0];
          for (int j = 1; j < kTileK; ++j) {
            amax = e.let(math::max(amax, math::abs(v[j])));
            total = e.let(total + v[j]);
          }
          xs[st_sbase[c]] = amax * (1.0f / 127.0f);
          xsum[st_sbase[c]] = total;
          const auto inv = e.let(127.0f / math::max(amax, e.f32(kAmaxFloor)));
          for (std::uint32_t w = 0; w < 4u; ++w) {
            auto word = e.let(e.u32(0));
            for (std::uint32_t b = 0; b < 4u; ++b) {
              const auto code = e.let(math::rint(v[w * 4u + b] * inv));
              const auto byte =
                  e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
              word = e.let(word + byte * (1u << (8 * b)));
            }
            xq[e.let(st_qbase[c] + w)] = word;
          }
        } else {
          // A row past the end still publishes a step and a sum, or the lanes
          // that read them take whatever the previous round left behind.
          xs[st_sbase[c]] = e.f32(0.0f);
          xsum[st_sbase[c]] = e.f32(0.0f);
          for (std::uint32_t w = 0; w < 4u; ++w) {
            xq[e.let(st_qbase[c] + w)] = e.u32(0);
          }
        }
      }
    }
    e.barrier();

    // One staged round holds several groups, and each keeps its own scales.
    for (std::uint32_t gi = 0; gi < round_groups; ++gi) {
      const auto g = e.let(rnd * round_groups + gi);
      const auto at = e.let(col_scales + g);
      const auto gscale = e.let(math::widen(a.scales[at]));
      const auto gbias = e.let(math::widen(a.biases[at]));
      for (std::uint32_t t = 0; t < slices; ++t) {
        const auto k0 =
            e.let(g * gsize + t * static_cast<std::uint32_t>(kTileK));
        const auto bf = e.local<kir::u32, 4>();
        for (std::uint32_t c = 0; c < 4u; ++c) bf[c] = e.u32(0);
        if (auto gd = e.when(live && bcol < n)) {
          fill_weights(e, a, bcol, k0, lanes, d.spec.bits, bf);
        }
        // The unpacked fragment now feeds every row block before it is
        // discarded, which is the whole point of the blocking.
        const std::uint32_t slot = gi * slices + t;
        for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
          const auto af = e.local<kir::u32, 4>();
          for (std::uint32_t c = 0; c < 4u; ++c) {
            af[c] = xq[e.let(lane_words[i] + (slot * 4u + c))].read();
          }
          auto acc = e.local<kir::i32, 8>();
          for (auto z : e.unroll(8u)) acc[z] = kir::cast<kir::i32>(e.u32(0));
          // One issue takes the whole K slice: the row declares A and B four
          // registers wide and chained=1, so a lane hands over its sixteen
          // contiguous K values at once.
          acc = math::mma<MmaRdna3>(af.value(), bf.value(), acc.value());
          // Drained here rather than carried per slice: the step is per slice,
          // so folding it in now costs eight fma and saves an accumulator for
          // every slice a group holds.
          for (std::uint32_t z = 0; z < 8u; ++z) {
            const auto at_s =
                e.let(slot_step[i] + (z * 2u * round_slices + slot));
            const auto term = e.let(gscale * xs[at_s].read());
            out[i * 8u + z] =
                math::fma(term, kir::cast<kir::f32>(acc[z].read()),
                          out[i * 8u + z].read()) +
                gbias * xsum[at_s].read();
          }
        }
      }
    }
    e.barrier();
  }

  for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
    for (std::uint32_t z = 0; z < 8u; ++z) {
      const auto row = e.let(slot_out[i] + z * 2u);
      if (auto gr = e.when(live && row < m && bcol < n)) {
        e.store(row * n + bcol, out[i * 8u + z].read());
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
    const std::uint32_t nblocks = (tiles_n + waves - 1u) / waves;
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] =
        ((m + kRowsPerGroup - 1u) / kRowsPerGroup) * nblocks;
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

  const std::optional<math::MatrixTarget> target = matrix_target(*s.device);
  if (!target.has_value() || *target != math::MatrixTarget::kRdna3) return nullptr;
  if (!math::has_cap(device_matrix_caps(*s.device), math::MatrixCap::kWmmaInt8)) {
    return nullptr;
  }
  if (s.intrinsics->find(MmaRdna3::kRow.key).empty()) return nullptr;
  return &kQuantWmma;
}

}  // namespace lse::kernels
