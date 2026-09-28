#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/dispatch/quant.hpp"
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
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kRowBlocks = 4u;  // measured champion (see sweep note)
constexpr std::uint32_t kColBlocks = 1u;
constexpr std::uint32_t kRowsPerGroup = kTileM * kRowBlocks;
template <math::MatrixTarget G>
using MmaFor = math::op::Mma<G, math::MatrixElem::kI32, math::MatrixElem::kSU8,
                             kTileM, kTileN, kTileK>;

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
  if (d.spec.bits != 4) return d;
  // Keep the specialization to groups spanning at least two full K slices.
  // The scale and bias must stay constant across each matrix instruction.
  if (d.spec.group_size % (2 * kTileK) != 0) return d;
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
template <int Frag, class A>
void fill_weights(env::Emit& e, const A& a, const kir::Val<kir::u32>& col,
                  const kir::Val<kir::u32>& k0,
                  const kir::Val<kir::u32>& k_lane, std::uint32_t lanes,
                  int bits, const kir::Local<kir::u32, Frag>& frag) {
  const std::uint32_t per_word = 32u / static_cast<std::uint32_t>(bits);
  const auto base = e.let(col * lanes + (k0 + k_lane) / per_word);
  for (std::uint32_t f = 0; f < static_cast<std::uint32_t>(Frag); ++f) {
    frag[f] = bits == 4 ? PackedCodes<4>::word(e, a.packed, base, f)
                        : PackedCodes<8>::word(e, a.packed, base, f);
  }
}

template <class A, math::MatrixTarget G, bool ActsLayout>
std::string emit_body(const KernelShapes& s, const Dims& d) {
  using Mma = MmaFor<G>;
  constexpr math::MatrixCoreRow kRow = Mma::kRow;
  constexpr TileGeometry kGeo = geometry_of(kRow);
  constexpr std::uint32_t kWave = kGeo.wave;
  constexpr std::uint32_t kSlots = kGeo.slots;
  constexpr int kSlotsI = kRow.c_len;
  constexpr std::uint32_t kFrag = kGeo.frag;
  constexpr int kFragI = kRow.a_len / kRow.chained;
  constexpr std::uint32_t kSlotStep = kGeo.slot_step;
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto gsize = static_cast<std::uint32_t>(d.spec.group_size);
  const std::uint32_t slices = gsize / kTileK;
  const std::uint32_t kGroupsPerRound = 1u;
  const std::uint32_t round_groups =
      groups < kGroupsPerRound ? groups : kGroupsPerRound;
  const std::uint32_t round_slices = slices * round_groups;
  const std::uint32_t words = gsize * round_groups / 4u;
  const std::uint32_t waves = kBlock / kWave;
  const std::uint32_t tiles_n = (n + kTileN - 1u) / kTileN;
  const std::uint32_t wg_tiles = waves * kColBlocks;
  const std::uint32_t nblocks = (tiles_n + wg_tiles - 1u) / wg_tiles;
  const std::uint32_t load_bytes = device_load_bytes(s.device);

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};
  const auto xq = e.lds<kir::u32>(kRowsPerGroup * words);
  const auto xamax = e.lds<kir::f32>(kRowsPerGroup * round_slices);
  const auto xssum = e.lds<kir::f32>(kRowsPerGroup * round_slices);
  // Per group, not per slice: one step for the whole group is what lets a
  // single accumulator span every slice in it.
  const auto xstep = e.lds<kir::f32>(kRowsPerGroup * round_groups);
  const auto xtot = e.lds<kir::f32>(kRowsPerGroup * round_groups);

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / kWave);
  const auto lane = e.let(lid % kWave);
  const auto lane_lo = e.let(lane % static_cast<std::uint32_t>(kTileN));
  const auto lane_hi = e.let(lane / static_cast<std::uint32_t>(kTileN));
  std::optional<kir::Val<kir::u32>> lane_rot;
  if constexpr (ActsLayout) lane_rot = e.let((lane_lo / 4u) * 4u);

  const auto wg = e.let(math::workgroup_id_x());
  const auto m0 = e.let((wg / nblocks) * kRowsPerGroup);
  std::vector<kir::Val<kir::u32>> ntile, n0, bcol;
  std::vector<kir::Val<kir::boolean>> live;
  for (std::uint32_t j = 0; j < kColBlocks; ++j) {
    ntile.push_back(e.let((wg % nblocks) * wg_tiles + j * waves + wave_id));
    n0.push_back(e.let(ntile[j] * static_cast<std::uint32_t>(kTileN)));
    bcol.push_back(e.let(n0[j] + lane_lo));
    live.push_back(e.let(ntile[j] < tiles_n));
  }

  std::vector<kir::LValue<kir::f32>> out;
  out.reserve(kRowBlocks * kColBlocks * kSlots);
  for (std::uint32_t i = 0; i < kRowBlocks * kColBlocks * kSlots; ++i) {
    out.push_back(e.var(0.0f));
  }
  std::vector<kir::Val<kir::u32>> slot_g;
  std::vector<kir::Val<kir::u32>> slot_out;
  std::vector<kir::Val<kir::u32>> lane_words;
  slot_g.reserve(kRowBlocks);
  slot_out.reserve(kRowBlocks);
  lane_words.reserve(kRowBlocks);
  for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
    const std::uint32_t rb = i * static_cast<std::uint32_t>(kTileM);
    lane_words.push_back(e.let((lane_lo + rb) * words));
    const auto half = e.let(lane_hi * kGeo.half_rows);
    slot_g.push_back(e.let((rb + half) * round_groups));
    slot_out.push_back(e.let(m0 + rb + half));
  }
  std::vector<kir::Val<kir::u32>> safe_col;
  for (std::uint32_t j = 0; j < kColBlocks; ++j) {
    safe_col.push_back(e.let(select(bcol[j] < n, bcol[j], e.u32(0))));
  }
  // Zero on a generation whose lane holds the whole step; on one that splits
  // it, the offset of this lane's half, in operand values and in packed words.
  const auto lane_k = e.let(lane_hi * (kGeo.split_k ? kGeo.lane_k : 0u));
  const auto lane_word =
      e.let(lane_hi * (kGeo.split_k ? kGeo.lane_k / 4u : 0u));
  std::vector<kir::Val<kir::u32>> col_scales;
  for (std::uint32_t j = 0; j < kColBlocks; ++j) {
    col_scales.push_back(e.let(safe_col[j] * groups));
  }

  const std::uint32_t items = kRowsPerGroup * round_slices;
  const std::uint32_t chunks = (items + kBlock - 1u) / kBlock;
  std::vector<std::vector<kir::LValue<kir::f32>>> vraw;
  vraw.reserve(chunks);
  for (std::uint32_t c = 0; c < chunks; ++c) {
    std::vector<kir::LValue<kir::f32>> row;
    row.reserve(static_cast<std::size_t>(kTileK));
    for (int j = 0; j < kTileK; ++j) row.push_back(e.var(0.0f));
    vraw.push_back(std::move(row));
  }
  std::vector<kir::Val<kir::u32>> st_t, st_row, st_xbase, st_qbase, st_sbase;
  std::vector<kir::Val<kir::u32>> st_slice0, st_gbase;
  std::vector<kir::Val<kir::boolean>> st_in;
  for (std::uint32_t c = 0; c < chunks; ++c) {
    const auto si = e.let(lid + c * kBlock);
    const auto sr = e.let(si / round_slices);
    const auto stt = e.let(si % round_slices);
    const auto gl = e.let(stt / slices);
    st_t.push_back(stt);
    st_row.push_back(e.let(m0 + sr));
    st_xbase.push_back(e.let(st_row[c] * k));
    if constexpr (ActsLayout) {
      // Quarter rotations preserve each aligned four-word store and
      // two-word fragment load within its activation row.
      const auto rot = e.let(((sr / 4u) % 4u) * 4u);
      st_qbase.push_back(e.let(sr * words + (stt * 4u + rot) % words));
    } else {
      st_qbase.push_back(e.let(sr * words + stt * 4u));
    }
    st_sbase.push_back(e.let(sr * round_slices + stt));
    st_slice0.push_back(e.let(sr * round_slices + gl * slices));
    st_gbase.push_back(e.let(sr * round_groups + gl));
    st_in.push_back(e.let(si < items));
  }
  for (auto rnd : e.range(0u, groups / round_groups, 1u)) {
    for (std::uint32_t c = 0; c < chunks; ++c) {
      if (auto stager = e.when(st_in[c])) {
        if (auto in_rows = e.when(st_row[c] < m)) {
          const auto base = e.let(st_xbase[c] +
                                  (rnd * (gsize * round_groups) +
                                   st_t[c] * static_cast<std::uint32_t>(kTileK)));
          const std::uint32_t wide = row_pack(kTileK, load_bytes, 4);
          for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(kTileK);
               j += wide) {
            const auto pack = e.load(a.x, e.let(base + j), load_bytes);
            for (std::uint32_t u = 0; u < wide; ++u) {
              vraw[c][j + u] = e.let(pack[static_cast<int>(u)]);
            }
          }
          auto amax = e.let(math::abs(vraw[c][0].read()));
          auto total = e.let(vraw[c][0].read());
          for (int j = 1; j < kTileK; ++j) {
            const auto vj = e.let(vraw[c][static_cast<std::size_t>(j)].read());
            amax = e.let(math::max(amax, math::abs(vj)));
            total = e.let(total + vj);
          }
          xamax[st_sbase[c]] = amax;
          xssum[st_sbase[c]] = total;
        } else {
          // A row past the end still publishes an amax and a sum, or the group
          // reduction below takes whatever the previous round left behind.
          xamax[st_sbase[c]] = e.f32(0.0f);
          xssum[st_sbase[c]] = e.f32(0.0f);
          for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(kTileK); ++j) {
            vraw[c][j] = e.f32(0.0f);
          }
        }
      }
    }
    e.barrier();
    for (std::uint32_t c = 0; c < chunks; ++c) {
      if (auto stager = e.when(st_in[c])) {
        auto gmax = e.let(xamax[st_slice0[c]].read());
        auto gsum = e.let(xssum[st_slice0[c]].read());
        for (std::uint32_t j = 1; j < slices; ++j) {
          gmax = e.let(math::max(gmax, xamax[e.let(st_slice0[c] + j)].read()));
          gsum = e.let(gsum + xssum[e.let(st_slice0[c] + j)].read());
        }
        xstep[st_gbase[c]] = gmax * (1.0f / 127.0f);
        xtot[st_gbase[c]] = gsum;
        const auto inv = e.let(127.0f / math::max(gmax, e.f32(kAmaxFloor)));
        std::optional<kir::Local<kir::u32, 4>> packed_words;
        if constexpr (ActsLayout) packed_words = e.local<kir::u32, 4>();
        for (std::uint32_t w = 0; w < 4u; ++w) {
          auto word = e.let(e.u32(0));
          for (std::uint32_t b = 0; b < 4u; ++b) {
            const auto raw = e.let(vraw[c][w * 4u + b].read());
            const auto code = e.let(math::rint(raw * inv));
            const auto byte =
                e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
            word = e.let(word + byte * (1u << (8 * b)));
          }
          if constexpr (ActsLayout) {
            (*packed_words)[w] = word;
          } else {
            xq[e.let(st_qbase[c] + w)] = word;
          }
        }
        if constexpr (ActsLayout) {
          const auto value = packed_words->value();
          kb.store_pack<kir::u32>(xq.id(), st_qbase[c],
              kir::Pack<kir::u32>(value.types(), value.body(), value.id(), 4), 16u);
        }
      }
    }
    e.barrier();

    for (std::uint32_t gi = 0; gi < round_groups; ++gi) {
      const auto g = e.let(rnd * round_groups + gi);
      std::vector<kir::Val<kir::f32>> gscale, gbias;
      for (std::uint32_t j = 0; j < kColBlocks; ++j) {
        const auto at = e.let(col_scales[j] + g);
        gscale.push_back(e.let(math::widen(a.scales[at])));
        gbias.push_back(e.let(math::widen(a.biases[at])));
      }
      std::vector<kir::Local<kir::i32, 8>> acc;
      acc.reserve(kRowBlocks * kColBlocks);
      for (std::uint32_t i = 0; i < kRowBlocks * kColBlocks; ++i) {
        acc.push_back(e.local<kir::i32, kSlotsI>());
        for (auto z : e.unroll(kSlots)) acc[i][z] = kir::cast<kir::i32>(e.u32(0));
      }

      for (std::uint32_t t = 0; t < slices; ++t) {
        const auto k0 =
            e.let(g * gsize + t * static_cast<std::uint32_t>(kTileK));
        std::vector<kir::Local<kir::u32, kFragI>> bf;
        bf.reserve(kColBlocks);
        for (std::uint32_t j = 0; j < kColBlocks; ++j) {
          bf.push_back(e.local<kir::u32, kFragI>());
          for (std::uint32_t c = 0; c < kFrag; ++c) bf[j][c] = e.u32(0);
          if (auto gd = e.when(live[j] && bcol[j] < n)) {
            fill_weights(e, a, bcol[j], k0, lane_k, lanes, d.spec.bits, bf[j]);
          } else {
            for (std::uint32_t c = 0; c < kFrag; ++c) bf[j][c] = e.u32(0);
          }
        }
        const std::uint32_t slot = gi * slices + t;
        for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
          const auto af = e.local<kir::u32, kFragI>();
          if constexpr (ActsLayout) {
            const auto abase = e.let(lane_words[i] +
                (lane_word + slot * (static_cast<std::uint32_t>(kRow.k) / 4u) +
                 *lane_rot) % words);
            const auto av = xq.load(abase, 8u);
            for (std::uint32_t c = 0; c < kFrag; ++c) af[c] = av[c];
          } else {
            for (std::uint32_t c = 0; c < kFrag; ++c) {
              // Four words hold a whole sixteen-value step; a split-K lane takes
              // the half of them its own half of the wave is responsible for.
              af[c] = xq[e.let(lane_words[i] + lane_word +
                               (slot * (static_cast<std::uint32_t>(kRow.k) / 4u) +
                                c))]
                          .read();
            }
          }
          // One issue takes the whole K slice. Each half-wave contributes
          // its two packed registers for eight K values on this split-K row.
          for (std::uint32_t j = 0; j < kColBlocks; ++j) {
            const std::uint32_t ij = i * kColBlocks + j;
            acc[ij] = math::mma<Mma>(af.value(), bf[j].value(),
                                     acc[ij].value());
          }
        }
      }

      for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
        for (std::uint32_t z = 0; z < kSlots; ++z) {
          const auto at_g =
              e.let(slot_g[i] + (z * kSlotStep * round_groups + gi));
          const auto step_v = e.let(xstep[at_g].read());
          const auto tot_v = e.let(xtot[at_g].read());
          for (std::uint32_t j = 0; j < kColBlocks; ++j) {
            const std::uint32_t ij = i * kColBlocks + j;
            const auto term = e.let(gscale[j] * step_v);
            out[ij * kSlots + z] =
                math::fma(term, kir::cast<kir::f32>(acc[ij][z].read()),
                          out[ij * kSlots + z].read()) +
                gbias[j] * tot_v;
          }
        }
      }
    }
    e.barrier();
  }

  for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
    for (std::uint32_t z = 0; z < kSlots; ++z) {
      const auto row = e.let(slot_out[i] + z * kSlotStep);
      for (std::uint32_t j = 0; j < kColBlocks; ++j) {
        const std::uint32_t ij = i * kColBlocks + j;
        if (auto gr = e.when(live[j] && row < m && bcol[j] < n)) {
          e.store(row * n + bcol[j], out[ij * kSlots + z].read());
        }
      }
    }
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}

template <bool ActsLayout>
struct QuantWmmaKernel final : graph::KernelPrimitive<QuantWmmaKernel<ActsLayout>> {
  static constexpr std::string_view kName = ActsLayout
      ? "quant_linear.wmma.acts_swizzle_v2.v1" : "quant_linear.wmma";
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
    const auto* row = dispatch::quant_plan(s).matrix;
    if (row == nullptr) return {};
    return with_matrix_target<std::string>(
        row->target, [&]<math::MatrixTarget G>() -> std::string {
          if constexpr (!math::has_matrix_core_row(
                            G, math::MatrixElem::kI32, math::MatrixElem::kSU8,
                            kTileM, kTileN, kTileK)) {
            return {};
          } else {
            return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
              return emit_body<Args<S>, G, ActsLayout>(s, d);
            });
          }
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
    tp.lds_bytes = dispatch::kQ4MatrixLdsBytes;
    return tp;
  }
};

const QuantWmmaKernel<false> kQuantWmma;
const QuantWmmaKernel<true> kQuantWmmaActs;

}  // namespace

const graph::KernelPrimitiveBase* wmma_quant_linear_for(const KernelShapes& s) {
  switch (dispatch::quant_plan(s).implementation) {
    case dispatch::QuantMatrix::kInt8Lds: return &kQuantWmmaActs;
    case dispatch::QuantMatrix::kInt8: return &kQuantWmma;
    default: return nullptr;
  }
}

}  // namespace lse::kernels
