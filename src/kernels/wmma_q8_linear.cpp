#include <algorithm>
#include <array>
#include <string>
#include <optional>
#include <vector>

#include "lse/dispatch/q8_matrix.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/kernels/wmma_q8_linear.hpp"

namespace lse::kernels {
namespace env = graph::env;
namespace kir = graph::kir;
using graph::KernelShapes;
using graph::ThreadPlan;
namespace {
constexpr int kTileM = 16, kTileN = 16, kTileK = 16;
constexpr std::uint32_t kBlock = 256, kColBlocks = 1;
constexpr float kAmaxFloor = 1e-30f;
template <math::MatrixTarget G>
using MmaFor = math::op::Mma<G, math::MatrixElem::kI32, math::MatrixElem::kSU8,
                             kTileM, kTileN, kTileK>;
template <class S>
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};
struct PackedArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> original_words;
  env::In<lse::bf16, env::Emit> original_scales;
  env::In<lse::bf16, env::Emit> original_biases;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales;
  env::In<lse::bf16, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};
KernelShapes original_shapes(const KernelShapes& s) {
  auto original = s;
  original.inputs = s.inputs.first(4);
  original.input_dtypes = s.input_dtypes.first(4);
  return original;
}
const graph::KernelPrimitiveBase* original_kernel(const KernelShapes& s) {
  const auto* kernel = dynamic_cast<const graph::KernelPrimitiveBase*>(
      graph::find_primitive("quant_linear"));
  return kernel ? kernel->specialize(s) : nullptr;
}
template <bool Packed, int Frag, class A>
void fill_weights(env::Emit& e, const A& a, const kir::Val<kir::u32>& col,
                  const kir::Val<kir::u32>& k0,
                  const kir::Val<kir::u32>& k_lane, std::uint32_t lanes,
                  const kir::Local<kir::u32, Frag>& frag) {
  if constexpr (Packed) {
    static_assert(Frag == 2);
    const auto base = e.let((col / 16u) * (lanes * 16u) +
                           (k0 / 16u) * 64u + (k_lane / 8u) * 16u + col % 16u);
    for (std::uint32_t f = 0; f < static_cast<std::uint32_t>(Frag); ++f)
      frag[static_cast<int>(f)] = e.let(a.packed[e.let(base + f * 32u)]);
  } else {
    const auto base = e.let(col * lanes + (k0 + k_lane) / 4u);
    for (std::uint32_t f = 0; f < static_cast<std::uint32_t>(Frag); ++f)
      frag[static_cast<int>(f)] = PackedCodes<8>::word(e, a.packed, base, f);
  }
}

template <class A, math::MatrixTarget G, std::uint32_t RowBlocks, bool Packed>
std::string emit_body(const KernelShapes& s, const dispatch::AffineMatrixPlan& d) {
  constexpr std::uint32_t kRowBlocks = RowBlocks;
  constexpr std::uint32_t kRowsPerGroup = kTileM * RowBlocks;
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
  const auto gsize = static_cast<std::uint32_t>(64u);
  const std::uint32_t slices = gsize / kTileK;
  const std::uint32_t round_groups = d.round_groups;
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

  const auto xstep = e.lds<kir::f32>(kRowsPerGroup * round_groups);
  const auto xtot = e.lds<kir::f32>(kRowsPerGroup * round_groups);

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / kWave);
  const auto lane = e.let(lid % kWave);
  const auto lane_lo = e.let(lane % static_cast<std::uint32_t>(kTileN));
  const auto lane_hi = e.let(lane / static_cast<std::uint32_t>(kTileN));

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

  const auto lane_k = e.let(lane_hi * (kGeo.split_k ? kGeo.lane_k : 0u));
  const auto lane_word =
      e.let(lane_hi * (kGeo.split_k ? kGeo.lane_k / 4u : 0u));
  std::vector<kir::Val<kir::u32>> col_scales;
  for (std::uint32_t j = 0; j < kColBlocks; ++j) {
    if constexpr (Packed)
      col_scales.push_back(e.let((safe_col[j] / 16u) * (groups * 16u) + safe_col[j] % 16u));
    else
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
    st_qbase.push_back(e.let(sr * words + stt * 4u));
    st_sbase.push_back(e.let(sr * round_slices + stt));
    st_slice0.push_back(e.let(sr * round_slices + gl * slices));
    st_gbase.push_back(e.let(sr * round_groups + gl));
    st_in.push_back(e.let(si < items));
  }
  for (auto rnd : e.range(0u, RowBlocks == 1 ? (groups + round_groups - 1u) / round_groups : groups / round_groups, 1u)) {
    for (std::uint32_t c = 0; c < chunks; ++c) {
      if (auto stager = e.when(st_in[c])) {
        auto row_live = st_row[c] < m;
        if constexpr (RowBlocks == 1)
          row_live = row_live && rnd * round_groups + st_t[c] / slices < groups;
        if (auto in_rows = e.when(row_live)) {
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
        for (std::uint32_t w = 0; w < 4u; ++w) {
          auto word = e.let(e.u32(0));
          for (std::uint32_t b = 0; b < 4u; ++b) {
            const auto raw = e.let(vraw[c][w * 4u + b].read());
            const auto code = e.let(math::rint(raw * inv));
            const auto byte =
                e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
            word = e.let(word + byte * (1u << (8 * b)));
          }
          xq[e.let(st_qbase[c] + w)] = word;
        }
      }
    }
    e.barrier();

    for (std::uint32_t gi = 0; gi < round_groups; ++gi) {
      const auto g = e.let(rnd * round_groups + gi);
      std::optional<env::Emit::Guard> live_group;
      if constexpr (RowBlocks == 1) live_group.emplace(e.k, g < groups);
      std::vector<kir::Val<kir::f32>> gscale, gbias;
      for (std::uint32_t j = 0; j < kColBlocks; ++j) {
        const auto at = e.let([&] {
          if constexpr (Packed) return col_scales[j] + g * 16u;
          else return col_scales[j] + g;
        }());
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
          for (std::uint32_t c = 0; c < kFrag; ++c) bf[j][static_cast<int>(c)] = e.u32(0);
          if (auto gd = e.when(live[j] && bcol[j] < n)) {
            fill_weights<Packed>(e, a, bcol[j], k0, lane_k, lanes, bf[j]);
          } else {
            for (std::uint32_t c = 0; c < kFrag; ++c) bf[j][static_cast<int>(c)] = e.u32(0);
          }
        }
        const std::uint32_t slot = gi * slices + t;
        for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
          const auto af = e.local<kir::u32, kFragI>();
          for (std::uint32_t c = 0; c < kFrag; ++c) {
            af[static_cast<int>(c)] = xq[e.let(lane_words[i] + lane_word + slot *
                (static_cast<std::uint32_t>(kRow.k) / 4u) + c)].read();
          }
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
                math::fma(term, kir::cast<kir::f32>(acc[ij][static_cast<int>(z)].read()),
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

template <bool Packed>
dispatch::AffineMatrixPlan selected_plan(const KernelShapes& s, std::uint32_t rows) {
  if constexpr (Packed) return dispatch::q8_packed_matrix_plan(s);
  return dispatch::q8_matrix_plan(s, rows);
}
template <std::uint32_t RowBlocks, bool Packed = false>
struct AffineWmmaKernel final : graph::KernelPrimitive<AffineWmmaKernel<RowBlocks, Packed>> {
  static constexpr std::uint32_t kRows = kTileM * RowBlocks;
  static constexpr std::string_view kName = Packed
      ? "quant_linear.q8.wmma16.packed.v1"
      : RowBlocks == 1
      ? "quant_linear.q8.wmma16.iu8_affine.v2"
      : "quant_linear.q8.wmma64.iu8_affine.v1";
  static constexpr std::string_view kEntry = "lse_quant_linear_affine_wmma";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return Packed ? 7 : 4; }
  bool owns_indexing() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != (Packed ? 7 : 4) || !in[0].rank() || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "affine WMMA requires x, packed, scales, biases");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  std::string emit_kernel(const KernelShapes& s) const override {
    const auto d = selected_plan<Packed>(s, kRows);
    if (!s.store || !s.types.scalar) return {};
    if (!d.matrix) {
      if constexpr (Packed) {
        if (s.inputs.size() != 7 || s.input_dtypes.size() != 7) return {};
        const auto original = original_shapes(s);
        const auto* kernel = original_kernel(original);
        return kernel ? kernel->emit_kernel(original) : std::string{};
      }
      return {};
    }
    return with_matrix_target<std::string>(d.matrix->target,
        [&]<math::MatrixTarget G>() -> std::string {
          if constexpr (math::has_matrix_core_row(G, math::MatrixElem::kI32,
                  math::MatrixElem::kSU8, kTileM, kTileN, kTileK)) {
            if constexpr (Packed) {
              if constexpr (MmaFor<G>::kRow.a_len / MmaFor<G>::kRow.chained == 2)
                return emit_body<PackedArgs, G, RowBlocks, true>(s, d);
            } else {
              return emit_body<Args<lse::bf16>, G, RowBlocks, false>(s, d);
            }
          }
          return {};
        });
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    const auto d = selected_plan<Packed>(s, kRows);
    ThreadPlan tp;
    if (!d.matrix) {
      if constexpr (Packed) {
        if (s.inputs.size() != 7 || s.input_dtypes.size() != 7) return tp;
        const auto original = original_shapes(s);
        const auto* kernel = original_kernel(original);
        return kernel ? kernel->plan(original) : ThreadPlan{};
      }
      return tp;
    }
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = ((d.m + kRows - 1u) / kRows) * ((d.n + 127u) / 128u);
    tp.lds_bytes = d.lds_bytes;
    return tp;
  }
};
const AffineWmmaKernel<1> kQ8Wmma16;
const AffineWmmaKernel<1, true> kQ8PackedWmma16;
const AffineWmmaKernel<4> kQ8Wmma64;
const graph::PrimitiveRegistrar kRegister16{&kQ8Wmma16};
const graph::PrimitiveRegistrar kRegisterPacked{&kQ8PackedWmma16};
const graph::PrimitiveRegistrar kRegister64{&kQ8Wmma64};
}  // namespace

const graph::KernelPrimitiveBase* wmma_q8_linear_for(
    const KernelShapes& s, std::uint32_t rows) {
  if (!dispatch::q8_matrix_plan(s, rows).matrix) return nullptr;
  return rows == 16 ? static_cast<const graph::KernelPrimitiveBase*>(&kQ8Wmma16)
                    : static_cast<const graph::KernelPrimitiveBase*>(&kQ8Wmma64);
}
}  // namespace lse::kernels
