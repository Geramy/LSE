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

// Row blocks a workgroup owns. More of them let one unpacked weight fragment
// feed several row blocks. On gfx1151 that never paid (measured 10.37 s,
// 10.50 s, 10.62 s at 1601 tokens for 1/2/4). On gfx1201 it pays outright:
// measured 3.29 s, 3.03 s, 2.68 s at 277 tokens for 1/2/4, the dominant
// prefill matmul 3.2x faster, verified correct on one device and on a
// two-member tensor split, short and long context. (Four was once falsely
// convicted of corrupting the split by a test prompt made of one sentence
// repeated — a correct greedy model continues the repetition, and the
// "corruption" was the model doing its job. Judge correctness on natural
// prompts.) Eight measured 319.9 warm prefill tokens/s against four's 381.3
// on the same day — its 46 KB of LDS per workgroup halves occupancy, and
// the sweep is closed: 1 < 2 < 4 > 8.
constexpr std::uint32_t kRowBlocks = 4u;  // measured champion (see sweep note)
// Column blocks a wave owns. The symmetric twin of the row blocking above:
// every A fragment a lane reads from LDS feeds kColBlocks different B
// columns before it is discarded. Measured on gfx1201 at 124 prompt tokens:
// 2.32 s at rb4/cb1, 2.44 s at rb4/cb2, 2.47 s at rb2/cb2 — the A reuse does
// not buy back the doubled B fills and register pressure, so one. The
// machinery stays because double-buffering may change the balance.
constexpr std::uint32_t kColBlocks = 1u;
constexpr std::uint32_t kRowsPerGroup = kTileM * kRowBlocks;

// The instruction this invocation gets. `G` is the generation the device
// reported; every width below is read off the row that selects, so nothing
// here names a generation or a register count of its own.
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
// `k_lane` is where this lane's own slice of the instruction's K starts. It is
// zero when a lane holds the whole step, and (lane / n) * k / halves when the
// wave splits the step between its halves -- the difference between the two
// generations, taken from the row rather than assumed.
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

template <class A, math::MatrixTarget G>
std::string emit_body(const KernelShapes& s, const Dims& d) {
  using Mma = MmaFor<G>;
  constexpr math::MatrixCoreRow kRow = Mma::kRow;
  // Lanes that cooperate on one tile, and how many accumulator slots a lane
  // holds -- wave32 with eight is RDNA, wave64 with four is CDNA, and neither
  // is spelled here.
  // Every width and every index rule comes from the row, through the one place
  // that reads a layout. Nothing below names a generation.
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

  // A workgroup owns one row block and `waves` column blocks of it, so the
  // sixteen rows are quantized once here and every wave reads them. Doing it
  // per wave repeated the amax, the rounding and the loads once per column
  // tile, which is what made the correct kernel three hundred times slower
  // than the loop it replaces.
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

  const auto wg = e.let(math::workgroup_id_x());
  const auto m0 = e.let((wg / nblocks) * kRowsPerGroup);
  // Column tile j of this wave: the workgroup's tiles are laid out so the
  // waves stay adjacent within each block, keeping their weight reads beside
  // each other, and a wave's own j-tiles stride by `waves`.
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

  // One base per row block: the accumulator slot's own offset is a constant,
  // so folding it in at the use costs an add the scheduler hides, where
  // hoisting all of them costs a live register each.
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
  // A column past the end still indexes the scales, because the group loop
  // reads them before it knows whether the lane will store anything. Its
  // result is discarded either way, so it reads column zero rather than off
  // the end of the plane: n only has to miss a multiple of waves * kTileN for
  // the last workgroup to carry lanes that are not columns at all.
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
  // The raw activations of the round stay in the stager thread's own
  // registers across the amax barrier: the same thread quantizes the slice it
  // loaded, so nothing else ever reads them, and the 16 KB of LDS the old
  // staging plane cost was the difference between two workgroups per CU and
  // several. Declared outside the guards so the values scope across them.
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

  // Measured and reverted: software-pipelining this loop two rounds deep
  // (staging round r+1 into a second LDS plane while round r's matrix work
  // runs) emitted correctly and lost — 369.8 warm prefill tokens/s against
  // 475.4 for this sequential order. Overlapping the stages keeps the staged
  // raws, the amax sums and the accumulators live at once, and the spills
  // cost more than the hidden load latency saved: this kernel is bound by
  // occupancy, not by memory latency.
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

    // The group's own step, reduced from the slice amaxes the pass above left.
    // Every slice of a group computes it and writes the same value, which is
    // cheaper than electing one thread to.
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
      std::vector<kir::Val<kir::f32>> gscale, gbias;
      for (std::uint32_t j = 0; j < kColBlocks; ++j) {
        const auto at = e.let(col_scales[j] + g);
        gscale.push_back(e.let(math::widen(a.scales[at])));
        gbias.push_back(e.let(math::widen(a.biases[at])));
      }

      // One accumulator per row block, spanning every slice of the group: the
      // step is constant across it, so the integer sum can run to the end
      // before anything is scaled.
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
          }
        }
        // The unpacked fragments feed every row block, and each row block's A
        // fragment feeds every column block, before either is discarded —
        // that is what both blockings buy.
        const std::uint32_t slot = gi * slices + t;
        for (std::uint32_t i = 0; i < kRowBlocks; ++i) {
          const auto af = e.local<kir::u32, kFragI>();
          for (std::uint32_t c = 0; c < kFrag; ++c) {
            // Four words hold a whole sixteen-value step; a split-K lane takes
            // the half of them its own half of the wave is responsible for.
            af[c] = xq[e.let(lane_words[i] + lane_word +
                             (slot * (static_cast<std::uint32_t>(kRow.k) / 4u) +
                              c))]
                        .read();
          }
          // One issue takes the whole K slice: the row declares A and B four
          // registers wide and chained=1, so a lane hands over its sixteen
          // contiguous K values at once.
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
    const std::optional<math::MatrixTarget> target = matrix_target(*s.device);
    if (!target.has_value()) return {};
    return with_matrix_target<std::string>(
        *target, [&]<math::MatrixTarget G>() -> std::string {
          if constexpr (!math::has_matrix_core_row(
                            G, math::MatrixElem::kI32, math::MatrixElem::kSU8,
                            kTileM, kTileN, kTileK)) {
            return {};
          } else {
            return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
              return emit_body<Args<S>, G>(s, d);
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

  // One row is decode, and a 16x16 tile would mask fifteen of its rows to
  // move the same weights -- but the dot4 path's cost is the dequant and dot
  // ALU per ROW, not the weight stream (measured: its GEMVs double from m=2
  // to m=4), so the crossover to the matrix core sits well below a full
  // tile. LSE_WMMA_MIN_M overrides the threshold for measurement.
  static const std::uint32_t min_m = [] {
    const char* v = std::getenv("LSE_WMMA_MIN_M");
    const long n = v != nullptr ? std::strtol(v, nullptr, 10) : kTileM;
    return static_cast<std::uint32_t>(std::clamp(n, 1L, 1024L));
  }();
  if (d.m < min_m) return nullptr;

  const std::optional<math::MatrixTarget> target = matrix_target(*s.device);
  if (!target.has_value()) return nullptr;

  // The row the device's generation selects, and whether it may be emitted at
  // all. A row whose lane mapping was never measured on the part is not a
  // slower path, it is a wrong answer that looks right, so the table declines
  // it and so does this. Nothing here names a generation: a part joins by
  // having its layout measured into the table.
  return with_matrix_target<const graph::KernelPrimitiveBase*>(
      *target, [&]<math::MatrixTarget G>() -> const graph::KernelPrimitiveBase* {
        // CDNA's MFMA int8 has no signedness immediates, so there is no mixed
        // row to name: unsigned codes there need the algebra shifted, not a
        // different spelling. It declines here rather than pretending.
        if constexpr (!math::has_matrix_core_row(
                          G, math::MatrixElem::kI32, math::MatrixElem::kSU8,
                          kTileM, kTileN, kTileK)) {
          return nullptr;
        } else {
        constexpr math::MatrixCoreRow kRow = MmaFor<G>::kRow;
        if constexpr (!kRow.emittable()) {
          return nullptr;
        } else {
          if (!math::has_cap(device_matrix_caps(*s.device), kRow.cap)) {
            return nullptr;
          }
          if (s.intrinsics->find(kRow.key).empty()) return nullptr;
          // The tile stages one K slice per lane and drains kTileM rows, so a
          // row whose instruction does not have that shape needs the staging
          // rewritten, not just a different spelling.
          if (kRow.m != kTileM || kRow.n != kTileN || kRow.k != kTileK) {
            return nullptr;
          }
          if (kRow.chained != 1) return nullptr;
          return &kQuantWmma;
        }
        }
      });
}

}  // namespace lse::kernels
