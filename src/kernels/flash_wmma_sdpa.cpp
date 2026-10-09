// Typed paged KV; FP32 matrix accumulators and online softmax state.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <optional>
#include <string>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_tuneconfig.h"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kv/block.hpp"
#include "lse/math.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/kernels/kv_storage.hpp"
#include "lse/kernels/sdpa.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kThreads = dispatch::attention_shapes::kFlashThreads;
constexpr std::uint32_t QTile = dispatch::attention_shapes::kFlashQueryTile;
constexpr std::uint32_t kKWin = dispatch::attention_shapes::kFlashKeyWindow;
using Dims = dispatch::FlashDims;

template <class E, kv::CacheDType Storage>
struct FlashArgs {
  env::In<kir::f32, E> q;
  env::In<KvElement<Storage>, E> k;
  env::In<KvElement<Storage>, E> v;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

template <class E, kv::CacheDType Storage>
struct FlashMeanArgs {
  env::In<kir::f32, E> q;
  env::In<KvElement<Storage>, E> k, v;
  env::In<kir::f32, E> meta, table, pooled, selected;
  env::Out<kir::f32, E> out;
};

// Q/P and decoded KV use F16 operands for F16 storage, BF16 otherwise.
// Both matrix accumulators, softmax state and output remain FP32.
//
// `Split`: a draft tree's pass (dispatch::flash_split_scope). Each workgroup
// takes one share of the key windows and writes, per query row, the share's
// running maximum, sum and unnormalized output as one split record
// [B, H, T, parts, 258] for attention.split_merge128.wg128c2.v1.
template <bool MeanCorrection, bool Split = false>
struct FlashWmmaImpl final : KernelPrimitive<FlashWmmaImpl<MeanCorrection, Split>> {
  static_assert(!(MeanCorrection && Split));
  static constexpr std::string_view kName = MeanCorrection ? "attention.flashprefill.wmma.v1"
      : Split ? "attention.flash_split.wmma16.v1" : "attention.flash.wmma16.v3";
  static constexpr std::string_view kEntry = MeanCorrection ? "lse_flashprefill_wmma_v1"
      : Split ? "lse_flash_split_wmma16_v1" : "lse_flash_wmma16_v3";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return MeanCorrection ? 7 : 5; }
  // Variant 1 stages values 16 keys at a time where 32 fit (emit_storage),
  // and variant 2 takes two query tiles per workgroup (emit_two_tiles), on a
  // contiguous-fragment generation; a split pass, whose records the two-tile
  // form does not write, offers only the first two. Each output is the same
  // value either way.
  std::uint32_t variants(const KernelShapes& s) const override {
    if (s.device == nullptr || !matrix_target(*s.device)) return 1;
    const auto* row = [&]() -> const math::MatrixCoreRow* {
      for (const auto& r : math::matrix_core_table())
        if (r.target == *matrix_target(*s.device) && r.wave == 32 && r.m == 16 &&
            r.n == 16 && r.k_step == 16 && r.acc == math::MatrixElem::kF32)
          return &r;
      return nullptr;
    }();
    if (row == nullptr || row->a_len != 16) return 1u;
    return Split ? 2u : 3u;
  }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  // Query heads of one key head that share a split pass's tile: the most
  // that divide the group and whose rows fill at most the tile's sixteen.
  static std::uint32_t split_heads_per_tile(const Dims& d) {
    if constexpr (!Split) return 1u;
    if (!d.valid || d.tq == 0 || d.group == 0) return 1u;
    std::uint32_t pack = 1;
    for (std::uint32_t n = 2; n <= d.group && n * d.tq <= QTile; ++n)
      if (d.group % n == 0) pack = n;
    return pack;
  }

  static KernelShapes dense_request(const KernelShapes& s) {
    auto dense=s;
    if constexpr (MeanCorrection) {
      if(s.inputs.size()>=5) dense.inputs=s.inputs.first(5);
      if(s.input_dtypes.size()>=5) dense.input_dtypes=s.input_dtypes.first(5);
    }
    if constexpr (Split) {
      if (s.output.rank() == 5 && s.output.dim(4) == dispatch::attention_shapes::kSplitRecord)
        dense.output = Shape{s.output.dim(0), s.output.dim(1), s.output.dim(2),
                             dispatch::attention_shapes::kSplitRecord - 2};
      else
        dense.output = Shape{};
    }
    return dense;
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    if constexpr (Split)
      if (!dispatch::flash_split_scope(dense_request(s))) return {};
    if (s.input_dtypes.size() != (MeanCorrection ? 7 : 5) || !dispatch::flash_wmma_supported(dense_request(s)) || !s.store ||
        (s.attrs[3] == 1.0f && (!std::isfinite(s.attrs[2]) || s.attrs[2] < 0)))
      return {};
    if constexpr(MeanCorrection)
      if(s.iattrs[0]!=1 || s.inputs[0].dim(3)!=256 || s.output.dim(3)!=256 ||
         !std::isfinite(s.attrs[2]) || s.attrs[2]<0 || s.attrs[2]>1) return {};
    // The device's matrix generation picks the row, and with it the fragment
    // layout every index below follows (see lse/math/matrix_rdna*.hpp).
    const auto target = matrix_target(*s.device);
    if (!target) return {};
    return with_kv_storage(s.input_dtypes[1], s.attrs[1],
        [&]<kv::CacheDType Storage>() {
          return with_matrix_target<std::string>(
              *target, [&]<math::MatrixTarget G>() -> std::string {
                constexpr auto operand = Storage == kv::CacheDType::kF16
                                             ? math::MatrixElem::kF16
                                             : math::MatrixElem::kBF16;
                if constexpr (!math::has_matrix_core_row(
                                  G, math::MatrixElem::kF32, operand, 16, 16, 16)) {
                  return {};
                } else if constexpr (math::matrix_core_row(
                                         G, math::MatrixElem::kF32, operand, 16,
                                         16, 16).wave != 32) {
                  return {};
                } else {
                  return s.variant == 2u ? emit_two_tiles<Storage, G>(s)
                                         : emit_storage<Storage, G>(s);
                }
              });
        });
  }

  // Variant 2: two 16-query tiles of one head per workgroup, on a
  // contiguous-fragment generation with 16-bit values. Each key and value
  // fragment the workgroup reads feeds both tiles, and each barrier serves
  // both. Every tile keeps its own FlashPrefill V2 selection and retention
  // vote: a tile that would have skipped a block sees -inf scores (no
  // selection) or skips its rows' softmax (no retention), which leaves its
  // rows exactly as skipping did. Each row's scores, softmax and products
  // are the single-tile kernel's, value for value and in the same order.
  // Scratch is tight for 32 rows, so the probabilities are narrowed into
  // scratch a quarter window at a time, from the kept scores and each row's
  // maximum.
  template <kv::CacheDType Storage, math::MatrixTarget G>
  std::string emit_two_tiles(const KernelShapes& s) const {
    constexpr auto operand = Storage == kv::CacheDType::kF16
                                 ? math::MatrixElem::kF16 : math::MatrixElem::kBF16;
    using Mma = math::op::Mma<G, math::MatrixElem::kF32, operand, 16, 16, 16>;
    using Narrow = math::matrix_scalar_t<Mma::kRow.a_elem>;
    constexpr TileGeometry kGeo = geometry_of(Mma::kRow);
    constexpr std::uint32_t kFrag = kGeo.lane_k;
    constexpr int kSlots = Mma::kRow.c_len;
    constexpr bool kWideKv =
        Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16;
    if constexpr (kGeo.split_k || !kWideKv || kFrag != 16u) {
      return {};
    } else {
    // Sixteen waves: one 16-key column tile of each window and one 16-wide
    // value tile each, two rows each in the row passes.
    constexpr std::uint32_t kTiles = 2, QR = QTile * kTiles, kW = 16, kT = kW * 32u;
    constexpr std::uint32_t kColTiles = kKWin / 16u / kW, kRowsPerWave = QR / kW;
    constexpr std::uint32_t kQuarter = 64, kProbRow = kQuarter + 8u, kVsub = 16;
    const Dims d = dispatch::flash_dimensions(dense_request(s));
    if (d.dh % 16u != 0 || d.dv % 16u != 0 || d.dv > 256u ||
        s.device->max_threads_per_workgroup < kT) return {};
    const auto mask = d.mask == 2 && d.window == 0u ? 1 : d.mask;
    // A draft tree's ancestor mask (mask 3) stays with the single-tile form.
    if (mask == 3) return {};
    const std::uint32_t ntiles = (d.tq + QTile - 1u) / QTile;
    const std::uint32_t npairs = (ntiles + kTiles - 1u) / kTiles;
    const std::uint32_t padded_depth = (d.dh + 15u) / 16u * 16u;
    const std::uint32_t qchunks = (QR * padded_depth + kT - 1u) / kT;
    const std::uint32_t dpt = (d.dv + kT - 1u) / kT;
    const std::uint32_t value_tiles = (d.dv / 16u + kW - 1u) / kW;
    const std::uint32_t vrow = d.dv;
    const std::uint64_t bytes = std::uint64_t{QR} * padded_depth * 2u +
        std::uint64_t{QR} * kKWin * 4u + 4u * QR * 4u + kVsub * vrow * 2u +
        std::uint64_t{QR} * kProbRow * 2u + 256u;
    if (bytes > workgroup_lds_bytes(s.device)) return {};
    const bool sparse = s.attrs[3] == 1.0f;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    std::conditional_t<MeanCorrection, FlashMeanArgs<env::Emit, Storage>,
                       FlashArgs<env::Emit, Storage>> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto qs = e.lds<Narrow>(QR * padded_depth);
    const auto sc = e.lds<kir::f32>(QR * kKWin);
    const auto mrow = e.lds<kir::f32>(QR);
    const auto drow = e.lds<kir::f32>(QR);
    const auto arow = e.lds<kir::f32>(QR);
    const auto smax = e.lds<kir::f32>(QR);
    const auto vs = e.lds<Narrow>(kVsub * vrow);
    const auto pb = e.lds<Narrow>(QR * kProbRow);

    const auto lid = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto qp = e.let(wg % npairs);
    const auto h = e.let((wg / npairs) % d.qh);
    const auto b = e.let(wg / (npairs * d.qh));
    const auto kh = e.let(h / d.group);
    const auto qt0 = e.let(qp * kTiles);
    const auto q0 = e.let(qt0 * QTile);
    const auto has1 = e.let(qt0 + 1u < ntiles);
    const auto obase = e.let(((b * d.qh + h) * d.tq) * d.dv);

    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    if (auto pad = e.when(b >= rows)) {
      for (std::uint32_t r = 0; r < QR; ++r)
        for (std::uint32_t p = 0; p < dpt; ++p) {
          const auto qrow = e.let(q0 + r);
          const auto dd = e.let(lid + p * kT);
          if (auto g = e.when(qrow < d.tq && dd < d.dv))
            e.store(obase + qrow * d.dv + dd, e.f32(0.0f));
        }
    }
    (void)e.ret_if(b >= rows);

    const auto mb =
        e.let(e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
              b * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow)));
    const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
    const auto capacity = e.u32(d.stride * d.ts);
    const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto row_len = e.let(select(loaded_len < capacity, loaded_len, capacity));
    const auto tb = e.let(b * d.stride);

    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kT);
      if (auto inb = e.when(idx < QR * padded_depth)) {
        const auto r = e.let(idx / padded_depth);
        const auto dd = e.let(idx % padded_depth);
        const auto qrow = e.let(q0 + r);
        qs[idx] = math::narrow<Narrow>(e.f32(0.0f));
        if (auto g = e.when(qrow < d.tq && dd < d.dh)) {
          const auto qvalue = e.let(a.q[e.let(((b * d.qh + h) * d.tq + qrow) * d.dh + dd)]);
          qs[idx] = math::narrow<Narrow>(qvalue);
        }
      }
    }
    if (auto g = e.when(lid < QR)) {
      mrow[lid] = math::neg_inf();
      drow[lid] = e.f32(0.0f);
    }

    const auto wave = e.let(lid / 32u);
    const auto lane = e.let(lid % 32u);
    const auto lane_lo = e.let(lane % 16u);
    const auto lane_hi = e.let(lane / 16u);
    // Rows a wave owns in the row passes, all in one tile.
    const auto wave_tile = e.let(wave * kRowsPerWave / QTile);
    const auto acc_row = [&](std::uint32_t f) {
      return e.let(f * kGeo.slot_step + lane_hi * kGeo.half_rows);
    };
    std::vector<decltype(e.local<kir::f32, kSlots>())> o;
    for (std::uint32_t i = 0; i < kTiles * value_tiles; ++i) {
      o.push_back(e.local<kir::f32, kSlots>());
      for (auto f : e.unroll(8u)) o.back()[f] = e.f32(0.0f);
    }
    const auto position_of = [&](std::uint32_t last_row) {
      const auto last = e.let(select(q0 + last_row < d.tq, q0 + last_row, e.u32(d.tq - 1u)));
      return e.let(kir::cast<std::int64_t>(offset) + kir::cast<std::int64_t>(last));
    };
    const auto last_position0 = position_of(QTile - 1u);
    const auto last_position = position_of(QR - 1u);
    e.barrier();

    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent(
        "kv_len", select(loaded_max < capacity, loaded_max, capacity));
    const auto nwin = e.let((kv_len + e.u32(kKWin - 1u)) / e.u32(kKWin));
    const auto loop_windows = [&] {
      if constexpr (MeanCorrection) return e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
      else return nwin;
    }();
    for (auto w : e.range(loop_windows)) {
      const auto wbase = e.let(w * kKWin);
      const auto signed_start = e.let(kir::cast<std::int64_t>(wbase));
      // Whether the window is one each tile, as its own workgroup, visits.
      const auto useful1 = mask == 0 ? wbase < row_len
                                     : wbase < row_len && signed_start <= last_position;
      const auto useful0 = mask == 0 ? wbase < row_len
                                     : wbase < row_len && signed_start <= last_position0;
      if (auto live_window = e.when(useful1 && w < nwin)) {
        const auto visits0 = e.let(useful0);
        const auto visits1 = e.let(has1);
        const auto sel0 = [&] {
          if constexpr (MeanCorrection) {
            const auto blocks = e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
            return e.let(select(visits0, a.selected[((b * d.qh + h) * ntiles + qt0) * blocks + w],
                                e.f32(0.0f)));
          } else {
            return e.let(select(visits0, e.f32(1.0f), e.f32(0.0f)));
          }
        }();
        const auto sel1 = [&] {
          if constexpr (MeanCorrection) {
            const auto blocks = e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
            const auto qt1 = e.let(select(has1, qt0 + 1u, qt0));
            return e.let(select(visits1, a.selected[((b * d.qh + h) * ntiles + qt1) * blocks + w],
                                e.f32(0.0f)));
          } else {
            return e.let(select(visits1, e.f32(1.0f), e.f32(0.0f)));
          }
        }();
        if (auto exact_block = e.when(sel0 != 0.0f || sel1 != 0.0f)) {
          auto write_score = [&](std::uint32_t tile, const kir::Val<kir::u32>& row,
                                 const kir::Val<kir::u32>& key,
                                 const kir::Val<kir::f32>& value) {
            const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                        kir::cast<std::int64_t>(q0) + kir::cast<std::int64_t>(row));
            const auto signed_key = e.let(kir::cast<std::int64_t>(key));
            const auto slot = e.let(row * kKWin + key - wbase);
            const auto chosen = tile == 0 ? sel0 : sel1;
            sc[slot] = math::neg_inf();
            if (auto live_score = e.when(key < row_len && q0 + row < d.tq && chosen != 0.0f)) {
              if (mask == 0) sc[slot] = value;
              else if (mask == 1) {
                if (auto causal = e.when(signed_key <= position)) sc[slot] = value;
              } else {
                const auto distance = e.let(position - signed_key);
                if (auto sliding = e.when(signed_key <= position && distance < kir::cast<std::int64_t>(e.u32(d.window))))
                  sc[slot] = value;
              }
            }
          };
          for (std::uint32_t column_tile = 0; column_tile < kColTiles; ++column_tile) {
            const auto key = e.let(wbase + (wave + column_tile * kW) * 16u + lane_lo);
            std::vector<decltype(e.local<kir::f32, 8>())> acc;
            for (std::uint32_t t = 0; t < kTiles; ++t) {
              acc.push_back(e.local<kir::f32, 8>());
              for (auto f : e.unroll(8u)) acc.back()[f] = e.f32(0.0f);
            }
            const auto safe_key = e.let(select(key < row_len, key, wbase));
            const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + safe_key / d.ts)]));
            const auto vector = e.let((page * d.kvh + kh) * d.ts + safe_key % d.ts);
            constexpr std::uint32_t kStepsPerBatch = 8;
            const auto kbase = e.let(vector * d.dh);
            const std::uint32_t steps = padded_depth / 16u;
            using KeyFrag = lse::vec<Narrow, 16>;
            for (std::uint32_t first = 0; first < steps; first += kStepsPerBatch) {
              const std::uint32_t last = std::min(steps, first + kStepsPerBatch);
              std::vector<kir::Pack<KvElement<Storage>>> keys;
              for (std::uint32_t step = first; step < last; ++step)
                keys.push_back(k.load_elems<KvElement<Storage>>(
                    a.k.b.id(), e.let(kbase + step * 16u), 16u));
              for (std::uint32_t step = first; step < last; ++step) {
                const kir::Val<KeyFrag> kf(&k.types(), &k.ir(), keys[step - first].id());
                for (std::uint32_t t = 0; t < kTiles; ++t) {
                  const auto qbase = e.let((t * QTile + lane_lo) * padded_depth + step * 16u);
                  const auto qlo = qs.load(qbase, 16u);
                  const auto qhi = qs.load(e.let(qbase + 8u), 16u);
                  auto qa = e.local<Narrow, 16>();
                  for (int j = 0; j < 8; ++j) {
                    qa[j] = qlo[j];
                    qa[8 + j] = qhi[j];
                  }
                  acc[t] = math::mma<Mma>(qa.value(), kf, acc[t].value());
                }
              }
            }
            for (std::uint32_t t = 0; t < kTiles; ++t)
              for (std::uint32_t f = 0; f < 8u; ++f)
                write_score(t, e.let(acc_row(f) + t * QTile), key,
                            e.let(acc[t][static_cast<int>(f)].read() * d.scale));
          }
          e.barrier();

          // Per tile: whether its rows take this block (its vote).
          auto retain0 = e.var(1.0f);
          auto retain1 = e.var(1.0f);
          if (sparse) {
            for (std::uint32_t wr = 0; wr < kRowsPerWave; ++wr) {
              const auto row = e.let(wave * kRowsPerWave + wr);
              auto block_max = e.var(math::neg_inf());
              for (auto f : e.unroll(8u))
                block_max = math::max(block_max.read(), sc[e.let(row * kKWin + lane + f * 32u)].read());
              for (std::uint32_t bit = 16; bit; bit /= 2)
                block_max = math::max(block_max.read(), math::shfl_xor(block_max.read(), e.u32(bit)));
              const auto running = e.let(math::max(mrow[row].read(), block_max.read()));
              const auto length = e.let(select(row_len > 0u, row_len, e.u32(1)));
              const auto raw_lambda = e.let(e.f32(s.attrs[2]) / kir::cast<kir::f32>(length));
              const auto lambda = e.let(select(raw_lambda < 1.0f, raw_lambda, e.f32(1)));
              const auto keep = e.let(block_max.read() != math::neg_inf() &&
                  math::exp(block_max.read() - running) >= lambda);
              if (auto leader = e.when(lane == 0u))
                arow[row] = select(keep, e.f32(1.0f), e.f32(0.0f));
            }
            e.barrier();
            retain0 = e.f32(0);
            retain1 = e.f32(0);
            for (auto row : e.range(QTile))
              if (auto votes = e.when(arow[row].read() != 0.0f)) retain0 = e.f32(1);
            for (auto row : e.range(QTile))
              if (auto votes = e.when(arow[e.let(row + QTile)].read() != 0.0f)) retain1 = e.f32(1);
            e.barrier();
          }
          // A tile's rows take the softmax where the tile would have: its
          // selection (else its scores are all -inf, which changes nothing)
          // and its vote.
          const auto row_retains = e.let(select(wave_tile == 0u, retain0.read(), retain1.read()));
          for (std::uint32_t wr = 0; wr < kRowsPerWave; ++wr) {
            const auto row = e.let(wave * kRowsPerWave + wr);
            if (auto softmax = e.when(row_retains != 0.0f)) {
              auto maximum = e.var(math::neg_inf());
              for (auto f : e.unroll(8u))
                maximum = math::max(maximum.read(), sc[e.let(row * kKWin + lane + f * 32u)].read());
              for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
                maximum = math::max(maximum.read(), math::shfl_xor(maximum.read(), e.u32(shift)));
              const auto old_max = e.let(mrow[row].read());
              const auto new_max = e.let(math::max(old_max, maximum.read()));
              const auto empty = e.let(new_max == math::neg_inf());
              const auto safe_max = e.let(select(empty, e.f32(0.0f), new_max));
              const auto alpha = e.let(select(empty, e.f32(1.0f), math::exp(old_max - safe_max)));
              auto total = e.var(0.0f);
              for (auto f : e.unroll(8u)) {
                const auto index = e.let(row * kKWin + lane + f * 32u);
                const auto probability = e.let(math::exp(sc[index].read() - safe_max));
                total = total.read() + probability;
              }
              for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
                total = total.read() + math::shfl_xor(total.read(), e.u32(shift));
              if (auto leader = e.when(lane == 0u)) {
                mrow[row] = new_max;
                arow[row] = alpha;
                smax[row] = safe_max;
                drow[row] = math::fma(drow[row].read(), alpha, total.read());
              }
            }
          }
          e.barrier();
          const auto any_retains = e.let(retain0.read() != 0.0f || retain1.read() != 0.0f);
          if (auto voted = e.when(any_retains)) {
            for (std::uint32_t t = 0; t < kTiles; ++t) {
              const auto tile_retains = t == 0 ? retain0.read() : retain1.read();
              if (auto rescale = e.when(tile_retains != 0.0f))
                for (std::uint32_t vt = 0; vt < value_tiles; ++vt)
                  for (std::uint32_t f = 0; f < 8u; ++f) {
                    const auto row = e.let(acc_row(f) + t * QTile);
                    auto& acc = o[t * value_tiles + vt];
                    acc[static_cast<int>(f)] = acc[static_cast<int>(f)].read() * arow[row].read();
                  }
            }
          }
          const std::uint32_t pieces_per_key = d.dv / 8u;
          const std::uint32_t pieces = kVsub * pieces_per_key;
          for (std::uint32_t quarter = 0; quarter < kKWin; quarter += kQuarter) {
            // The quarter's probabilities, narrowed: zero for a tile that
            // does not take the block.
            if (auto voted = e.when(any_retains)) {
              for (std::uint32_t c = 0; c < QR * kQuarter / kT; ++c) {
                const auto idx = e.let(lid + c * kT);
                const auto row = e.let(idx / kQuarter);
                const auto key = e.let(idx % kQuarter);
                const auto takes = e.let(select(row < QTile, retain0.read(), retain1.read()));
                const auto probability = e.let(math::exp(
                    sc[e.let(row * kKWin + quarter + key)].read() - smax[row].read()));
                pb[e.let(row * kProbRow + key)] = math::narrow<Narrow>(
                    select(takes != 0.0f, probability, e.f32(0.0f)));
              }
            }
            e.barrier();
            for (std::uint32_t sub = 0; sub < kQuarter; sub += kVsub) {
              if (auto voted = e.when(any_retains)) {
                for (std::uint32_t c = 0; c * kT < pieces; ++c) {
                  const auto piece = e.let(select(lid + c * kT < pieces,
                                                  lid + c * kT, e.u32(pieces - 1u)));
                  const auto slot_key = e.let(piece / pieces_per_key);
                  const auto part = e.let(piece % pieces_per_key);
                  const auto key = e.let(wbase + quarter + sub + slot_key);
                  const auto safe_key = e.let(select(key < row_len, key, wbase));
                  const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + safe_key / d.ts)]));
                  const auto vector = e.let((page * d.kvh + kh) * d.ts + safe_key % d.ts);
                  const auto row_v = e.load(a.v, e.let(vector * d.dv + part * 8u), 16u);
                  k.store_pack<Narrow>(vs.id(), e.let(slot_key * vrow + part * 8u), row_v, 16u);
                }
              }
              e.barrier();
              if (auto voted = e.when(any_retains)) {
                using Frag = lse::vec<Narrow, 16>;
                std::vector<kir::Val<Frag>> pf;
                for (std::uint32_t t = 0; t < kTiles; ++t) {
                  const auto ap = pb.load_elems(e.let((t * QTile + lane_lo) * kProbRow + sub), 16u);
                  pf.push_back(kir::Val<Frag>(&k.types(), &k.ir(), ap.id()));
                }
                for (std::uint32_t vt = 0; vt < value_tiles; ++vt) {
                  const auto dimension = e.let((wave + vt * kW) * 16u + lane_lo);
                  const auto safe_dim = e.let(select(dimension < d.dv, dimension, e.u32(d.dv - 1u)));
                  auto bf = e.local<Narrow, 16>();
                  for (auto f : e.unroll(16u)) bf[f] = vs[e.let(f * vrow + safe_dim)].read();
                  for (std::uint32_t t = 0; t < kTiles; ++t) {
                    auto& acc = o[t * value_tiles + vt];
                    acc = math::mma<Mma>(pf[t], bf.value(), acc.value());
                  }
                }
              }
              e.barrier();
            }
          }
        }  // exact block
        if constexpr (MeanCorrection) {
          const auto blocks = e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
          const auto pool = e.let(((b * d.kvh + kh) * blocks + w) * 513u);
          // Per tile, as its own workgroup would have: a block it visits and
          // did not select takes the pooled correction.
          const auto corrects0 = e.let(visits0 && sel0 == 0.0f);
          const auto corrects1 = e.let(visits1 && sel1 == 0.0f);
          const auto row_corrects = e.let(select(wave_tile == 0u, corrects0, corrects1));
          for (std::uint32_t wr = 0; wr < kRowsPerWave; ++wr) {
            const auto row = e.let(wave * kRowsPerWave + wr);
            if (auto correction = e.when(row_corrects)) {
              auto score = e.var(0.0f);
              if (auto live = e.when(q0 + row < d.tq)) {
                const auto qb = e.let(((b * d.qh + h) * d.tq + q0 + row) * d.dh);
                for (std::uint32_t c = 0; c < 8; ++c)
                  score = math::fma(a.q[qb + lane + c * 32u], a.pooled[pool + lane + c * 32u], score.read());
              }
              for (std::uint32_t bit = 16; bit; bit /= 2)
                score = score.read() + math::shfl_xor(score.read(), e.u32(bit));
              const auto logit = e.let(score.read() * d.scale + e.f32(std::log(256.0f)));
              const auto old = e.let(mrow[row].read());
              const auto updated = e.let(math::max(old, logit));
              const auto alpha = e.let(math::exp(old - updated));
              const auto weight = e.let(math::exp(logit - updated));
              if (auto leader = e.when(lane == 0u)) {
                mrow[row] = updated; arow[row] = alpha;
                sc[row] = weight; drow[row] = math::fma(drow[row].read(), alpha, weight);
              }
            }
          }
          e.barrier();
          for (std::uint32_t t = 0; t < kTiles; ++t) {
            if (auto correction_values = e.when(t == 0 ? corrects0 : corrects1)) {
              for (std::uint32_t vt = 0; vt < value_tiles; ++vt) {
                const auto dimension = e.let((wave + vt * kW) * 16u + lane_lo);
                const auto value = e.let(a.pooled[pool + 256u + dimension]);
                auto& acc = o[t * value_tiles + vt];
                for (std::uint32_t f = 0; f < 8u; ++f) {
                  const auto row = e.let(acc_row(f) + t * QTile);
                  acc[static_cast<int>(f)] = math::fma(sc[row].read(), value,
                      acc[static_cast<int>(f)].read() * arow[row].read());
                }
              }
            }
          }
          e.barrier();
        }
      }
    }

    for (std::uint32_t t = 0; t < kTiles; ++t)
      for (std::uint32_t vt = 0; vt < value_tiles; ++vt) {
        const auto dimension = e.let((wave + vt * kW) * 16u + lane_lo);
        for (std::uint32_t f = 0; f < 8u; ++f) {
          const auto row = e.let(acc_row(f) + t * QTile);
          const auto qrow = e.let(q0 + row);
          const auto den = e.let(drow[row].read());
          const auto inv = e.let(select(den == 0.0f, e.f32(1.0f), den));
          if (auto live_query = e.when(qrow < d.tq && dimension < d.dv))
            e.store(obase + qrow * d.dv + dimension, o[t * value_tiles + vt][static_cast<int>(f)].read() / inv);
        }
      }
    if (!k.lds().ok()) return {};
    return k.str();
    }
  }

  // One body for every wave32 WMMA generation. What differs between them is
  // the row's fragment layout: RDNA4 splits each 16-wide K step across the
  // half-waves (8 operand values per lane, the upper half-wave reading K 8..15)
  // and gives each half-wave a block of 8 accumulator rows; RDNA3/3.5 gives
  // every lane the whole K step (16 values, the half-waves repeating the same
  // rows) and interleaves the half-waves' accumulator rows. kGeo carries that,
  // and the few places that index a lane's slice branch on it at compile time.
  template <kv::CacheDType Storage, math::MatrixTarget G>
  std::string emit_storage(const KernelShapes& s) const {
    constexpr auto operand = Storage == kv::CacheDType::kF16
                                 ? math::MatrixElem::kF16 : math::MatrixElem::kBF16;
    using Mma = math::op::Mma<G, math::MatrixElem::kF32, operand, 16, 16, 16>;
    using Narrow = math::matrix_scalar_t<Mma::kRow.a_elem>;
    constexpr TileGeometry kGeo = geometry_of(Mma::kRow);
    // Operand values per lane per instruction, and accumulator slots.
    constexpr std::uint32_t kFrag = kGeo.lane_k;
    constexpr int kSlots = Mma::kRow.c_len;
    static_assert(Mma::kRow.k == 16 && Mma::kRow.n == 16 && kSlots == 8 &&
                  static_cast<std::uint32_t>(Mma::kRow.a_len) == kFrag &&
                  (kFrag == 8 || kFrag == 16) && Mma::kRow.chained == 1,
                  "the flash tile is written for 16x16x16 wave32 rows with "
                  "eight accumulator slots");
    static_assert(kGeo.split_k == (kFrag == 8),
                  "a split-K row holds half the step, a contiguous one all of it");
    const Dims d = dispatch::flash_dimensions(dense_request(s));
    const auto mask = d.mask == 2 && d.window == 0u ? 1 : d.mask;
    const std::uint32_t ntiles = (d.tq + QTile - 1u) / QTile;
    const std::uint32_t padded_depth = (d.dh + 15u) / 16u * 16u;
    const std::uint32_t qchunks = (QTile * padded_depth + kThreads - 1u) / kThreads;
    const std::uint32_t dpt = (d.dv + kThreads - 1u) / kThreads;
    const std::uint32_t value_tiles = (d.dv + 127u) / 128u;

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    std::conditional_t<MeanCorrection, FlashMeanArgs<env::Emit, Storage>,
                       FlashArgs<env::Emit, Storage>> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto qs = e.lds<Narrow>(QTile * padded_depth);
    // The window's scores, then the window's probabilities in place.
    const auto sc = e.lds<kir::f32>(QTile * kKWin);
    const auto mrow = e.lds<kir::f32>(QTile);
    const auto drow = e.lds<kir::f32>(QTile);
    const auto arow = e.lds<kir::f32>(QTile);
    // Value staging: a block of keys' whole value rows, read coalesced from
    // the cache once and then gathered by every wave from scratch. The block
    // is 32 keys, or 16 when scratch cannot hold 32 (64 measured no faster on
    // gfx1201 and halves the workgroups scratch admits).
    constexpr bool kWideKv =
        Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16;
    const bool stage_values = kWideKv && d.dv % 8u == 0u;
    const std::uint32_t vrow = d.dv + 8u;
    std::uint32_t vsub = 0;
    if (stage_values) {
      const std::uint64_t used = dispatch::attention_shapes::flash_wmma_lds_bytes(d.dh, 0u) + 512u +
          (kGeo.split_k ? 0u : std::uint64_t{QTile} * (kKWin + 8u) * 2u);
      const std::uint64_t budget = workgroup_lds_bytes(s.device);
      // Variant 1 stages 16 keys where 32 fit: less scratch per workgroup,
      // so more of them resident, for twice the staging rounds.
      for (std::uint32_t keys : {32u, 16u})
        if (vsub == 0 && !(s.variant == 1u && keys == 32u) &&
            used + std::uint64_t{keys} * vrow * 2u <= budget) vsub = keys;
    }
    const auto vs = e.lds<Narrow>(stage_values && vsub ? vsub * vrow : 8u);
    // On a contiguous-fragment generation (gfx11: a lane carries all sixteen
    // K of an operand) the window's probabilities are also kept narrowed, in
    // rows padded by 16 bytes so a fragment read's sixteen rows fall on
    // distinct bank groups: a lane's P fragment is then two 16-byte reads
    // instead of sixteen scalar reads and conversions, of the same values.
    constexpr bool kContiguous = !kGeo.split_k;
    constexpr std::uint32_t kProbRow = kKWin + 8u;
    std::optional<kir::Tile<Narrow>> pb;
    if constexpr (kContiguous) pb.emplace(e.lds<Narrow>(QTile * kProbRow));
    // Reuse alpha scratch for the uniform block-retention vote before softmax.
    const bool sparse = s.attrs[3] == 1.0f;

    const auto lid = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    // A split's workgroups are its shares of each tile's windows.
    const std::uint32_t parts = Split ? static_cast<std::uint32_t>(s.output.dim(3)) : 1u;
    const std::uint32_t share = Split ? dispatch::attention_shapes::flash_split_share(
        std::uint64_t{d.stride} * d.ts, parts) : 0u;
    // A split numbers its workgroups share by share, and within a share key
    // head by key head: the tiles of every query head reading one key head's
    // windows are neighbours, so they run together and the share's keys and
    // values come from the cache after the first of them reads them. Ordered
    // tile-major instead, the query heads of one key head were a whole row of
    // shares apart and each read the share from memory: at 64K keys an 8-row
    // tree pass read the cache six times over. Each workgroup computes what
    // it did before; only the order they start in changes.
    //
    // A split pass of at most eight rows also packs the query heads of one
    // key head into the sixteen rows of a tile (split_heads_per_tile): eight
    // rows of two heads, four of three, two of six. Every row is computed by
    // the same instructions on the same operands as in a tile of its own --
    // a matrix product's element depends only on its row and column, the
    // softmax runs per row -- so each record is the value it was; the keys
    // and values of a window are read once per tile for several heads.
    const std::uint32_t pack = split_heads_per_tile(d);
    const std::uint32_t per_key_head = d.group / pack * ntiles;
    const auto tile_wg = [&] {
      if constexpr (Split) return e.let(wg % per_key_head);
      else return wg;
    }();
    const auto part = [&] {
      if constexpr (Split) return e.let((wg / (per_key_head * d.kvh)) % parts);
      else return wg;
    }();
    (void)part;
    const auto qt = e.let(tile_wg % ntiles);
    const auto h = [&] {
      if constexpr (Split)
        return e.let(((wg / per_key_head) % d.kvh) * d.group + tile_wg / ntiles * pack);
      else return e.let((tile_wg / ntiles) % d.qh);
    }();
    const auto b = [&] {
      if constexpr (Split) return e.let(wg / (per_key_head * d.kvh * parts));
      else return e.let(tile_wg / (ntiles * d.qh));
    }();
    const auto kh = e.let(h / d.group);
    const auto q0 = e.let(qt * QTile);
    // Tile row r: its query row, its head, and whether it holds one.
    const auto row_query = [&](const auto& r) {
      if (pack > 1u) return e.let(r % d.tq);
      return e.let(q0 + r);
    };
    const auto row_head = [&](const auto& r) {
      if (pack > 1u) return e.let(h + r / d.tq);
      return e.let(h + 0u);
    };
    const auto row_live = [&](const auto& r) {
      if (pack > 1u) return e.let(r < pack * d.tq);
      return e.let(q0 + r < d.tq);
    };
    // Tile row r's record of this share.
    const auto record = [&](const auto& r) {
      return e.let((((b * d.qh + row_head(r)) * d.tq + row_query(r)) * parts + part) *
                   dispatch::attention_shapes::kSplitRecord);
    };
    const auto obase = e.let(((b * d.qh + h) * d.tq) * d.dv);

    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    // Padded rows write zero without reading a block-table row.
    if (auto pad = e.when(b >= rows)) {
      for (std::uint32_t r = 0; r < QTile; ++r) {
        for (std::uint32_t p = 0; p < dpt; ++p) {
          const auto qrow = e.let(q0 + r);
          const auto dd = e.let(lid + p * kThreads);
          if (auto g = e.when(row_live(e.u32(r)) && dd < d.dv)) {
            if constexpr (Split) {
              // An empty share: zero sum and output, never read for its maximum.
              const auto rec = record(e.u32(r));
              e.store(e.let(rec + 2u + dd), e.f32(0.0f));
              if (auto lead = e.when(dd == 0u)) {
                e.store(rec, e.f32(0.0f));
                e.store(e.let(rec + 1u), e.f32(0.0f));
              }
            } else {
              e.store(obase + qrow * d.dv + dd, e.f32(0.0f));
            }
          }
        }
      }
    }
    (void)e.ret_if(b >= rows);

    const auto mb =
        e.let(e.u32(static_cast<std::uint32_t>(kv::kStepMetaHeader)) +
              b * e.u32(static_cast<std::uint32_t>(kv::kStepMetaPerRow)));
    const auto offset = e.let(kir::cast<kir::u32>(a.meta[mb]));
    // Clamp live metadata to the block-table capacity.
    const auto capacity = e.u32(d.stride * d.ts);
    const auto loaded_len = e.let(kir::cast<kir::u32>(a.meta[mb + 1u]));
    const auto row_len = e.let(select(loaded_len < capacity, loaded_len, capacity));
    const auto tb = e.let(b * d.stride);

    // Stage each query once.
    for (std::uint32_t c = 0; c < qchunks; ++c) {
      const auto idx = e.let(lid + c * kThreads);
      if (auto inb = e.when(idx < QTile * padded_depth)) {
        const auto r = e.let(idx / padded_depth);
        const auto dd = e.let(idx % padded_depth);
        const auto qrow = row_query(r);
        qs[idx] = math::narrow<Narrow>(e.f32(0.0f));
        if (auto g = e.when(row_live(r) && dd < d.dh)) {
          const auto qvalue = e.let(a.q[e.let(((b * d.qh + row_head(r)) * d.tq + qrow) * d.dh + dd)]);
          qs[idx] = math::narrow<Narrow>(qvalue);
        }
      }
    }
    if (auto g = e.when(lid < QTile)) {
      mrow[lid] = math::neg_inf();
      drow[lid] = e.f32(0.0f);
    }

    const auto wave = e.let(lid / 32u);
    const auto lane = e.let(lid % 32u);
    const auto lane_lo = e.let(lane % 16u);
    const auto lane_hi = e.let(lane / 16u);
    // The output row accumulator slot f of this lane holds: block layout
    // f + 8 * half-wave, pair layout 2f + half-wave.
    const auto acc_row = [&](const auto& f) {
      if constexpr (kGeo.slot_step == 1u)
        return e.let(f + lane_hi * kGeo.half_rows);
      else
        return e.let(f * kGeo.slot_step + lane_hi * kGeo.half_rows);
    };
    std::vector<decltype(e.local<kir::f32, kSlots>())> o;
    o.reserve(value_tiles);
    for (std::uint32_t tile = 0; tile < value_tiles; ++tile)
      o.push_back(e.local<kir::f32, kSlots>());
    for (auto& fragment : o)
      for (auto f : e.unroll(8u)) fragment[f] = e.f32(0.0f);
    const auto last_query = e.let(select(q0 + (QTile - 1u) < d.tq,
                                        q0 + (QTile - 1u), e.u32(d.tq - 1u)));
    const auto first_position = e.let(kir::cast<std::int64_t>(offset) +
                                      kir::cast<std::int64_t>(q0));
    const auto last_position = e.let(kir::cast<std::int64_t>(offset) +
                                     kir::cast<std::int64_t>(last_query));
    e.barrier();

    // The window bound is shared by the workgroup.
    const auto loaded_max = e.let(kir::cast<kir::u32>(a.meta[1u]));
    const auto kv_len = e.runtime_extent(
        "kv_len", select(loaded_max < capacity, loaded_max, capacity));
    const auto nwin = e.let((kv_len + e.u32(kKWin - 1u)) / e.u32(kKWin));
    // A static capacity bound keeps pooled address arithmetic provably narrow.
    // Runtime metadata still gates every live window.
    const auto loop_windows = [&] {
      if constexpr (MeanCorrection) return e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
      else return nwin;
    }();
    auto windows = [&] {
      if constexpr (Split) {
        const auto first_window = e.let(part * share);
        return e.range(first_window, e.let(first_window + share), 1u);
      } else {
        return e.range(loop_windows);
      }
    }();
    for (auto w : windows) {
      const auto wbase = e.let(w * kKWin);

      const auto signed_start = e.let(kir::cast<std::int64_t>(wbase));
      const auto useful = mask == 0 ? wbase < row_len
                                   : wbase < row_len && signed_start <= last_position;
      if (auto live_window = e.when(useful && w < nwin)) {
        const auto selected=[&] {
          if constexpr(MeanCorrection) {
            const auto blocks=e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
            return e.let(a.selected[((b*d.qh+h)*ntiles+qt)*blocks+w]);
          } else return e.f32(1);
        }();
        if(auto exact_block=e.when(selected!=0.0f)) {
        auto write_score = [&](const kir::Val<kir::u32>& row,
                               const kir::Val<kir::u32>& key,
                               const kir::Val<kir::f32>& value) {
          const auto query_row = row_query(row);
          const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                      kir::cast<std::int64_t>(query_row));
          const auto signed_key = e.let(kir::cast<std::int64_t>(key));
          const auto slot = e.let(row * kKWin + key - wbase);
          sc[slot] = math::neg_inf();
          if (auto live_score = e.when(key < row_len && row_live(row))) {
            if (mask == 0) sc[slot] = value;
            else if (mask == 1) {
              if (auto causal = e.when(signed_key <= position)) sc[slot] = value;
            } else if (mask == 3) {
              // A tree pass: keys before it are visible, its own rows only
              // where the ancestor mask says so (kv::tree_meta_elems).
              const auto before = e.let(key < offset);
              const auto column = e.let(kir::cast<kir::u32>(select(before, kir::cast<std::int64_t>(e.u32(0)),
                  kir::cast<std::int64_t>(key) - kir::cast<std::int64_t>(offset))));
              const auto seen = e.let(a.meta[e.let(e.u32(static_cast<std::uint32_t>(kv::tree_mask_offset(1))) +
                                                   query_row * d.tq + column)]);
              if (auto visible = e.when(signed_key <= position && (before || seen != 0.0f)))
                sc[slot] = value;
            } else {
              const auto distance = e.let(position - signed_key);
              if (auto sliding = e.when(signed_key <= position && distance < kir::cast<std::int64_t>(e.u32(d.window))))
                sc[slot] = value;
            }
          }
        };
        for (std::uint32_t column_tile = 0; column_tile < 2; ++column_tile) {
          const auto key = e.let(wbase + (wave + column_tile * 8u) * 16u + lane_lo);
          auto acc = e.local<kir::f32, 8>();
          for (auto f : e.unroll(8u)) acc[f] = e.f32(0.0f);
          if (kWideKv && d.dh % 16u == 0) {
            // A key past the row reads the window's first key instead (a live
            // window starts inside the row): its score is replaced by -inf
            // below, so only finiteness matters, and the
            // loads stay unconditional -- all of them issue before the first
            // matrix instruction waits on one.
            const auto safe_key = e.let(select(key < row_len, key, wbase));
            const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + safe_key / d.ts)]));
            const auto vector = e.let((page * d.kvh + kh) * d.ts + safe_key % d.ts);
            if constexpr (kGeo.split_k) {
            const auto kbase = e.let(vector * d.dh + lane_hi * 8u);
            std::vector<kir::Pack<KvElement<Storage>>> keys;
            for (std::uint32_t depth = 0; depth < padded_depth; depth += 16u)
              keys.push_back(e.load(a.k, e.let(kbase + depth), 16u));
            using Frag = lse::vec<Narrow, 8>;
            for (std::uint32_t step = 0; step < keys.size(); ++step) {
              const auto qa = qs.load(e.let(lane_lo * padded_depth + step * 16u + lane_hi * 8u), 16u);
              acc = math::mma<Mma>(kir::Val<Frag>(&k.types(), &k.ir(), qa.id()),
                                   kir::Val<Frag>(&k.types(), &k.ir(), keys[step].id()),
                                   acc.value());
            }
            } else {
            // Every lane takes all sixteen depths of its key and query rows,
            // two 16-byte moves each; the half-waves read the same rows. The
            // key fragment is twice RDNA4's, so the loads issue a batch of
            // steps at a time: eight steps hold as many key registers in
            // flight as RDNA4's whole 256-deep row does.
            if constexpr (kWideKv) {
            constexpr std::uint32_t kStepsPerBatch = 8;
            const auto kbase = e.let(vector * d.dh);
            const std::uint32_t steps = padded_depth / 16u;
            for (std::uint32_t first = 0; first < steps; first += kStepsPerBatch) {
              const std::uint32_t last = std::min(steps, first + kStepsPerBatch);
              // Each key fragment is one sixteen-element load, which the
              // target splits into its two 16-byte loads straight into the
              // operand registers; built a half at a time it cost a bit move
              // per half (4096-token prefill 466.5 -> 474.7 tok/s on the
              // 8060S). The query fragment stays half by half: loaded whole
              // from scratch, the target holds every step's query at once
              // and the kernel spills.
              using KeyFrag = lse::vec<Narrow, 16>;
              std::vector<kir::Pack<KvElement<Storage>>> keys;
              for (std::uint32_t step = first; step < last; ++step)
                keys.push_back(k.load_elems<KvElement<Storage>>(
                    a.k.b.id(), e.let(kbase + step * 16u), 16u));
              for (std::uint32_t step = first; step < last; ++step) {
                const auto qbase = e.let(lane_lo * padded_depth + step * 16u);
                const auto qlo = qs.load(qbase, 16u);
                const auto qhi = qs.load(e.let(qbase + 8u), 16u);
                auto qa = e.local<Narrow, 16>();
                for (int j = 0; j < 8; ++j) {
                  qa[j] = qlo[j];
                  qa[8 + j] = qhi[j];
                }
                acc = math::mma<Mma>(
                    qa.value(),
                    kir::Val<KeyFrag>(&k.types(), &k.ir(), keys[step - first].id()),
                    acc.value());
              }
            }
            }
            }
          } else {
          for (auto depth : e.range(0u, padded_depth, 16u)) {
            auto af = e.local<Narrow, static_cast<int>(kFrag)>();
            auto bf = e.local<Narrow, static_cast<int>(kFrag)>();
            const auto kk = [&] {
              if constexpr (kGeo.split_k) return e.let(depth + lane_hi * 8u);
              else return e.let(depth + 0u);
            }();
            for (auto f : e.unroll(kFrag)) {
              af[f] = qs[e.let(lane_lo * padded_depth + kk + f)].read();
              bf[f] = math::narrow<Narrow>(e.f32(0.0f));
            }
            if (auto key_live = e.when(key < row_len)) {
              const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + key / d.ts)]));
              const auto vector = e.let((page * d.kvh + kh) * d.ts + key % d.ts);
              const auto base = e.let(vector * d.dh + kk);
              if constexpr (Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16) {
                if (d.dh % 16u == 0) {
                  const auto packed = e.load(a.k, base, 16u);
                  for (auto f : e.unroll(8u)) bf[f] = packed[f];
                  if constexpr (kFrag == 16u) {
                    const auto upper = e.load(a.k, e.let(base + 8u), 16u);
                    for (int f = 0; f < 8; ++f) bf[8 + f] = upper[f];
                  }
                } else {
                  for (auto f : e.unroll(kFrag))
                    if (auto tail = e.when(kk + f < d.dh)) bf[f] = a.k[e.let(base + f)];
                }
              } else {
                for (auto f : e.unroll(kFrag)) {
                  auto load_key = [&] {
                    bf[f] = math::narrow<Narrow>(kv_load_vector<Storage>(e, a.k, vector, e.let(kk + f), d.dh));
                  };
                  if (d.dh % 16u == 0) load_key();
                  else if (auto tail = e.when(kk + f < d.dh)) load_key();
                }
              }
            }
            acc = math::mma<Mma>(af.value(), bf.value(), acc.value());
          }
          }
          for (auto f : e.unroll(8u))
            write_score(acc_row(f), key, e.let(acc[f].read() * d.scale));
        }
        e.barrier();

        auto retain = e.var(1.0f);
        if (sparse) {
          // All rows in this query tile must vote to omit it. Invalid query
          // rows and fully masked rows cannot prevent an otherwise safe skip.
          for (std::uint32_t wr = 0; wr < 2; ++wr) {
            const auto row = e.let(wave * 2u + wr);
            auto block_max = e.var(math::neg_inf());
            for (auto f : e.unroll(8u))
              block_max = math::max(block_max.read(), sc[e.let(row * kKWin + lane + f * 32u)].read());
            for (std::uint32_t bit = 16; bit; bit /= 2)
              block_max = math::max(block_max.read(), math::shfl_xor(block_max.read(), e.u32(bit)));
            const auto running = e.let(math::max(mrow[row].read(), block_max.read()));
            const auto length = e.let(select(row_len > 0u, row_len, e.u32(1)));
            const auto raw_lambda = e.let(e.f32(s.attrs[2]) / kir::cast<kir::f32>(length));
            const auto lambda = e.let(select(raw_lambda < 1.0f, raw_lambda, e.f32(1)));
            const auto keep = e.let(block_max.read() != math::neg_inf() &&
                math::exp(block_max.read() - running) >= lambda);
            if (auto leader = e.when(lane == 0u))
              arow[row] = select(keep, e.f32(1.0f), e.f32(0.0f));
          }
          e.barrier();
          retain = e.f32(0);
          for (auto row : e.range(QTile))
            if (auto votes = e.when(arow[row].read() != 0.0f)) retain = e.f32(1);
          // Complete all vote reads before any wave reuses arow for alpha.
          e.barrier();
        }
        // Uniform across the workgroup: every nested barrier is convergent.
        if (auto retained = e.when(retain.read() != 0.0f)) {
        // A wave owns two rows; each lane owns eight contiguous-bank keys.
        for (std::uint32_t wave_row = 0; wave_row < 2; ++wave_row) {
          const auto row = e.let(wave * 2u + wave_row);
          auto maximum = e.var(math::neg_inf());
          for (auto f : e.unroll(8u)) {
            const auto slot = e.let(lane + f * 32u);
            maximum = math::max(maximum.read(), sc[e.let(row * kKWin + slot)].read());
          }
          for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
            maximum = math::max(maximum.read(), math::shfl_xor(maximum.read(), e.u32(shift)));
          const auto old_max = e.let(mrow[row].read());
          const auto new_max = e.let(math::max(old_max, maximum.read()));
          const auto empty = e.let(new_max == math::neg_inf());
          const auto safe_max = e.let(select(empty, e.f32(0.0f), new_max));
          const auto alpha = e.let(select(empty, e.f32(1.0f), math::exp(old_max - safe_max)));
          auto total = e.var(0.0f);
          for (auto f : e.unroll(8u)) {
            const auto slot = e.let(lane + f * 32u);
            const auto index = e.let(row * kKWin + slot);
            const auto probability = e.let(math::exp(sc[index].read() - safe_max));
            sc[index] = probability;
            if constexpr (kContiguous)
              (*pb)[e.let(row * kProbRow + slot)] = math::narrow<Narrow>(probability);
            total = total.read() + probability;
          }
          for (std::uint32_t shift = 16u; shift > 0; shift /= 2u)
            total = total.read() + math::shfl_xor(total.read(), e.u32(shift));
          if (auto leader = e.when(lane == 0u)) {
            mrow[row] = new_max;
            arow[row] = alpha;
            drow[row] = math::fma(drow[row].read(), alpha, total.read());
          }
        }
        }  // retained softmax
        e.barrier();

        // A pruned block's vote is uniform across the workgroup, but the
        // target cannot prove it of a value read from scratch, and the staged
        // value loop's barriers would sit inside that branch. On a contiguous
        // fragment generation the staged loop runs unconditionally with only
        // its work under the vote (the barriers are then outside it); the
        // RDNA4 form keeps its branch, which its target accepts.
        if (sparse && stage_values && vsub != 0 && !kGeo.split_k) {
          const std::uint32_t pieces_per_key = d.dv / 8u;
          const std::uint32_t pieces = vsub * pieces_per_key;
          if (auto voted = e.when(retain.read() != 0.0f)) {
            for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
              for (auto f : e.unroll(8u)) {
                const auto row = acc_row(f);
                o[column_tile][f] = o[column_tile][f].read() * arow[row].read();
              }
            }
          }
          for (auto sub : e.range(0u, kKWin, vsub)) {
            if (auto voted = e.when(retain.read() != 0.0f)) {
            // Every value row this block needs, 16 bytes per thread per
            // piece. A key past the row reads the window's first key: its
            // probability is exactly zero, so only finiteness matters.
            std::vector<kir::Pack<KvElement<Storage>>> rows;
            std::vector<kir::Val<kir::u32>> slots;
            for (std::uint32_t c = 0; c * kThreads < pieces; ++c) {
              const auto piece = e.let(select(lid + c * kThreads < pieces,
                                              lid + c * kThreads, e.u32(pieces - 1u)));
              const auto slot_key = e.let(piece / pieces_per_key);
              const auto part = e.let(piece % pieces_per_key);
              const auto key = e.let(wbase + sub + slot_key);
              const auto safe_key = e.let(select(key < row_len, key, wbase));
              const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + safe_key / d.ts)]));
              const auto vector = e.let((page * d.kvh + kh) * d.ts + safe_key % d.ts);
              rows.push_back(e.load(a.v, e.let(vector * d.dv + part * 8u), 16u));
              slots.push_back(e.let(slot_key * vrow + part * 8u));
            }
            if constexpr (kWideKv)
              for (std::size_t c = 0; c < rows.size(); ++c)
                k.store_pack<Narrow>(vs.id(), slots[c], rows[c], 16u);
            }
            e.barrier();
            if (auto voted = e.when(retain.read() != 0.0f)) {
            for (std::uint32_t tile = 0; tile < vsub; tile += 16u) {
              // This lane's keys of the 16-key step: its half on a split
              // layout, all sixteen on a contiguous one.
              const auto key_half = [&] {
                if constexpr (kGeo.split_k) return e.let(lane_hi * 8u);
                else return e.u32(0);
              }();
              using Frag = lse::vec<Narrow, static_cast<int>(kFrag)>;
              auto af = e.local<Narrow, static_cast<int>(kFrag)>();
              if constexpr (kContiguous) {
                const auto ap = pb->load_elems(e.let(lane_lo * kProbRow + sub + tile), kFrag);
                af = kir::Val<Frag>(&k.types(), &k.ir(), ap.id());
              } else {
                for (auto f : e.unroll(kFrag))
                  af[f] = math::narrow<Narrow>(
                      sc[e.let(lane_lo * kKWin + sub + tile + key_half + f)].read());
              }
              for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
                const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
                const auto safe_dim = e.let(select(dimension < d.dv, dimension, e.u32(d.dv - 1u)));
                auto bf = e.local<Narrow, static_cast<int>(kFrag)>();
                for (auto f : e.unroll(kFrag)) {
                  if constexpr (kGeo.split_k)
                    bf[f] = vs[e.let((tile + key_half + f) * vrow + safe_dim)].read();
                  else
                    bf[f] = vs[e.let((tile + f) * vrow + safe_dim)].read();
                }
                o[column_tile] = math::mma<Mma>(af.value(), bf.value(), o[column_tile].value());
              }
            }
            }
            e.barrier();
          }
        } else
        if (auto retained_values = e.when(retain.read() != 0.0f)) {
        if (stage_values && vsub != 0) {
          for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
            for (auto f : e.unroll(8u)) {
              const auto row = acc_row(f);
              o[column_tile][f] = o[column_tile][f].read() * arow[row].read();
            }
          }
          const std::uint32_t pieces_per_key = d.dv / 8u;
          const std::uint32_t pieces = vsub * pieces_per_key;
          for (auto sub : e.range(0u, kKWin, vsub)) {
            // Every value row this block needs, 16 bytes per thread per
            // piece. A key past the row reads the window's first key: its
            // probability is exactly zero, so only finiteness matters.
            std::vector<kir::Pack<KvElement<Storage>>> rows;
            std::vector<kir::Val<kir::u32>> slots;
            for (std::uint32_t c = 0; c * kThreads < pieces; ++c) {
              const auto piece = e.let(select(lid + c * kThreads < pieces,
                                              lid + c * kThreads, e.u32(pieces - 1u)));
              const auto slot_key = e.let(piece / pieces_per_key);
              const auto part = e.let(piece % pieces_per_key);
              const auto key = e.let(wbase + sub + slot_key);
              const auto safe_key = e.let(select(key < row_len, key, wbase));
              const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + safe_key / d.ts)]));
              const auto vector = e.let((page * d.kvh + kh) * d.ts + safe_key % d.ts);
              rows.push_back(e.load(a.v, e.let(vector * d.dv + part * 8u), 16u));
              slots.push_back(e.let(slot_key * vrow + part * 8u));
            }
            if constexpr (kWideKv)
              for (std::size_t c = 0; c < rows.size(); ++c)
                k.store_pack<Narrow>(vs.id(), slots[c], rows[c], 16u);
            e.barrier();
            for (std::uint32_t tile = 0; tile < vsub; tile += 16u) {
              // This lane's keys of the 16-key step: its half on a split
              // layout, all sixteen on a contiguous one.
              const auto key_half = [&] {
                if constexpr (kGeo.split_k) return e.let(lane_hi * 8u);
                else return e.u32(0);
              }();
              using Frag = lse::vec<Narrow, static_cast<int>(kFrag)>;
              auto af = e.local<Narrow, static_cast<int>(kFrag)>();
              if constexpr (kContiguous) {
                const auto ap = pb->load_elems(e.let(lane_lo * kProbRow + sub + tile), kFrag);
                af = kir::Val<Frag>(&k.types(), &k.ir(), ap.id());
              } else {
                for (auto f : e.unroll(kFrag))
                  af[f] = math::narrow<Narrow>(
                      sc[e.let(lane_lo * kKWin + sub + tile + key_half + f)].read());
              }
              for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
                const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
                const auto safe_dim = e.let(select(dimension < d.dv, dimension, e.u32(d.dv - 1u)));
                auto bf = e.local<Narrow, static_cast<int>(kFrag)>();
                for (auto f : e.unroll(kFrag)) {
                  if constexpr (kGeo.split_k)
                    bf[f] = vs[e.let((tile + key_half + f) * vrow + safe_dim)].read();
                  else
                    bf[f] = vs[e.let((tile + f) * vrow + safe_dim)].read();
                }
                o[column_tile] = math::mma<Mma>(af.value(), bf.value(), o[column_tile].value());
              }
            }
            e.barrier();
          }
        } else {
        // A wave owns one D16 tile per 128-channel slice; C remains FP32.
        for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
          const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
          for (auto f : e.unroll(8u)) {
            const auto row = acc_row(f);
            o[column_tile][f] = o[column_tile][f].read() * arow[row].read();
          }
          for (auto tile : e.range(0u, kKWin, 16u)) {
            auto af = e.local<Narrow, static_cast<int>(kFrag)>();
            auto bf = e.local<Narrow, static_cast<int>(kFrag)>();
            const auto key_half = [&] {
              if constexpr (kGeo.split_k) return e.let(lane_hi * 8u);
              else return e.u32(0);
            }();
            for (auto f : e.unroll(kFrag)) {
              const auto key_slot = [&] {
                if constexpr (kGeo.split_k) return e.let(tile + key_half + f);
                else return e.let(tile + f);
              }();
              const auto key = e.let(wbase + key_slot);
              af[f] = math::narrow<Narrow>(sc[e.let(lane_lo * kKWin + key_slot)].read());
              bf[f] = math::narrow<Narrow>(e.f32(0.0f));
              auto load_value = [&] {
                const auto page = e.let(kir::cast<kir::u32>(a.table[e.let(tb + key / d.ts)]));
                const auto vector = e.let((page * d.kvh + kh) * d.ts + key % d.ts);
                if constexpr (Storage == kv::CacheDType::kF16 || Storage == kv::CacheDType::kBF16)
                  bf[f] = a.v[e.let(vector * d.dv + dimension)];
                else
                  bf[f] = math::narrow<Narrow>(kv_load_vector<Storage>(e, a.v, vector, dimension, d.dv));
              };
              if (auto live_key = e.when(key < row_len && dimension < d.dv)) {
                if (mask == 0) load_value();
                else {
                  const auto signed_key = e.let(kir::cast<std::int64_t>(key));
                  if (mask == 1 || mask == 3) {
                    if (auto used = e.when(signed_key <= last_position)) load_value();
                  } else {
                    if (auto used = e.when(signed_key <= last_position &&
                          first_position - signed_key < kir::cast<std::int64_t>(e.u32(d.window))))
                      load_value();
                  }
                }
              }
            }
            o[column_tile] = math::mma<Mma>(af.value(), bf.value(), o[column_tile].value());
          }
        }
        }  // staged values
        }  // retained block
        e.barrier();
        }  // exact block
        if constexpr(MeanCorrection) {
          const auto blocks=e.u32(static_cast<std::uint32_t>(s.inputs[5].dim(2)));
          const auto pool=e.let(((b*d.kvh+kh)*blocks+w)*513u);
          if(auto correction=e.when(selected==0.0f)) {
            // Selector only omits complete blocks visible to every query.
            // log(256) accounts for their full mass in both softmax sums.
            for(std::uint32_t wr=0;wr<2;++wr) {
              const auto row=e.let(wave*2u+wr);
              auto score=e.var(0.0f);
              if(auto live=e.when(q0+row<d.tq)) {
                const auto qb=e.let(((b*d.qh+h)*d.tq+q0+row)*d.dh);
                for(std::uint32_t c=0;c<8;++c)
                  score=math::fma(a.q[qb+lane+c*32u],a.pooled[pool+lane+c*32u],score.read());
              }
              for(std::uint32_t bit=16;bit;bit/=2)
                score=score.read()+math::shfl_xor(score.read(),e.u32(bit));
              const auto logit=e.let(score.read()*d.scale+e.f32(std::log(256.0f)));
              const auto old=e.let(mrow[row].read());
              const auto updated=e.let(math::max(old,logit));
              const auto alpha=e.let(math::exp(old-updated));
              const auto weight=e.let(math::exp(logit-updated));
              if(auto leader=e.when(lane==0u)) {
                mrow[row]=updated; arow[row]=alpha;
                sc[row]=weight; drow[row]=math::fma(drow[row].read(),alpha,weight);
              }
            }
          }
          e.barrier();
          if(auto correction_values=e.when(selected==0.0f)) {
            for(std::uint32_t column_tile=0;column_tile<value_tiles;++column_tile) {
              const auto dimension=e.let((wave+column_tile*8u)*16u+lane_lo);
              const auto value=e.let(a.pooled[pool+256u+dimension]);
              for(auto f:e.unroll(8u)) {
                const auto row=acc_row(f);
                o[column_tile][f]=math::fma(sc[row].read(),value,
                    o[column_tile][f].read()*arow[row].read());
              }
            }
          }
          e.barrier();
        }
      }
    }

    if constexpr (Split) {
      for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
        const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
        for (auto f : e.unroll(8u)) {
          const auto row = acc_row(f);
          if (auto live_query = e.when(row_live(row) && dimension < d.dv))
            e.store(e.let(record(row) + 2u + dimension), o[column_tile][f].read());
        }
      }
      if (auto lead = e.when(lid < QTile && row_live(lid))) {
        const auto rec = record(lid);
        e.store(rec, mrow[lid].read());
        e.store(e.let(rec + 1u), drow[lid].read());
      }
    } else {
    for (std::uint32_t column_tile = 0; column_tile < value_tiles; ++column_tile) {
      const auto dimension = e.let((wave + column_tile * 8u) * 16u + lane_lo);
      for (auto f : e.unroll(8u)) {
        const auto row = acc_row(f);
        const auto qrow = e.let(q0 + row);
        const auto den = e.let(drow[row].read());
        const auto inv = e.let(select(den == 0.0f, e.f32(1.0f), den));
        if (auto live_query = e.when(qrow < d.tq && dimension < d.dv))
          e.store(obase + qrow * d.dv + dimension, o[column_tile][f].read() / inv);
      }
    }
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape>) const override {
    // The attention builder knows the packed storage tag and logical V width.
    return LSE_ERROR(kInvalidArgument, "WMMA attention is selected through sdpa_paged");
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const Dims d = dispatch::flash_dimensions(dense_request(s));
    const std::uint32_t ntiles = d.valid ? (d.tq + QTile - 1u) / QTile : 1u;
    const std::uint32_t parts =
        Split && s.output.rank() == 5 ? static_cast<std::uint32_t>(s.output.dim(3)) : 1u;
    // Variant 2 takes two query tiles per workgroup, on sixteen waves.
    const std::uint32_t tiles = s.variant == 2u ? (ntiles + 1u) / 2u : ntiles;
    tp.workgroup_size[0] = s.variant == 2u ? 2u * kThreads : kThreads;
    tp.workgroup_count[0] = d.valid ? d.bsz * (d.qh / split_heads_per_tile(d)) * tiles * parts : 1u;
    tp.workgroup_count[1] = 1;
    tp.workgroup_count[2] = 1;
    return tp;
  }
};
using FlashWmma=FlashWmmaImpl<false>;
using FlashMean=FlashWmmaImpl<true>;
using FlashSplit=FlashWmmaImpl<false, true>;
LSE_REGISTER_PRIMITIVE(FlashWmma);
LSE_REGISTER_PRIMITIVE(FlashMean);
LSE_REGISTER_PRIMITIVE(FlashSplit);
const FlashWmma kFlashWmma{};
}  // namespace

const KernelPrimitiveBase* flash_wmma_sdpa() { return &kFlashWmma; }
}  // namespace lse::kernels
