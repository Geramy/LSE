// Typed paged KV; FP32 matrix accumulators and online softmax state.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>
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
template <bool MeanCorrection>
struct FlashWmmaImpl final : KernelPrimitive<FlashWmmaImpl<MeanCorrection>> {
  static constexpr std::string_view kName = MeanCorrection ? "attention.flashprefill.wmma.v1" : "attention.flash.wmma16.v3";
  static constexpr std::string_view kEntry = MeanCorrection ? "lse_flashprefill_wmma_v1" : "lse_flash_wmma16_v3";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return MeanCorrection ? 7 : 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  static KernelShapes dense_request(const KernelShapes& s) {
    auto dense=s;
    if constexpr (MeanCorrection) {
      if(s.inputs.size()>=5) dense.inputs=s.inputs.first(5);
      if(s.input_dtypes.size()>=5) dense.input_dtypes=s.input_dtypes.first(5);
    }
    return dense;
  }
  std::string emit_kernel(const KernelShapes& s) const override {
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
                  return emit_storage<Storage, G>(s);
                }
              });
        });
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
      const std::uint64_t used = dispatch::attention_shapes::flash_wmma_lds_bytes(d.dh, 0u) + 512u;
      const std::uint64_t budget = workgroup_lds_bytes(s.device);
      for (std::uint32_t keys : {32u, 16u})
        if (vsub == 0 && used + std::uint64_t{keys} * vrow * 2u <= budget) vsub = keys;
    }
    const auto vs = e.lds<Narrow>(stage_values && vsub ? vsub * vrow : 8u);
    // Reuse alpha scratch for the uniform block-retention vote before softmax.
    const bool sparse = s.attrs[3] == 1.0f;

    const auto lid = e.let(math::local_id());
    const auto wg = e.let(math::workgroup_id_x());
    const auto qt = e.let(wg % ntiles);
    const auto h = e.let((wg / ntiles) % d.qh);
    const auto b = e.let(wg / (ntiles * d.qh));
    const auto kh = e.let(h / d.group);
    const auto q0 = e.let(qt * QTile);
    const auto obase = e.let(((b * d.qh + h) * d.tq) * d.dv);

    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    // Padded rows write zero without reading a block-table row.
    if (auto pad = e.when(b >= rows)) {
      for (std::uint32_t r = 0; r < QTile; ++r) {
        for (std::uint32_t p = 0; p < dpt; ++p) {
          const auto qrow = e.let(q0 + r);
          const auto dd = e.let(lid + p * kThreads);
          if (auto g = e.when(qrow < d.tq && dd < d.dv)) {
            e.store(obase + qrow * d.dv + dd, e.f32(0.0f));
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
        const auto qrow = e.let(q0 + r);
        qs[idx] = math::narrow<Narrow>(e.f32(0.0f));
        if (auto g = e.when(qrow < d.tq && dd < d.dh)) {
          const auto qvalue = e.let(a.q[e.let(((b * d.qh + h) * d.tq + qrow) * d.dh + dd)]);
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
    for (auto w : e.range(loop_windows)) {
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
          const auto position = e.let(kir::cast<std::int64_t>(offset) +
                                      kir::cast<std::int64_t>(q0) + kir::cast<std::int64_t>(row));
          const auto signed_key = e.let(kir::cast<std::int64_t>(key));
          const auto slot = e.let(row * kKWin + key - wbase);
          sc[slot] = math::neg_inf();
          if (auto live_score = e.when(key < row_len && q0 + row < d.tq)) {
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
              auto af = e.local<Narrow, static_cast<int>(kFrag)>();
              for (auto f : e.unroll(kFrag)) {
                if constexpr (kGeo.split_k)
                  af[f] = math::narrow<Narrow>(
                      sc[e.let(lane_lo * kKWin + sub + tile + key_half + f)].read());
                else
                  af[f] = math::narrow<Narrow>(
                      sc[e.let(lane_lo * kKWin + sub + tile + f)].read());
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
                  if (mask == 1) {
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
    tp.workgroup_size[0] = kThreads;
    tp.workgroup_count[0] = d.valid ? d.bsz * d.qh * ntiles : 1u;
    tp.workgroup_count[1] = 1;
    tp.workgroup_count[2] = 1;
    return tp;
  }
};
using FlashWmma=FlashWmmaImpl<false>;
using FlashMean=FlashWmmaImpl<true>;
LSE_REGISTER_PRIMITIVE(FlashWmma);
LSE_REGISTER_PRIMITIVE(FlashMean);
const FlashWmma kFlashWmma{};
}  // namespace

const KernelPrimitiveBase* flash_wmma_sdpa() { return &kFlashWmma; }
}  // namespace lse::kernels
