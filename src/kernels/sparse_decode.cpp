// Block-sparse decode attention: the page summaries, their scores against a
// pass's query rows, the selection of the pages to read, and the shifted step
// descriptor. The attention itself is the exact split/flash kernel run over
// the selected pages' block table (graph::sparse_decode_attention).
//
// Summaries are per 16-token page and key head: the center and radius of the
// page's keys per channel, bf16, so a query row's bound on any key of the
// page is q . center + |q| . radius (Quest's max(q*min, q*max) per channel).
#include "lse/dispatch/sparse_decode_tuneconfig.h"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/kv_storage.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/kv/block.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;
namespace sd = dispatch::sparse_decode;

namespace {

constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kPositionBound = sd::kMaxPages * sd::kPage * 2u;
using Summary = lse::bf16;

bool positive_u32_shape(const Shape& shape) {
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
    if (shape.dim(axis) <= 0 || static_cast<std::uint64_t>(shape.dim(axis)) > UINT32_MAX / count)
      return false;
    count *= static_cast<std::uint64_t>(shape.dim(axis));
  }
  return shape.rank() > 0;
}

// a - b, or 0 when b > a. A u32 subtraction whose wrap could be observed has
// no Loom form, and a wide signed difference makes every index after it
// wide; positions, pages and lanes are far below 2^24, where f32 is exact,
// and a position read from the step descriptor is an f32 anyway.
kir::Val<kir::u32> minus(env::Emit& e, const kir::Val<kir::u32>& a, const kir::Val<kir::u32>& b) {
  const auto d = e.let(kir::cast<kir::f32>(a) - kir::cast<kir::f32>(b));
  // Bounded, so Loom keeps every index derived from it narrow.
  return e.let(math::min(kir::cast<kir::u32>(e.let(math::max(d, e.f32(0.0f)))),
                         e.u32(kPositionBound)));
}
kir::Val<kir::u32> minus(env::Emit& e, const kir::Val<kir::u32>& a, std::uint32_t b) {
  return minus(e, a, e.u32(b));
}

// A row's live length and page count from the step descriptor.
struct RowMeta {
  kir::Val<kir::u32> first, live, pages;
};
RowMeta row_meta(env::Emit& e, const env::In<kir::f32, env::Emit>& meta,
                 const kir::Val<kir::u32>& b) {
  const auto mb = e.let(e.u32(kv::kStepMetaHeader) + b * e.u32(kv::kStepMetaPerRow));
  // Clamped like every paged kernel's reads, which keeps derived indices
  // narrow; a real position is far below the bound.
  const auto first = e.let(math::min(kir::cast<kir::u32>(meta[mb]), e.u32(kPositionBound)));
  const auto live = e.let(math::min(kir::cast<kir::u32>(meta[mb + 1u]), e.u32(kPositionBound)));
  return {first, live, e.let((live + (sd::kPage - 1u)) / sd::kPage)};
}

// ---------------------------------------------------------------------------
// Summary refresh: in place, the pages from kRefreshSlackTokens before the
// pass's first row to its last live position, from the keys as written.
template <class E, kv::CacheDType Storage>
struct SummaryArgs {
  env::In<Summary, E> summary;  // in place
  env::In<KvElement<Storage>, E> keys;
  env::In<KvElement<Storage>, E> values;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<Summary, E> out;
};

struct SparseDecodeSummary final : KernelPrimitive<SparseDecodeSummary> {
  static constexpr std::string_view kName = "attention.sparse_decode.summary.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_summary_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }

  static bool valid(const KernelShapes& s) {
    if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 || s.input_dtypes[0] != DType::kBF16 ||
        s.output_dtype != DType::kBF16 || s.input_dtypes[1] != s.input_dtypes[2] ||
        s.input_dtypes[3] != DType::kF32 || s.input_dtypes[4] != DType::kF32 ||
        s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 || s.inputs[2].rank() != 4 ||
        s.inputs[4].rank() != 2 || s.output != s.inputs[0] || s.iattrs[0] <= 0)
      return false;
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return false;
    const auto& sum = s.inputs[0];
    const auto& keys = s.inputs[1];
    const auto& values = s.inputs[2];
    const bool row_major_values = values == keys;
    const bool dim_major_values = values.dim(0) == keys.dim(0) && values.dim(1) == keys.dim(1) &&
                                  values.dim(2) == kWidth && values.dim(3) == sd::kPage;
    return sum.dim(0) >= keys.dim(0) && sum.dim(1) == keys.dim(1) && sum.dim(2) == sd::kSummaryRows &&
           sum.dim(3) == kWidth && keys.dim(2) == sd::kPage && keys.dim(3) == kWidth &&
           (row_major_values || dim_major_values);
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!valid(s) || !s.store || !s.intrinsics || !s.types.scalar) return {};
    const auto storage = kv::cache_dtype(s.input_dtypes[1], s.attrs[1]);
    if (storage != kv::CacheDType::kBF16 && storage != kv::CacheDType::kF16) return {};
    return with_kv_storage(s.input_dtypes[1], s.attrs[1], [&]<kv::CacheDType Storage>() {
      if constexpr (Storage != kv::CacheDType::kBF16 && Storage != kv::CacheDType::kF16)
        return std::string{};
      else return emit_storage<Storage>(s);
    });
  }

  template <kv::CacheDType Storage>
  std::string emit_storage(const KernelShapes& s) const {
    const auto kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
    const bool dim_major = s.inputs[2] != s.inputs[1];
    const auto window = sd::refresh_pages(static_cast<std::uint32_t>(s.iattrs[0]));
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    SummaryArgs<env::Emit, Storage> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto i = e.let(wg % window);
    const auto kh = e.let((wg / window) % kvh);
    const auto b = e.let(wg / (window * kvh));
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(b >= rows);
    const auto m = row_meta(e, a.meta, b);
    (void)e.ret_if(m.live == 0u);
    const auto start = e.let(minus(e, m.first, sd::kRefreshSlackTokens) / sd::kPage);
    const auto page = e.let(start + i);
    (void)e.ret_if(page >= m.pages || page >= stride);
    const auto blk = e.let(kv_block_index<Storage>(a.table[b * stride + page]));
    auto lo = e.var(3.0e38f), hi = e.var(-3.0e38f), ksum = e.var(0.0f), vsum = e.var(0.0f),
         count = e.var(0.0f);
    for (auto slot : e.range(sd::kPage)) {
      if (auto live = e.when(page * sd::kPage + slot < m.live)) {
        const auto x = e.let(kv_load<Storage>(
            e, a.keys, e.let(((blk * kvh + kh) * sd::kPage + slot) * kWidth + lane), kWidth));
        const auto v = e.let(dim_major
            ? math::widen(a.values[e.let(((blk * kvh + kh) * kWidth + lane) * sd::kPage + slot)])
            : kv_load<Storage>(e, a.values, e.let(((blk * kvh + kh) * sd::kPage + slot) * kWidth + lane), kWidth));
        lo = math::min(lo.read(), x);
        hi = math::max(hi.read(), x);
        ksum = ksum.read() + x;
        vsum = vsum.read() + v;
        count = count.read() + 1.0f;
      }
    }
    const auto base = e.let((blk * kvh + kh) * sd::kSummaryRows * kWidth + lane);
    const auto inverse = e.let(1.0f / math::max(count.read(), e.f32(1.0f)));
    a.out[e.let(base + sd::kRowCenter * kWidth)] = kir::cast<Summary>(e.let((lo.read() + hi.read()) * 0.5f));
    a.out[e.let(base + sd::kRowRadius * kWidth)] = kir::cast<Summary>(e.let((hi.read() - lo.read()) * 0.5f));
    a.out[e.let(base + sd::kRowKeyMean * kWidth)] = kir::cast<Summary>(e.let(ksum.read() * inverse));
    a.out[e.let(base + sd::kRowValueMean * kWidth)] = kir::cast<Summary>(e.let(vsum.read() * inverse));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5) return LSE_ERROR(kInvalidArgument, "sparse decode summary takes 5 inputs");
    return in[0];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kBF16; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = kWidth;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.inputs[4].dim(0) * s.inputs[1].dim(1)) *
                           sd::refresh_pages(static_cast<std::uint32_t>(s.iattrs[0]));
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeSummary);

// ---------------------------------------------------------------------------
// Scores: for each far page of each sequence and key head, the largest bound
// over the key head's query heads and the pass's rows, relative to each row's
// bound over the newest pages, in logit units. One thread per page.
template <class E>
struct ScoreArgs {
  env::In<kir::f32, E> q;
  env::In<Summary, E> summary;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

struct SparseDecodeScore final : KernelPrimitive<SparseDecodeScore> {
  static constexpr std::string_view kName = "attention.sparse_decode.score.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_score_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  static std::uint32_t chunk_rows(const KernelShapes& s) {
    const auto rows = static_cast<std::uint32_t>(s.inputs[0].dim(1) / s.inputs[1].dim(1) *
                                                 s.inputs[0].dim(2));
    return rows < sd::kScoreRowChunk ? rows : sd::kScoreRowChunk;
  }
  static std::uint32_t tiles(const KernelShapes& s) {
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    return (stride + sd::kScorePages - 1u) / sd::kScorePages;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 || s.input_dtypes[0] != DType::kF32 ||
        s.input_dtypes[1] != DType::kBF16 || s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[3].rank() != 2 || s.inputs[0].dim(3) != kWidth || s.inputs[1].dim(2) != sd::kSummaryRows ||
        s.inputs[1].dim(3) != kWidth || s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
        !s.device || !s.store || !s.intrinsics || !s.types.scalar ||
        s.inputs[3].dim(1) > static_cast<std::int64_t>(sd::kMaxPages) ||
        s.output != Shape{s.inputs[0].dim(0) * s.inputs[1].dim(1), s.inputs[3].dim(1)} ||
        s.iattrs[0] <= static_cast<std::int32_t>(sd::kSinkPages + sd::kRecentPages) ||
        !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f || !std::isfinite(s.attrs[1]) || s.attrs[1] < 0.0f)
      return {};
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return {};
    if (s.device->wavefront_size != 32 || s.intrinsics->find("wave.shfl_xor").empty()) return {};
    // A wave per page: each lane holds eight channels of the page's center
    // and radius (coalesced reads), the query rows come from LDS, and a
    // butterfly sums each row's bound.
    constexpr std::uint32_t kThreads = sd::kScoreThreads, kPagesPer = sd::kScorePages;
    constexpr std::uint32_t kWaves = kThreads / 32u;
    static_assert(kWaves == sd::kReferencePages && kPagesPer % kWaves == 0);
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto queries = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto group = heads / kvh;
    const auto total_rows = group * queries;
    const auto rc = chunk_rows(s);
    const auto chunks = (total_rows + rc - 1u) / rc;
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    const auto ntiles = tiles(s);
    const auto budget = static_cast<std::uint32_t>(s.iattrs[0]);
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    ScoreArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto qs = e.lds<kir::f32>(rc * kWidth);
    const auto refs = e.lds<kir::f32>(kWaves * rc);
    const auto ref = e.lds<kir::f32>(rc);
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto wl = e.let(lane % 32u), wave = e.let(lane / 32u);
    const auto tile = e.let(wg % ntiles);
    const auto kh = e.let((wg / ntiles) % kvh);
    const auto b = e.let(wg / (ntiles * kvh));
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(b >= rows);
    const auto m = row_meta(e, a.meta, b);
    // Nothing to score while every page is read (selection copies the table)
    // or for a tile wholly outside the far range; uniform per workgroup.
    const auto far_end = minus(e, m.pages, sd::kRecentPages);
    const auto tile_begin = e.let(tile * kPagesPer);
    (void)e.ret_if(m.pages <= budget || tile_begin >= far_end ||
                   tile_begin + kPagesPer <= sd::kSinkPages);
    const auto channel = e.let(wl * 8u);
    // Few rows (a decode step's query heads): a wave per page with the
    // channels over the lanes. Many (verify passes): a lane per row.
    const bool lanes_rows = total_rows >= 16u;
    // Row rr of the key head's rows: query head kh*group + rr/queries, row rr%queries.
    const auto q_index = [&](const kir::Val<kir::u32>& rr, const kir::Val<kir::u32>& c) {
      return e.let(((b * heads + kh * group + rr / queries) * queries + rr % queries) * kWidth + c);
    };
    // This lane's eight center and radius values of a page.
    struct Slice { std::vector<kir::Val<kir::f32>> center, radius; };
    const auto slice_of = [&](const kir::Val<kir::u32>& page) {
      const auto blk = e.let(kir::cast<kir::u32>(a.table[b * stride + page]));
      const auto sb = e.let((blk * kvh + kh) * sd::kSummaryRows * kWidth + channel);
      const auto c = e.load(a.summary, sb, 16u);
      const auto r = e.load(a.summary, e.let(sb + kWidth), 16u);
      Slice out;
      for (int j = 0; j < 8; ++j) {
        out.center.push_back(e.let(math::widen(c[j])));
        out.radius.push_back(e.let(math::widen(r[j])));
      }
      return out;
    };
    // Row r's bound on the page, summed over the wave (every lane holds it).
    const auto bound = [&](const Slice& sl, const kir::Val<kir::u32>& r) {
      const auto at = e.let(r * kWidth + channel);
      const auto lo = qs.load(at, 16u);
      const auto hi = qs.load(e.let(at + 4u), 16u);
      auto u = e.var(0.0f);
      for (int j = 0; j < 8; ++j) {
        const auto qv = e.let(j < 4 ? lo[j] : hi[j - 4]);
        u = math::fma(qv, sl.center[static_cast<std::size_t>(j)], u.read());
        u = math::fma(e.let(math::abs(qv) * s.attrs[1]), sl.radius[static_cast<std::size_t>(j)], u.read());
      }
      for (std::uint32_t bit = 16; bit; bit /= 2) u = u.read() + math::shfl_xor(u.read(), e.u32(bit));
      return e.let(u.read());
    };
    std::vector<kir::LValue<kir::f32>> best;
    for (std::uint32_t i = 0; i < kPagesPer / kWaves; ++i) best.push_back(e.var(math::neg_inf()));
    for (auto chunk : e.range(chunks)) {
      const auto r0 = e.let(chunk * rc);
      e.barrier();
      for (auto idx : e.range(lane, e.u32(rc * kWidth), kThreads)) {
        const auto rr = e.let(r0 + idx / kWidth);
        auto value = e.var(0.0f);
        if (auto live = e.when(rr < total_rows)) value = a.q[q_index(rr, e.let(idx % kWidth))];
        qs[idx] = value.read();
      }
      e.barrier();
      // Reference bounds: wave w takes the w-th newest page.
      {
        const auto sl = slice_of(minus(e, minus(e, m.pages, 1u), wave));
        for (auto r : e.range(rc)) {
          const auto u = bound(sl, r);
          if (auto lead = e.when(wl == 0u)) refs[e.let(wave * rc + r)] = u;
        }
      }
      e.barrier();
      if (auto owner = e.when(lane < rc)) {
        auto top = e.var(math::neg_inf());
        for (std::uint32_t w = 0; w < kWaves; ++w)
          top = math::max(top.read(), refs[e.let(e.u32(w * rc) + lane)].read());
        ref[lane] = top.read();
      }
      e.barrier();
      for (std::uint32_t i = 0; i < kPagesPer / kWaves; ++i) {
        const auto page = e.let(tile_begin + wave + i * kWaves);
        const auto mine = e.let(page >= sd::kSinkPages && page < far_end && page < stride);
        if (auto live = e.when(mine)) {
          if (!lanes_rows) {
            const auto sl = slice_of(page);
            for (auto r : e.range(rc))
              if (auto row = e.when(r0 + r < total_rows))
                best[i] = math::max(best[i].read(), bound(sl, r) - ref[r].read());
          } else {
            // Many rows: a lane per row, the page's summary read by the
            // whole wave at once (one address), no per-row reduction.
            const auto blk = e.let(kir::cast<kir::u32>(a.table[b * stride + page]));
            const auto sb = e.let((blk * kvh + kh) * sd::kSummaryRows * kWidth);
            for (std::uint32_t pass = 0; pass < (rc + 31u) / 32u; ++pass) {
              const auto rr = e.let(wl + pass * 32u);
              const auto at = e.let(select(rr < rc, rr, e.u32(0)) * kWidth);
              auto even = e.var(0.0f), odd = e.var(0.0f);
              for (auto c8 : e.range(kWidth / 8u)) {
                const auto c = e.load(a.summary, e.let(sb + c8 * 8u), 16u);
                const auto r = e.load(a.summary, e.let(sb + kWidth + c8 * 8u), 16u);
                const auto lo = qs.load(e.let(at + c8 * 8u), 16u);
                const auto hi = qs.load(e.let(at + c8 * 8u + 4u), 16u);
                for (int j = 0; j < 8; ++j) {
                  const auto qv = e.let(j < 4 ? lo[j] : hi[j - 4]);
                  auto& sum = (j % 2 == 0) ? even : odd;
                  sum = math::fma(qv, math::widen(c[j]), sum.read());
                  sum = math::fma(e.let(math::abs(qv) * s.attrs[1]), math::widen(r[j]), sum.read());
                }
              }
              auto value = e.var(math::neg_inf());
              if (auto row = e.when(rr < rc && r0 + rr < total_rows))
                value = even.read() + odd.read() - ref[rr].read();
              for (std::uint32_t bit = 16; bit; bit /= 2)
                value = math::max(value.read(), math::shfl_xor(value.read(), e.u32(bit)));
              best[i] = math::max(best[i].read(), value.read());
            }
          }
        }
      }
    }
    for (std::uint32_t i = 0; i < kPagesPer / kWaves; ++i) {
      const auto page = e.let(tile_begin + wave + i * kWaves);
      if (auto live = e.when(wl == 0u && page >= sd::kSinkPages && page < far_end && page < stride))
        e.store(e.let((b * kvh + kh) * stride + page), e.let(best[i].read() * s.attrs[0]));
    }
    return k.lds().ok() ? k.str() : std::string{};
  }

  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument, "sparse decode scores are built by sparse_decode_attention");
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = sd::kScoreThreads;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0)) * tiles(s);
    const auto rc = chunk_rows(s);
    p.lds_bytes = (rc * kWidth + (sd::kScoreThreads / 32u) * rc + rc) * 4u;
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeScore);

// ---------------------------------------------------------------------------
// Scores on the matrix core, for passes of many rows (verify chains and
// trees): the bound q . center + |q| . radius over 16 rows and 16 pages is
// two accumulated 16x16 products per 16 channels, [q | |q|] (bf16, LDS)
// against [center; radius] read as a key tile would be. A wave takes 16
// pages; every row tile of the chunk accumulates beside the others. The
// reference bounds are the same products over the newest pages (wave 0).
struct SparseDecodeScoreMma final : KernelPrimitive<SparseDecodeScoreMma> {
  static constexpr std::string_view kName = "attention.sparse_decode.score_mma.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_score_mma_v1";
  static constexpr std::string_view kSource = {};
  static constexpr std::uint32_t kThreads = sd::kScoreThreads, kWaves = kThreads / 32u;
  static constexpr std::uint32_t kPagesPer = 16u * kWaves;
  static constexpr std::uint32_t kRowStride = kWidth + 8u;
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  static std::uint32_t chunk_rows(const KernelShapes& s) {
    const auto rows = static_cast<std::uint32_t>(s.inputs[0].dim(1) / s.inputs[1].dim(1) *
                                                 s.inputs[0].dim(2));
    const auto chunk = rows < sd::kScoreRowChunk ? rows : sd::kScoreRowChunk;
    return (chunk + 15u) / 16u * 16u;
  }
  static std::uint32_t tiles(const KernelShapes& s) {
    return static_cast<std::uint32_t>((s.inputs[3].dim(1) + kPagesPer - 1) / kPagesPer);
  }
  static std::uint32_t lds_bytes(const KernelShapes& s) {
    const auto rc = chunk_rows(s);
    return 2u * rc * kRowStride * 2u + 2u * rc * 4u;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 || s.input_dtypes[0] != DType::kF32 ||
        s.input_dtypes[1] != DType::kBF16 || s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[3].rank() != 2 || s.inputs[0].dim(3) != kWidth || s.inputs[1].dim(2) != sd::kSummaryRows ||
        s.inputs[1].dim(3) != kWidth || s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
        !s.device || !s.store || !s.intrinsics || !s.types.scalar ||
        s.inputs[3].dim(1) > static_cast<std::int64_t>(sd::kMaxPages) ||
        s.output != Shape{s.inputs[0].dim(0) * s.inputs[1].dim(1), s.inputs[3].dim(1)} ||
        s.iattrs[0] <= static_cast<std::int32_t>(sd::kSinkPages + sd::kRecentPages) ||
        !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f || !std::isfinite(s.attrs[1]) || s.attrs[1] < 0.0f ||
        s.device->wavefront_size != 32 ||
        s.intrinsics->find("wave.shfl_xor").empty() ||
        lds_bytes(s) > workgroup_lds_bytes(s.device))
      return {};
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return {};
    const auto target = matrix_target(*s.device);
    if (!target) return {};
    return with_matrix_target<std::string>(*target, [&]<math::MatrixTarget G>() -> std::string {
      constexpr auto operand = math::MatrixElem::kBF16;
      if constexpr (!math::has_matrix_core_row(G, math::MatrixElem::kF32, operand, 16, 16, 16)) {
        return {};
      } else if constexpr (math::matrix_core_row(G, math::MatrixElem::kF32, operand, 16, 16, 16).wave != 32) {
        return {};
      } else {
        return emit_target<G>(s);
      }
    });
  }

  template <math::MatrixTarget G>
  std::string emit_target(const KernelShapes& s) const {
    using Mma = math::op::Mma<G, math::MatrixElem::kF32, math::MatrixElem::kBF16, 16, 16, 16>;
    using Narrow = math::matrix_scalar_t<Mma::kRow.a_elem>;
    constexpr TileGeometry kGeo = geometry_of(Mma::kRow);
    constexpr std::uint32_t kFrag = kGeo.lane_k;
    const auto caps = device_matrix_caps(*s.device);
    if (!math::has_cap(caps, Mma::kRow.cap) || s.intrinsics->find(Mma::kRow.key).empty()) return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto queries = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto group = heads / kvh;
    const auto total_rows = group * queries;
    const auto rc = chunk_rows(s);
    const auto row_tiles = rc / 16u;
    const auto chunks = (total_rows + rc - 1u) / rc;
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    const auto ntiles = tiles(s);
    const auto budget = static_cast<std::uint32_t>(s.iattrs[0]);
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    ScoreArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto qs = e.lds<Narrow>(rc * kRowStride);
    const auto qa = e.lds<Narrow>(rc * kRowStride);
    const auto ref = e.lds<kir::f32>(rc);
    const auto scratch = e.lds<kir::f32>(rc);
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto wave = e.let(lane / 32u), wl = e.let(lane % 32u);
    const auto lane_lo = e.let(wl % 16u), lane_hi = e.let(wl / 16u);
    const auto tile = e.let(wg % ntiles);
    const auto kh = e.let((wg / ntiles) % kvh);
    const auto b = e.let(wg / (ntiles * kvh));
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(b >= rows);
    const auto m = row_meta(e, a.meta, b);
    const auto far_end = minus(e, m.pages, sd::kRecentPages);
    const auto tile_begin = e.let(tile * kPagesPer);
    (void)e.ret_if(m.pages <= budget || tile_begin >= far_end ||
                   tile_begin + kPagesPer <= sd::kSinkPages);
    const auto acc_row = [&](std::uint32_t f) {
      return e.let(f * kGeo.slot_step + lane_hi * kGeo.half_rows);
    };
    const auto q_index = [&](const kir::Val<kir::u32>& rr, const kir::Val<kir::u32>& c) {
      return e.let(((b * heads + kh * group + rr / queries) * queries + rr % queries) * kWidth + c);
    };
    // This lane's column of a 16-page tile, its summary row base.
    const auto column_base = [&](const kir::Val<kir::u32>& page) {
      const auto blk = e.let(kir::cast<kir::u32>(a.table[b * stride + page]));
      return e.let((blk * kvh + kh) * sd::kSummaryRows * kWidth);
    };
    // Every row tile's accumulators over one 16-page column tile.
    const auto products = [&](const kir::Val<kir::u32>& sb) {
      std::vector<decltype(e.local<kir::f32, 8>())> acc;
      for (std::uint32_t t = 0; t < row_tiles; ++t) {
        acc.push_back(e.local<kir::f32, 8>());
        for (auto f : e.unroll(8u)) acc.back()[f] = e.f32(0.0f);
      }
      for (std::uint32_t half = 0; half < 2; ++half) {  // center, then radius
        const auto& lds = half == 0 ? qs : qa;
        for (auto depth : e.range(0u, kWidth, 16u)) {
          const auto kk = [&] {
            if constexpr (kGeo.split_k) return e.let(depth + lane_hi * 8u);
            else return e.let(depth + 0u);
          }();
          auto bf = e.local<Narrow, static_cast<int>(kFrag)>();
          const auto base = e.let(sb + half * kWidth + kk);
          const auto packed = e.load(a.summary, base, 16u);
          for (int f = 0; f < 8; ++f) bf[f] = packed[f];
          if constexpr (kFrag == 16u) {
            const auto upper = e.load(a.summary, e.let(base + 8u), 16u);
            for (int f = 0; f < 8; ++f) bf[8 + f] = upper[f];
          }
          for (std::uint32_t t = 0; t < row_tiles; ++t) {
            auto af = e.local<Narrow, static_cast<int>(kFrag)>();
            for (auto f : e.unroll(kFrag))
              af[f] = lds[e.let((e.u32(t * 16u) + lane_lo) * kRowStride + kk + f)].read();
            acc[t] = math::mma<Mma>(af.value(), bf.value(), acc[t].value());
          }
        }
      }
      return acc;
    };
    std::vector<kir::LValue<kir::f32>> best;
    best.push_back(e.var(math::neg_inf()));
    const auto page = e.let(tile_begin + wave * 16u + lane_lo);
    const auto mine = e.let(page >= sd::kSinkPages && page < far_end && page < stride);
    for (auto chunk : e.range(chunks)) {
      const auto r0 = e.let(chunk * rc);
      e.barrier();
      for (auto idx : e.range(lane, e.u32(rc * kWidth), kThreads)) {
        const auto rr = e.let(idx / kWidth), c = e.let(idx % kWidth);
        auto value = e.var(0.0f);
        if (auto live = e.when(r0 + rr < total_rows)) value = a.q[q_index(e.let(r0 + rr), c)];
        qs[e.let(rr * kRowStride + c)] = math::narrow<Narrow>(value.read());
        qa[e.let(rr * kRowStride + c)] = math::narrow<Narrow>(e.let(math::abs(value.read()) * s.attrs[1]));
      }
      if (auto clear = e.when(lane < rc)) scratch[lane] = math::neg_inf();
      e.barrier();
      // Reference: wave 0 over the 16 newest pages, the 8 newest kept.
      if (auto first_wave = e.when(wave == 0u)) {
        const auto ref_page = minus(e, minus(e, m.pages, 1u), lane_lo);
        auto acc = products(column_base(ref_page));
        for (std::uint32_t t = 0; t < row_tiles; ++t)
          for (std::uint32_t f = 0; f < 8; ++f) {
            auto v = e.var(select(lane_lo < sd::kReferencePages, acc[t][static_cast<int>(f)].read(),
                                  math::neg_inf()));
            for (std::uint32_t bit = 8; bit; bit /= 2)
              v = math::max(v.read(), math::shfl_xor(v.read(), e.u32(bit)));
            if (auto lead = e.when(lane_lo == 0u)) ref[e.let(e.u32(t * 16u) + acc_row(f))] = v.read();
          }
      }
      e.barrier();
      {
        auto acc = products(column_base(e.let(select(mine, page, e.u32(0)))));
        for (std::uint32_t t = 0; t < row_tiles; ++t)
          for (std::uint32_t f = 0; f < 8; ++f) {
            const auto rr = e.let(e.u32(t * 16u) + acc_row(f));
            if (auto row = e.when(r0 + rr < total_rows))
              best[0] = math::max(best[0].read(), acc[t][static_cast<int>(f)].read() - ref[rr].read());
          }
      }
    }
    // The two halves of the wave hold the same page's other rows.
    const auto total = e.let(math::max(best[0].read(), math::shfl_xor(best[0].read(), e.u32(16))));
    if (auto live = e.when(mine && lane_hi == 0u))
      e.store(e.let((b * kvh + kh) * stride + page), e.let(total * s.attrs[0]));
    return k.lds().ok() ? k.str() : std::string{};
  }

  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument, "sparse decode scores are built by sparse_decode_attention");
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = kThreads;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0)) * tiles(s);
    p.lds_bytes = lds_bytes(s);
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeScoreMma);

// ---------------------------------------------------------------------------
// Selection: one workgroup per sequence and key head. The far pages' scores
// become unique keys (relative mass bits, page index in the low bits); a
// bitwise bisection finds the threshold that keeps exactly the budget's far
// pages, and a scan writes their block ids in page order between the sink
// pages and the recent window. With no more pages than the budget the table
// is copied as it is.
template <class E>
struct SelectArgs {
  env::In<kir::f32, E> scores;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

struct SparseDecodeSelect final : KernelPrimitive<SparseDecodeSelect> {
  static constexpr std::string_view kName = "attention.sparse_decode.select.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_select_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    constexpr std::uint32_t threads = sd::kSelectThreads;
    if (s.inputs.size() != 3 || s.inputs[0].rank() != 2 || s.inputs[2].rank() != 2 ||
        !s.device || s.device->max_threads_per_workgroup < threads ||
        s.device->wavefront_size != 32 || !s.store || !s.intrinsics || !s.types.scalar ||
        !has_shfl(s))
      return {};
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return {};
    const auto kvh = static_cast<std::uint32_t>(s.iattrs[2]);
    const auto budget = static_cast<std::uint32_t>(s.iattrs[0]);
    const auto stride = static_cast<std::uint32_t>(s.inputs[2].dim(1));
    if (kvh == 0 || s.inputs[0].dim(0) % kvh || s.inputs[0].dim(1) != s.inputs[2].dim(1) ||
        stride > sd::kMaxPages || budget <= sd::kSinkPages + sd::kRecentPages ||
        s.output != Shape{s.inputs[0].dim(0), static_cast<std::int64_t>(budget) + s.inputs[2].dim(1)})
      return {};
    const bool joint = s.iattrs[1] != 0;
    const auto far_budget = budget - sd::kSinkPages - sd::kRecentPages;
    constexpr std::uint32_t waves = threads / 32u;
    constexpr int kPer = static_cast<int>(sd::kMaxPages / threads);
    const auto per = (stride + threads - 1u) / threads;
    if (per > static_cast<std::uint32_t>(kPer)) return {};
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SelectArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto partial = e.lds<kir::f32>(2u * waves);
    const auto scan = e.lds<kir::f32>(2u * threads);
    const auto lane = e.let(math::local_id()), row = e.let(math::workgroup_id_x());
    const auto wl = e.let(lane % 32u), wi = e.let(lane / 32u);
    const auto kh = e.let(row % kvh), b = e.let(row / kvh);
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(b >= rows);
    const auto m = row_meta(e, a.meta, b);
    const auto tb = e.let(b * stride);
    // A row is the budget's block ids, then a flag per page: 1 where a far
    // page was left out (the mean correction's pages).
    const auto ob = e.let(row * (budget + stride));
    // Every page fits: the table as it is (slots past the row are never read).
    const auto all = e.let(m.pages <= budget);
    if (auto copy = e.when(all))
      for (auto j : e.range(lane, e.u32(budget), threads))
        e.store(e.let(ob + j), a.table[e.let(tb + select(j < stride, j, e.u32(0)))]);
    (void)e.ret_if(all);

    const auto far_end = minus(e, m.pages, sd::kRecentPages);
    // Workgroup reductions through `partial` (two halves, alternating, so a
    // reduction needs one barrier).
    std::uint32_t half = 0;
    auto reduce = [&](const kir::Val<kir::f32>& v, bool maximum) {
      auto x = e.var(v);
      for (std::uint32_t bit = 16; bit; bit /= 2) {
        const auto other = e.let(math::shfl_xor(x.read(), e.u32(bit)));
        x = maximum ? math::max(x.read(), other) : x.read() + other;
      }
      const auto base = e.u32(half * waves);
      half ^= 1u;
      if (auto lead = e.when(wl == 0u)) partial[e.let(base + wi)] = x.read();
      e.barrier();
      auto total = e.var(maximum ? math::neg_inf() : e.f32(0.0f));
      for (std::uint32_t w = 0; w < waves; ++w) {
        const auto pw = e.let(partial[e.let(base + w)].read());
        total = maximum ? math::max(total.read(), pw) : total.read() + pw;
      }
      return e.let(total.read());
    };
    // A far page's score (or its largest over key heads); pages outside
    // the far range are not candidates.
    auto far = [&](const kir::Val<kir::u32>& p) { return e.let(p >= sd::kSinkPages && p + sd::kRecentPages < m.pages && p < stride); };
    auto score_of = [&](const kir::Val<kir::u32>& p) {
      if (!joint) return e.let(a.scores[e.let(row * stride + p)]);
      auto v = e.var(math::neg_inf());
      for (std::uint32_t h = 0; h < kvh; ++h)
        v = math::max(v.read(), a.scores[e.let((b * kvh + h) * stride + p)]);
      return e.let(v.read());
    };
    auto top = e.var(math::neg_inf());
    for (int i = 0; i < static_cast<int>(per); ++i) {
      const auto p = e.let(lane * per + static_cast<std::uint32_t>(i));
      if (auto live = e.when(far(p))) top = math::max(top.read(), score_of(p));
    }
    const auto peak = reduce(top.read(), true);
    auto key = e.local<kir::u32, kPer>();
    for (int i = 0; i < static_cast<int>(per); ++i) {
      const auto p = e.let(lane * per + static_cast<std::uint32_t>(i));
      const auto candidate = far(p);
      const auto value = e.let(select(candidate, score_of(e.let(select(p < stride, p, e.u32(0)))), peak));
      // Relative mass in (0, 1]: its bits order as the values do. The page
      // index (+1, so no live key is zero) makes every key unique.
      const auto mass = e.let(math::exp(value - peak));
      const auto bits = e.let(math::bit_and(math::bits_of<lse::f32>(mass),
                                            e.u32(~((1u << sd::kPageIndexBits) - 1u))));
      // Both arms in the word domain (bits_of's): Loom will not select
      // between a word and an index.
      const auto none = e.let(math::bit_and(bits, e.u32(0)));
      key[i] = select(candidate, e.let(bits + p + 1u), none);
    }
    // The largest threshold with at least far_budget keys at or above it;
    // keys are unique, so exactly far_budget are. Every key is below 2^30
    // (a mass of at most 1.0f is 0x3f800000, plus a page index), so the
    // search starts at bit 29 and no comparison sees a sign bit.
    auto threshold = e.var(e.u32(0));
    for (int bit = 29; bit >= 0; --bit) {
      const auto candidate = e.let(threshold.read() + (1u << bit));
      auto count = e.var(0.0f);
      for (int i = 0; i < static_cast<int>(per); ++i)
        count = count.read() + select(key[i].read() >= candidate, e.f32(1.0f), e.f32(0.0f));
      const auto total = reduce(count.read(), false);
      threshold = select(total >= static_cast<float>(far_budget), candidate, threshold.read());
    }
    // Exclusive scan of each thread's kept count (Hillis-Steele, two buffers).
    auto kept = e.var(0.0f);
    for (int i = 0; i < static_cast<int>(per); ++i)
      kept = kept.read() + select(key[i].read() >= threshold.read(), e.f32(1.0f), e.f32(0.0f));
    scan[lane] = kept.read();
    e.barrier();
    std::uint32_t from = 0;
    for (std::uint32_t offset = 1; offset < threads; offset *= 2) {
      const auto src = e.u32(from * threads), dst = e.u32((from ^ 1u) * threads);
      auto v = e.var(scan[e.let(src + lane)].read());
      if (auto has = e.when(lane >= offset)) v = v.read() + scan[e.let(src + minus(e, lane, offset))].read();
      scan[e.let(dst + lane)] = v.read();
      e.barrier();
      from ^= 1u;
    }
    const auto inclusive = e.let(scan[e.let(e.u32(from * threads) + lane)].read());
    auto slot = e.var(kir::cast<kir::u32>(e.let(inclusive - kept.read())));
    for (int i = 0; i < static_cast<int>(per); ++i) {
      const auto p = e.let(lane * per + static_cast<std::uint32_t>(i));
      const auto take = e.let(key[i].read() >= threshold.read());
      if (auto kept_page = e.when(take)) {
        e.store(e.let(ob + sd::kSinkPages + slot.read()), a.table[e.let(tb + p)]);
        slot = slot.read() + 1u;
      }
      if (auto real = e.when(p < stride))
        e.store(e.let(ob + budget + p),
                select(far(p) && key[i].read() < threshold.read(), e.f32(1.0f), e.f32(0.0f)));
    }
    if (auto sink = e.when(lane < sd::kSinkPages))
      e.store(e.let(ob + lane), a.table[e.let(tb + lane)]);
    if (auto recent = e.when(lane < sd::kRecentPages))
      e.store(e.let(ob + sd::kSinkPages + far_budget + lane), a.table[e.let(tb + far_end + lane)]);
    return k.lds().ok() ? k.str() : std::string{};
  }

  static bool has_shfl(const KernelShapes& s) {
    return !s.intrinsics->find("wave.shfl_xor").empty() && !s.intrinsics->find("bits.f32").empty() &&
           !s.intrinsics->find("and.u32").empty();
  }
  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument, "sparse decode selection is built by sparse_decode_attention");
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = sd::kSelectThreads;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0));
    p.lds_bytes = (2u * (sd::kSelectThreads / 32u) + 2u * sd::kSelectThreads) * 4u;
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeSelect);

// ---------------------------------------------------------------------------
// The omitted far pages' mean correction: for each query row and head, every
// far page the selection left out contributes 16 * exp(scale * q . kmean)
// with value vmean (FlashPrefill V2's correction, per page), accumulated as
// split records the merge combines with the exact ones. A workgroup per
// sequence, key head and 256 pages; a wave per group of rows, the page's
// key and value means over the lanes.
template <class E>
struct TailArgs {
  env::In<kir::f32, E> q;
  env::In<Summary, E> summary;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::In<kir::f32, E> selected;
  env::Out<kir::f32, E> out;
};

struct SparseDecodeTail final : KernelPrimitive<SparseDecodeTail> {
  static constexpr std::string_view kName = "attention.sparse_decode.tail.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_tail_v1";
  static constexpr std::string_view kSource = {};
  static constexpr std::uint32_t kThreads = 256, kWaves = kThreads / 32u;
  // Pages a wave walks (one split record per row each), and the rows it
  // carries in registers at a time.
  static constexpr std::uint32_t kPagesPerWave = sd::kTailPagesPerWave, kRowGroup = 6;
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  // Split records per row: one per workgroup (eight waves' runs of pages).
  static std::uint32_t records(std::int64_t stride) {
    return static_cast<std::uint32_t>((stride + kPagesPerWave * kWaves - 1) / (kPagesPerWave * kWaves));
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 || s.input_dtypes[1] != DType::kBF16 ||
        s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 || s.inputs[3].rank() != 2 ||
        s.inputs[4].rank() != 2 || s.inputs[0].dim(3) != kWidth || s.inputs[1].dim(2) != sd::kSummaryRows ||
        s.inputs[1].dim(3) != kWidth || s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
        !s.device || s.device->wavefront_size != 32 || !s.store || !s.intrinsics || !s.types.scalar ||
        s.intrinsics->find("wave.shfl_xor").empty() || !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f)
      return {};
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return {};
    const auto batch = static_cast<std::uint32_t>(s.inputs[0].dim(0));
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto queries = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto group = heads / kvh;
    
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    const auto nrecords = records(stride);
    const auto width = static_cast<std::uint32_t>(s.inputs[4].dim(1));
    if (width <= stride || s.inputs[4].dim(0) != batch * kvh) return {};
    const auto budget = width - stride;
    if (budget <= sd::kSinkPages + sd::kRecentPages) return {};
    const Shape want = queries == 1 ? Shape{batch, heads, nrecords, sd::kSplitRecord}
                                    : Shape{batch, heads, queries, nrecords, sd::kSplitRecord};
    if (s.output != want) return {};
    // A verify pass's rows of one query head share one correction, from
    // their mean query: the omitted pages' mass is a smooth correction, and
    // per-row logits would cost the pass's width again.
    const auto tail_rows = group;
    const auto groups = (tail_rows + kRowGroup - 1u) / kRowGroup;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    TailArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto wave = e.let(lane / 32u), wl = e.let(lane % 32u);
    const auto tile = e.let(wg % nrecords);  // this workgroup's record
    const auto run = e.let(tile * kWaves + wave);  // this wave's run of pages
    const auto kh = e.let((wg / nrecords) % kvh);
    const auto b = e.let(wg / (nrecords * kvh));
    const auto partial = e.lds<kir::f32>(kWaves * kRowGroup * sd::kSplitRecord);
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    const auto m = row_meta(e, a.meta, b);
    const auto first_page = e.let(run * kPagesPerWave);
    const auto far_end = minus(e, m.pages, sd::kRecentPages);
    const auto flags = e.let((b * kvh + kh) * width + budget);
    const auto record = [&](const kir::Val<kir::u32>& rr) {
      const auto h = e.let(kh * group + rr / queries), t = e.let(rr % queries);
      return e.let((((b * heads + h) * queries + t) * nrecords + tile) * sd::kSplitRecord);
    };
    // No far page here (or every page is read): empty records.
    const auto live = e.let(b < rows && m.pages > budget && first_page < far_end);
    const float log_count = std::log(static_cast<float>(sd::kPage));
    const auto channel = e.let(wl * 8u);
    for (std::uint32_t g = 0; g < groups; ++g) {
      const auto count = std::min(kRowGroup, tail_rows - g * kRowGroup);
      std::vector<std::vector<kir::Val<kir::f32>>> qv;
      std::vector<kir::LValue<kir::f32>> top, denom;
      std::vector<std::vector<kir::LValue<kir::f32>>> num;
      for (std::uint32_t r = 0; r < count; ++r) {
        const auto h = e.u32(g * kRowGroup + r);  // query head within the group
        std::vector<kir::LValue<kir::f32>> mean;
        for (int j = 0; j < 8; ++j) mean.push_back(e.var(0.0f));
        for (std::uint32_t t = 0; t < queries; ++t) {
          const auto qb = e.let(((b * heads + kh * group + h) * queries + t) * kWidth + channel);
          const auto lo = e.load(a.q, qb, 16u);
          const auto hi = e.load(a.q, e.let(qb + 4u), 16u);
          for (int j = 0; j < 8; ++j)
            mean[static_cast<std::size_t>(j)] = mean[static_cast<std::size_t>(j)].read() +
                                                 e.let(j < 4 ? lo[j] : hi[j - 4]) * (1.0f / static_cast<float>(queries));
        }
        qv.emplace_back();
        for (int j = 0; j < 8; ++j) qv.back().push_back(e.let(mean[static_cast<std::size_t>(j)].read()));
        top.push_back(e.var(math::neg_inf()));
        denom.push_back(e.var(0.0f));
        num.emplace_back();
        for (int j = 0; j < 8; ++j) num.back().push_back(e.var(0.0f));
      }
      if (auto any = e.when(live)) {
        // Four pages per step, their loads issued before any arithmetic, so a
        // wave keeps several reads in flight instead of one.
        constexpr std::uint32_t kStep = 4;
        for (auto i0 : e.range(0u, kPagesPerWave, kStep)) {
          std::vector<kir::Val<kir::boolean>> out;
          std::vector<std::vector<kir::Val<kir::f32>>> kf, vf;
          for (std::uint32_t u = 0; u < kStep; ++u) {
            const auto page = e.let(first_page + i0 + u);
            const auto real = e.let(page < stride);
            const auto safe = e.let(select(real, page, e.u32(0)));
            const auto flag = e.let(a.selected[e.let(flags + safe)]);
            out.push_back(e.let(real && flag != 0.0f));
            const auto blk = e.let(kir::cast<kir::u32>(a.table[e.let(b * stride + safe)]));
            const auto sb = e.let((blk * kvh + kh) * sd::kSummaryRows * kWidth + channel);
            const auto km = e.load(a.summary, e.let(sb + sd::kRowKeyMean * kWidth), 16u);
            const auto vm = e.load(a.summary, e.let(sb + sd::kRowValueMean * kWidth), 16u);
            kf.emplace_back();
            vf.emplace_back();
            for (int j = 0; j < 8; ++j) {
              kf.back().push_back(e.let(math::widen(km[j])));
              vf.back().push_back(e.let(math::widen(vm[j])));
            }
          }
          for (std::uint32_t u = 0; u < kStep; ++u) {
            if (auto take = e.when(out[u])) {
              for (std::uint32_t r = 0; r < count; ++r) {
                auto dot = e.var(0.0f);
                for (std::size_t j = 0; j < 8; ++j) dot = math::fma(qv[r][j], kf[u][j], dot.read());
                for (std::uint32_t bit = 16; bit; bit /= 2) dot = dot.read() + math::shfl_xor(dot.read(), e.u32(bit));
                const auto logit = e.let(dot.read() * s.attrs[0] + log_count);
                const auto updated = e.let(math::max(top[r].read(), logit));
                const auto alpha = e.let(math::exp(top[r].read() - updated));
                const auto weight = e.let(math::exp(logit - updated));
                top[r] = updated;
                denom[r] = math::fma(denom[r].read(), alpha, weight);
                for (std::size_t j = 0; j < 8; ++j)
                  num[r][j] = math::fma(num[r][j].read(), alpha, e.let(weight * vf[u][j]));
              }
            }
          }
        }
      }
      // The eight waves' records of each row, combined in LDS into the
      // workgroup's one (stable rescaling to the largest maximum).
      for (std::uint32_t r = 0; r < count; ++r) {
        const auto at = e.let((wave * kRowGroup + r) * sd::kSplitRecord);
        for (std::uint32_t j = 0; j < 8; ++j) partial[e.let(at + 2u + channel + j)] = num[r][j].read();
        if (auto lead = e.when(wl == 0u)) {
          partial[at] = top[r].read();
          partial[e.let(at + 1u)] = denom[r].read();
        }
      }
      e.barrier();
      if (auto owner = e.when(wave < count)) {
        auto peak = e.var(math::neg_inf());
        for (std::uint32_t w = 0; w < kWaves; ++w)
          peak = math::max(peak.read(), partial[e.let(e.u32(w * kRowGroup) * sd::kSplitRecord + wave * sd::kSplitRecord)].read());
        const auto safe = e.let(select(peak.read() != math::neg_inf(), peak.read(), e.f32(0.0f)));
        auto d = e.var(0.0f);
        std::vector<kir::LValue<kir::f32>> n;
        for (int j = 0; j < 8; ++j) n.push_back(e.var(0.0f));
        for (std::uint32_t w = 0; w < kWaves; ++w) {
          const auto at = e.let((e.u32(w * kRowGroup) + wave) * sd::kSplitRecord);
          const auto scale_w = e.let(math::exp(partial[at].read() - safe));
          d = math::fma(partial[e.let(at + 1u)].read(), scale_w, d.read());
          for (std::uint32_t j = 0; j < 8; ++j)
            n[j] = math::fma(partial[e.let(at + 2u + channel + j)].read(), scale_w, n[j].read());
        }
        for (std::uint32_t t = 0; t < queries; ++t) {
          const auto base = record(e.let((e.u32(g * kRowGroup) + wave) * queries + t));
          for (std::uint32_t j = 0; j < 8; ++j) e.store(e.let(base + 2u + channel + j), n[j].read());
          if (auto lead = e.when(wl == 0u)) {
            e.store(base, peak.read());
            e.store(e.let(base + 1u), d.read());
          }
        }
      }
      e.barrier();
    }
    return k.lds().ok() ? k.str() : std::string{};
  }
  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument, "sparse decode tail records are built by sparse_decode_attention");
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = kThreads;
    const auto n = records(s.inputs[3].dim(1));
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.inputs[0].dim(0) * s.inputs[1].dim(1)) * n;
    p.lds_bytes = kWaves * kRowGroup * sd::kSplitRecord * 4u;
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeTail);

// ---------------------------------------------------------------------------
// The step descriptor of the selected pages: every row with more pages than
// the budget has its positions moved down by the pages left out, so its rows
// and the recent window keep their offsets in the compacted table and the
// causal and tree masks hold as they are. The tree mask is copied.
template <class E>
struct MetaArgs {
  env::In<kir::f32, E> meta;
  env::Out<kir::f32, E> out;
};

struct SparseDecodeMeta final : KernelPrimitive<SparseDecodeMeta> {
  static constexpr std::string_view kName = "attention.sparse_decode.meta.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_meta_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const auto batch = static_cast<std::uint32_t>(s.iattrs[1]);
    const auto budget = static_cast<std::uint32_t>(s.iattrs[0]);
    if (s.inputs.size() != 1 || s.output != s.inputs[0] || batch == 0 || budget == 0 ||
        s.inputs[0].elem_count() < static_cast<std::size_t>(kv::step_meta_elems(static_cast<std::int32_t>(batch))) ||
        s.inputs[0].elem_count() > UINT32_MAX || !s.store || !s.intrinsics || !s.types.scalar)
      return {};
    const auto count = static_cast<std::uint32_t>(s.inputs[0].elem_count());
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    MetaArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto lane = e.let(math::local_id());
    // Each row's shift and shifted pair; the header takes their largest.
    std::vector<kir::Val<kir::f32>> first, live;
    auto top_first = e.var(0.0f), top_live = e.var(0.0f);
    for (std::uint32_t r = 0; r < batch; ++r) {
      const auto f = e.let(a.meta[e.u32(kv::kStepMetaHeader + kv::kStepMetaPerRow * r)]);
      const auto l = e.let(a.meta[e.u32(kv::kStepMetaHeader + kv::kStepMetaPerRow * r + 1u)]);
      const auto pages = e.let(kir::cast<kir::f32>(e.let(
          (kir::cast<kir::u32>(l) + (sd::kPage - 1u)) / sd::kPage)));
      const auto shift = e.let(select(pages > static_cast<float>(budget),
                                      (pages - static_cast<float>(budget)) * static_cast<float>(sd::kPage),
                                      e.f32(0.0f)));
      first.push_back(e.let(select(l > 0.0f, f - shift, f)));
      live.push_back(e.let(l - shift));
      if (auto real = e.when(l > 0.0f)) {
        top_first = math::max(top_first.read(), first.back());
        top_live = math::max(top_live.read(), live.back());
      }
    }
    for (auto i : e.range(lane, e.u32(count), 256u)) {
      auto v = e.var(a.meta[i]);
      if (auto h0 = e.when(i == 0u)) v = top_first.read();
      if (auto h1 = e.when(i == 1u)) v = top_live.read();
      for (std::uint32_t r = 0; r < batch; ++r) {
        if (auto f = e.when(i == kv::kStepMetaHeader + kv::kStepMetaPerRow * r)) v = first[r];
        if (auto l = e.when(i == kv::kStepMetaHeader + kv::kStepMetaPerRow * r + 1u)) v = live[r];
      }
      e.store(i, v.read());
    }
    return k.str();
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "sparse decode meta takes 1 input");
    return in[0];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes&) {
    ThreadPlan p;
    p.workgroup_size[0] = 256;
    p.workgroup_count[0] = 1;
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeMeta);

}  // namespace
}  // namespace lse::kernels
