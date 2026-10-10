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
#include "lse/kv/block.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/device_info.hpp"

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
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<Summary, E> out;
};

struct SparseDecodeSummary final : KernelPrimitive<SparseDecodeSummary> {
  static constexpr std::string_view kName = "attention.sparse_decode.summary.v1";
  static constexpr std::string_view kEntry = "lse_sparse_decode_summary_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }

  static bool valid(const KernelShapes& s) {
    if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 || s.input_dtypes[0] != DType::kBF16 ||
        s.output_dtype != DType::kBF16 || s.input_dtypes[2] != DType::kF32 ||
        s.input_dtypes[3] != DType::kF32 || s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[3].rank() != 2 || s.output != s.inputs[0] || s.iattrs[0] <= 0)
      return false;
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return false;
    const auto& sum = s.inputs[0];
    const auto& keys = s.inputs[1];
    return sum.dim(0) >= keys.dim(0) && sum.dim(1) == keys.dim(1) && sum.dim(2) == 2 &&
           sum.dim(3) == kWidth && keys.dim(2) == sd::kPage && keys.dim(3) == kWidth;
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
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
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
    auto lo = e.var(3.0e38f), hi = e.var(-3.0e38f);
    for (auto slot : e.range(sd::kPage)) {
      if (auto live = e.when(page * sd::kPage + slot < m.live)) {
        const auto x = e.let(kv_load<Storage>(
            e, a.keys, e.let(((blk * kvh + kh) * sd::kPage + slot) * kWidth + lane), kWidth));
        lo = math::min(lo.read(), x);
        hi = math::max(hi.read(), x);
      }
    }
    const auto base = e.let((blk * kvh + kh) * 2u * kWidth + lane);
    a.out[base] = kir::cast<Summary>(e.let((lo.read() + hi.read()) * 0.5f));
    a.out[e.let(base + kWidth)] = kir::cast<Summary>(e.let((hi.read() - lo.read()) * 0.5f));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4) return LSE_ERROR(kInvalidArgument, "sparse decode summary takes 4 inputs");
    return in[0];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kBF16; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    p.workgroup_size[0] = kWidth;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.inputs[3].dim(0) * s.inputs[1].dim(1)) *
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

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 || s.input_dtypes[0] != DType::kF32 ||
        s.input_dtypes[1] != DType::kBF16 || s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[3].rank() != 2 || s.inputs[0].dim(3) != kWidth || s.inputs[1].dim(2) != 2 ||
        s.inputs[1].dim(3) != kWidth || s.inputs[0].dim(1) % s.inputs[1].dim(1) ||
        !s.device || !s.store || !s.intrinsics || !s.types.scalar ||
        s.inputs[3].dim(1) > static_cast<std::int64_t>(sd::kMaxPages) ||
        s.output != Shape{s.inputs[0].dim(0) * s.inputs[1].dim(1), s.inputs[3].dim(1)} ||
        s.iattrs[0] <= static_cast<std::int32_t>(sd::kSinkPages + sd::kRecentPages) ||
        !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f)
      return {};
    for (const auto& shape : s.inputs) if (!positive_u32_shape(shape)) return {};
    const auto heads = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto queries = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto kvh = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto group = heads / kvh;
    const auto total_rows = group * queries;
    const auto rc = chunk_rows(s);
    const auto chunks = (total_rows + rc - 1u) / rc;
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    const auto tiles = (stride + sd::kScoreThreads - 1u) / sd::kScoreThreads;
    const auto budget = static_cast<std::uint32_t>(s.iattrs[0]);
    constexpr std::uint32_t kRef = sd::kReferencePages;
    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    ScoreArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto qs = e.lds<kir::f32>(rc * kWidth);
    const auto refs = e.lds<kir::f32>(kRef * rc);
    const auto ref = e.lds<kir::f32>(rc);
    const auto lane = e.let(math::local_id()), wg = e.let(math::workgroup_id_x());
    const auto tile = e.let(wg % tiles);
    const auto kh = e.let((wg / tiles) % kvh);
    const auto b = e.let(wg / (tiles * kvh));
    const auto rows = e.runtime_extent("rows", kir::cast<kir::u32>(a.meta[2u]));
    (void)e.ret_if(b >= rows);
    const auto m = row_meta(e, a.meta, b);
    // Nothing to score while every page is read (selection copies the table)
    // or for a tile wholly outside the far range; uniform per workgroup.
    const auto far_end = minus(e, m.pages, sd::kRecentPages);
    const auto tile_begin = e.let(tile * sd::kScoreThreads);
    (void)e.ret_if(m.pages <= budget || tile_begin >= far_end ||
                   tile_begin + sd::kScoreThreads <= sd::kSinkPages);
    const auto page = e.let(tile_begin + lane);
    const auto mine = e.let(page >= sd::kSinkPages && page < far_end && page < stride);
    const auto own_blk = e.let(kir::cast<kir::u32>(
        a.table[b * stride + select(page < stride, page, e.u32(0))]));
    const auto summary_base = [&](const kir::Val<kir::u32>& blk) {
      return e.let((blk * kvh + kh) * 2u * kWidth);
    };
    // Row rr of the key head's rows: query head kh*group + rr/queries, row rr%queries.
    const auto q_index = [&](const kir::Val<kir::u32>& rr, const kir::Val<kir::u32>& c) {
      return e.let(((b * heads + kh * group + rr / queries) * queries + rr % queries) * kWidth + c);
    };
    auto best = e.var(math::neg_inf());
    for (auto chunk : e.range(chunks)) {
      const auto r0 = e.let(chunk * rc);
      e.barrier();
      for (auto idx : e.range(lane, e.u32(rc * kWidth), sd::kScoreThreads)) {
        const auto rr = e.let(r0 + idx / kWidth);
        auto value = e.var(0.0f);
        if (auto live = e.when(rr < total_rows)) value = a.q[q_index(rr, e.let(idx % kWidth))];
        qs[idx] = value.read();
      }
      e.barrier();
      // Reference bounds: (newest page i, chunk row r) pairs over the threads.
      for (auto idx : e.range(lane, e.u32(kRef * rc), sd::kScoreThreads)) {
        const auto i = e.let(idx / rc), r = e.let(idx % rc);
        const auto ref_page = minus(e, minus(e, m.pages, 1u), i);
        const auto blk = e.let(kir::cast<kir::u32>(a.table[b * stride + ref_page]));
        const auto sb = summary_base(blk);
        auto u = e.var(0.0f);
        for (auto c8 : e.range(kWidth / 8u)) {
          const auto center = e.load(a.summary, e.let(sb + c8 * 8u), 16u);
          const auto radius = e.load(a.summary, e.let(sb + kWidth + c8 * 8u), 16u);
          const auto lo = qs.load(e.let(r * kWidth + c8 * 8u), 16u);
          const auto hi = qs.load(e.let(r * kWidth + c8 * 8u + 4u), 16u);
          for (int j = 0; j < 8; ++j) {
            const auto qv = e.let(j < 4 ? lo[j] : hi[j - 4]);
            u = math::fma(qv, math::widen(center[j]), u.read());
            u = math::fma(math::abs(qv), math::widen(radius[j]), u.read());
          }
        }
        refs[idx] = u.read();
      }
      e.barrier();
      if (auto owner = e.when(lane < rc)) {
        auto top = e.var(math::neg_inf());
        for (std::uint32_t i = 0; i < kRef; ++i)
          top = math::max(top.read(), refs[e.let(e.u32(i * rc) + lane)].read());
        ref[lane] = top.read();
      }
      e.barrier();
      if (auto live = e.when(mine)) {
        const auto sb = summary_base(own_blk);
        std::vector<kir::LValue<kir::f32>> acc;
        for (std::uint32_t r = 0; r < rc; ++r) acc.push_back(e.var(0.0f));
        for (auto c8 : e.range(kWidth / 8u)) {
          const auto center = e.load(a.summary, e.let(sb + c8 * 8u), 16u);
          const auto radius = e.load(a.summary, e.let(sb + kWidth + c8 * 8u), 16u);
          std::vector<kir::Val<kir::f32>> cv, rv;
          for (int j = 0; j < 8; ++j) {
            cv.push_back(e.let(math::widen(center[j])));
            rv.push_back(e.let(math::widen(radius[j])));
          }
          for (std::uint32_t r = 0; r < rc; ++r) {
            const auto lo = qs.load(e.let(e.u32(r * kWidth) + c8 * 8u), 16u);
            const auto hi = qs.load(e.let(e.u32(r * kWidth) + c8 * 8u + 4u), 16u);
            for (int j = 0; j < 8; ++j) {
              const auto qv = e.let(j < 4 ? lo[j] : hi[j - 4]);
              acc[r] = math::fma(qv, cv[static_cast<std::size_t>(j)], acc[r].read());
              acc[r] = math::fma(math::abs(qv), rv[static_cast<std::size_t>(j)], acc[r].read());
            }
          }
        }
        for (std::uint32_t r = 0; r < rc; ++r)
          if (auto row = e.when(r0 + r < total_rows))
            best = math::max(best.read(), acc[r].read() - ref[e.u32(r)].read());
      }
    }
    if (auto live = e.when(mine))
      e.store(e.let((b * kvh + kh) * stride + page), e.let(best.read() * s.attrs[0]));
    return k.lds().ok() ? k.str() : std::string{};
  }

  Result<Shape> infer_shape(std::span<const Shape>) const override {
    return LSE_ERROR(kInvalidArgument, "sparse decode scores are built by sparse_decode_attention");
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan p;
    const auto stride = static_cast<std::uint32_t>(s.inputs[3].dim(1));
    p.workgroup_size[0] = sd::kScoreThreads;
    p.workgroup_count[0] = static_cast<std::uint32_t>(s.output.dim(0)) *
                           ((stride + sd::kScoreThreads - 1u) / sd::kScoreThreads);
    const auto rc = chunk_rows(s);
    p.lds_bytes = (rc * kWidth + sd::kReferencePages * rc + rc) * 4u;
    return p;
  }
};
LSE_REGISTER_PRIMITIVE(SparseDecodeScore);

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
        s.output != Shape{s.inputs[0].dim(0), static_cast<std::int64_t>(budget)})
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
    const auto ob = e.let(row * budget);
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
      if (auto take = e.when(key[i].read() >= threshold.read())) {
        e.store(e.let(ob + sd::kSinkPages + slot.read()), a.table[e.let(tb + p)]);
        slot = slot.read() + 1u;
      }
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
