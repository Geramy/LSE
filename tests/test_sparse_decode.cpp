// Block-sparse decode attention on the GPU against a host reference: page
// summaries, per-key-head selection (sink, recent window, top-scoring far
// pages), the shifted step descriptor, and the exact kernels over the
// selected pages, for decode (1 row), a verify chain (8 rows) and a draft
// tree (8 rows), plus the identity regime where every page fits the budget.
//
//   test_sparse_decode          host-side shape checks
//   test_sparse_decode --gpu    the device reference cases
#include "harness.hpp"
#include "lse/dispatch/arch/tuning.hpp"
#include "lse/dispatch/sparse_decode_tuneconfig.h"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace sd = lse::dispatch::sparse_decode;

namespace {

float bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return std::bit_cast<float>(bits & 0xffff0000u);
}

constexpr int kKvh = 4, kHeads = 24, kGroup = 6, kWidth = 256, kPages = 512, kBlocks = 520;
constexpr int kCapacity = kPages * 16;
constexpr int kBudget = 1024, kBudgetPages = kBudget / 16;
constexpr float kScale = 0.0625f;

struct Fixture {
  std::vector<float> q, k, v, table;
  std::vector<std::vector<int>> hot;  // per key head
};

Fixture make_fixture(int rows) {
  Fixture f;
  std::uint32_t seed = 7;
  auto random = [&]() { seed = seed * 1664525u + 1013904223u; return float(int(seed >> 9) % 2001 - 1000) / 1000.0f; };
  f.q.resize(std::size_t(kHeads) * rows * kWidth);
  for (auto& x : f.q) x = random();
  f.k.resize(std::size_t(kBlocks) * kKvh * 16 * kWidth);
  f.v.resize(f.k.size());
  for (auto& x : f.k) x = bf16(random() * 0.25f);
  for (auto& x : f.v) x = bf16(random());
  f.table.resize(kPages);
  for (int p = 0; p < kPages; ++p) f.table[p] = float((p * 7 + 3) % kBlocks);
  // Ten far pages per key head whose keys follow one of its query heads.
  f.hot.resize(kKvh);
  for (int g = 0; g < kKvh; ++g)
    for (int j = 0; j < 10; ++j) {
      const int page = 20 + g * 11 + j * 41;
      f.hot[g].push_back(page);
      const int blk = int(f.table[page]);
      const int h = g * kGroup + j % kGroup;
      for (int slot = 0; slot < 16; ++slot)
        for (int c = 0; c < kWidth; ++c)
          f.k[((std::size_t(blk) * kKvh + g) * 16 + slot) * kWidth + c] =
              bf16(f.q[(std::size_t(h) * rows + 0) * kWidth + c] * 0.8f);
    }
  return f;
}

float key_at(const Fixture& f, int g, int pos, int c) {
  const int blk = int(f.table[pos / 16]);
  return f.k[((std::size_t(blk) * kKvh + g) * 16 + pos % 16) * kWidth + c];
}
// Values are dimension-major in the pool the engine uses; this fixture keeps
// both pools row-major (the kernels read either).
float value_at(const Fixture& f, int g, int pos, int c) {
  const int blk = int(f.table[pos / 16]);
  return f.v[((std::size_t(blk) * kKvh + g) * 16 + pos % 16) * kWidth + c];
}

// Host scores, the score kernel's formula on bf16 summaries.
std::vector<double> host_scores(const Fixture& f, int rows, int live, int g) {
  const int pages = (live + 15) / 16;
  std::vector<std::array<std::vector<float>, 2>> summary(pages);
  for (int p = 0; p < pages; ++p) {
    summary[p][0].resize(kWidth); summary[p][1].resize(kWidth);
    for (int c = 0; c < kWidth; ++c) {
      float lo = 3e38f, hi = -3e38f;
      for (int s = 0; s < 16 && p * 16 + s < live; ++s) {
        lo = std::min(lo, key_at(f, g, p * 16 + s, c));
        hi = std::max(hi, key_at(f, g, p * 16 + s, c));
      }
      summary[p][0][c] = bf16((lo + hi) * 0.5f);
      summary[p][1][c] = bf16((hi - lo) * 0.5f);
    }
  }
  auto bound = [&](int p, int h, int t) {
    double u = 0;
    for (int c = 0; c < kWidth; ++c) {
      const double qv = f.q[(std::size_t(h) * rows + t) * kWidth + c];
      u += qv * summary[p][0][c] + std::abs(qv) * summary[p][1][c];
    }
    return u;
  };
  std::vector<double> ref(kGroup * rows, -INFINITY);
  for (int i = 0; i < int(sd::kReferencePages); ++i)
    for (int r = 0; r < kGroup * rows; ++r)
      ref[r] = std::max(ref[r], bound(pages - 1 - i, g * kGroup + r / rows, r % rows));
  std::vector<double> out(pages, -INFINITY);
  for (int p = int(sd::kSinkPages); p < pages - int(sd::kRecentPages); ++p) {
    double best = -INFINITY;
    for (int r = 0; r < kGroup * rows; ++r) best = std::max(best, bound(p, g * kGroup + r / rows, r % rows) - ref[r]);
    out[p] = best * kScale;
  }
  return out;
}

}  // namespace

LSE_TEST(sparse_decode_refresh_window_covers_slack_and_rows) {
  LSE_EXPECT(sd::refresh_pages(1) * sd::kPage >= 1 + sd::kRefreshSlackTokens + sd::kPage);
  LSE_EXPECT(sd::refresh_pages(1024) * sd::kPage >= 1024 + sd::kRefreshSlackTokens + sd::kPage);
  LSE_EXPECT(sd::kRecentPages * sd::kPage >= 64 + sd::kPage);
  LSE_EXPECT(sd::kMinBudgetTokens > (sd::kSinkPages + sd::kRecentPages) * sd::kPage);
}

int gpu_check() {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const auto& d = scheduler->backend().device_info();
  auto upload = [&](Shape shape, const std::vector<float>& data) {
    auto allocation = scheduler->backend().allocate(data.size() * 4, backend::MemoryClass::kDevice);
    if (!allocation.ok()) return Array{};
    auto buf = allocation.release();
    LSE_EXPECT_OK(scheduler->backend().copy(buf, data.data(), data.size() * 4));
    return Array::from_buffer(std::move(buf), shape, DType::kF32);
  };
  unsigned cases = 0;
  struct Case { const char* name; int rows; int live; MaskKind mask; };
  const Case cases_list[] = {
      {"decode", 1, 8000, MaskKind::kCausal},
      {"chain", 8, 8000, MaskKind::kCausal},
      {"tree", 8, 8000, MaskKind::kTree},
      {"decode-partial-page", 1, 7990, MaskKind::kCausal},
      {"identity", 1, 900, MaskKind::kCausal},
  };
  const int parent[8] = {-1, 0, 1, 2, 0, 4, 1, 6};
  for (const Case& cs : cases_list) {
    const int rows = cs.rows, live = cs.live, first = live - rows;
    Fixture f = make_fixture(rows);
    auto aq = upload({1, kHeads, rows, kWidth}, f.q);
    auto ak = cast(upload({kBlocks, kKvh, 16, kWidth}, f.k), DType::kBF16);
    auto av = cast(upload({kBlocks, kKvh, 16, kWidth}, f.v), DType::kBF16);
    std::vector<float> meta{float(first), float(live), 1, float(first), float(live)};
    std::vector<std::vector<bool>> visible(rows, std::vector<bool>(rows, false));
    for (int r = 0; r < rows; ++r) {
      if (cs.mask == MaskKind::kTree) {
        for (int a = r; a >= 0; a = parent[a]) visible[r][a] = true;
      } else {
        for (int a = 0; a <= r; ++a) visible[r][a] = true;
      }
    }
    if (cs.mask == MaskKind::kTree)
      for (int r = 0; r < rows; ++r) for (int c = 0; c < rows; ++c) meta.push_back(visible[r][c] ? 1.0f : 0.0f);
    auto am = upload({std::int64_t(meta.size())}, meta);
    auto at = upload({1, kPages}, f.table);
    // Summaries of every page: a refresh "pass" from position 0 over the whole context.
    auto summary_pool = cast(upload({kBlocks, kKvh, 2, kWidth}, std::vector<float>(std::size_t(kBlocks) * kKvh * 2 * kWidth)), DType::kBF16);
    auto refresh_meta = upload({5}, {0.0f, float(live), 1, 0.0f, float(live)});
    auto summary = sparse_decode_summary(summary_pool, ak, refresh_meta, at, live, kv::CacheDType::kBF16);
    auto out = sparse_decode_attention(aq, ak, av, kScale, cs.mask, am, at, 16, &d, kv::CacheDType::kBF16,
                                       summary, kBudget, false);
    if (!out.ok()) { std::fprintf(stderr, "%s: %s\n", cs.name, out.status().to_string().c_str()); return 1; }
    const NodePtr partial = out->node()->inputs[0];
    const NodePtr selected = partial->inputs[4];
    const NodePtr shifted = partial->inputs[3];
    const NodePtr roots[]{out->node(), selected, shifted};
    scheduler->reset_accumulated_trace();
    if (auto status = scheduler->eval(roots, false); !status.ok()) {
      std::fprintf(stderr, "%s eval: %s\n", cs.name, status.to_string().c_str());
      return 1;
    }
    LSE_EXPECT_OK(scheduler->drain());
    LSE_EXPECT_EQ(scheduler->last_trace().host_groups, 0u);
    LSE_EXPECT_EQ(scheduler->last_trace().host_fallbacks, 0u);
    std::vector<float> got(f.q.size()), picked(std::size_t(kKvh) * kBudgetPages), shifted_meta(meta.size());
    LSE_EXPECT_OK(out->to_host(got.data(), got.size() * 4));
    LSE_EXPECT_OK(Array(selected).to_host(picked.data(), picked.size() * 4));
    LSE_EXPECT_OK(Array(shifted).to_host(shifted_meta.data(), shifted_meta.size() * 4));

    const int pages = (live + 15) / 16;
    const bool all = pages <= kBudgetPages;
    const int shift = all ? 0 : (pages - kBudgetPages) * 16;
    LSE_EXPECT_EQ(shifted_meta[0], float(first - shift));
    LSE_EXPECT_EQ(shifted_meta[1], float(live - shift));
    LSE_EXPECT_EQ(shifted_meta[3], float(first - shift));
    LSE_EXPECT_EQ(shifted_meta[4], float(live - shift));
    for (std::size_t i = 5; i < meta.size(); ++i) LSE_EXPECT_EQ(shifted_meta[i], meta[i]);

    std::vector<int> page_of(kBlocks, -1);
    for (int p = 0; p < kPages; ++p) page_of[int(f.table[p])] = p;
    double max_error = 0, max_dense_gap = 0;
    unsigned hot_missed = 0, rank_violations = 0;
    for (int g = 0; g < kKvh; ++g) {
      // The positions each key head's rows read.
      std::vector<int> positions;
      if (all) {
        for (int j = 0; j < live; ++j) positions.push_back(j);
      } else {
        std::set<int> far;
        for (int i = 0; i < kBudgetPages; ++i) {
          const int page = page_of[int(picked[g * kBudgetPages + i])];
          if (i < int(sd::kSinkPages)) LSE_EXPECT_EQ(page, i);
          else if (i >= kBudgetPages - int(sd::kRecentPages))
            LSE_EXPECT_EQ(page, pages - int(sd::kRecentPages) + (i - (kBudgetPages - int(sd::kRecentPages))));
          else {
            LSE_EXPECT(page >= int(sd::kSinkPages) && page < pages - int(sd::kRecentPages));
            LSE_EXPECT(far.insert(page).second);
          }
          for (int s = 0; s < 16; ++s) if (page * 16 + s < live) positions.push_back(page * 16 + s);
        }
        for (int page : f.hot[g]) hot_missed += far.count(page) == 0;
        const auto scores = host_scores(f, rows, live, g);
        double kept_min = INFINITY, dropped_max = -INFINITY;
        for (int p = int(sd::kSinkPages); p < pages - int(sd::kRecentPages); ++p)
          (far.count(p) ? kept_min : dropped_max) = far.count(p) ? std::min(kept_min, scores[p])
                                                                 : std::max(dropped_max, scores[p]);
        rank_violations += kept_min + 1e-2 < dropped_max;
      }
      for (int hl = 0; hl < kGroup; ++hl) {
        const int h = g * kGroup + hl;
        for (int t = 0; t < rows; ++t) {
          auto attend = [&](const std::vector<int>& keys) {
            std::vector<double> logits;
            std::vector<int> used;
            for (int j : keys) {
              if (j >= first) {  // the pass's own rows
                if (!visible[t][j - first]) continue;
              }
              double dot = 0;
              for (int c = 0; c < kWidth; ++c) dot += double(f.q[(std::size_t(h) * rows + t) * kWidth + c]) * key_at(f, g, j, c);
              logits.push_back(dot * kScale);
              used.push_back(j);
            }
            const double top = *std::max_element(logits.begin(), logits.end());
            std::vector<double> o(kWidth, 0.0);
            double denom = 0;
            for (std::size_t i = 0; i < used.size(); ++i) {
              const double w = std::exp(logits[i] - top);
              denom += w;
              for (int c = 0; c < kWidth; ++c) o[c] += w * value_at(f, g, used[i], c);
            }
            for (auto& x : o) x /= denom;
            return o;
          };
          const auto want = attend(positions);
          std::vector<int> every(live);
          for (int j = 0; j < live; ++j) every[j] = j;
          const auto dense = attend(every);
          for (int c = 0; c < kWidth; ++c) {
            const double value = got[(std::size_t(h) * rows + t) * kWidth + c];
            LSE_EXPECT(std::isfinite(value));
            max_error = std::max(max_error, std::abs(value - want[c]));
            max_dense_gap = std::max(max_dense_gap, std::abs(value - dense[c]));
          }
        }
      }
    }
    LSE_EXPECT(max_error < 2e-2);
    LSE_EXPECT_EQ(hot_missed, 0u);
    LSE_EXPECT_EQ(rank_violations, 0u);
    std::printf("sparse decode %-20s rows=%d live=%d max_error=%.3g vs-dense=%.3g hot_missed=%u rank_violations=%u\n",
                cs.name, rows, live, max_error, max_dense_gap, hot_missed, rank_violations);
    ++cases;
  }
  std::printf("Sparse decode GPU reference cases: %u\n", cases);
  return test::Registry::get().failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--gpu") return gpu_check();
  return test::run_all();
}
