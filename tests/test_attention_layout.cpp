// Dimension-major value pools (kv::dimension_major_values) give the same
// attention output, byte for byte, as row-major ones holding the same values,
// on every path the model takes: split decode, short split, tree flash split.
#include <cstring>
#include <random>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"

using namespace lse;

namespace {
graph::Array filled(Shape shape, DType dtype, const std::vector<float>& v) {
  graph::Array a = graph::Array::zeros(shape, dtype);
  auto* sched = graph::default_scheduler();
  graph::Node& n = *a.node();
  if (!graph::interpreter::ensure_output_buffer(n, sched->backend()).ok()) return {};
  for (std::size_t i = 0; i < v.size(); ++i) graph::interpreter::store_element(n, i, v[i]);
  n.materialized = true;
  if (!graph::interpreter::sync_to_device(n, sched->backend()).ok()) return {};
  return a;
}
}  // namespace

LSE_TEST(dimension_major_values_match_row_major_values) {
  auto* scheduler = graph::default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler) return;
  if (!scheduler->backend().emitter()) LSE_SKIP("requires native attention dispatch");
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  const auto& device = scheduler->backend().device_info();
  constexpr int heads = 24, kvh = 4, dim = 256, block = 16;
  for (const auto storage : {kv::CacheDType::kBF16, kv::CacheDType::kF16}) {
    const DType dtype = kv::storage_dtype(storage);
    for (const int capacity : {2048, 8192}) {
      const int live = capacity - 37, blocks = capacity / block;
      std::mt19937 rng(77);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<float> keys(static_cast<std::size_t>(blocks) * kvh * block * dim), values(keys.size()),
          transposed(keys.size());
      for (auto& x : keys) x = nd(rng) * 0.5f;
      for (auto& x : values) x = nd(rng) * 0.5f;
      for (int b = 0; b < blocks; ++b)
        for (int h = 0; h < kvh; ++h)
          for (int t = 0; t < block; ++t)
            for (int d = 0; d < dim; ++d)
              transposed[((static_cast<std::size_t>(b) * kvh + h) * dim + d) * block + t] =
                  values[((static_cast<std::size_t>(b) * kvh + h) * block + t) * dim + d];
      auto k = filled(Shape{blocks, kvh, block, dim}, dtype, keys);
      auto v = filled(Shape{blocks, kvh, block, dim}, dtype, values);
      auto vt = filled(Shape{blocks, kvh, dim, block}, dtype, transposed);
      LSE_EXPECT(dispatch::dimension_major_values(k.shape(), vt.shape()));
      std::vector<float> table(static_cast<std::size_t>(blocks));
      for (int i = 0; i < blocks; ++i) table[static_cast<std::size_t>(i)] = static_cast<float>((i * 7 + 3) % blocks);
      auto t = filled(Shape{1, blocks}, DType::kF32, table);
      for (const int rows : {1, 2, 3, 4, 8, 9, 15, 31}) {
        for (const bool tree : {false, true}) {
          if ((tree && rows < 2) || (!tree && rows > 8)) continue;
          std::vector<float> q(static_cast<std::size_t>(heads) * rows * dim);
          for (auto& x : q) x = nd(rng) * 2.0f;
          auto query = filled(Shape{1, heads, rows, dim}, DType::kF32, q);
          const int first = live - rows;
          std::vector<float> meta(static_cast<std::size_t>(tree ? kv::tree_meta_elems(1, rows)
                                                                : kv::step_meta_elems(1)), 0.0f);
          meta[0] = static_cast<float>(first); meta[1] = static_cast<float>(live); meta[2] = 1.0f;
          meta[kv::kStepMetaHeader] = static_cast<float>(first);
          meta[kv::kStepMetaHeader + 1] = static_cast<float>(live);
          if (tree) {
            std::vector<int> parent(static_cast<std::size_t>(rows), -1);
            for (int r = 1; r < rows; ++r) parent[static_cast<std::size_t>(r)] = static_cast<int>(rng() % static_cast<unsigned>(r));
            for (int r = 0; r < rows; ++r)
              for (int a = r; a >= 0; a = parent[static_cast<std::size_t>(a)])
                meta[static_cast<std::size_t>(kv::tree_mask_offset(1) + r * rows + a)] = 1.0f;
          }
          auto m = filled(Shape{static_cast<std::int64_t>(meta.size())}, DType::kF32, meta);
          std::vector<float> out[2];
          for (int layout = 0; layout < 2; ++layout) {
            auto o = graph::sdpa_paged(query, k, layout ? vt : v, 0.0625f,
                                       tree ? graph::MaskKind::kTree : graph::MaskKind::kCausal,
                                       0, m, t, block, &device, storage);
            graph::Program program;
            const graph::NodePtr roots[]{o.node()};
            LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
            const auto trace = scheduler->last_trace();
            LSE_EXPECT_EQ(trace.host_groups, 0u);
            out[layout].resize(static_cast<std::size_t>(o.shape().elem_count()));
            LSE_EXPECT_OK(o.to_host(out[layout].data(), out[layout].size() * sizeof(float)));
          }
          LSE_EXPECT(!out[0].empty() && out[0].size() == out[1].size() &&
                     std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0);
        }
      }
    }
  }
}

LSE_TEST_MAIN()
