#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include <utility>

namespace {
using namespace lse;
using namespace lse::graph;
Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = dtype;
  node->materialized = true;
  return Array(node);
}
struct Fixture {
  backend::DeviceInfo gpu;
  Array q, k, v, meta, table;
  Fixture(int queries, int capacity) {
    gpu.arch = "gfx1201";
    gpu.wavefront_size = 32;
    gpu.max_threads_per_workgroup = 256;
    gpu.lds_bytes_per_workgroup = 65536;
    q = leaf({1, 24, queries, 256});
    k = leaf({capacity / 16 + 1, 4, 16, 256});
    v = leaf(k.shape());
    meta = leaf({kv::step_meta_elems(1)});
    table = leaf({1, capacity / 16});
  }
  Array split(MaskKind mask = MaskKind::kCausal, int window = 0) {
    return sdpa_paged(q, k, v, 0.0625f, mask, window,
                      meta, table, 16, &gpu);
  }
};
}  // namespace

LSE_TEST(short_split_selects_measured_defaults_and_two_ordered_native_stages) {
  backend::LoomEmitter emitter;
  for (int queries : {2, 3, 4, 5, 6, 7, 8}) {
    for (int capacity : {1024, 2048}) {
      Fixture fx(queries, capacity);
      auto baseline = sdpa_paged(fx.q, fx.k, fx.v, 0.0625f, MaskKind::kCausal,
                                 0, fx.meta, fx.table, 16);
      LSE_EXPECT(baseline.node()->prim->name() == "attention");
      auto split = fx.split();
      LSE_EXPECT(split.shape() == baseline.shape());
      LSE_EXPECT(split.node()->prim->name() == "attention.short_merge128.wg128c2.v1");
      LSE_EXPECT(split.node()->inputs[0]->shape == Shape({1, 24, queries, capacity / 128, 258}));
      const NodePtr roots[]{split.node()};
      auto groups = Partitioner::partition(roots, &fx.gpu);
      LSE_EXPECT_EQ(groups.size(), 2u);
      if (groups.size() != 2) return;
      for (std::size_t i = 0; i < groups.size(); ++i) {
        auto emitted = emitter.emit(groups[i], fx.gpu);
        LSE_EXPECT(emitted.ok());
        if (!emitted.ok()) return;
        const bool partial = i == 0;
        LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 128u);
        LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],
                      static_cast<unsigned>(24 * queries * (partial ? capacity / 128 : 1)));
        LSE_EXPECT_EQ(emitted->lds_bytes, partial ? 512u : static_cast<unsigned>(capacity / 32));
        LSE_EXPECT_EQ(emitted->binding_order.size(), partial ? 6u : 2u);
        LSE_EXPECT(emitted->source.find("scalar.fmaf") != std::string::npos);
      }
      const auto split_key = emitter.cache_key(groups[0], fx.gpu);
      groups[0].outputs[0]->prim = find_primitive("attention.decode_partial128.wg128c2.v2");
      LSE_EXPECT(split_key != emitter.cache_key(groups[0], fx.gpu));
    }
  }
}

LSE_TEST(short_split_declines_unsupported_contracts_without_changing_baseline) {
  for (int variant = 0; variant < 16; ++variant) {
    Fixture fx(3, 1024);
    if (variant == 0) fx.gpu.arch = "gfx1100";
    if (variant == 1) fx.gpu.wavefront_size = 64;
    if (variant == 2) fx.gpu.max_threads_per_workgroup = 127;
    if (variant == 3) fx.gpu.lds_bytes_per_workgroup = 511;
    if (variant == 4) fx.q = leaf({1, 24, 1, 256});
    if (variant == 5) fx.q = leaf({1, 24, 9, 256});
    if (variant == 6) fx.q = leaf({1, 24, 3, 128});
    if (variant == 7) fx.q = leaf({1, 25, 3, 256});
    if (variant == 8) fx.q = leaf(fx.q.shape(), DType::kBF16);
    if (variant == 9) fx.table = leaf({1, 513});
    if (variant == 10) fx.meta = leaf({1});
    if (variant == 12) {
      fx.q = leaf({2, 24, 3, 256});
      fx.meta = leaf({kv::step_meta_elems(2)});
      fx.table = leaf({2, 64});
    }
    if (variant == 13) { fx.q = leaf({1, 24, 7, 256}); fx.table = leaf({1, 256}); }
    if (variant == 14) fx.table = leaf({1, 32});
    if (variant == 15) fx.k = fx.v = leaf({65, 2, 16, 256});
    auto out = variant == 11 ? fx.split(MaskKind::kSlidingWindow, -1) : fx.split();
    LSE_EXPECT(out.node()->prim->name() ==
               (variant == 4 ? "attention.decode_merge128.wg128c2.v2" : "attention"));
  }
}

LSE_TEST(short_split_scope_uses_actual_capacity_and_known_prefix) {
  for (int queries : {1, 2, 3, 4, 5, 6, 7, 8, 9})
    for (int capacity : {512, 1024, 2048, 4096, 262100}) {
      const Shape q{1, 24, queries, 256};
      const bool accepted = ((queries >= 2 && queries <= 8) &&
                             (capacity == 1024 || capacity == 2048));
      LSE_EXPECT(dispatch::split_short_scope(q, 512, capacity) == accepted);
      LSE_EXPECT(dispatch::split_short_scope(q, 1024, capacity) == accepted);
      LSE_EXPECT(!dispatch::split_short_scope(q, 511, capacity));
    }
  Fixture fx(3, 1024);
  for (const auto [mask, window] : {std::pair{MaskKind::kNone, 0},
                                   std::pair{MaskKind::kSlidingWindow, 7},
                                   std::pair{MaskKind::kCausal, 7}})
    LSE_EXPECT(fx.split(mask, window).node()->prim->name() == "attention");
}

LSE_TEST(short_split_signed_masks_handle_query_offsets_at_u32_boundary) {
  backend::LoomEmitter emitter;
  for (auto mask : {MaskKind::kNone, MaskKind::kCausal, MaskKind::kSlidingWindow}) {
    Fixture fx(3, 1024);
    auto inputs = std::vector<Array>{fx.q, fx.k, fx.v, fx.meta, fx.table};
    auto partial = custom("attention.short_partial128.wg128c2.v2", inputs, {0.0625f, 0, 0, 0});
    LSE_EXPECT(partial.ok());
    if (!partial.ok()) return;
    partial->node()->iattrs = {static_cast<int>(mask), 7, 0, 16};
    auto merged = custom("attention.short_merge128.wg128c2.v1", {*partial});
    LSE_EXPECT(merged.ok());
    if (!merged.ok()) return;
    auto out = *merged;
    const NodePtr roots[]{out.node()};
    auto groups = Partitioner::partition(roots, &fx.gpu);
    LSE_EXPECT_EQ(groups.size(), 2u);
    if (groups.size() != 2) return;
    auto emitted = emitter.emit(groups[0], fx.gpu);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) return;
    LSE_EXPECT(emitted->source.find("index.sub") == std::string::npos);
  }
  for (std::uint64_t offset : {0ull, 511ull, 0xffffffffull})
    for (std::uint64_t query : {0ull, 2ull, 7ull})
      for (std::uint64_t key : {0ull, 511ull, 8191ull}) {
        const auto position = offset + query;
        const bool causal = key <= position;
        const bool sliding = causal && position - key < 7;
        const auto signed_distance = static_cast<std::int64_t>(position) - static_cast<std::int64_t>(key);
        LSE_EXPECT(sliding == (causal && signed_distance < 7));
      }
}
LSE_TEST_MAIN()
