#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_shapes.hpp"
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

LSE_TEST(short_split_derives_capacity_and_two_ordered_native_stages) {
  backend::LoomEmitter emitter;
  for (int queries : {2, 3, 4, 5, 6, 7, 8}) {
    for (int capacity : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 69632, 131072, 262144}) {
      Fixture fx(queries, capacity);
      auto baseline = sdpa_paged(fx.q, fx.k, fx.v, 0.0625f, MaskKind::kCausal,
                                 0, fx.meta, fx.table, 16);
      LSE_EXPECT(baseline.node()->prim->name() == "attention");
      auto split = fx.split();
      LSE_EXPECT(split.shape() == baseline.shape());
      LSE_EXPECT(split.node()->prim->name() == "attention.split_merge128.wg128c2.v1");
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
        const auto tile = dispatch::attention_shapes::short_query_tile(
            static_cast<std::uint32_t>(queries), static_cast<std::uint32_t>(capacity));
        const auto head_tile = dispatch::attention_shapes::short_head_tile(
            static_cast<std::uint32_t>(queries), 24u, 4u, static_cast<std::uint32_t>(capacity));
        LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], partial
                      ? (24u / head_tile) * ((static_cast<unsigned>(queries) + tile - 1u) / tile) * static_cast<unsigned>(capacity / 128)
                      : 24u * static_cast<unsigned>(queries));
        LSE_EXPECT_EQ(emitted->lds_bytes, partial ? tile * head_tile * 512u : static_cast<unsigned>(capacity / 32));
        LSE_EXPECT_EQ(emitted->binding_order.size(), partial ? 6u : 2u);
        LSE_EXPECT(emitted->source.find("scalar.fmaf") != std::string::npos);
      }
      const auto split_key = emitter.cache_key(groups[0], fx.gpu);
      groups[0].outputs[0]->prim = find_primitive("attention");
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
    if (variant == 9) fx.table = leaf({1, 1025});
    if (variant == 10) fx.meta = leaf({1});
    if (variant == 12) {
      fx.q = leaf({2, 24, 3, 256});
      fx.meta = leaf({kv::step_meta_elems(2)});
      fx.table = leaf({2, 64});
    }
    if (variant == 13) fx.v = leaf({65, 4, 16, 128});
    if (variant == 14) fx.table = leaf({1, 32});
    if (variant == 15) fx.k = fx.v = leaf({65, 2, 16, 256});
    auto out = variant == 11 ? fx.split(MaskKind::kSlidingWindow, -1) : fx.split();
    LSE_EXPECT(out.node()->prim->name() ==
               (variant == 4 ? "attention.split_merge128.wg128c2.v1" :
                variant == 9 ? "attention.split_merge128.wg128c2.v1" : "attention"));
  }
}

LSE_TEST(short_split_scope_uses_actual_capacity_and_known_prefix) {
  for (int queries : {1, 2, 3, 4, 5, 6, 7, 8, 9})
    for (int capacity : {512, 1024, 2048, 4096, 8192, 16384, 16385, 32768, 65536, 262100}) {
      const Shape q{1, 24, queries, 256};
      const bool accepted = ((queries >= 2 && queries <= 8) &&
                             (capacity >= 1024));
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

LSE_TEST(short_split_merge_covers_more_partitions_than_weight_writers) {
  backend::LoomEmitter emitter;
  Fixture fx(6, 16384);
  for (int partitions : {64, 65, 128, 129, 256, 512, 544, 1024, 2048}) {
    auto partial = leaf({1, 24, 6, partitions, 258});
    auto merged = custom("attention.split_merge128.wg128c2.v1", {partial});
    LSE_EXPECT(merged.ok());
    if (!merged.ok()) return;
    const NodePtr roots[]{merged->node()};
    auto groups = Partitioner::partition(roots, &fx.gpu);
    LSE_EXPECT_EQ(groups.size(), 1u);
    if (groups.size() != 1) return;
    auto emitted = emitter.emit(groups[0], fx.gpu);
    LSE_EXPECT(emitted.ok());
    if (emitted.ok()) {
      LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 128u);
      LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], 24u * 6u);
      LSE_EXPECT_EQ(emitted->lds_bytes, static_cast<unsigned>(((partitions * 4 + 15) / 16) * 16));
    }
  }
}

LSE_TEST(short_split_large_declared_limit_uses_actual_table_geometry) {
  backend::LoomEmitter emitter;
  constexpr int declared_limit = 262100;
  for (const auto [live, capacity] : {std::pair{5610, 8192}, std::pair{14000, 16384}}) {
    LSE_EXPECT(kv::blocks_for(live, 16) <= capacity / 16);
    LSE_EXPECT(capacity < declared_limit);
    Fixture fx(4, capacity);
    LSE_EXPECT(dispatch::split_short_scope(fx.q.shape(), live - 4, capacity));
    LSE_EXPECT(dispatch::split_short_scope(fx.q.shape(), live - 4, declared_limit));
    auto split = fx.split();
    LSE_EXPECT(split.node()->prim->name() == "attention.split_merge128.wg128c2.v1");
    const auto partial = split.node()->inputs[0];
    LSE_EXPECT(partial->shape == Shape({1, 24, 4, capacity / 128, 258}));
    LSE_EXPECT_EQ(partial->element_count() * sizeof(float),
                  static_cast<std::size_t>(24 * 4 * (capacity / 128) * 258 * 4));
  }
  Fixture ragged(6, 8208);
  auto split = ragged.split();
  LSE_EXPECT(split.node()->prim->name() == "attention.split_merge128.wg128c2.v1");
  LSE_EXPECT(split.node()->inputs[0]->shape == Shape({1, 24, 6, 65, 258}));
  const NodePtr roots[]{split.node()};
  auto groups = Partitioner::partition(roots, &ragged.gpu);
  LSE_EXPECT_EQ(groups.size(), 2u);
  if (groups.size() != 2) return;
  auto partial = emitter.emit(groups[0], ragged.gpu);
  auto merge = emitter.emit(groups[1], ragged.gpu);
  LSE_EXPECT(partial.ok() && merge.ok());
  if (!partial.ok() || !merge.ok()) return;
  LSE_EXPECT_EQ(partial->dims.workgroup_count[0], 12u * 2u * 65u);
  LSE_EXPECT_EQ(merge->lds_bytes, 272u);
}

LSE_TEST(short_split_signed_masks_handle_query_offsets_at_u32_boundary) {
  backend::LoomEmitter emitter;
  for (auto mask : {MaskKind::kNone, MaskKind::kCausal, MaskKind::kSlidingWindow}) {
    Fixture fx(3, 1024);
    auto inputs = std::vector<Array>{fx.q, fx.k, fx.v, fx.meta, fx.table};
    auto partial = custom("attention.split_partial128.wg128c2.v1", inputs, {0.0625f, 0, 0, 0});
    LSE_EXPECT(partial.ok());
    if (!partial.ok()) return;
    partial->node()->iattrs = {static_cast<int>(mask), 7, 0, 16};
    auto merged = custom("attention.split_merge128.wg128c2.v1", {*partial});
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
      for (std::uint64_t key : {0ull, 511ull, 8191ull, 16383ull}) {
        const auto position = offset + query;
        const bool causal = key <= position;
        const bool sliding = causal && position - key < 7;
        const auto signed_distance = static_cast<std::int64_t>(position) - static_cast<std::int64_t>(key);
        LSE_EXPECT(sliding == (causal && signed_distance < 7));
      }
}
LSE_TEST(short_split_query_tile_uses_width_and_matching_lds_contract) {
  for (int queries : {1, 2, 3, 4, 5, 6, 7, 8, 9})
    for (int capacity : {1024, 2048, 4096, 8192, 8208, 16384, 32768, 65536, 69632, 262144}) {
      const auto tile = dispatch::attention_shapes::short_query_tile(
          static_cast<std::uint32_t>(queries), static_cast<std::uint32_t>(capacity));
      const bool measured = queries >= 4 && queries <= 8 &&
                            capacity >= 8192;
      LSE_EXPECT_EQ(tile, measured ? 4u : 1u);
    }
  for (int capacity : {8192, 16384})
    for (unsigned lds : {512u, 2048u, 4095u, 4096u}) {
      Fixture fx(4, capacity);
      fx.gpu.lds_bytes_per_workgroup = lds;
      auto out = fx.split();
      LSE_EXPECT(out.node()->prim->name() ==
          (lds >= 4096u ? "attention.split_merge128.wg128c2.v1" : "attention"));
    }
}
LSE_TEST(short_split_empty_partition_shortcut_follows_query_tile) {
  for (std::uint32_t queries : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u})
    for (std::uint32_t capacity : {1024u, 2048u, 4096u, 8192u, 8208u, 16384u, 32768u, 65536u, 69632u, 262144u})
      LSE_EXPECT(dispatch::attention_shapes::short_skips_empty_partitions(queries, capacity) ==
                 (queries >= 4u && queries <= 8u &&
                  capacity >= 8192u));
}

LSE_TEST(short_split_checks_merge_lds_before_selecting_both_stages) {
  for (unsigned lds : {2048u, 2175u, 2176u, 4095u, 4096u, 8192u}) {
    Fixture fx(8, 69632);
    fx.gpu.lds_bytes_per_workgroup = lds;
    const auto output = fx.split();
    LSE_EXPECT(output.node()->prim->name() ==
        (lds >= 4096u ? "attention.split_merge128.wg128c2.v1" : "attention"));
  }
  Fixture fx(8, 262144);
  fx.gpu.lds_bytes_per_workgroup = 8191;
  LSE_EXPECT(fx.split().node()->prim->name() == "attention");
  fx.gpu.lds_bytes_per_workgroup = 8192;
  LSE_EXPECT(fx.split().node()->prim->name() == "attention.split_merge128.wg128c2.v1");
}

LSE_TEST_MAIN()
