// Flash selection, typed metadata and aliased input binding without a GPU.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/cache.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"

LSE_TEST(flash_attention_handles_long_ragged_queries_and_bound_cache_windows) {
  using namespace lse;
  using namespace lse::graph;
  auto leaf = [](Shape shape) {
    auto node = std::make_shared<Node>();
    node->shape = shape;
    node->dtype = DType::kF32;
    node->materialized = true;
    return Array(node);
  };
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  device.lds_bytes_per_workgroup = 65536;
  for (int seq : {2, 3, 7, 8, 17, 33, 128}) {
    for (int capacity : {128, 320}) {
      for (auto mask : {MaskKind::kCausal, MaskKind::kSlidingWindow})
        for (bool alias : {false, true}) {
          auto q = leaf(Shape{2, 2, seq, 16});
          auto keys = leaf(Shape{2 * capacity / 16, 1, 16, 16});
          auto values = alias ? keys : leaf(keys.shape());
          auto meta = leaf(Shape{kv::step_meta_elems(2)});
          auto table = leaf(Shape{2, capacity / 16});
          auto output = sdpa_paged(q, keys, values, 0.25f, mask,
                                   5, meta, table, 16);
          const NodePtr roots[] = {output.node()};
          auto groups = Partitioner::partition(roots);
          LSE_EXPECT(groups.size() == 1);
          for (const auto& group : groups) {
            auto emitted = backend::LoomEmitter{}.emit(group, device);
            LSE_EXPECT(emitted.ok());
            if (emitted.ok()) {
              LSE_EXPECT(emitted->lds_bytes > 0);
              LSE_EXPECT(emitted->dims.workgroup_size[0] == 256);
              const unsigned tile = seq >= 12 ? 12u : 8u;
              LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],
                            4u * ((static_cast<unsigned>(seq) + tile - 1u) / tile));
              LSE_EXPECT(emitted->binding_order.size() == (alias ? 5u : 6u));
              LSE_EXPECT(emitted->source.find("extent_zero") != std::string::npos);
              LSE_EXPECT(emitted->source.find("kernel.barrier") != std::string::npos);
            }
          }
        }
    }
  }
}

LSE_TEST(attention_selected_implementation_changes_persistent_cache_identity) {
  using namespace lse;
  using namespace lse::graph;
  auto leaf = [](Shape shape) {
    auto node = std::make_shared<Node>();
    node->shape = shape;
    node->dtype = DType::kF32;
    node->materialized = true;
    return Array(node);
  };
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  device.lds_bytes_per_workgroup = 65536;
  auto output = sdpa_paged(leaf({1, 24, 1, 256}), leaf({128, 4, 16, 256}),
                           leaf({128, 4, 16, 256}), 0.0625f,
                           MaskKind::kCausal, 0, leaf({kv::step_meta_elems(1)}),
                           leaf({1, 128}), 16);
  FusionGroup group;
  group.nodes = {output.node()};
  group.outputs = group.nodes;
  group.inputs = output.node()->inputs;
  group.anchor = output.node()->kind;
  group.anchor_class = output.node()->fclass;
  const auto* shared = find_primitive("attention.decode_shared");
  const auto* shared_exp = find_primitive("attention.decode_shared_exp");
  LSE_EXPECT(shared != nullptr);
  LSE_EXPECT(shared_exp != nullptr);
  if (shared == nullptr || shared_exp == nullptr) return;
  const auto types = backend::loom_types();
  const auto intrinsics = backend::loom_sources();
  backend::LoomEmitter emitter;
  output.node()->prim = shared;
  const auto shared_policy = dispatch::specialization_cache_key(
      0, group, device, types, intrinsics);
  const auto shared_key = emitter.cache_key(group, device);
  output.node()->prim = shared_exp;
  const auto exp_policy = dispatch::specialization_cache_key(
      0, group, device, types, intrinsics);
  const auto exp_key = emitter.cache_key(group, device);
  LSE_EXPECT(shared_policy != exp_policy);
  LSE_EXPECT(shared_key != exp_key);
}


LSE_TEST(split_decode_attention_orders_two_barriers_and_versions_both_kernels) {
  using namespace lse;
  using namespace lse::graph;
  auto leaf = [](Shape shape, DType dtype = DType::kF32) {
    auto n = std::make_shared<Node>();
    n->shape = shape; n->dtype = dtype; n->materialized = true;
    return Array(n);
  };
  backend::DeviceInfo device;
  device.arch = "gfx1201"; device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  device.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter emitter;
  constexpr auto partial_name = "attention.decode_partial128.wg128c2.v2";
  constexpr auto merge_name = "attention.decode_merge128.wg128c2.v2";
  for (int capacity : {128, 320, 2048}) {
    for (auto mask : {MaskKind::kNone, MaskKind::kCausal, MaskKind::kSlidingWindow}) {
      auto q = leaf({2, 24, 1, 256});
      auto k = leaf({2 * capacity / 16, 4, 16, 256});
      auto v = leaf(k.shape());
      auto meta = leaf({kv::step_meta_elems(2)});
      auto table = leaf({2, capacity / 16});
      auto output = sdpa_paged(q, k, v, 0.0625f, mask, 65, meta, table, 16, &device);
      LSE_EXPECT(output.shape() == (Shape{2, 24, 1, 256}));
      LSE_EXPECT(output.node()->prim->name() == merge_name);
      auto partial = output.node()->inputs[0];
      LSE_EXPECT(partial->prim->name() == partial_name);
      LSE_EXPECT(partial->shape == (Shape{2, 24, (capacity + 127) / 128, 258}));
      const NodePtr roots[] = {output.node()};
      auto groups = Partitioner::partition(roots);
      LSE_EXPECT_EQ(groups.size(), 2u);
      if (groups.size() != 2) continue;
      LSE_EXPECT(groups[0].outputs[0] == partial);
      LSE_EXPECT(groups[1].inputs[0] == partial);
      const auto partial_key = emitter.cache_key(groups[0], device);
      const auto merge_key = emitter.cache_key(groups[1], device);
      struct Revision final : KernelPrimitiveBase {
        const KernelPrimitiveBase& base;
        std::string revised_name;
        explicit Revision(const KernelPrimitiveBase& p)
            : base(p), revised_name(std::string(p.name()) + ".test-forwarding") {}
        std::string_view name() const noexcept override { return revised_name; }
        std::string_view entry_name() const noexcept override { return base.entry_name(); }
        std::size_t arity() const noexcept override { return base.arity(); }
        Result<Shape> infer_shape(std::span<const Shape> in) const override { return base.infer_shape(in); }
        DType infer_dtype(std::span<const DType> in) const override { return base.infer_dtype(in); }
        std::string emit_kernel(const KernelShapes& in) const override { return base.emit_kernel(in); }
        ThreadPlan plan(const KernelShapes& in) const override { return base.plan(in); }
        void eval_cpu(std::span<const float* const>, float*, std::size_t,
                      const std::array<float, 4>&) const override {}
      };
      const NodePtr stages[] = {partial, output.node()};
      const std::uint64_t original_keys[] = {partial_key, merge_key};
      for (std::size_t i = 0; i < 2; ++i) {
        const auto* original = stages[i]->prim;
        const auto* kernel = dynamic_cast<const KernelPrimitiveBase*>(original);
        LSE_EXPECT(kernel != nullptr);
        if (kernel == nullptr) continue;
        Revision revision(*kernel);
        stages[i]->prim = &revision;
        LSE_EXPECT(emitter.cache_key(groups[i], device) != original_keys[i]);
        stages[i]->prim = original;
      }
      LSE_EXPECT(partial_key != merge_key);
      for (std::size_t i = 0; i < groups.size(); ++i) {
        auto emitted = emitter.emit(groups[i], device);
        LSE_EXPECT(emitted.ok());
        if (!emitted.ok()) continue;
        LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 128u);
        LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],
            i == 0 ? 48u * static_cast<unsigned>((capacity + 127) / 128) : 48u);
        LSE_EXPECT_EQ(emitted->lds_bytes,
            i == 0 ? 512u : (4u * static_cast<unsigned>((capacity + 127) / 128) + 15u) / 16u * 16u);
        LSE_EXPECT(emitted->source.find("kernel.barrier") != std::string::npos);
      }
      if (capacity == 128 && mask == MaskKind::kNone) {
        const auto* legacy_partial = find_primitive("attention.decode_partial128.v1");
        const auto* legacy_merge = find_primitive("attention.decode_merge128.v1");
        LSE_EXPECT(legacy_partial != nullptr && legacy_merge != nullptr);
        if (legacy_partial && legacy_merge) {
          const auto* selected_partial = partial->prim;
          const auto* selected_merge = output.node()->prim;
          partial->prim = legacy_partial;
          output.node()->prim = legacy_merge;
          for (std::size_t i = 0; i < groups.size(); ++i) {
            LSE_EXPECT(emitter.cache_key(groups[i], device) != original_keys[i]);
            const auto emitted = emitter.emit(groups[i], device);
            LSE_EXPECT(emitted.ok());
            if (emitted.ok()) {
              LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 256u);
              LSE_EXPECT_EQ(emitted->lds_bytes, i == 0 ? 512u : 16u);
              LSE_EXPECT(emitted->source.find("kernel.barrier") != std::string::npos);
            }
          }
          partial->prim = selected_partial;
          output.node()->prim = selected_merge;
        }
      }
      auto baseline = sdpa_paged(q, k, v, 0.0625f, mask, 65, meta, table, 16);
      const NodePtr baseline_roots[] = {baseline.node()};
      auto baseline_groups = Partitioner::partition(baseline_roots);
      LSE_EXPECT_EQ(baseline_groups.size(), 1u);
      LSE_EXPECT(emitter.cache_key(baseline_groups[0], device) != partial_key);
      LSE_EXPECT(emitter.cache_key(baseline_groups[0], device) != merge_key);
    }
  }
}

LSE_TEST(split_decode_attention_preserves_portable_graph_for_unsupported_requests) {
  using namespace lse;
  using namespace lse::graph;
  auto leaf = [](Shape shape, DType dtype = DType::kF32) {
    auto n = std::make_shared<Node>();
    n->shape = shape; n->dtype = dtype; n->materialized = true;
    return Array(n);
  };
  backend::DeviceInfo device;
  device.arch = "gfx1201"; device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  device.lds_bytes_per_workgroup = 65536;
  for (int which = 0; which < 8; ++which) {
    auto d = device;
    if (which == 0) d.arch = "gfx1100";
    if (which == 1) d.max_threads_per_workgroup = 128;
    if (which == 7) d.wavefront_size = 64;
    auto q = leaf({1, 24, which == 2 ? 2 : 1, 256}, which == 3 ? DType::kF16 : DType::kF32);
    auto k = leaf({128, 4, 16, 256});
    auto meta = leaf({which == 4 ? 1 : kv::step_meta_elems(1)});
    auto table = leaf({1, which == 5 ? 1024 : 128});
    auto out = sdpa_paged(q, k, k, 0.0625f, MaskKind::kSlidingWindow,
                          which == 6 ? -1 : 65, meta, table, 16, &d);
    LSE_EXPECT(out.node()->prim->name() == "attention");
    LSE_EXPECT(out.node()->kind == OpKind::kAttention);
  }
}

LSE_TEST_MAIN()
