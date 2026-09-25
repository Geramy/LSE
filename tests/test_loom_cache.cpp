#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

#include <chrono>

namespace {
using namespace lse;
using namespace lse::graph;

Array leaf(Shape shape, DType type = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  node->materialized = true;
  return Array(node);
}

FusionGroup subtraction(bool reverse = false, bool alias = false) {
  auto a = leaf(Shape{128}), b = alias ? a : leaf(Shape{128});
  auto out = reverse ? b - a : a - b;
  FusionGroup group;
  group.nodes = {out.node()};
  group.inputs = alias ? std::vector<NodePtr>{a.node()}
                       : std::vector<NodePtr>{a.node(), b.node()};
  group.outputs = {out.node()};
  group.anchor = out.node()->kind;
  group.anchor_class = out.node()->fclass;
  return group;
}

backend::DeviceInfo device() {
  backend::DeviceInfo info;
  info.arch = "gfx1201";
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  return info;
}
}  // namespace

LSE_TEST(loom_cache_rebinds_new_graphs_without_retaining_old_nodes) {
  backend::LoomEmitter emitter;
  auto info = device();
  std::weak_ptr<Node> old_input;
  std::string source;
  {
    auto group = subtraction();
    old_input = group.inputs[0];
    auto emitted = emitter.emit(group, info);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) return;
    source = emitted->source;
  }
  LSE_EXPECT(old_input.expired());
  auto rebuilt = subtraction();
  auto cached = emitter.emit(rebuilt, info);
  auto fresh = backend::LoomEmitter{}.emit(rebuilt, info);
  LSE_EXPECT(cached.ok() && fresh.ok());
  if (!cached.ok() || !fresh.ok()) return;
  LSE_EXPECT(cached->source == source && cached->source == fresh->source);
  LSE_EXPECT(cached->binding_order == fresh->binding_order);
  LSE_EXPECT(cached->binding_order.front() == rebuilt.inputs.front());
  LSE_EXPECT_EQ(emitter.cache_stats().hits, 1u);
  LSE_EXPECT_EQ(emitter.cache_stats().entries, 1u);
}

LSE_TEST(loom_cache_identity_distinguishes_edges_aliases_constants_and_devices) {
  backend::LoomEmitter emitter;
  auto info = device();
  auto normal = subtraction(), reversed = subtraction(true);
  // The old display signature omits edges and collides for a-b and b-a.
  LSE_EXPECT_EQ(normal.signature(), reversed.signature());
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(reversed, info));
  auto aliased = subtraction(false, true);
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(aliased, info));
  auto wider = info;
  wider.wavefront_size = 64;
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(normal, wider));
  wider = info;
  wider.lds_bytes_per_workgroup /= 2;
  LSE_EXPECT(emitter.cache_key(normal, info) != emitter.cache_key(normal, wider));
  for (const auto* group : {&normal, &reversed, &aliased}) {
    auto result = emitter.emit(*group, info);
    LSE_EXPECT(result.ok());
  }
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 3u);
  LSE_EXPECT_EQ(emitter.cache_stats().hits, 0u);
  auto constant = subtraction();
  constant.inputs[0]->set_kind(OpKind::kConstant);
  constant.inputs[0]->attrs[0] = 1.0f;
  const auto one = emitter.cache_key(constant, info);
  constant.inputs[0]->attrs[0] = 2.0f;
  LSE_EXPECT(one != emitter.cache_key(constant, info));
}

LSE_TEST(loom_cache_reuses_q6_kernel_source_and_launch_metadata) {
  auto info = device();
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  auto out = quant_linear(leaf(Shape{1, 4096}),
      leaf(Shape{17, 768}, DType::kU32),
      leaf(Shape{17, 64}, DType::kBF16),
      leaf(Shape{17, 64}, DType::kBF16), 6, 64);
  const NodePtr roots[] = {out.node()};
  const auto groups = Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 1u);
  if (groups.size() != 1) return;
  backend::LoomEmitter emitter;
  auto cold_start = std::chrono::steady_clock::now();
  auto cold = emitter.emit(groups[0], info);
  auto cold_end = std::chrono::steady_clock::now();
  LSE_EXPECT(cold.ok());
  if (!cold.ok()) return;
  constexpr unsigned repeats = 100;
  for (unsigned i = 0; i < repeats; ++i) {
    auto hot = emitter.emit(groups[0], info);
    LSE_EXPECT(hot.ok());
    if (!hot.ok()) return;
    LSE_EXPECT(hot->source == cold->source);
    LSE_EXPECT(hot->binding_order == cold->binding_order);
    LSE_EXPECT_EQ(hot->lds_bytes, cold->lds_bytes);
    LSE_EXPECT_EQ(hot->dims.workgroup_count[0], cold->dims.workgroup_count[0]);
    LSE_EXPECT_EQ(hot->constants.total_bytes, cold->constants.total_bytes);
  }
  const auto hot_end = std::chrono::steady_clock::now();
  LSE_EXPECT_EQ(emitter.cache_stats().hits, repeats);
  LSE_EXPECT_EQ(emitter.cache_stats().misses, 1u);
  std::printf("    Q6 source emission: cold %.1f us, warm mean %.1f us (%u calls; host only)\n",
      std::chrono::duration<double, std::micro>(cold_end - cold_start).count(),
      std::chrono::duration<double, std::micro>(hot_end - cold_end).count() / repeats,
      repeats);
}

LSE_TEST_MAIN()
