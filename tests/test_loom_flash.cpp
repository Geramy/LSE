// Flash selection, typed metadata and aliased input binding without a GPU.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
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
  for (int seq : {8, 17, 33, 128}) {
    for (int capacity : {128, 320}) {
      for (bool alias : {false, true}) {
        auto q = leaf(Shape{2, 2, seq, 16});
        auto keys = leaf(Shape{2 * capacity / 16, 1, 16, 16});
        auto values = alias ? keys : leaf(keys.shape());
        auto meta = leaf(Shape{kv::step_meta_elems(2)});
        auto table = leaf(Shape{2, capacity / 16});
        auto output = sdpa_paged(q, keys, values, 0.25f, MaskKind::kCausal,
                                 0, meta, table, 16);
        const NodePtr roots[] = {output.node()};
        auto groups = Partitioner::partition(roots);
        LSE_EXPECT(groups.size() == 1);
        for (const auto& group : groups) {
          auto emitted = backend::LoomEmitter{}.emit(group, device);
          LSE_EXPECT(emitted.ok());
          if (emitted.ok()) {
            LSE_EXPECT(emitted->lds_bytes > 0);
            LSE_EXPECT(emitted->dims.workgroup_size[0] == 256);
            LSE_EXPECT(emitted->binding_order.size() == (alias ? 5u : 6u));
            LSE_EXPECT(emitted->source.find("extent_zero") != std::string::npos);
            LSE_EXPECT(emitted->source.find("kernel.barrier") != std::string::npos);
          }
        }
      }
    }
  }
}

LSE_TEST_MAIN()
