// Convolution tail lowering across the retained-history boundary, without a
// GPU.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

LSE_TEST(convolution_tail_merges_guarded_values_for_decode_and_prefill) {
  using namespace lse;
  using namespace lse::graph;
  auto input = [](Shape shape) {
    auto n = std::make_shared<Node>();
    n->shape = shape;
    n->dtype = DType::kF32;
    n->materialized = true;
    return Array(n);
  };
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  for (int seq : {1, 2, 3, 7}) {
    auto tail = input(Shape{2, 3, 17});
    auto x = input(Shape{2, seq, 17});
    auto output = conv_tail(tail, x);
    LSE_EXPECT(output.node()->prim != nullptr);
    const NodePtr roots[] = {output.node()};
    for (const auto& group : Partitioner::partition(roots)) {
      auto result = backend::LoomEmitter{}.emit(group, device);
      LSE_EXPECT(result.ok());
      if (result.ok()) {
        LSE_EXPECT(result->source.find("view.load") != std::string::npos);
        LSE_EXPECT(result->source.find("view.store") != std::string::npos);
        if (seq < 3) {
          LSE_EXPECT(result->source.find("scf.yield") != std::string::npos);
        }
      }
    }
  }
}

LSE_TEST_MAIN()
