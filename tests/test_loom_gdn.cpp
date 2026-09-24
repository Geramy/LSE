// GDN emission is host-only; no GPU runtime or device allocation is used.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

namespace {
using namespace lse;
using namespace lse::graph;
Array input(Shape shape) {
  auto n = std::make_shared<Node>();
  n->shape = shape;
  n->dtype = DType::kF32;
  n->materialized = true;
  return Array(n);
}
void check(int dim, int seq, int wave, int outputs) {
  auto q = input(Shape{1, seq, 48, dim});
  auto k = input(q.shape()), v = input(q.shape());
  auto alpha = input(Shape{1, seq, 48}), beta = input(alpha.shape());
  auto state = input(Shape{1, 48, dim, dim});
  Array next;
  auto out = gated_delta_step(q, k, v, alpha, beta, state, &next);
  std::vector<NodePtr> roots;
  if (outputs != 1) roots.push_back(out.node());
  if (outputs != 0) roots.push_back(next.node());
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.wavefront_size = static_cast<std::uint32_t>(wave);
  device.max_threads_per_workgroup = 1024;
  std::size_t emitted_count = 0;
  for (const auto& group : Partitioner::partition(roots)) {
    auto result = backend::LoomEmitter{}.emit(group, device);
    if (!result.ok()) {
      test::fail(__FILE__, __LINE__, result.status().to_string());
      continue;
    }
    ++emitted_count;
    LSE_EXPECT(result->source.find("scf.for") != std::string::npos);
    LSE_EXPECT(result->source.find("= scf.for") != std::string::npos);
    LSE_EXPECT(result->source.find("scf.yield") != std::string::npos);
    LSE_EXPECT(result->source.find("kernel.subgroup.shuffle<xor>") != std::string::npos);
    LSE_EXPECT(result->source.find("buffer.alloca") == std::string::npos);
    LSE_EXPECT(result->source.find("float s[") == std::string::npos);
    LSE_EXPECT(result->source.find("view.store") != std::string::npos);
  }
  LSE_EXPECT(emitted_count != 0);
}
}

LSE_TEST(gdn_register_state_is_typed_for_decode_prefill_and_all_outputs) {
  for (int dim : {16, 32, 64, 128}) {
    for (int seq : {1, 6}) {
      for (int wave : {32, 64}) {
        for (int outputs : {0, 1, 2}) check(dim, seq, wave, outputs);
      }
    }
  }
}
LSE_TEST_MAIN()
