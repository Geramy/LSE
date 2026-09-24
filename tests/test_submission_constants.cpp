#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include <cstring>

namespace {
using namespace lse;
using namespace lse::graph;
Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->set_kind(OpKind::kBuffer);
  node->shape = shape; node->dtype = dtype; node->materialized = true;
  return Array(node);
}
void payload(Array& array, std::initializer_list<float> values) {
  array.node()->host_mirror.resize(values.size() * sizeof(float));
  std::memcpy(array.node()->host_mirror.data(), values.begin(), values.size() * sizeof(float));
}
}

LSE_TEST(loom_decode_push_constants_exclude_token_and_position_payloads) {
  using namespace lse;
  using namespace lse::graph;
  backend::DeviceInfo info;
  info.arch = "gfx1201"; info.compute_units = 64; info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024; info.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId; info.extension = &amd;
  auto ids = leaf(Shape{1, 1}), position = leaf(Shape{1});
  auto meta = leaf(Shape{kv::step_meta_elems(1)}), table = leaf(Shape{1, 8});
  auto q = leaf(Shape{1, 24, 1, 256}), pool = leaf(Shape{8, 4, 16, 256});
  auto gq = leaf(Shape{1, 1, 48, 128});
  Array next_state;
  std::vector<Array> outputs{
      quant_embedding(leaf(Shape{248320, 960}, DType::kU32),
          leaf(Shape{248320, 80}, DType::kBF16), leaf(Shape{248320, 80}, DType::kBF16), ids, 6, 64),
      rope(q, leaf(Shape{128, 256}), leaf(Shape{128, 256}), position),
      sdpa_paged(q, pool, pool, 0.0625f, MaskKind::kCausal, 0, meta, table, 16),
      kv_page_write(pool, leaf(Shape{1, 4, 1, 256}), meta, table, 16),
      gated_delta_step(gq, leaf(gq.shape()), leaf(gq.shape()), leaf(Shape{1, 1, 48}),
          leaf(Shape{1, 1, 48}), leaf(Shape{1, 48, 128, 128}), &next_state)};
  outputs.push_back(next_state);
  backend::LoomEmitter emitter;
  for (auto& output : outputs) {
    const NodePtr roots[]{output.node()};
    auto groups = Partitioner::partition(roots, &info);
    LSE_EXPECT(!groups.empty());
    for (const auto& group : groups) {
      std::string initial_source;
      std::size_t initial_count = 0;
      for (unsigned step = 0; step < 3; ++step) {
        payload(ids, {static_cast<float>(100 + step)});
        payload(position, {static_cast<float>(5 + step)});
        payload(meta, {static_cast<float>(5 + step), static_cast<float>(6 + step), 1.0f, 1.0f});
        auto emitted = emitter.emit(group, info);
        LSE_EXPECT(emitted.ok());
        if (!emitted.ok()) {
          std::fprintf(stderr, "%s\n", emitted.status().to_string().c_str());
          return;
        }
        LSE_EXPECT_EQ(emitted->constants.fields.size(), 1u);
        LSE_EXPECT_EQ(emitted->constants.total_bytes, 4u);
        if (emitted->constants.fields.size() != 1) return;
        LSE_EXPECT(emitted->constants.fields[0].name == "count");
        LSE_EXPECT_EQ(emitted->constants.fields[0].offset, 0u);
        LSE_EXPECT_EQ(emitted->constants.fields[0].size, 4u);
        const auto count = group.nodes.back()->element_count();
        if (step == 0) { initial_source = emitted->source; initial_count = count; }
        else {
          LSE_EXPECT(emitted->source == initial_source);
          LSE_EXPECT_EQ(count, initial_count);
        }
      }
    }
  }
}
LSE_TEST_MAIN()
