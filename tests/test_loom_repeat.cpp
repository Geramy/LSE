// Emission regressions; no HRX library, Loom compiler, or GPU is needed.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

namespace {
using namespace lse;
using namespace lse::graph;
FusionGroup group(Shape shape, int count, int axis, DType dtype = DType::kF32) {
  auto input = std::make_shared<Node>();
  input->shape = shape;
  input->dtype = dtype;
  input->materialized = true;
  auto output = repeat(Array(input), count, axis).node();
  FusionGroup result;
  result.inputs = {input};
  result.nodes = {output};
  result.outputs = {output};
  result.anchor = OpKind::kRepeat;
  result.anchor_class = output->fclass;
  return result;
}
Result<EmittedKernel> emit(const FusionGroup& group) {
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.max_threads_per_workgroup = 1024;
  device.wavefront_size = 32;
  return backend::LoomEmitter{}.emit(group, device);
}
}  // namespace

LSE_TEST(repeat_axes_and_counts_emit_bound_pointer_loads) {
  for (int axis : {0, 1, 2, -1})
    for (int count : {1, 2, 6}) {
      const auto g = group(Shape{2, 3, 5}, count, axis);
      const auto result = emit(g);
      LSE_EXPECT(result.ok());
      if (result.ok()) {
        LSE_EXPECT(result->binding_order.size() == 2);
        LSE_EXPECT(result->source.find("index.div") != std::string::npos);
        LSE_EXPECT(result->source.find("index.rem") != std::string::npos);
        LSE_EXPECT(result->source.find("view.load %b0_view") !=
                   std::string::npos);
        LSE_EXPECT(result->source.find("range(") != std::string::npos);
      }
    }
}

LSE_TEST(repeat_direct_store_preserves_native_bits) {
  for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16, DType::kU32,
                     DType::kI32, DType::kU8, DType::kI8}) {
    const auto result = emit(group(Shape{2, 3, 5}, 3, 1, dtype));
    LSE_EXPECT(result.ok());
    if (result.ok()) {
      LSE_EXPECT(result->source.find("view.store %krepeat_value") !=
                 std::string::npos);
      LSE_EXPECT(result->source.find("scalar.fptoui") == std::string::npos);
      LSE_EXPECT(result->source.find("scalar.fptosi") == std::string::npos);
      LSE_EXPECT(result->source.find("scalar.truncf") == std::string::npos);
    }
  }
}

LSE_TEST(repeat_rejects_malformed_shapes_before_unsafe_index_assumptions) {
  for (int invalid = 0; invalid < 7; ++invalid) {
    auto g = group(Shape{2, 3, 5}, 3, 1);
    auto& n = g.nodes.back();
    if (invalid == 0) n->iattrs[1] = 0;
    if (invalid == 1) n->iattrs[1] = -1;
    if (invalid == 2) n->iattrs[0] = 3;
    if (invalid == 3) n->shape = Shape{2, 8, 5};
    if (invalid == 4) n->dtype = DType::kU32;
    if (invalid == 5) {
      g.inputs[0]->shape = Shape{2, 0, 5};
      n->shape = Shape{2, 0, 5};
    }
    if (invalid == 6) {
      g.inputs[0]->shape = Shape{65536, 65536, 5};
      n->shape = Shape{65536, 196608, 5};
    }
    LSE_EXPECT(!emit(g).ok());
  }
}

LSE_TEST(narrow_repeat_load_is_bounded_when_sibling_output_is_wider) {
  auto g = group(Shape{2, 3, 5}, 2, 1);
  auto wider = repeat(Array(g.inputs[0]), 6, 1).node();
  g.nodes.push_back(wider);
  g.outputs.push_back(wider);
  const auto result = emit(g);
  LSE_EXPECT(result.ok());
  if (result.ok()) {
    LSE_EXPECT(result->source.find("index.rem %i,") != std::string::npos);
    LSE_EXPECT(result->source.find("index.cmp ult, %i,") != std::string::npos);
  }
}

LSE_TEST_MAIN()
