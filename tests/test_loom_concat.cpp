#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/ops.hpp"

#include <array>

using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>(); node->shape = shape;
  node->dtype = dtype; node->materialized = true;
  return Array(node);
}
std::size_t occurrences(const std::string& source, std::string_view needle) {
  std::size_t count = 0, at = 0;
  while ((at = source.find(needle, at)) != std::string::npos) {
    ++count; at += needle.size();
  }
  return count;
}
}
LSE_TEST(concat_five_and_eight_taps_emit_one_pass_in_input_order) {
  backend::DeviceInfo device;
  device.arch = "gfx1201"; device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  for (const std::size_t count : {2u, 5u, 8u}) {
    std::vector<Array> parts;
    std::int64_t width = 0;
    for (std::size_t p = 0; p < count; ++p) {
      const auto part_width = static_cast<std::int64_t>(p + 1);
      parts.push_back(leaf({2, 3, part_width})); width += part_width;
    }
    auto output = concat(parts, -1);
    LSE_EXPECT(output.shape() == Shape({2, 3, width}));
    const NodePtr roots[] = {output.node()};
    const auto groups = Partitioner::partition(roots);
    LSE_EXPECT_EQ(groups.size(), 1u);
    if (groups.size() != 1) continue;
    auto emitted = backend::LoomEmitter{}.emit(groups.front(), device);
    LSE_EXPECT_OK(emitted.status());
    if (!emitted.ok()) continue;
    LSE_EXPECT_EQ(emitted->binding_order.size(), count + 1);
    for (std::size_t p = 0; p < count; ++p)
      LSE_EXPECT(emitted->binding_order[p] == parts[p].node());
    LSE_EXPECT(emitted->binding_order.back() == output.node());
    LSE_EXPECT_EQ(occurrences(emitted->source, "view.load "), count);
    LSE_EXPECT_EQ(occurrences(emitted->source, "view.store "), 1u);
    LSE_EXPECT_EQ(emitted->lds_bytes, 0u);
    LSE_EXPECT(emitted->source.find("kernel.barrier") == std::string::npos);
  }
}
LSE_TEST(concat_declines_more_than_eight_inputs_and_checks_new_slot_dtypes) {
  const auto* primitive = dynamic_cast<const KernelPrimitiveBase*>(find_primitive("concat"));
  LSE_EXPECT(primitive != nullptr);
  if (!primitive) return;
  LSE_EXPECT_EQ(primitive->arity(), 2u);
  std::array<Shape, 9> inputs;
  std::array<DType, 9> dtypes;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = Shape{1, 2, 1}; dtypes[i] = DType::kF32;
  }
  auto intrinsics = backend::loom_sources();
  KernelShapes shapes;
  shapes.inputs = inputs; shapes.input_dtypes = dtypes;
  shapes.output = Shape{1, 2, 9}; shapes.iattrs[0] = 2;
  shapes.types = backend::loom_types(); shapes.intrinsics = &intrinsics;
  LSE_EXPECT(primitive->emit_kernel(shapes).empty());
  shapes.inputs = std::span<const Shape>(inputs.data(), 8);
  shapes.input_dtypes = std::span<const DType>(dtypes.data(), 8);
  shapes.output = Shape{1, 2, 8};
  LSE_EXPECT(!primitive->emit_kernel(shapes).empty());
  for (std::size_t slot = 4; slot < 8; ++slot) {
    dtypes[slot] = DType::kBF16;
    LSE_EXPECT(primitive->emit_kernel(shapes).empty());
    dtypes[slot] = DType::kF32;
  }
}
LSE_TEST_MAIN()
