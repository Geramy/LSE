#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include <cstdint>
#include <limits>

namespace {
using namespace lse;
using namespace lse::graph;
backend::DeviceInfo device() {
  backend::DeviceInfo out;
  out.arch = "gfx1201";
  out.wavefront_size = 32;
  out.max_threads_per_workgroup = 1024;
  out.lds_bytes_per_workgroup = 65536;
  return out;
}
Array leaf(Shape shape) {
  auto node = std::make_shared<Node>();
  node->set_kind(OpKind::kBuffer);
  node->shape = shape;
  node->dtype = DType::kF32;
  node->materialized = true;
  return Array(node);
}
struct Fixture {
  backend::DeviceInfo gpu = device();
  std::vector<Shape> shapes;
  std::vector<DType> dtypes = std::vector<DType>(5, DType::kF32);
  Array out;
  explicit Fixture(int capacity = 128, int batch = 2) {
    shapes = {Shape{batch, 24, 1, 256}, Shape{capacity / 16 + 1, 4, 16, 256},
              Shape{capacity / 16 + 1, 4, 16, 256},
              Shape{kv::step_meta_elems(batch)}, Shape{batch, capacity / 16}};
    out =
        sdpa_paged(leaf(shapes[0]), leaf(shapes[1]), leaf(shapes[2]), 0.0625f,
                   MaskKind::kCausal, 0, leaf(shapes[3]), leaf(shapes[4]), 16);
  }
  KernelShapes request() const {
    KernelShapes s;
    s.inputs = shapes;
    s.input_dtypes = dtypes;
    s.output = out.shape();
    s.device = &gpu;
    s.attrs = out.node()->attrs;
    s.iattrs = out.node()->iattrs;
    return s;
  }
  const KernelPrimitiveBase *primitive() const {
    return dynamic_cast<const KernelPrimitiveBase *>(out.node()->prim);
  }
};
} // namespace

LSE_TEST(decode_attention_specialization_reports_exact_resources) {
  for (int capacity : {128, 512, 8192}) {
    Fixture fx(capacity);
    auto shapes = fx.request();
    auto *base = fx.primitive();
    LSE_EXPECT(base != nullptr);
    if (!base)
      return;
    auto *selected = base->specialize(shapes);
    LSE_EXPECT(selected != nullptr && selected != base);
    if (!selected)
      return;
    LSE_EXPECT(selected->owns_indexing());
    const auto plan = selected->plan(shapes);
    LSE_EXPECT_EQ(plan.workgroup_size[0], 256u);
    LSE_EXPECT_EQ(plan.workgroup_count[0], 48u);
    LSE_EXPECT_EQ(plan.lds_bytes, static_cast<unsigned>(capacity) * 4u);
  }
}

LSE_TEST(decode_attention_declines_unqualified_contracts) {
  for (int variant = 0; variant < 9; ++variant) {
    Fixture fx;
    if (variant == 0)
      fx.gpu.arch = "gfx1100";
    if (variant == 1)
      fx.gpu.lds_bytes_per_workgroup = 0;
    if (variant == 2)
      fx.gpu.lds_bytes_per_workgroup = 511;
    if (variant == 3)
      fx.gpu.max_threads_per_workgroup = 128;
    if (variant == 4)
      fx.shapes[0] = Shape{2, 24, 2, 256};
    if (variant == 5)
      fx.shapes[0] = Shape{2, 25, 1, 256};
    if (variant == 6)
      fx.dtypes[0] = DType::kBF16;
    if (variant == 7)
      fx.shapes[4] = Shape{2, 1024};
    auto shapes = fx.request();
    if (variant == 8) {
      shapes.iattrs[0] = 2;
      shapes.iattrs[1] = -1;
    }
    const auto *base = fx.primitive();
    LSE_EXPECT(base != nullptr);
    if (base)
      LSE_EXPECT(base->specialize(shapes) == base);
  }
}

LSE_TEST(decode_attention_masks_keep_binding_and_launch_abi) {
  backend::LoomEmitter emitter;
  for (int mode : {0, 1, 2})
    for (int window : {0, 7, std::numeric_limits<int>::max()}) {
      Fixture fx;
      fx.out.node()->iattrs[0] = mode;
      fx.out.node()->iattrs[1] = window;
      const NodePtr roots[]{fx.out.node()};
      const auto groups = Partitioner::partition(roots, &fx.gpu);
      LSE_EXPECT_EQ(groups.size(), 1u);
      if (groups.size() != 1)
        return;
      auto emitted = emitter.emit(groups[0], fx.gpu);
      LSE_EXPECT(emitted.ok());
      if (!emitted.ok())
        return;
      LSE_EXPECT_EQ(emitted->binding_order.size(), 6u);
      for (std::size_t i = 0; i < 5; ++i)
        LSE_EXPECT(emitted->binding_order[i] == fx.out.node()->inputs[i]);
      LSE_EXPECT(emitted->binding_order[5] == fx.out.node());
      LSE_EXPECT_EQ(emitted->constants.fields.size(), 1u);
      LSE_EXPECT_EQ(emitted->constants.total_bytes, 4u);
      LSE_EXPECT(emitted->constants.fields[0].name == "count");
      LSE_EXPECT_EQ(emitted->constants.fields[0].offset, 0u);
      LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], 48u);
      LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 256u);
      LSE_EXPECT_EQ(emitted->lds_bytes, 512u);
      LSE_EXPECT(emitted->source.find("kernel.barrier<workgroup>") !=
                 std::string::npos);
      LSE_EXPECT(emitted->source.find("scalar.fmaf") != std::string::npos);
      LSE_EXPECT(emitted->source.find("index.sub") == std::string::npos);
    }
}

LSE_TEST(decode_attention_sliding_mask_equivalence_at_integer_boundaries) {
  constexpr std::uint32_t keys[]{0, 1, 7, 127, 255, 256, 511, 512, 8191};
  constexpr std::uint32_t offsets[]{
      0, 1, 7, 128, 512, 8191, 0x7fffffffu, 0x80000000u, 0xffffffffu};
  constexpr std::uint32_t windows[]{0, 1, 7, 128, 512, 8192, 0x7fffffffu};
  for (auto key : keys)
    for (auto offset : offsets)
      for (auto window : windows) {
        LSE_EXPECT(std::uint64_t(key) + window <=
                   std::numeric_limits<std::uint32_t>::max());
        const bool original = key <= offset && offset - key < window;
        const bool shared = key <= offset && key + window > offset;
        LSE_EXPECT(original == shared);
      }
}
LSE_TEST_MAIN()
