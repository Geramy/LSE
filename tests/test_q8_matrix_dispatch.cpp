#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/cache.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/wmma_q8_linear.hpp"

#include <algorithm>
#include <array>
#include <limits>

using namespace lse;
using namespace lse::graph;
namespace {
struct Fixture {
  std::array<Shape, 4> inputs;
  std::array<DType, 4> dtypes{DType::kF32, DType::kU32, DType::kBF16, DType::kBF16};
  backend::AmdDeviceInfo amd;
  backend::DeviceInfo device;
  DialectSourceTable intrinsics = backend::loom_sources();
  KernelShapes shapes;
  Fixture(std::int64_t m, std::int64_t n, std::int64_t k)
      : inputs{Shape{m, k}, Shape{n, k / 4}, Shape{n, k / 64}, Shape{n, k / 64}} {
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    shapes.inputs = inputs; shapes.input_dtypes = dtypes;
    shapes.output = Shape{m, n}; shapes.iattrs = {8, 64, 0, 0};
    shapes.device = &device; shapes.intrinsics = &intrinsics;
    shapes.types = backend::loom_types();
  }
  std::uint32_t rows() const { return dispatch::q8_matrix_rows(shapes); }
};
Array leaf(Shape shape, DType dtype) {
  auto node = std::make_shared<Node>(); node->shape = shape;
  node->dtype = dtype; node->materialized = true;
  return Array(node);
}
struct GraphFixture {
  Array output;
  FusionGroup group;
  explicit GraphFixture(const Fixture& f) {
    output = quant_linear(leaf(f.inputs[0], f.dtypes[0]),
                          leaf(f.inputs[1], f.dtypes[1]),
                          leaf(f.inputs[2], f.dtypes[2]),
                          leaf(f.inputs[3], f.dtypes[3]), 8, 64);
    group.nodes = {output.node()}; group.outputs = group.nodes;
    group.inputs = output.node()->inputs;
    group.anchor = output.node()->kind;
    group.anchor_class = output.node()->fclass;
  }
};
}

LSE_TEST(q8_matrix_dispatch_selects_measured_small_shapes_and_prefill_boundary) {
  struct Case { std::int64_t m, n, k; std::uint32_t rows; };
  const std::array cases{
      Case{1, 5120, 10240, 16}, Case{2, 5120, 10240, 16},
      Case{1, 17408, 5120, 16}, Case{8, 17408, 5120, 16},
      Case{1, 5120, 17408, 0}, Case{2, 5120, 17408, 0},
      Case{3, 5120, 17408, 16}, Case{8, 5120, 17408, 16},
      Case{9, 17408, 5120, 0}, Case{63, 17408, 5120, 0},
      Case{64, 17408, 5120, 64}, Case{65, 17408, 5120, 64},
      Case{512, 5120, 10240, 64}, Case{512, 5120, 17408, 64},
      Case{3, 5120, 5120, 0}, Case{3, 129, 128, 0},
      Case{7, 4096, 5120, 0}, Case{8, 4096, 5120, 16},
      Case{7, 5120, 4096, 0}, Case{8, 5120, 4096, 16},
      Case{7, 1024, 5120, 0}, Case{8, 1024, 5120, 16},
      Case{7, 1280, 5120, 0}, Case{8, 1280, 5120, 16},
      Case{1, 5120, 25600, 0}, Case{2, 5120, 25600, 0},
      Case{3, 5120, 25600, 16}, Case{7, 5120, 25600, 16},
      Case{8, 5120, 25600, 0},
      Case{64, 127, 128, 0}, Case{64, 128, 128, 64},
  };
  for (const auto& c : cases) {
    Fixture f(c.m, c.n, c.k);
    LSE_EXPECT_EQ(f.rows(), c.rows);
    if (c.rows == 0) continue;
    const auto* primitive = kernels::wmma_q8_linear_for(f.shapes, c.rows);
    LSE_EXPECT(primitive != nullptr);
    if (!primitive) continue;
    const auto plan = primitive->plan(f.shapes);
    LSE_EXPECT_EQ(plan.workgroup_size[0], 256u);
    LSE_EXPECT_EQ(plan.workgroup_count[0],
        static_cast<std::uint32_t>((c.m + c.rows - 1) / c.rows * ((c.n + 127) / 128)));
    LSE_EXPECT_EQ(plan.lds_bytes, 6656u);
    const auto matrix = dispatch::q8_matrix_plan(f.shapes, c.rows);
    LSE_EXPECT(matrix.matrix != nullptr);
    if (matrix.matrix) {
      LSE_EXPECT(matrix.matrix->acc == math::MatrixElem::kI32);
      LSE_EXPECT(matrix.matrix->operand == math::MatrixElem::kSU8);
    }
  }
}

LSE_TEST(q8_matrix_dispatch_refuses_invalid_formats_shapes_and_device_resources) {
  Fixture f(3, 17408, 5120);
  LSE_EXPECT_EQ(f.rows(), 16u);
  f.shapes.iattrs[0] = 4; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.iattrs[0] = 8;
  f.shapes.iattrs[1] = 32; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.iattrs[1] = 64;
  for (const auto dtype : {DType::kF16, DType::kF32, DType::kU32}) {
    f.dtypes[2] = f.dtypes[3] = dtype; LSE_EXPECT_EQ(f.rows(), 0u);
  }
  f.dtypes[2] = f.dtypes[3] = DType::kBF16;
  f.dtypes[0] = DType::kBF16; LSE_EXPECT_EQ(f.rows(), 0u); f.dtypes[0] = DType::kF32;
  f.dtypes[1] = DType::kI32; LSE_EXPECT_EQ(f.rows(), 0u); f.dtypes[1] = DType::kU32;
  f.shapes.output_dtype = DType::kBF16; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.output_dtype = DType::kF32;
  const auto valid_output = f.shapes.output;
  f.shapes.output = Shape{1, 3, 17408}; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.output = valid_output;
  const auto packed = f.inputs[1]; f.inputs[1] = Shape{17408, 1279};
  LSE_EXPECT_EQ(f.rows(), 0u); f.inputs[1] = packed;
  const auto scales = f.inputs[2]; f.inputs[2] = Shape{17408, 79};
  LSE_EXPECT_EQ(f.rows(), 0u); f.inputs[2] = scales;
  f.inputs[0] = Shape{0, 5120}; LSE_EXPECT_EQ(f.rows(), 0u); f.inputs[0] = Shape{3, 5120};
  f.shapes.staged = {"foreign", 5120}; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.staged = {};
  f.shapes.staged_quant.codes = "foreign"; LSE_EXPECT_EQ(f.rows(), 0u); f.shapes.staged_quant = {};
  f.device.arch = "gfx1200"; LSE_EXPECT_EQ(f.rows(), 0u); f.device.arch = "gfx1201";
  f.device.wavefront_size = 64; LSE_EXPECT_EQ(f.rows(), 0u); f.device.wavefront_size = 32;
  f.device.max_threads_per_workgroup = 255; LSE_EXPECT_EQ(f.rows(), 0u);
  f.device.max_threads_per_workgroup = 256; LSE_EXPECT_EQ(f.rows(), 16u);
  f.device.lds_bytes_per_workgroup = 6655; LSE_EXPECT_EQ(f.rows(), 0u);
  f.device.lds_bytes_per_workgroup = 6656; LSE_EXPECT_EQ(f.rows(), 16u);
  f.amd.matrix_core = backend::MatrixCore::kNone; LSE_EXPECT_EQ(f.rows(), 0u);
  f.amd.matrix_core = backend::MatrixCore::kWMMA;
  const auto* intrinsics = f.shapes.intrinsics;
  DialectSourceTable empty; f.shapes.intrinsics = &empty; LSE_EXPECT_EQ(f.rows(), 0u);
  f.shapes.intrinsics = intrinsics;
  Fixture prefill(64, 128, 128);
  prefill.device.lds_bytes_per_workgroup = 6655; LSE_EXPECT_EQ(prefill.rows(), 0u);
  prefill.device.lds_bytes_per_workgroup = 6656; LSE_EXPECT_EQ(prefill.rows(), 64u);
  Fixture activation_overflow(65536, 128, 65536);
  LSE_EXPECT_EQ(activation_overflow.rows(), 0u);
  Fixture packed_overflow(64, 1 << 29, 64);
  LSE_EXPECT_EQ(packed_overflow.rows(), 0u);
}

LSE_TEST(q8_matrix_round_preparation_tracks_group_count_and_preserves_prefill) {
  for (std::uint32_t groups : {1u, 2u, 3u, 4u, 7u}) {
    Fixture f(7,129,static_cast<std::int64_t>(groups)*64);
    const auto plan=dispatch::q8_matrix_plan(f.shapes,16);
    LSE_EXPECT(plan.matrix!=nullptr);
    LSE_EXPECT_EQ(plan.round_groups,std::min(groups,4u));
    LSE_EXPECT_EQ(plan.lds_bytes,1664u*std::min(groups,4u));
    f.device.lds_bytes_per_workgroup=plan.lds_bytes-1;
    LSE_EXPECT(dispatch::q8_matrix_plan(f.shapes,16).matrix==nullptr);
    f.device.lds_bytes_per_workgroup=plan.lds_bytes;
    LSE_EXPECT(dispatch::q8_matrix_plan(f.shapes,16).matrix!=nullptr);
  }
  Fixture prefill(512,17408,5120);
  const auto plan=dispatch::q8_matrix_plan(prefill.shapes,64);
  LSE_EXPECT_EQ(plan.round_groups,1u);
  LSE_EXPECT_EQ(plan.lds_bytes,6656u);
}

LSE_TEST(q8_matrix_selection_and_same_shape_variants_change_cache_identity) {
  Fixture f(3, 17408, 5120); GraphFixture graph(f);
  const auto* scalar = dynamic_cast<const KernelPrimitiveBase*>(graph.output.node()->prim);
  LSE_EXPECT(scalar != nullptr);
  if (!scalar) return;
  const auto* selected = scalar->specialize(f.shapes);
  LSE_EXPECT(selected != nullptr && selected->name() == "quant_linear.q8.wmma16.iu8_affine.v2");
  backend::LoomEmitter emitter;
  const auto policy = dispatch::specialization_cache_key(0, graph.group,
      f.device, f.shapes.types, f.intrinsics);
  const auto cache = emitter.cache_key(graph.group, f.device);
  f.amd.matrix_core = backend::MatrixCore::kNone;
  LSE_EXPECT(scalar->specialize(f.shapes) == scalar);
  LSE_EXPECT(policy != dispatch::specialization_cache_key(0, graph.group,
      f.device, f.shapes.types, f.intrinsics));
  LSE_EXPECT(cache != emitter.cache_key(graph.group, f.device));
  f.amd.matrix_core = backend::MatrixCore::kWMMA;
  LSE_EXPECT_EQ(policy, dispatch::specialization_cache_key(0, graph.group,
      f.device, f.shapes.types, f.intrinsics));
  LSE_EXPECT_EQ(cache, emitter.cache_key(graph.group, f.device));
  Fixture prefill(512, 17408, 5120); GraphFixture big(prefill);
  const auto* p16 = kernels::wmma_q8_linear_for(prefill.shapes, 16);
  const auto* p64 = kernels::wmma_q8_linear_for(prefill.shapes, 64);
  LSE_EXPECT(p16 != nullptr && p64 != nullptr);
  if (!p16 || !p64) return;
  big.output.node()->prim = p16;
  const auto row16_policy = dispatch::specialization_cache_key(0, big.group,
      prefill.device, prefill.shapes.types, prefill.intrinsics);
  const auto row16_cache = emitter.cache_key(big.group, prefill.device);
  big.output.node()->prim = p64;
  LSE_EXPECT(row16_policy != dispatch::specialization_cache_key(0, big.group,
      prefill.device, prefill.shapes.types, prefill.intrinsics));
  LSE_EXPECT(row16_cache != emitter.cache_key(big.group, prefill.device));
}
LSE_TEST_MAIN()
