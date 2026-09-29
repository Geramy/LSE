#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/attention.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

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
Array leaf(Shape shape, DType dtype = DType::kF32) {
  auto node = std::make_shared<Node>();
  node->set_kind(OpKind::kBuffer);
  node->shape = shape;
  node->dtype = dtype;
  node->materialized = true;
  return Array(node);
}
struct Fixture {
  backend::DeviceInfo gpu = device();
  std::vector<Shape> shapes;
  std::vector<DType> dtypes = std::vector<DType>(5, DType::kF32);
  explicit Fixture(int capacity = 128, int batch = 2) {
    shapes = {Shape{batch, 24, 1, 256}, Shape{capacity / 16 + 1, 4, 16, 256},
              Shape{capacity / 16 + 1, 4, 16, 256},
              Shape{kv::step_meta_elems(batch)}, Shape{batch, capacity / 16}};
  }
  KernelShapes request() const {
    KernelShapes s;
    s.inputs = shapes;
    s.input_dtypes = dtypes;
    s.output = shapes[0];
    s.device = &gpu;
    s.attrs = {0.0625f, 0, 0, 0};
    s.iattrs = {1, 0, 0, 16};
    return s;
  }
  Array graph(MaskKind mask = MaskKind::kCausal, int window = 0,
              bool with_device = true, kv::CacheDType storage = kv::CacheDType::kF32) const {
    return sdpa_paged(leaf(shapes[0], dtypes[0]), leaf(shapes[1], dtypes[1]),
                      leaf(shapes[2], dtypes[2]), 0.0625f, mask, window,
                      leaf(shapes[3]), leaf(shapes[4]), 16,
                      with_device ? &gpu : nullptr, storage);
  }
};
} // namespace

LSE_TEST(decode_attention_uses_split_stages_at_short_and_long_capacities) {
  backend::LoomEmitter emitter;
  for (int capacity : {16, 128, 512, 8192, 16384, 32768, 262144}) {
    Fixture fx(capacity);
    const auto request = fx.request();
    LSE_EXPECT(dispatch::split_decode_supported(request));
    const auto output = fx.graph();
    LSE_EXPECT(output.node()->prim->name() == "attention.decode_merge128.wg128c2.v3");
    const auto partial = output.node()->inputs[0];
    const unsigned parts = (static_cast<unsigned>(capacity) + 127u) / 128u;
    LSE_EXPECT(partial->prim->name() == "attention.decode_partial128.wg128c2.v3");
    LSE_EXPECT(partial->shape == Shape({2, 24, parts, 258}));
    const NodePtr roots[]{output.node()};
    const auto groups = Partitioner::partition(roots, &fx.gpu);
    LSE_EXPECT_EQ(groups.size(), 2u);
    if (groups.size() != 2) return;
    for (std::size_t i = 0; i < 2; ++i) {
      const auto emitted = emitter.emit(groups[i], fx.gpu);
      LSE_EXPECT(emitted.ok());
      if (!emitted.ok()) return;
      LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 128u);
      LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], i == 0 ? 8u * parts : 48u);
      LSE_EXPECT_EQ(emitted->lds_bytes, i == 0 ? 3072u : (parts * 4u + 15u) / 16u * 16u);
    }
  }
  LSE_EXPECT(find_primitive("attention.decode_shared") == nullptr);
  LSE_EXPECT(find_primitive("attention.decode_shared_exp") == nullptr);
}

LSE_TEST(decode_attention_admission_uses_actual_resources_and_partial_extent) {
  Fixture grouped(8192, 1);
  grouped.gpu.lds_bytes_per_workgroup = 3072;
  LSE_EXPECT(dispatch::split_decode_supported(grouped.request()));
  grouped.gpu.lds_bytes_per_workgroup = 3071;
  LSE_EXPECT(!dispatch::split_decode_supported(grouped.request()));
  Fixture fx(262144, 1);
  auto request = fx.request();
  fx.gpu.max_threads_per_workgroup = 128;
  fx.gpu.lds_bytes_per_workgroup = 8192;
  LSE_EXPECT(dispatch::split_decode_supported(request));
  fx.gpu.lds_bytes_per_workgroup = 8191;
  LSE_EXPECT(!dispatch::split_decode_supported(request));
  fx.gpu.lds_bytes_per_workgroup = 65536;
  fx.gpu.max_threads_per_workgroup = 127;
  LSE_EXPECT(!dispatch::split_decode_supported(request));
  fx.gpu.max_threads_per_workgroup = 128;
  fx.gpu.wavefront_size = 64;
  LSE_EXPECT(!dispatch::split_decode_supported(request));
  fx.gpu.wavefront_size = 32;
  request.output = Shape{1, 24, 1, 128};
  LSE_EXPECT(!dispatch::split_decode_supported(request));
  request.output = fx.shapes[0];
  request.attrs[0] = std::numeric_limits<float>::infinity();
  LSE_EXPECT(!dispatch::split_decode_supported(request));
  fx.shapes[0] = Shape{4096, 24, 1, 256};
  fx.shapes[3] = Shape{kv::step_meta_elems(4096)};
  fx.shapes[4] = Shape{4096, 262144 / 16};
  LSE_EXPECT(!dispatch::split_decode_supported(fx.request()));
}

LSE_TEST(decode_attention_scope_covers_every_nonnegative_offset) {
  const Shape query{1, 24, 1, 256};
  for (int capacity : {1, 512, 8192, 16384, 32768, 262144})
    for (int offset : {0, 1, 510, 511, 8191, 16383, 262143})
      LSE_EXPECT(dispatch::split_decode_scope(query, offset, capacity));
  LSE_EXPECT(!dispatch::split_decode_scope(query, -1, 8192));
  LSE_EXPECT(!dispatch::split_decode_scope(query, 0, 0));
  LSE_EXPECT(!dispatch::split_decode_scope(Shape{1, 24, 2, 256}, 512, 8192));
}

LSE_TEST(decode_attention_merge_initializes_partitions_beyond_one_wavegroup) {
  Fixture fx;
  const auto* merge = dynamic_cast<const KernelPrimitiveBase*>(
      find_primitive("attention.decode_merge128.wg128c2.v3"));
  LSE_EXPECT(merge != nullptr);
  if (!merge) return;
  for (int parts : {1, 64, 128, 129, 256, 2048, 16384, 16385}) {
    const std::array<Shape, 1> inputs{Shape{1, 24, parts, 258}};
    const std::array<DType, 1> dtypes{DType::kF32};
    KernelShapes request;
    request.inputs = inputs;
    request.input_dtypes = dtypes;
    request.output = Shape{1, 24, 1, 256};
    request.device = &fx.gpu;
    LSE_EXPECT(dispatch::split_decode_merge_supported(request) == (parts <= 16384));
    if (parts <= 16384) {
      const auto plan = merge->plan(request);
      LSE_EXPECT_EQ(plan.workgroup_size[0], 128u);
      LSE_EXPECT_EQ(plan.workgroup_count[0], 24u);
      LSE_EXPECT_EQ(plan.lds_bytes, static_cast<unsigned>(parts) * 4u);
    }
  }
}

LSE_TEST(decode_attention_masks_keep_binding_and_launch_abi) {
  backend::LoomEmitter emitter;
  for (int mode : {0, 1, 2})
    for (int window : {0, 7, std::numeric_limits<int>::max()}) {
      Fixture fx(8192);
      const auto output = fx.graph(static_cast<MaskKind>(mode), window);
      const NodePtr roots[]{output.node()};
      const auto groups = Partitioner::partition(roots, &fx.gpu);
      LSE_EXPECT_EQ(groups.size(), 2u);
      if (groups.size() != 2) return;
      for (std::size_t stage = 0; stage < groups.size(); ++stage) {
        const auto emitted = emitter.emit(groups[stage], fx.gpu);
        LSE_EXPECT(emitted.ok());
        if (!emitted.ok()) return;
        LSE_EXPECT_EQ(emitted->binding_order.size(), stage == 0 ? 6u : 2u);
        LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], stage == 0 ? 8u * 64u : 48u);
        LSE_EXPECT_EQ(emitted->dims.workgroup_size[0], 128u);
        LSE_EXPECT(emitted->source.find("kernel.barrier<workgroup>") != std::string::npos);
        LSE_EXPECT(emitted->source.find("scalar.fmaf") != std::string::npos);
        LSE_EXPECT(emitted->source.find("index.sub") == std::string::npos);
      }
    }
}

LSE_TEST(decode_attention_keeps_every_kv_storage_format_on_the_split_route) {
  for (const auto storage : {kv::CacheDType::kF32, kv::CacheDType::kF16,
                             kv::CacheDType::kBF16, kv::CacheDType::kFP8,
                             kv::CacheDType::kBF8}) {
    Fixture fx(32768, 1);
    fx.shapes[1] = fx.shapes[2] = Shape{2049, 4, 16, kv::storage_width(storage, 256)};
    fx.dtypes[1] = fx.dtypes[2] = kv::storage_dtype(storage);
    auto request = fx.request();
    request.attrs[1] = static_cast<float>(storage);
    LSE_EXPECT(dispatch::split_decode_supported(request));
    LSE_EXPECT(fx.graph(MaskKind::kCausal, 0, true, storage).node()->prim->name() ==
               "attention.decode_merge128.wg128c2.v3");
  }
}

LSE_TEST(attention_dispatch_matrix_tiles_follow_device_and_lds_limits) {
  Fixture fx;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(fx.gpu, amd);
  fx.gpu.extension_id = backend::AmdDeviceInfo::kExtensionId;
  fx.gpu.extension = &amd;
  fx.shapes[0] = Shape{2, 24, 512, 256};
  auto request = fx.request();
  const auto intrinsics = backend::loom_sources();
  request.types = backend::loom_types();
  request.intrinsics = &intrinsics;
  using dispatch::AttentionPlan;
  for (const auto [bytes, expected] : {
       std::pair{65536u, AttentionPlan::kFlashWmma},
       std::pair{24768u, AttentionPlan::kFlashWmma},
       std::pair{24767u, AttentionPlan::kScalar}}) {
    fx.gpu.lds_bytes_per_workgroup = bytes;
    LSE_EXPECT(dispatch::attention_plan(request) == expected);
  }
  fx.gpu.lds_bytes_per_workgroup = 65536;
  fx.gpu.arch = "gfx1100";
  LSE_EXPECT(dispatch::attention_plan(request) == AttentionPlan::kScalar);
  fx.gpu.arch = "gfx1201";
  for (int rows : {2, 7, 11, 17, 512}) {
    fx.shapes[0] = Shape{2, 24, rows, 256};
    request.output = fx.shapes[0];
    LSE_EXPECT(dispatch::attention_plan(request) == AttentionPlan::kFlashWmma);
  }
  fx.gpu.extension = nullptr;
  LSE_EXPECT(dispatch::attention_plan(request) == AttentionPlan::kScalar);
}

LSE_TEST(attention_dispatch_rejects_malformed_paged_input_contracts) {
  using dispatch::AttentionPlan;
  for (int variant = 0; variant < 10; ++variant) {
    Fixture fx;
    fx.shapes[0] = Shape{2, 24, 512, 256};
    if (variant == 0) fx.dtypes[1] = DType::kF16;
    if (variant == 1) fx.shapes[0] = Shape{-1, 24, 512, 256};
    if (variant == 2) fx.shapes[1] = Shape{9, 4, 16, 128};
    if (variant == 3) fx.shapes[2] = Shape{8, 4, 16, 256};
    if (variant == 4) fx.shapes[4] = Shape{1, 8};
    if (variant == 5) fx.shapes[4] = Shape{2, 1, 8};
    if (variant == 6) fx.shapes[3] = Shape{1};
    if (variant == 7) fx.shapes[4] = Shape{2, std::int64_t{UINT32_MAX} / 16 + 1};
    if (variant == 8) fx.shapes[0] = Shape{2, 24, 512, std::int64_t{UINT32_MAX} + 1};
    auto request = fx.request();
    if (variant == 9) request.output_dtype = DType::kBF16;
    LSE_EXPECT(dispatch::attention_plan(request) == AttentionPlan::kScalar);
    LSE_EXPECT(!dispatch::flash_dimensions(request).valid);
    LSE_EXPECT(!dispatch::flash_wmma_supported(request));
    LSE_EXPECT(!dispatch::split_decode_supported(request));
  }
}

LSE_TEST(decode_attention_sliding_mask_equivalence_at_integer_boundaries) {
  constexpr std::uint32_t keys[]{0, 1, 7, 127, 255, 256, 511, 512, 8191, 16383, 32767, 262143};
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
LSE_TEST(decode_attention_narrow_storage_uses_wave_qk_and_preserves_partial_abi) {
  backend::LoomEmitter emitter;
  for (const auto storage : {kv::CacheDType::kF16, kv::CacheDType::kBF16}) {
    for (int capacity : {8192, 16384}) {
      Fixture fx(capacity, 1);
      fx.dtypes[1] = fx.dtypes[2] = kv::storage_dtype(storage);
      const auto output = fx.graph(MaskKind::kCausal, 0, true, storage);
      const unsigned parts = static_cast<unsigned>(capacity) / 128u;
      LSE_EXPECT(output.node()->inputs[0]->shape == Shape({1, 24, parts, 258}));
      const NodePtr roots[]{output.node()};
      const auto groups = Partitioner::partition(roots, &fx.gpu);
      LSE_EXPECT_EQ(groups.size(), 2u);
      if (groups.size() != 2) return;
      const auto partial = emitter.emit(groups[0], fx.gpu);
      const auto merge = emitter.emit(groups[1], fx.gpu);
      LSE_EXPECT(partial.ok());
      LSE_EXPECT(merge.ok());
      if (!partial.ok() || !merge.ok()) return;
      LSE_EXPECT_EQ(partial->dims.workgroup_size[0], 128u);
      LSE_EXPECT_EQ(partial->dims.workgroup_count[0], 4u * parts);
      LSE_EXPECT_EQ(partial->lds_bytes, 3072u);
      LSE_EXPECT_EQ(partial->binding_order.size(), 6u);
      LSE_EXPECT(partial->source.find("kernel.subgroup.shuffle<xor>") != std::string::npos);
      LSE_EXPECT(partial->source.find("scalar.fmaf") != std::string::npos);
      LSE_EXPECT(partial->source.find(storage == kv::CacheDType::kF16 ? "xf16" : "xbf16") != std::string::npos);
      LSE_EXPECT_EQ(merge->dims.workgroup_count[0], 24u);
      LSE_EXPECT_EQ(merge->lds_bytes, parts * sizeof(float));
    }
  }
}

LSE_TEST_MAIN()
