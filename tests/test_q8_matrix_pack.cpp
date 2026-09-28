#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/cache.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/graph/ops.hpp"
#include "lse/quant/q8_matrix_pack.hpp"

#include <array>
#include <cstring>
#include <limits>

using namespace lse;
using namespace lse::graph;
namespace {
constexpr std::string_view kPacked = "quant_linear.q8.wmma16.packed.v1";
struct Fixture {
  std::array<Shape, 7> inputs;
  std::array<DType, 7> dtypes{DType::kF32,  DType::kU32, DType::kBF16,
                              DType::kBF16, DType::kU32, DType::kBF16,
                              DType::kBF16};
  backend::AmdDeviceInfo amd;
  backend::DeviceInfo device;
  DialectSourceTable intrinsics = backend::loom_sources();
  KernelShapes shapes;
  Fixture(std::int64_t n = 17408, std::int64_t k = 5120)
      : inputs{Shape{1, 1, k},   Shape{n, k / 4}, Shape{n, k / 64},
               Shape{n, k / 64}, Shape{n, k / 4}, Shape{n, k / 64},
               Shape{n, k / 64}} {
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    shapes.inputs = inputs;
    shapes.input_dtypes = dtypes;
    shapes.output = Shape{1, 1, n};
    shapes.iattrs = {8, 64, 0, 0};
    shapes.device = &device;
    shapes.intrinsics = &intrinsics;
    shapes.types = backend::loom_types();
    shapes.store = [](std::string_view index, std::string_view value) {
      return "out[" + std::string(index) + "] = " + std::string(value) + ";";
    };
  }
  KernelShapes original() const {
    auto s = shapes;
    s.inputs = shapes.inputs.first(4);
    s.input_dtypes = shapes.input_dtypes.first(4);
    return s;
  }
};
Array leaf(Shape shape, DType dtype) {
  static std::uint64_t next = 1;
  backend::DeviceBuffer buffer;
  buffer.handle = next++;
  buffer.size_bytes = dtype_storage_bytes(dtype, shape.elem_count());
  buffer.residency = backend::DeviceIndex{1};
  buffer.member = 0;
  buffer.storage = std::make_shared<int>(0);
  return Array::from_buffer(std::move(buffer), shape, dtype);
}
struct Weights {
  Array words, scales, biases;
  std::shared_ptr<PackedQuantMatrix> storage;
  explicit Weights(const Fixture &f)
      : words(leaf(f.inputs[1], f.dtypes[1])),
        scales(leaf(f.inputs[2], f.dtypes[2])),
        biases(leaf(f.inputs[3], f.dtypes[3])),
        storage(std::make_shared<PackedQuantMatrix>()) {
    const std::array sources{words.node(), scales.node(), biases.node()};
    for (std::size_t i = 0; i < 3; ++i) {
      storage->sources[i] = sources[i];
      storage->source_buffers[i] = sources[i]->buffer;
      storage->packed[i] = leaf(f.inputs[4 + i], f.dtypes[4 + i]).node();
    }
    auto planes = std::make_shared<QuantPlanes>();
    planes->scales = scales.node();
    planes->biases = biases.node();
    planes->bits = 8;
    planes->group_size = 64;
    planes->in_features = f.inputs[0].dim(2);
    planes->matrix_storage = storage;
    words.node()->quant = std::move(planes);
  }
  Array projection(const Array &x) const {
    return quant_linear(x, words, scales, biases, 8, 64);
  }
};
FusionGroup group(const Array &output) {
  FusionGroup g;
  g.nodes = {output.node()};
  g.outputs = g.nodes;
  g.inputs = output.node()->inputs;
  g.anchor = output.node()->kind;
  g.anchor_class = output.node()->fclass;
  return g;
}
} // namespace

LSE_TEST(q8_weight_pack_preserves_all_raw_words_affine_bits_and_padding) {
  constexpr std::uint32_t n = 17, k = 192, lanes = k / 4, groups = k / 64;
  std::vector<std::byte> words(n * lanes * 4 + 1), scales(n * groups * 2 + 1),
      biases(scales.size());
  for (std::uint32_t c = 0; c < n; ++c) {
    for (std::uint32_t w = 0; w < lanes; ++w) {
      const auto raw = 0x8193a7bdu ^ (c * 0x01234567u + w * 0x01020304u);
      std::memcpy(words.data() + 1 + (c * lanes + w) * 4, &raw, 4);
    }
    for (std::uint32_t g = 0; g < groups; ++g) {
      const auto scale =
          static_cast<std::uint16_t>(0x7f80u ^ (c * 313u + g * 971u));
      const auto bias =
          static_cast<std::uint16_t>(0xffc1u ^ (c * 733u + g * 127u));
      std::memcpy(scales.data() + 1 + (c * groups + g) * 2, &scale, 2);
      std::memcpy(biases.data() + 1 + (c * groups + g) * 2, &bias, 2);
    }
  }
  const auto before_words = words, before_scales = scales,
             before_biases = biases;
  auto result = quant::pack_q8_matrix(n, k, std::span(words).subspan(1),
                                      std::span(scales).subspan(1),
                                      std::span(biases).subspan(1));
  LSE_EXPECT(result.ok());
  if (!result.ok())
    return;
  const auto &packed = *result;
  LSE_EXPECT_EQ(packed.padded_columns, 32u);
  for (std::uint32_t c = 0; c < 32; ++c) {
    const auto source = c < n ? c : 0u;
    for (std::uint32_t w = 0; w < lanes; ++w) {
      std::uint32_t raw;
      std::memcpy(&raw, words.data() + 1 + (source * lanes + w) * 4, 4);
      const auto dst = (c / 16) * lanes * 16 + (w / 4) * 64 + (w % 2) * 32 +
                       ((w / 2) % 2) * 16 + c % 16;
      LSE_EXPECT_EQ(packed.words[dst], raw);
    }
    for (std::uint32_t g = 0; g < groups; ++g) {
      std::uint16_t raw_scale, raw_bias;
      std::memcpy(&raw_scale, scales.data() + 1 + (source * groups + g) * 2, 2);
      std::memcpy(&raw_bias, biases.data() + 1 + (source * groups + g) * 2, 2);
      const auto dst = (c / 16) * groups * 16 + g * 16 + c % 16;
      LSE_EXPECT_EQ(packed.scales[dst], raw_scale);
      LSE_EXPECT_EQ(packed.biases[dst], raw_bias);
    }
  }
  LSE_EXPECT(words == before_words && scales == before_scales &&
             biases == before_biases);
  LSE_EXPECT(!quant::pack_q8_matrix(0, k, {}, {}, {}).ok());
  LSE_EXPECT(!quant::pack_q8_matrix(n, 65, {}, {}, {}).ok());
  LSE_EXPECT(!quant::pack_q8_matrix(UINT32_MAX, k, {}, {}, {}).ok());
  LSE_EXPECT(!quant::pack_q8_matrix(1u << 30, 64, {}, {}, {}).ok());
  LSE_EXPECT(!quant::pack_q8_matrix(n, k, std::span(words).subspan(2),
                                    std::span(scales).subspan(1),
                                    std::span(biases).subspan(1))
                  .ok());
}
LSE_TEST(
    q8_packed_memory_reserves_original_slabs_and_declines_unknown_or_short_budget) {
  constexpr std::size_t page = 4096, slab = 4 * page, packed = page;
  const std::array<std::size_t, 1> small{1};
  LSE_EXPECT(
      !quant::q8_packed_memory_admitted(std::nullopt, small, slab, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(0, {}, slab, packed));
  LSE_EXPECT(quant::q8_packed_memory_admitted(packed, {}, slab, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(packed - 1, {}, slab, packed));
  LSE_EXPECT(
      quant::q8_packed_memory_admitted(slab + packed, small, slab, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(slab + packed - 1, small, slab,
                                               packed));

  // An oversized original gets an exact allocation in addition to regular
  // slabs.
  const std::array<std::size_t, 2> mixed{5 * page, 1};
  const auto reserve = 5 * page + slab;
  LSE_EXPECT(
      quant::q8_packed_memory_admitted(reserve + packed, mixed, slab, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(reserve + packed - 1, mixed,
                                               slab, packed));

  // Fragmentation must be reserved even when original payload bytes fit.
  const std::array<std::size_t, 3> fragmented{3 * page, 3 * page, 3 * page};
  LSE_EXPECT(!quant::q8_packed_memory_admitted(3 * slab + packed, fragmented,
                                               slab, packed));
  LSE_EXPECT(quant::q8_packed_memory_admitted(5 * slab + packed, fragmented,
                                              slab, packed));

  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  const std::array<std::size_t, 1> unroundable{limit};
  const std::array<std::size_t, 2> overflowing{limit & ~(page - 1),
                                               limit & ~(page - 1)};
  LSE_EXPECT(
      !quant::q8_packed_memory_admitted(limit, unroundable, slab, packed));
  LSE_EXPECT(
      !quant::q8_packed_memory_admitted(limit, overflowing, slab, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(limit, small, slab + 1, packed));
  LSE_EXPECT(!quant::q8_packed_memory_admitted(limit, small, page - 1, packed));
  LSE_EXPECT(
      quant::q8_packed_memory_admitted(limit, {}, limit & ~(page - 1), packed));
}
LSE_TEST(q8_packed_matrix_dispatch_is_limited_to_measured_M1_shapes) {
  for (const auto pair :
       {std::array<std::int64_t, 2>{17408, 5120}, {5120, 10240}}) {
    Fixture f(pair[0], pair[1]);
    LSE_EXPECT(dispatch::q8_packed_matrix_shape(f.original()));
    LSE_EXPECT(dispatch::q8_packed_weight_device(f.device));
    const auto p = dispatch::q8_packed_matrix_plan(f.shapes);
    LSE_EXPECT(p.matrix != nullptr);
    LSE_EXPECT_EQ(p.lds_bytes, 6656u);
    LSE_EXPECT_EQ(p.round_groups, 4u);
    f.inputs[0] = Shape{1, 2, pair[1]};
    f.shapes.output = Shape{1, 2, pair[0]};
    LSE_EXPECT(!dispatch::q8_packed_matrix_shape(f.original()));
    f.inputs[0] = Shape{1, 1, pair[1]};
    f.shapes.output = Shape{1, 1, pair[0]};
    f.inputs[4] = Shape{pair[0], pair[1] / 4 - 1};
    LSE_EXPECT(!dispatch::q8_packed_matrix_plan(f.shapes).matrix);
  }
  Fixture f;
  f.device.wavefront_size = 64;
  LSE_EXPECT(!dispatch::q8_packed_weight_device(f.device));
  f.device.wavefront_size = 32;
  f.amd.matrix_core = backend::MatrixCore::kNone;
  LSE_EXPECT(!dispatch::q8_packed_matrix_plan(f.shapes).matrix);
  f.amd.matrix_core = backend::MatrixCore::kWMMA;
  f.device.arch = "gfx1151";
  LSE_EXPECT(!dispatch::q8_packed_weight_device(f.device));
  f.device.arch = "gfx1201";
  f.shapes.staged.name = "foreign";
  LSE_EXPECT(!dispatch::q8_packed_matrix_plan(f.shapes).matrix);
  f.shapes.staged = {};
  f.dtypes[5] = DType::kF16;
  LSE_EXPECT(!dispatch::q8_packed_matrix_plan(f.shapes).matrix);
  Fixture other(5120, 17408);
  LSE_EXPECT(!dispatch::q8_packed_matrix_shape(other.original()));
}
LSE_TEST(
    q8_packed_leaves_are_reused_without_cycles_and_rebound_sources_decline) {
  Fixture f;
  Weights w(f);
  const auto x = leaf(f.inputs[0], f.dtypes[0]);
  auto a = w.projection(x), b = w.projection(x);
  LSE_EXPECT(a.node()->prim && a.node()->prim->name() == kPacked);
  LSE_EXPECT_EQ(a.node()->inputs.size(), 7u);
  for (std::size_t i = 4; i < 7; ++i) {
    LSE_EXPECT(a.node()->inputs[i] == b.node()->inputs[i]);
    LSE_EXPECT(a.node()->inputs[i]->kind == OpKind::kBuffer);
    LSE_EXPECT(a.node()->inputs[i]->materialized);
  }
  const auto original = w.words.node()->buffer;
  ++w.words.node()->buffer.offset;
  LSE_EXPECT_EQ(w.projection(x).node()->inputs.size(), 4u);
  w.words.node()->buffer = original;
  const auto copy = Array(std::make_shared<Node>(*w.words.node()));
  LSE_EXPECT_EQ(
      quant_linear(x, copy, w.scales, w.biases, 8, 64).node()->inputs.size(),
      4u);
  w.scales.node()->host_dirty = true;
  LSE_EXPECT_EQ(w.projection(x).node()->inputs.size(), 4u);
  w.scales.node()->host_dirty = false;
  ++w.storage->packed[0]->buffer.residency.value;
  LSE_EXPECT_EQ(w.projection(x).node()->inputs.size(), 4u);
  --w.storage->packed[0]->buffer.residency.value;
  const auto weak = std::weak_ptr<Node>(w.words.node());
  a = {};
  b = {};
  w.words = {};
  LSE_EXPECT(weak.expired());
}
LSE_TEST(
    q8_packed_cache_tracks_capability_and_fallback_matches_legacy_body_plan) {
  Fixture f;
  Weights w(f);
  const auto output = w.projection(leaf(f.inputs[0], f.dtypes[0]));
  const auto g = group(output);
  const auto *kernel =
      dynamic_cast<const KernelPrimitiveBase *>(output.node()->prim);
  const auto *legacy =
      dynamic_cast<const KernelPrimitiveBase *>(find_primitive("quant_linear"));
  LSE_EXPECT(kernel && legacy);
  if (!kernel || !legacy)
    return;
  backend::LoomEmitter loom;
  backend::HipEmitter hip;
  const auto loom_key = loom.cache_key(g, f.device),
             hip_key = hip.cache_key(g, f.device);
  const auto packed_body = kernel->emit_kernel(f.shapes);
  LSE_EXPECT(!packed_body.empty());
  const auto packed_plan = kernel->plan(f.shapes);
  LSE_EXPECT_EQ(packed_plan.workgroup_count[0], 136u);
  LSE_EXPECT_EQ(packed_plan.workgroup_size[0], 256u);
  f.amd.matrix_core = backend::MatrixCore::kNone;
  LSE_EXPECT(loom.cache_key(g, f.device) != loom_key &&
             hip.cache_key(g, f.device) != hip_key);
  const auto original = f.original();
  const auto *baseline = legacy->specialize(original);
  LSE_EXPECT(kernel->emit_kernel(f.shapes) == baseline->emit_kernel(original));
  const auto actual = kernel->plan(f.shapes),
             expected = baseline->plan(original);
  for (std::size_t i = 0; i < 3; ++i) {
    LSE_EXPECT_EQ(actual.workgroup_count[i], expected.workgroup_count[i]);
    LSE_EXPECT_EQ(actual.workgroup_size[i], expected.workgroup_size[i]);
  }
  LSE_EXPECT_EQ(actual.lds_bytes, expected.lds_bytes);
  f.amd.matrix_core = backend::MatrixCore::kWMMA;
  LSE_EXPECT_EQ(loom.cache_key(g, f.device), loom_key);
  LSE_EXPECT_EQ(hip.cache_key(g, f.device), hip_key);
}
LSE_TEST_MAIN()
