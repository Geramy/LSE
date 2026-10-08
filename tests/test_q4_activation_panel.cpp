#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/kernels/lds_linear.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {
constexpr auto kProducer = "quant_activation.q4_shared_panel.v1";
constexpr auto kConsumer = "quant_linear.q4_global_panel.v1";
constexpr auto kRows8Consumer = "quant_linear.q4_global_panel.rows8.v2";
using Projection = std::array<std::size_t, 2>;
constexpr std::array kFFNProjections{Projection{17408, 5120},
                                     Projection{5120, 17408}};
constexpr std::array kAdditionalProjections{
    Projection{10240, 5120}, Projection{6144, 5120}, Projection{12288, 5120},
    Projection{5120, 6144}, Projection{248320, 5120}};
constexpr std::array<Projection, 1> kM7Head{Projection{248320, 5120}};
constexpr std::array kWidePanelProjections{
    Projection{17408, 5120}, Projection{5120, 17408}, Projection{10240, 5120},
    Projection{6144, 5120},  Projection{12288, 5120}, Projection{5120, 6144},
    Projection{248320, 5120}};
Array leaf(Shape shape, DType type) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  node->materialized = true;
  return Array(node);
}
template <class T>
Array filled(Shape shape, DType type, const std::vector<T> &values) {
  auto *scheduler = default_scheduler();
  if (!scheduler)
    return {};
  auto storage = scheduler->backend().allocate(values.size() * sizeof(T),
                                               backend::MemoryClass::kDevice);
  if (!storage.ok())
    return {};
  auto buffer = storage.release();
  if (!scheduler->backend()
           .copy(buffer, values.data(), values.size() * sizeof(T))
           .ok())
    return {};
  return Array::from_buffer(std::move(buffer), shape, type);
}
template <class T> std::vector<T> read(Array value) {
  std::vector<T> data(value.shape().elem_count());
  LSE_EXPECT_OK(value.to_host(data.data(), data.size() * sizeof(T)));
  return data;
}
Array panel(const Array &x) {
  auto result = custom(kProducer, {x});
  LSE_EXPECT(result.ok());
  return result.ok() ? result.release() : Array{};
}
std::vector<std::uint32_t> codec(const std::vector<float> &x, std::size_t k) {
  const auto stride = k * 25 / 64;
  std::vector<std::uint32_t> result(x.size() / k * stride);
  for (std::size_t row = 0; row < x.size() / k; ++row)
    for (std::size_t group = 0; group < k / 64; ++group) {
      std::array<float, 8> sums{};
      for (std::size_t chunk = 0; chunk < 8; ++chunk) {
        const auto c = group * 8 + chunk;
        const auto *v = x.data() + row * k + c * 8;
        float amax = std::abs(v[0]);
        sums[chunk] = v[0];
        for (std::size_t j = 1; j < 8; ++j) {
          amax = std::max(amax, std::abs(v[j]));
          sums[chunk] = sums[chunk] + v[j];
        }
        const float inverse = 127.0f / std::max(amax, 1e-30f);
        std::array<std::int32_t, 8> codes{};
        for (std::size_t j = 0; j < 8; ++j)
          codes[j] = static_cast<std::int32_t>(std::nearbyint(v[j] * inverse));
        for (std::size_t plane = 0; plane < 2; ++plane) {
          std::uint32_t word = 0;
          for (std::size_t byte = 0; byte < 4; ++byte)
            word |= (static_cast<std::uint32_t>(codes[2 * byte + plane]) & 255u)
                    << (8 * byte);
          result[row * stride + c * 2 + plane] = word;
        }
        result[row * stride + k / 4 + c] =
            std::bit_cast<std::uint32_t>(amax * (1.0f / 127.0f));
      }
      const float left = (sums[0] + sums[1]) + (sums[2] + sums[3]);
      const float right = (sums[4] + sums[5]) + (sums[6] + sums[7]);
      result[row * stride + 3 * k / 8 + group] =
          std::bit_cast<std::uint32_t>(left + right);
    }
  return result;
}
Array contraction(const Array &x, std::int64_t n, std::int64_t k) {
  return quant_linear(x, leaf({n, k / 8}, DType::kU32),
                      leaf({n, k / 64}, DType::kBF16),
                      leaf({n, k / 64}, DType::kBF16), 4, 64);
}
} // namespace

LSE_TEST(q4_panel_graph_shares_siblings_but_rejects_cloned_input_cache) {
  auto x = leaf({1, 4, 5120}, DType::kF32);
  auto a = contraction(x, 17408, 5120), b = contraction(x, 17408, 5120);
  LSE_EXPECT_EQ(a.node()->inputs.size(), 5u);
  LSE_EXPECT(a.node()->prim && a.node()->prim->name() == kConsumer);
  LSE_EXPECT(a.node()->inputs[4] == b.node()->inputs[4]);
  LSE_EXPECT(a.node()->inputs[4]->shape == Shape{4, 2000});
  LSE_EXPECT_EQ(a.node()->inputs[4]->consumer_count, 2u);
  auto clone = std::make_shared<Node>(*x.node());
  auto c = contraction(Array(clone), 17408, 5120);
  LSE_EXPECT(c.node()->inputs[4] != a.node()->inputs[4]);
  LSE_EXPECT(c.node()->inputs[4]->inputs[0] == clone);
  const auto weak = x.node()->quant_activation_panel;
  a = {};
  b = {};
  LSE_EXPECT(weak.expired());
}
LSE_TEST(q4_panel_single_token_ffn_uses_shared_activation_panel) {
  auto x = leaf({1, 5120}, DType::kF32);
  auto gate = contraction(x, 17408, 5120);
  auto up = contraction(x, 17408, 5120);
  LSE_EXPECT_EQ(gate.node()->inputs.size(), 5u);
  LSE_EXPECT(gate.node()->prim && gate.node()->prim->name() == kConsumer);
  LSE_EXPECT(gate.node()->inputs[4] == up.node()->inputs[4]);
  LSE_EXPECT(gate.node()->inputs[4]->shape == Shape{1, 2000});
}
LSE_TEST(q4_panel_graph_keeps_unmeasured_shapes_and_formats_on_legacy_route) {
  for (auto m : {2, 3, 7, 9, 512}) {
    auto y = contraction(leaf({m, 5120}, DType::kF32), 17408, 5120);
    LSE_EXPECT_EQ(y.node()->inputs.size(), 4u);
    LSE_EXPECT(y.node()->prim && y.node()->prim->name() == "quant_linear");
  }
  auto x = leaf({4, 5120}, DType::kF32);
  auto unmeasured = contraction(x, 4096, 5120);
  LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
  auto f32 = quant_linear(x, leaf({17408, 640}, DType::kU32),
                          leaf({17408, 80}, DType::kF32),
                          leaf({17408, 80}, DType::kF32), 4, 64);
  LSE_EXPECT_EQ(f32.node()->inputs.size(), 4u);
}
LSE_TEST(q4_panel_single_token_down_uses_panel_without_widening_admission) {
  auto down = contraction(leaf({1, 17408}, DType::kF32), 5120, 17408);
  LSE_EXPECT_EQ(down.node()->inputs.size(), 5u);
  LSE_EXPECT(down.node()->prim && down.node()->prim->name() == kConsumer);
  LSE_EXPECT(down.node()->inputs[4]->shape == Shape{1, 6800});
  for (const auto& dims : kAdditionalProjections) {
    auto value = contraction(leaf({1, static_cast<std::int64_t>(dims[1])}, DType::kF32),
                             static_cast<std::int64_t>(dims[0]),
                             static_cast<std::int64_t>(dims[1]));
    LSE_EXPECT_EQ(value.node()->inputs.size(), 4u);
  }
}
LSE_TEST(
    q4_panel_additional_projections_share_M4_panels_and_reject_unmeasured_widths) {
  auto x = leaf({1, 4, 5120}, DType::kF32);
  const auto ffn = contraction(x, 17408, 5120);
  for (const auto &pair : kAdditionalProjections) {
    const auto n = static_cast<std::int64_t>(pair[0]);
    const auto k = static_cast<std::int64_t>(pair[1]);
    const auto input = k == 5120 ? x : leaf({1, 4, k}, DType::kF32);
    const auto output = contraction(input, n, k);
    LSE_EXPECT_EQ(output.node()->inputs.size(), 5u);
    LSE_EXPECT(output.node()->prim && output.node()->prim->name() == kConsumer);
    LSE_EXPECT(output.node()->inputs[4]->shape == Shape{4, k * 25 / 64});
    if (k == 5120)
      LSE_EXPECT(output.node()->inputs[4] == ffn.node()->inputs[4]);
    const auto fewer = contraction(leaf({1, 3, k}, DType::kF32), n, k);
    const auto wider = contraction(leaf({1, 9, k}, DType::kF32), n, k);
    LSE_EXPECT_EQ(fewer.node()->inputs.size(), 4u);
    LSE_EXPECT_EQ(wider.node()->inputs.size(), 4u);
  }
}
LSE_TEST(
    q4_panel_M7_head_uses_seven_panel_rows_and_shape_local_specialization) {
  auto x = leaf({1, 7, 5120}, DType::kF32);
  const auto a = contraction(x, 248320, 5120);
  const auto b = contraction(x, 248320, 5120);
  LSE_EXPECT_EQ(a.node()->inputs.size(), 5u);
  LSE_EXPECT(a.node()->prim && a.node()->prim->name() == kConsumer);
  LSE_EXPECT(a.node()->inputs[4] == b.node()->inputs[4]);
  LSE_EXPECT(a.node()->inputs[4]->shape == Shape{7, 2000});
  for (const auto n : {6144, 10240, 12288, 17408}) {
    const auto unmeasured = contraction(x, n, 5120);
    LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
    LSE_EXPECT(unmeasured.node()->prim &&
               unmeasured.node()->prim->name() == "quant_linear");
  }
  for (const auto m : {1, 3, 5, 9}) {
    const auto unmeasured =
        contraction(leaf({1, m, 5120}, DType::kF32), 248320, 5120);
    LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
  }
}
LSE_TEST(q4_panel_M6_measured_projections_share_six_row_panels) {
  auto x = leaf({1, 6, 5120}, DType::kF32);
  const auto ffn = contraction(x, 17408, 5120);
  for (const auto &pair : kWidePanelProjections) {
    const auto n = static_cast<std::int64_t>(pair[0]);
    const auto k = static_cast<std::int64_t>(pair[1]);
    const auto input = k == 5120 ? x : leaf({1, 6, k}, DType::kF32);
    const auto output = contraction(input, n, k);
    LSE_EXPECT_EQ(output.node()->inputs.size(), 5u);
    LSE_EXPECT(output.node()->prim && output.node()->prim->name() == kConsumer);
    LSE_EXPECT(output.node()->inputs[4]->shape == Shape{6, (k / 64) * 25});
    if (k == 5120)
      LSE_EXPECT(output.node()->inputs[4] == ffn.node()->inputs[4]);
    for (const auto m : {3, 5, 9}) {
      const auto unmeasured = contraction(leaf({1, m, k}, DType::kF32), n, k);
      LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
    }
  }
  const auto unmeasured = contraction(x, 4096, 5120);
  LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
}
LSE_TEST(q4_panel_M8_measured_projections_share_eight_row_panels) {
  auto x = leaf({1, 8, 5120}, DType::kF32);
  const auto ffn = contraction(x, 17408, 5120);
  for (const auto &pair : kWidePanelProjections) {
    const auto n = static_cast<std::int64_t>(pair[0]);
    const auto k = static_cast<std::int64_t>(pair[1]);
    const auto input = k == 5120 ? x : leaf({1, 8, k}, DType::kF32);
    const auto output = contraction(input, n, k);
    LSE_EXPECT_EQ(output.node()->inputs.size(), 5u);
    const bool matrix = n == 5120 && k == 17408;
    LSE_EXPECT(output.node()->prim &&
               output.node()->prim->name() ==
                   (matrix ? "quant_linear.q4_matrix_panel.v1" : kConsumer));
    LSE_EXPECT(output.node()->inputs[4]->shape ==
               (matrix ? Shape{k / 64, dispatch::kQ4MatrixPanelGroupWords}
                       : Shape{8, (k / 64) * 25}));
    if (k == 5120)
      LSE_EXPECT(output.node()->inputs[4] == ffn.node()->inputs[4]);
    const auto unmeasured = contraction(leaf({1, 9, k}, DType::kF32), n, k);
    LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
    LSE_EXPECT(unmeasured.node()->prim &&
               unmeasured.node()->prim->name() == "quant_linear");
  }
  const auto unmeasured = contraction(x, 4096, 5120);
  LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
}
LSE_TEST(q4_panel_M8_rate_projections_share_existing_GDN_source_panel) {
  auto x = leaf({1, 8, 5120}, DType::kF32);
  const auto qkv = contraction(x, 10240, 5120);
  const auto a = contraction(x, 48, 5120);
  const auto b = contraction(x, 48, 5120);
  const auto gate = contraction(x, 6144, 5120);
  const auto shared = qkv.node()->inputs.at(4);
  for (const auto& y : {qkv, a, b, gate}) {
    LSE_EXPECT_EQ(y.node()->inputs.size(), 5u);
    LSE_EXPECT(y.node()->prim && y.node()->prim->name() == kConsumer);
    LSE_EXPECT(y.node()->inputs[4] == shared);
  }
  LSE_EXPECT(shared->shape == Shape{8, 2000});
  LSE_EXPECT(shared->inputs.size() == 1u && shared->inputs[0] == x.node());
  LSE_EXPECT_EQ(shared->consumer_count, 4u);
  const auto alpha = exp(neg(mul(leaf({48}, DType::kF32),
                               softplus(add(a, leaf({48}, DType::kBF16))))));
  const auto beta = clamp(sigmoid(b), 1e-4f, 1.0f - 1e-4f);
  LSE_EXPECT(alpha.shape() == Shape{1, 8, 48});
  LSE_EXPECT(beta.shape() == alpha.shape());
  std::vector<NodePtr> visited;
  std::size_t producers = 0;
  const auto visit = [&](auto&& self, const NodePtr& node) -> void {
    if (std::find(visited.begin(), visited.end(), node) != visited.end()) return;
    visited.push_back(node);
    if (node->prim && node->prim->name() == kProducer) ++producers;
    for (const auto& input : node->inputs) self(self, input);
  };
  for (const auto& root : {qkv, alpha, beta, gate}) visit(visit, root.node());
  LSE_EXPECT_EQ(producers, 1u);
  auto clone = std::make_shared<Node>(*x.node());
  const auto cloned = contraction(Array(clone), 48, 5120);
  LSE_EXPECT(cloned.node()->inputs[4] != shared);
  LSE_EXPECT(cloned.node()->inputs[4]->inputs[0] == clone);
  for (const auto m : {1, 6, 7, 9}) {
    const auto unmeasured = contraction(leaf({1, m, 5120}, DType::kF32), 48, 5120);
    LSE_EXPECT_EQ(unmeasured.node()->inputs.size(), 4u);
  }
  LSE_EXPECT_EQ(contraction(leaf({1, 4, 5120}, DType::kF32), 48, 5120)
                    .node()->inputs.size(), 5u);
  LSE_EXPECT_EQ(contraction(x, 49, 5120).node()->inputs.size(), 4u);
  LSE_EXPECT_EQ(contraction(leaf({1, 8, 6144}, DType::kF32), 48, 6144)
                    .node()->inputs.size(), 4u);
}
LSE_TEST(q4_panel_typed_host_codec_and_retained_replay_are_exact) {
  auto *scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler)
    return;
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  std::vector<float> data(4 * 128);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = std::sin(static_cast<float>(i) * .137f) * .731f;
  for (std::size_t i = 0; i < 64; ++i)
    data[i] = 0;
  auto x = filled({1, 4, 128}, DType::kF32, data);
  auto y = panel(x);
  const NodePtr roots[]{y.node()};
  Program program;
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT(read<std::uint32_t>(y) == codec(data, 128));
  for (auto &value : data)
    value = value * -.617f + .031f;
  LSE_EXPECT_OK(scheduler->backend().copy(x.node()->buffer, data.data(),
                                          data.size() * sizeof(float)));
  x.node()->host_dirty = false;
  x.node()->device_dirty = true;
  program.reset_compute();
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT(read<std::uint32_t>(y) == codec(data, 128));
}
LSE_TEST(q4_panel_consumer_cpu_reference_uses_original_activations) {
  auto *scheduler = default_scheduler();
  LSE_EXPECT(scheduler != nullptr);
  if (!scheduler)
    return;
  scheduler->set_mode(Scheduler::Mode::kHostOnly);
  constexpr std::size_t m = 4, n = 3, k = 64;
  std::vector<float> data(m * k);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = std::sin(static_cast<float>(i) * .173f) * .613f;
  std::vector<std::uint32_t> packed(n * k / 8);
  for (std::size_t i = 0; i < packed.size(); ++i)
    packed[i] = 0x13579bdfu + static_cast<std::uint32_t>(i) * 0x01010101u;
  const std::vector<bfloat16_t> scales(n, bfloat16_t(.0237f)),
      biases(n, bfloat16_t(-.131f));
  auto x = filled({1, m, k}, DType::kF32, data);
  auto base = quant_linear(x, filled({n, k / 8}, DType::kU32, packed),
                           filled({n, 1}, DType::kBF16, scales),
                           filled({n, 1}, DType::kBF16, biases), 4, 64);
  auto modified = std::make_shared<Node>(*base.node());
  modified->prim = find_primitive(kConsumer);
  modified->inputs.push_back(panel(x).node());
  // Replace the panel with deliberately unrelated words: the builtin CPU
  // reference reads x.
  modified->inputs.back() =
      filled({m, 25}, DType::kU32,
             std::vector<std::uint32_t>(m * 25, 0xffffffffu))
          .node();
  const auto actual = read<float>(Array(modified)),
             expected = read<float>(base);
  LSE_EXPECT(actual.size() == expected.size());
  LSE_EXPECT(std::memcmp(actual.data(), expected.data(),
                         actual.size() * sizeof(float)) == 0);
}
LSE_TEST(q4_panel_native_emit_uses_zero_lds_and_legacy_nontarget_plan) {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  device.arch = "gfx1201";
  backend::apply_arch_defaults(device, amd);
  device.extension_id = backend::AmdDeviceInfo::kExtensionId;
  device.extension = &amd;
  device.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter loom;
  backend::HipEmitter hip;
  const auto loom_intrinsics = loom.sources();
  const auto check_legacy = [&](const FusionGroup &group,
                                IKernelEmitter &emitter,
                                const EmittedKernel &emitted) {
    const auto *consumer =
        dynamic_cast<const KernelPrimitiveBase *>(group.outputs[0]->prim);
    const auto *legacy = dynamic_cast<const KernelPrimitiveBase *>(
        find_primitive("quant_linear"));
    LSE_EXPECT(consumer && legacy);
    if (!consumer || !legacy)
      return;
    std::vector<Shape> shapes;
    std::vector<DType> types;
    for (const auto &input : group.outputs[0]->inputs) {
      shapes.push_back(input->shape);
      types.push_back(input->dtype);
    }
    const auto intrinsics = emitter.sources();
    KernelShapes full;
    full.inputs = shapes;
    full.input_dtypes = types;
    full.output = group.outputs[0]->shape;
    full.iattrs = {4, 64, 0, 0};
    full.device = &device;
    full.intrinsics = &intrinsics;
    full.types = emitter.dialect() == Dialect::kLoom ? backend::loom_types()
                                                     : backend::hip_types();
    full.store = [](std::string_view index, std::string_view value) {
      return "out[" + std::string(index) + "] = " + std::string(value) + ";";
    };
    auto original = full;
    original.inputs = full.inputs.first(4);
    original.input_dtypes = full.input_dtypes.first(4);
    LSE_EXPECT(!dispatch::quant_plan(original).shared_activation_panel);
    const auto reference_body = legacy->emit_kernel(original);
    LSE_EXPECT(!reference_body.empty());
    LSE_EXPECT(consumer->emit_kernel(full) == reference_body);
    const auto expected = legacy->plan(original), actual = consumer->plan(full);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      LSE_EXPECT_EQ(actual.workgroup_size[axis], expected.workgroup_size[axis]);
      LSE_EXPECT_EQ(actual.workgroup_count[axis],
                    expected.workgroup_count[axis]);
    }
    LSE_EXPECT_EQ(actual.lds_bytes, expected.lds_bytes);
    LSE_EXPECT_EQ(emitted.lds_bytes, expected.lds_bytes);
  };
  for (const auto dimensions : {std::array<std::int64_t, 3>{4, 17408, 5120},
                                {4, 5120, 17408},
                                {4, 10240, 5120},
                                {4, 6144, 5120},
                                {4, 12288, 5120},
                                {4, 5120, 6144},
                                {4, 248320, 5120},
                                {7, 248320, 5120},
                                {6, 17408, 5120},
                                {6, 5120, 17408},
                                {6, 10240, 5120},
                                {6, 6144, 5120},
                                {6, 12288, 5120},
                                {6, 5120, 6144},
                                {6, 248320, 5120},
                                {8, 17408, 5120},
                                {8, 5120, 17408},
                                {8, 10240, 5120},
                                {8, 6144, 5120},
                                {8, 5120, 6144}}) {
    auto out = contraction(leaf({1, dimensions[0], dimensions[2]}, DType::kF32),
                           dimensions[1], dimensions[2]);
    std::vector<Shape> input_shapes;
    std::vector<DType> input_types;
    for (const auto &input : out.node()->inputs) {
      input_shapes.push_back(input->shape);
      input_types.push_back(input->dtype);
    }
    KernelShapes invocation;
    invocation.inputs = input_shapes;
    invocation.input_dtypes = input_types;
    invocation.output = out.shape();
    invocation.iattrs = {4, 64, 0, 0};
    invocation.device = &device;
    invocation.intrinsics = &loom_intrinsics;
    invocation.types = backend::loom_types();
    auto original_shape = invocation;
    original_shape.inputs = invocation.inputs.first(4);
    original_shape.input_dtypes = invocation.input_dtypes.first(4);
    // gfx1201 runs six- and seven-row passes on the eight-row rules
    // (dispatch::verify_rows).
    const auto rows = dispatch::verify_rows(device.arch, dimensions[0]);
    LSE_EXPECT_EQ(rows, dimensions[0] >= 6 ? 8 : dimensions[0]);
    {
      // A padded pass is scheduled as the pass it runs as, so each of its
      // rows accumulates in that pass's order.
      std::vector<Shape> as_rows(input_shapes.begin(), input_shapes.begin() + 4);
      as_rows[0] = Shape{1, rows, dimensions[2]};
      auto measured = original_shape;
      measured.inputs = as_rows;
      measured.output = Shape{1, rows, dimensions[1]};
      const auto a = lse::kernels::dot4_schedule(original_shape);
      const auto b = lse::kernels::dot4_schedule(measured);
      LSE_EXPECT(a.valid());
      LSE_EXPECT_EQ(a.rows, b.rows);
      LSE_EXPECT_EQ(a.chunks_per_lane, b.chunks_per_lane);
      LSE_EXPECT_EQ(a.k_splits, b.k_splits);
    }
    const bool adjacent =
        (rows == 4 && dimensions[2] == 5120 &&
         (dimensions[1] == 17408 || dimensions[1] == 10240 || dimensions[1] == 6144)) ||
        (rows == 8 &&
         ((dimensions[2] == 5120 &&
           (dimensions[1] == 17408 || dimensions[1] == 10240 || dimensions[1] == 6144)) ||
          (dimensions[1] == 5120 &&
           (dimensions[2] == 17408 || dimensions[2] == 6144))));
    const bool four_chunks =
        dimensions[2] == 5120 &&
        rows == 8 && (dimensions[1] == 17408 || dimensions[1] == 10240 ||
                      dimensions[1] == 12288 || dimensions[1] == 248320);
    LSE_EXPECT_EQ(dispatch::q4_shared_panel_load_chunks(original_shape),
                  four_chunks ? 4u : adjacent ? 2u : 1u);
    // The vocabulary projection at six to eight rows gives each wave four
    // columns and the 4-row down projection two; every other shape keeps one.
    const std::uint32_t columns =
        dimensions[1] == 248320 && dimensions[0] >= 6 ? 4u
        : dimensions[0] == 4 && dimensions[1] == 5120 && dimensions[2] == 17408
            ? 2u : 1u;
    LSE_EXPECT_EQ(dispatch::q4_shared_panel_columns(original_shape), columns);
    const auto *consumer =
        dynamic_cast<const KernelPrimitiveBase *>(out.node()->prim);
    LSE_EXPECT(consumer != nullptr);
    if (consumer) {
      const auto *selected = consumer->specialize(invocation);
      LSE_EXPECT(selected->name() ==
                 (dimensions[0] == 8 && dimensions[1] == 5120 &&
                          dimensions[2] == 17408
                      ? "quant_linear.q4_matrix_panel.v1"
                      : dimensions[0] == 4 ? kConsumer : kRows8Consumer));
      if (dimensions[0] != 4) {
        auto original = invocation;
        original.inputs = invocation.inputs.first(4);
        original.input_dtypes = invocation.input_dtypes.first(4);
        LSE_EXPECT_EQ(dispatch::q4_shared_panel_rows(original), 8u);
        LSE_EXPECT_EQ(selected->plan(invocation).workgroup_count[1], 1u);
      }
    }
    const NodePtr roots[]{out.node()};
    const auto groups = Partitioner::partition(roots);
    LSE_EXPECT_EQ(groups.size(), 2u);
    for (IKernelEmitter *emitter : {static_cast<IKernelEmitter *>(&loom),
                                    static_cast<IKernelEmitter *>(&hip)})
      for (const auto &group : groups) {
        const auto emitted = emitter->emit(group, device);
        LSE_EXPECT(emitted.ok());
        if (!emitted.ok()) {
          std::fprintf(stderr, "%s\n", emitted.status().to_string().c_str());
          continue;
        }
        LSE_EXPECT_EQ(emitted->lds_bytes, 0u);
        // The 8-row matrix panel narrows to four waves to cover every unit.
        LSE_EXPECT_EQ(emitted->dims.workgroup_size[0],
                      group.outputs[0]->prim->name() == "quant_linear.q4_matrix_panel.v1"
                          ? 128u : 256u);
        if (group.outputs[0]->prim->name() == kConsumer) {
          LSE_EXPECT_EQ(emitted->dims.workgroup_count[1], 1u);
          LSE_EXPECT_EQ(emitted->dims.workgroup_count[0],
                        static_cast<std::uint32_t>(
                            (dimensions[1] + 8 * columns - 1) / (8 * columns)));
        }
      }
    for (const auto &group : groups) {
      if (group.outputs[0]->prim->name() != kConsumer)
        continue;
      const auto before_key = hip.cache_key(group, device);
      const auto before = hip.emit(group, device);
      amd.has_dot4_iu8 = false;
      const auto after_key = hip.cache_key(group, device);
      const auto after = hip.emit(group, device);
      LSE_EXPECT(before_key != after_key);
      LSE_EXPECT(before.ok() && after.ok());
      if (before.ok() && after.ok()) {
        LSE_EXPECT(before->source != after->source);
        LSE_EXPECT_EQ(before->lds_bytes, 0u);
        check_legacy(group, hip, *after);
      }
      amd.has_dot4_iu8 = true;
      const auto restored = hip.emit(group, device);
      LSE_EXPECT(restored.ok());
      if (restored.ok() && before.ok())
        LSE_EXPECT(restored->source == before->source);
    }
    device.arch = "gfx1100";
    backend::apply_arch_defaults(device, amd);
    for (const auto &group : groups) {
      if (group.outputs[0]->prim->name() != kConsumer)
        continue;
      const auto emitted = loom.emit(group, device);
      LSE_EXPECT(emitted.ok());
      if (emitted.ok())
        check_legacy(group, loom, *emitted);
    }
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
  }
}

int gpu_panel(std::size_t m, std::span<const Projection> projections) {
  auto *scheduler = default_scheduler();
  if (!scheduler)
    return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  for (const auto &dims : projections) {
    struct Guarded {
      backend::DeviceBuffer allocation;
      std::vector<std::byte> bytes;
      bool readonly;
    };
    std::vector<Guarded> allocations;
    const auto guard = [&](const NodePtr &node, const void *data,
                           bool readonly) {
      const auto bytes =
          dtype_storage_bytes(node->dtype, node->element_count());
      Guarded g{
          {}, std::vector<std::byte>(bytes + 128, std::byte{0xa5}), readonly};
      if (data) {
        std::memcpy(g.bytes.data() + 64, data, bytes);
      } else if (node->dtype == DType::kF32) {
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < node->element_count(); ++i)
          std::memcpy(g.bytes.data() + 64 + i * sizeof(float), &nan,
                      sizeof(float));
      }
      auto storage = scheduler->backend().allocate(
          g.bytes.size(), backend::MemoryClass::kDevice);
      LSE_EXPECT(storage.ok());
      if (!storage.ok())
        return;
      g.allocation = storage.release();
      LSE_EXPECT_OK(scheduler->backend().copy(g.allocation, g.bytes.data(),
                                              g.bytes.size()));
      node->buffer = g.allocation;
      node->buffer.offset += 64;
      node->buffer.size_bytes = bytes;
      node->host_mirror.clear();
      node->host_dirty = false;
      node->device_dirty = data != nullptr;
      allocations.push_back(std::move(g));
    };
    const auto n = dims[0], k = dims[1];
    std::vector<float> data(m * k);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = std::sin(static_cast<float>(i) * .173f) * .613f;
    std::vector<std::uint32_t> words(n * k / 8);
    for (std::size_t i = 0; i < words.size(); ++i)
      words[i] = 0x13579bdfu + static_cast<std::uint32_t>(i) * 0x01010101u;
    std::vector<bfloat16_t> scales(n * k / 64), biases(scales.size());
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = bfloat16_t(.0237f + static_cast<float>(i % 31) * .0013f);
      biases[i] = bfloat16_t(-.131f + static_cast<float>(i % 17) * .0027f);
    }
    auto fast = quant_linear(
        filled({1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)},
               DType::kF32, data),
        filled({static_cast<std::int64_t>(n), static_cast<std::int64_t>(k / 8)},
               DType::kU32, words),
        filled(
            {static_cast<std::int64_t>(n), static_cast<std::int64_t>(k / 64)},
            DType::kBF16, scales),
        filled(
            {static_cast<std::int64_t>(n), static_cast<std::int64_t>(k / 64)},
            DType::kBF16, biases),
        4, 64);
    auto base = std::make_shared<Node>(*fast.node());
    base->prim = find_primitive("quant_linear");
    base->inputs.resize(4);
    guard(fast.node()->inputs[0], data.data(), true);
    guard(fast.node()->inputs[1], words.data(), true);
    guard(fast.node()->inputs[2], scales.data(), true);
    guard(fast.node()->inputs[3], biases.data(), true);
    guard(fast.node()->inputs[4], nullptr, false);
    guard(base, nullptr, false);
    guard(fast.node(), nullptr, false);
    const NodePtr roots[]{base, fast.node(), fast.node()->inputs[4]};
    scheduler->reset_accumulated_trace();
    LSE_EXPECT_OK(scheduler->eval(roots, false));
    LSE_EXPECT_OK(scheduler->drain());
    const auto trace = scheduler->last_trace();
    LSE_EXPECT(trace.device_groups > 0);
    LSE_EXPECT_EQ(trace.host_groups, 0u);
    LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
    const auto expected = read<float>(Array(base)), actual = read<float>(fast);
    LSE_EXPECT(actual.size() == expected.size());
    for (float value : actual)
      LSE_EXPECT(std::isfinite(value));
    for (float value : expected)
      LSE_EXPECT(std::isfinite(value));
    LSE_EXPECT(std::memcmp(actual.data(), expected.data(),
                           actual.size() * sizeof(float)) == 0);
    LSE_EXPECT(read<std::uint32_t>(Array(fast.node()->inputs[4])) ==
               codec(data, k));
    std::vector<float> residual(m * n);
    for (std::size_t i = 0; i < residual.size(); ++i)
      residual[i] = std::cos(static_cast<float>(i) * .091f) * .113f;
    auto r =
        filled({1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(n)},
               DType::kF32, residual);
    guard(r.node(), residual.data(), true);
    fast.node()->materialized = false;
    base->materialized = false;
    auto fast_epilogue = add(fast, r), base_epilogue = add(Array(base), r);
    guard(fast_epilogue.node(), nullptr, false);
    guard(base_epilogue.node(), nullptr, false);
    const NodePtr epilogue_roots[]{base_epilogue.node(), fast_epilogue.node()};
    const auto groups = Partitioner::partition(epilogue_roots);
    unsigned fused = 0;
    for (const auto &group : groups)
      if (group.anchor == OpKind::kQuantMatMul && group.nodes.size() > 1)
        ++fused;
    LSE_EXPECT_EQ(fused, 2u);
    scheduler->reset_accumulated_trace();
    LSE_EXPECT_OK(scheduler->eval(epilogue_roots, false));
    LSE_EXPECT_OK(scheduler->drain());
    const auto epilogue_trace = scheduler->last_trace();
    LSE_EXPECT(epilogue_trace.device_groups > 0);
    LSE_EXPECT_EQ(epilogue_trace.host_groups, 0u);
    LSE_EXPECT_EQ(epilogue_trace.host_fallbacks, 0u);
    const auto epilogue_base = read<float>(base_epilogue);
    const auto epilogue_fast = read<float>(fast_epilogue);
    LSE_EXPECT(epilogue_fast.size() == epilogue_base.size());
    for (float value : epilogue_fast)
      LSE_EXPECT(std::isfinite(value));
    for (float value : epilogue_base)
      LSE_EXPECT(std::isfinite(value));
    LSE_EXPECT(std::memcmp(epilogue_base.data(), epilogue_fast.data(),
                           epilogue_base.size() * sizeof(float)) == 0);
    for (const auto &allocation : allocations) {
      std::vector<std::byte> bytes(allocation.bytes.size());
      LSE_EXPECT_OK(scheduler->backend().copy(
          bytes.data(), allocation.allocation, bytes.size()));
      for (std::size_t i = 0; i < 64; ++i) {
        LSE_EXPECT(bytes[i] == std::byte{0xa5});
        LSE_EXPECT(bytes[bytes.size() - 64 + i] == std::byte{0xa5});
      }
      if (allocation.readonly)
        LSE_EXPECT(bytes == allocation.bytes);
    }
    std::printf(
        "Q4 panel M%zu N%zu K%zu: device=%u host=%u fallback=%u complete "
        "output/epilogue bits and panel exact; guards preserved\n",
        m, n, k, trace.device_groups, trace.host_groups, trace.host_fallbacks);
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}
int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel-m1-down") {
    const std::array<Projection, 1> measured{Projection{5120, 17408}};
    return gpu_panel(1, measured);
  }
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel")
    return gpu_panel(4, kFFNProjections);
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel-extensions")
    return gpu_panel(4, kAdditionalProjections);
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel-m7")
    return gpu_panel(7, kM7Head);
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel-m6")
    return gpu_panel(6, kWidePanelProjections);
  if (argc == 3 && std::string_view(argv[1]) == "--gpu-panel-verify") {
    // Every projection of the 27B model at a draft tree's verify width.
    const std::array measured{Projection{17408, 5120}, Projection{5120, 17408},
                              Projection{10240, 5120}, Projection{6144, 5120},
                              Projection{12288, 5120}, Projection{5120, 6144},
                              Projection{1024, 5120},  Projection{248320, 5120}};
    return gpu_panel(std::strtoul(argv[2], nullptr, 10), measured);
  }
  if (argc == 2 && std::string_view(argv[1]) == "--gpu-panel-m8-rowpairs") {
    const std::array measured{Projection{17408, 5120}, Projection{10240, 5120}};
    return gpu_panel(8, measured);
  }
  return lse::test::run_all();
}
