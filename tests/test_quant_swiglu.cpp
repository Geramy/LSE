#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/graph/quant_swiglu.hpp"
#include <array>
#include <bit>
#include <cstring>
using namespace lse;
using namespace lse::graph;
namespace {
Array leaf(Shape shape, DType dtype) {
  auto n = std::make_shared<Node>();
  n->shape = shape;
  n->dtype = dtype;
  n->materialized = true;
  return Array(n);
}
struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  backend::LoomEmitter emitter;
  Array x, gate, up, result;
  explicit Fixture(std::int64_t m = 8) {
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    device.lds_bytes_per_workgroup = 65536;
    x = leaf({1, m, 5120}, DType::kF32);
    const auto project = [&] {
      return quant_linear(x, leaf({17408, 640}, DType::kU32),
                          leaf({17408, 80}, DType::kBF16),
                          leaf({17408, 80}, DType::kBF16), 4, 64);
    };
    gate = project();
    up = project();
    result = silu(gate) * up;
  }
  std::size_t run() {
    const std::array roots{result.node()};
    return optimize_quant_swiglu(roots, device, emitter, 0);
  }
};
} // namespace
LSE_TEST(swiglu_pair_replaces_two_projections_and_preserves_root_identity) {
  Fixture f;
  auto root = f.result.node();
  auto panel = f.gate.node()->inputs[4];
  LSE_EXPECT_EQ(f.run(), 1u);
  if (!root->prim || root->inputs.size() != 8)
    return;
  LSE_EXPECT(f.result.node() == root && root->shape == Shape{1, 8, 17408});
  LSE_EXPECT(root->prim &&
             root->prim->name() == "quant_swiglu.q4_shared_panel.v1");
  LSE_EXPECT_EQ(root->inputs.size(), 8u);
  LSE_EXPECT(root->inputs[7] == panel);
  const std::array roots{root};
  const auto order = Partitioner::unmaterialized(roots);
  LSE_EXPECT_EQ(order.size(), 2u);
  LSE_EXPECT_EQ(f.run(), 0u);
  LSE_EXPECT(root->prim->infer_shape(std::array<Shape, 1>{Shape{1}}).ok() ==
             false);
}
LSE_TEST(swiglu_pair_fuses_single_token_decode) {
  Fixture f(1);
  auto root = f.result.node();
  LSE_EXPECT_EQ(f.run(), 1u);
  LSE_EXPECT(root->prim &&
             root->prim->name() == "quant_swiglu.q4_shared_panel.v1");
  LSE_EXPECT(root->shape == Shape{1, 1, 17408});
  LSE_EXPECT_EQ(root->inputs.size(), 8u);
  LSE_EXPECT(root->inputs[7] == f.gate.node()->inputs[4]);
}
LSE_TEST(
    swiglu_pair_preserves_escaping_intermediates_and_materialized_outputs) {
  {
    Fixture f;
    const std::array roots{f.result.node(), f.gate.node()};
    LSE_EXPECT_EQ(optimize_quant_swiglu(roots, f.device, f.emitter, 0), 0u);
  }
  {
    Fixture f;
    auto consumer = silu(f.gate);
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->materialized = true;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.result.node()->materialized = true;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->member = 1;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.gate.node()->inputs[1]->member = 1;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
}
LSE_TEST(swiglu_pair_requires_qualified_geometry_storage_and_device) {
  for (auto rows : {4, 6, 1024}) {
    Fixture f(rows);
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.device.arch = "gfx1100";
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.device.wavefront_size = 64;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.amd.has_dot4_iu8 = false;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.device.max_threads_per_workgroup = 128;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->inputs[4] = std::make_shared<Node>(*f.up.node()->inputs[4]);
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->inputs[2]->dtype = DType::kF32;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->inputs[1]->shape = Shape{17408, 639};
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.up.node()->iattrs[0] = 6;
    LSE_EXPECT_EQ(f.run(), 0u);
  }
  {
    Fixture f;
    f.result.node()->shape = Shape{1, 8, 1};
    LSE_EXPECT_EQ(f.run(), 0u);
  }
}
LSE_TEST(swiglu_pair_typed_reference_rejects_invalid_buffers_before_access) {
  Fixture f;
  LSE_EXPECT_EQ(f.run(), 1u);
  if (!f.result.node()->prim)
    return;
  const auto &n = *f.result.node();
  std::vector<HostTensorView> in;
  for (const auto &i : n.inputs)
    in.push_back({{}, i->shape, i->dtype});
  LSE_EXPECT(n.prim->has_typed_host_impl());
  LSE_EXPECT(
      !n.prim->eval_cpu_typed(in, {{}, n.shape, n.dtype}, n.attrs, n.iattrs)
           .ok());
}
LSE_TEST(
    swiglu_pair_typed_reference_preserves_signed_codes_and_distinct_weights) {
  const auto *pair = find_primitive("quant_swiglu.q4_shared_panel.v1");
  LSE_EXPECT(pair != nullptr);
  if (!pair)
    return;
  const auto check_rows = [&](std::size_t rows) {
    std::vector<float> x(rows * 5120), output(rows * 17408);
    std::vector<std::uint32_t> gw(17408 * 640), uw(gw.size()),
        panel(rows * 2000);
    std::vector<std::uint16_t> gs(17408 * 80, 0x3980), us(gs.size(), 0x3a00),
        biases(gs.size());
    for (std::size_t col = 0; col < 17408; ++col) {
      std::fill_n(gw.begin() + col * 640, 640,
                  col % 2 ? 0x77777777u : 0x33333333u);
      std::fill_n(uw.begin() + col * 640, 640,
                  col % 2 ? 0x22222222u : 0x55555555u);
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const auto code = static_cast<int>(row) - 3;
      const auto byte = static_cast<std::uint8_t>(code);
      const auto packed = static_cast<std::uint32_t>(byte) * 0x01010101u;
      std::fill_n(panel.begin() + row * 2000, 1280, packed);
      std::fill_n(panel.begin() + row * 2000 + 1280, 640,
                  std::bit_cast<std::uint32_t>(1.0f));
      std::fill_n(panel.begin() + row * 2000 + 1920, 80,
                  std::bit_cast<std::uint32_t>(64.0f * code));
    }
    const auto view = [](const auto &v, Shape shape, DType type) {
      return HostTensorView{std::as_bytes(std::span(v)), shape, type};
    };
    const std::array in{
        view(x, {1, static_cast<std::int64_t>(rows), 5120}, DType::kF32),
        view(gw, {17408, 640}, DType::kU32),
        view(gs, {17408, 80}, DType::kBF16),
        view(biases, {17408, 80}, DType::kBF16),
        view(uw, {17408, 640}, DType::kU32),
        view(us, {17408, 80}, DType::kBF16),
        view(biases, {17408, 80}, DType::kBF16),
        view(panel, {static_cast<std::int64_t>(rows), 2000}, DType::kU32)};
    LSE_EXPECT_OK(
        pair->eval_cpu_typed(in,
                             {std::as_writable_bytes(std::span(output)),
                              {1, static_cast<std::int64_t>(rows), 17408},
                              DType::kF32},
                             {}, {4, 64, 0, 0}));
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < 17408; ++col) {
        const auto code = static_cast<float>(static_cast<int>(row) - 3);
        const float gate = 5120.0f * (col % 2 ? 7.0f : 3.0f) * code / 4096.0f;
        const float up = 5120.0f * (col % 2 ? 2.0f : 5.0f) * code / 2048.0f;
        const float expected = (gate / (1.0f + std::exp(-gate))) * up;
        if (std::abs(output[row * 17408 + col] - expected) >
            1e-5f * std::max(1.0f, std::abs(expected))) {
          LSE_EXPECT(false);
          return;
        }
      }
  };
  check_rows(1);
  check_rows(8);
}

LSE_TEST_MAIN()
