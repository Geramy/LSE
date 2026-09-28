#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/ops.hpp"

#include <cstdlib>

using namespace lse;
namespace {
graph::FusionGroup make_group(int m, int n, int k) {
  using namespace graph;
  auto leaf = [](Shape shape, DType dtype) {
    auto node = std::make_shared<Node>();
    node->shape = shape;
    node->dtype = dtype;
    return Array(node);
  };
  auto out = quant_linear(leaf({m, k}, DType::kF32),
                          leaf({n, k * 6 / 32}, DType::kU32),
                          leaf({n, k / 64}, DType::kBF16),
                          leaf({n, k / 64}, DType::kBF16), 6, 64);
  const NodePtr roots[] = {out.node()};
  for (auto group : Partitioner::partition(roots))
    if (group.anchor == OpKind::kQuantMatMul) return group;
  std::abort();
}

struct Fixture {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  backend::HipEmitter hip;
  backend::LoomEmitter loom;
  backend::LoomcCompiler compiler;

  Fixture() {
    device.arch = "gfx1201";
    device.wavefront_size = 32;
    device.compute_units = 64;
    device.max_threads_per_workgroup = 1024;
    device.lds_bytes_per_workgroup = 65536;
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
  }

  void check(int m, int n, int k, bool matrix, bool native = true) {
    const auto group = make_group(m, n, k);
    const auto hip_key = hip.cache_key(group, device);
    const auto loom_key = loom.cache_key(group, device);
    auto h = hip.emit(group, device), l = loom.emit(group, device);
    LSE_EXPECT(h.ok());
    LSE_EXPECT(l.ok());
    if (!h.ok() || !l.ok()) return;
    LSE_EXPECT((h->source.find("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12") !=
                std::string::npos) == matrix);
    LSE_EXPECT((l->source.find("vector.mma") != std::string::npos) == matrix);
    LSE_EXPECT(l->source.find("element_format=fp8") == std::string::npos);
    LSE_EXPECT(l->source.find("element_format=bf8") == std::string::npos);
    if (matrix) {
      LSE_EXPECT_EQ(l->lds_bytes, 16384u);
      LSE_EXPECT_EQ(l->dims.workgroup_size[0], 128u);
      LSE_EXPECT(l->source.find("element_format=bf16") != std::string::npos);
    }
    auto cached_hip = hip.emit(group, device), cached_loom = loom.emit(group, device);
    LSE_EXPECT_EQ(hip.cache_key(group, device), hip_key);
    LSE_EXPECT_EQ(loom.cache_key(group, device), loom_key);
    LSE_EXPECT(cached_hip.ok() && cached_hip->source == h->source);
    LSE_EXPECT(cached_loom.ok() && cached_loom->source == l->source);
    if (native) {
      LSE_EXPECT(compiler.available());
      if (!compiler.available()) return;
      auto code = compiler.compile(l->source, "gfx1201");
      LSE_EXPECT(code.ok());
      if (!code.ok())
        std::fprintf(stderr, "M%d N%d K%d: %s\n", m, n, k,
                     code.status().to_string().c_str());
      else
        LSE_EXPECT(!code->code.empty());
    }
  }
};
}

LSE_TEST(q6_m512_ffn_defaults_emit_bf16_matrix_code) {
  Fixture fixture;
  for (int m : {64, 512, 1024}) {
    fixture.check(m, 17408, 5120, m == 512);
    fixture.check(m, 5120, 17408, m == 512);
  }
  fixture.check(512, 10240, 5120, false);
  fixture.check(512, 248320, 5120, false);
}

LSE_TEST(q6_other_shapes_and_unsupported_devices_keep_scalar) {
  Fixture fixture;
  fixture.check(17, 19, 1088, false);
  fixture.check(1, 17408, 5120, false);
  fixture.device.arch = "unsupported";
  fixture.check(512, 17408, 5120, false, false);
  fixture.device.arch = "gfx1201";
  fixture.device.lds_bytes_per_workgroup = 8192;
  fixture.check(512, 17408, 5120, false, false);
  fixture.device.lds_bytes_per_workgroup = 65536;
  fixture.device.max_threads_per_workgroup = 127;
  fixture.check(512, 17408, 5120, false, false);
}
LSE_TEST_MAIN()
