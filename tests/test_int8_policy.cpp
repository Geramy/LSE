#include "loom_dot_fixture.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kernels/int8_policy.hpp"
#include <cstdio>
#include <stdexcept>
using namespace lse;
using namespace lse::graph;
namespace {
void require(bool yes, const char *message) {
  if (!yes)
    throw std::runtime_error(message);
}
Array leaf(Shape shape, DType type) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  return Array(node);
}
FusionGroup projection(int rows, int bits) {
  auto out = quant_linear(
      leaf({rows, 64}, DType::kF32), leaf({17, 64 * bits / 32}, DType::kU32),
      leaf({17, 1}, DType::kBF16), leaf({17, 1}, DType::kBF16), bits, 64);
  const NodePtr roots[]{out.node()};
  for (const auto &group : Partitioner::partition(roots))
    if (group.anchor == OpKind::kQuantMatMul)
      return group;
  throw std::runtime_error("quantized projection group missing");
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2 && (std::string_view(argv[1]) == "0" ||
                          std::string_view(argv[1]) == "1"),
            "expected policy argument 0 or 1");
    const bool enabled = std::string_view(argv[1]) == "1";
    require(!kernels::parse_activation_int8(nullptr), "absent policy enabled");
    for (const char *raw : {"", "0", "true", "01", " 1", "1 ", "-1", "2"})
      require(!kernels::parse_activation_int8(raw), "malformed policy enabled");
    require(kernels::parse_activation_int8("1"), "explicit enable rejected");
    require(kernels::activation_int8_enabled() == enabled,
            "process policy mismatch");
    // A later environment edit cannot reinterpret retained dispatch plans.
    setenv("LSE_HRX_INT8", enabled ? "0" : "1", 1);
    require(kernels::activation_int8_enabled() == enabled,
            "process policy changed after selection");
    backend::DeviceInfo device;
    backend::AmdDeviceInfo amd;
    device.arch = "gfx1201";
    device.wavefront_size = 32;
    device.compute_units = 64;
    device.lds_bytes_per_workgroup = 65536;
    device.max_threads_per_workgroup = 1024;
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    backend::HipEmitter hip;
    backend::LoomEmitter loom;
    backend::LoomcCompiler compiler;
    require(compiler.available(), "native Loom compiler unavailable");
    for (int rows : {1, 32})
      for (int bits : {4, 6, 8}) {
        auto group = projection(rows, bits);
        const auto hip_key = hip.cache_key(group, device),
                   loom_key = loom.cache_key(group, device);
        auto h = hip.emit(group, device), l = loom.emit(group, device);
        if (!h.ok())
          throw std::runtime_error(h.status().to_string());
        if (!l.ok())
          throw std::runtime_error(l.status().to_string());
        const bool expected = enabled && bits == 4;
        const bool hip_dot =
            h->source.find("__builtin_amdgcn_sudot4") != std::string::npos;
        const bool hip_matrix =
            h->source.find("__builtin_amdgcn_wmma_i32") != std::string::npos;
        const bool loom_dot =
            l->source.find("vector.dot4i<s8u8>") != std::string::npos;
        const bool loom_matrix =
            l->source.find("vector.mma") != std::string::npos;
        require((hip_dot || hip_matrix) == expected,
                "HIP activation-conversion selection mismatch");
        require((loom_dot || loom_matrix) == expected,
                "Loom activation-conversion selection mismatch");
        if (expected) {
          require(rows == 1 ? hip_dot : hip_matrix, "HIP wrong INT8 route");
          require(rows == 1 ? loom_dot : loom_matrix, "Loom wrong INT8 route");
        }
        auto cached = loom.emit(group, device);
        require(cached.ok() && cached->source == l->source,
                "cached emission changed");
        auto code = compiler.compile(l->source, device.arch);
        if (!code.ok())
          throw std::runtime_error(code.status().to_string());
        require(!code->code.empty(), "native shader empty");
        std::printf("KEY m%d_q%d hip=%llu loom=%llu policy=%d\n", rows, bits,
                    static_cast<unsigned long long>(hip_key),
                    static_cast<unsigned long long>(loom_key), enabled);
      }
    // Raw integer arithmetic is not activation conversion and stays available.
    auto raw = dot_fixture::body(0);
    require(raw.ok() &&
                raw->text.find("vector.dot4i<s8u8>") != std::string::npos,
            "raw integer dot was disabled");
    std::puts("PASS INT8 policy, shared HIP/Loom selection, cached replay, and "
              "six native shaders; no GPU opened");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
