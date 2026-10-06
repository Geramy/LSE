#include "loom_dot_fixture.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/ops.hpp"
#include <cstdio>
#include <utility>
#include <vector>
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
int main() {
  try {
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
    std::vector<std::pair<int, int>> cases;
    for (int rows : {1, 32, 512})
      for (int bits : {4, 6, 8}) cases.emplace_back(rows, bits);
    for (int rows : {2, 3, 7, 8}) cases.emplace_back(rows, 4);
    for (auto [rows, bits] : cases) {
        auto group = projection(rows, bits);
        const auto hip_key = hip.cache_key(group, device),
                   loom_key = loom.cache_key(group, device);
        auto h = hip.emit(group, device), l = loom.emit(group, device);
        if (!h.ok())
          throw std::runtime_error(h.status().to_string());
        if (!l.ok())
          throw std::runtime_error(l.status().to_string());
        // Loom runs a prefill-width 4- or 8-bit projection as the tiled f16
        // matrix-core GEMM over an f16 copy of the activation; the legacy
        // HIP dialect keeps its INT8 matrix route.
        const bool loom_gemm = rows >= static_cast<int>(dispatch::kQ4GemmMinRows) &&
                               (bits == 4 || bits == 8);
        const bool expected = bits == 4 && (rows <= 8 || rows >= 16);
        const bool hip_dot =
            h->source.find("__builtin_amdgcn_sudot4") != std::string::npos;
        const bool hip_matrix =
            h->source.find("__builtin_amdgcn_wmma_i32") != std::string::npos;
        const bool loom_dot =
            l->source.find("vector.dot4i<s8u8>") != std::string::npos;
        const bool loom_matrix =
            l->source.find("vector.mma") != std::string::npos &&
            l->source.find("element_format=i8") != std::string::npos &&
            l->source.find("element_format=u8") != std::string::npos;
        require((hip_dot || hip_matrix) == expected,
                "HIP activation-conversion selection mismatch");
        const bool loom_f16_matrix =
            l->source.find("vector.mma") != std::string::npos &&
            l->source.find("vector<8xf16>") != std::string::npos &&
            l->source.find("element_format=f16") != std::string::npos;
        require((loom_dot || loom_matrix) == (expected && !loom_gemm),
                "Loom activation-conversion selection mismatch");
        require(loom_f16_matrix == loom_gemm, "Loom wrong prefill GEMM route");
        if (expected) {
          require(rows < 16 ? hip_dot : hip_matrix, "HIP wrong INT8 route");
          if (!loom_gemm)
            require(rows < 16 ? loom_dot : loom_matrix, "Loom wrong INT8 route");
        }
        if (!loom_gemm) {
          for (unsigned axis = 0; axis < 3; ++axis) {
            require(h->dims.workgroup_size[axis] == l->dims.workgroup_size[axis] &&
                        h->dims.workgroup_count[axis] == l->dims.workgroup_count[axis],
                    "HIP/Loom launch plans differ");
          }
          require(h->lds_bytes == l->lds_bytes, "HIP/Loom LDS plans differ");
        }
        auto declared = backend::HipEmitter::shared_bytes(h->source);
        require(declared.ok() && *declared == h->lds_bytes &&
                    h->lds_bytes <= device.lds_bytes_per_workgroup,
                "declared LDS differs from launch plan");
        if (rows >= 2 && rows <= 8 && bits == 4) {
          amd.has_dot4_iu8 = false;
          require(hip.cache_key(group, device) != hip_key &&
                      loom.cache_key(group, device) != loom_key,
                  "INT8 and FP32 specialization cache keys collide");
          amd.has_dot4_iu8 = true;
        }
        auto cached = loom.emit(group, device);
        require(cached.ok() && cached->source == l->source,
                "cached emission changed");
        auto code = compiler.compile(l->source, device.arch);
        if (!code.ok())
          throw std::runtime_error(code.status().to_string());
        require(!code->code.empty(), "native shader empty");
        std::printf("KEY m%d_q%d hip=%llu loom=%llu\n", rows, bits,
                    static_cast<unsigned long long>(hip_key),
                    static_cast<unsigned long long>(loom_key));
      }
    // Raw integer arithmetic is not activation conversion and stays available.
    auto raw = dot_fixture::body(0);
    require(raw.ok() &&
                raw->text.find("vector.dot4i<s8u8>") != std::string::npos,
            "raw integer dot was disabled");
    std::puts("PASS quantization defaults, HIP/Loom selection, cached replay, and "
              "thirteen native shaders; no GPU opened");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
