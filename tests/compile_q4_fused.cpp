// EXPERIMENT: emit and compile the prefill MLP's fused groups (rms_norm ->
// gate/up GEMMs with silu*up, down GEMM with residual adds) for a described
// device, offline. compile_q4_fused <dir> <arch> [M]
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
using namespace lse;
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  backend::DeviceInfo info;
  info.arch = argv[2];
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  backend::AmdDeviceInfo amd;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  const std::int64_t m = argc > 3 ? std::atoll(argv[3]) : 1024, h = 5120, f = 17408;
  auto leaf = [](Shape shape, DType type) {
    auto node = std::make_shared<graph::Node>();
    node->shape = shape;
    node->dtype = type;
    node->materialized = true;
    return graph::Array(node);
  };
  auto x = leaf(Shape{1, m, h}, DType::kF32);
  auto nw = leaf(Shape{h}, DType::kBF16);
  auto q = [&](std::int64_t n, std::int64_t k, graph::Array in) {
    return graph::quant_linear(in, leaf(Shape{n, k / 8}, DType::kU32),
                               leaf(Shape{n, k / 64}, DType::kBF16),
                               leaf(Shape{n, k / 64}, DType::kBF16), 4, 64);
  };
  auto n = graph::rms_norm(x, nw, 1e-6f);
  auto act = graph::silu(q(f, h, n)) * q(f, h, n);
  auto r = leaf(Shape{1, m, h}, DType::kF32);
  auto y = q(h, f, act) + x + r;
  const graph::NodePtr roots[] = {y.node()};
  auto groups = graph::Partitioner::partition(roots, &info);
  backend::LoomcCompiler compiler;
  int i = 0, bad = 0;
  for (const auto& g : groups) {
    backend::LoomEmitter emitter;
    auto emitted = emitter.emit(g, info);
    const std::string label = std::string(argv[1]) + "/g" + std::to_string(i++);
    if (!emitted.ok()) { std::printf("%s emit: %s\n", label.c_str(), emitted.status().to_string().c_str()); ++bad; continue; }
    if (emitted->source.find("vector.mma") == std::string::npos && !std::getenv("ALL")) continue;
    std::ofstream(label + ".loom") << emitted->source;
    const auto t0 = std::chrono::steady_clock::now();
    auto obj = compiler.compile(emitted->source, info.arch);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!obj.ok()) { std::printf("%s %zu B: FAIL %.1fs %s\n", label.c_str(), emitted->source.size(), s, obj.status().to_string().substr(0, 300).c_str()); ++bad; continue; }
    std::ofstream(label + ".hsaco", std::ios::binary).write(reinterpret_cast<const char*>(obj->code.data()), obj->code.size());
    for (const auto& res : obj->resources)
      std::printf("%s %zu B: ok %.1fs vgpr %u private %u vspill %u\n", label.c_str(), emitted->source.size(), s,
                  res.vector_registers.value_or(0), res.private_segment_bytes.value_or(0), res.vector_spills.value_or(0));
  }
  return bad ? 1 : 0;
}
