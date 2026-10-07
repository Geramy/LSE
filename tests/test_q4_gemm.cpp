// The tiled 4-bit prefill GEMM against a double-precision oracle, on the GPU.
//
// Without arguments this checks only the graph-construction contract, which
// needs no device. `--gpu M N K [reps]` runs one contraction on the device,
// compares a sample of outputs against the oracle and reports the mean time
// of `reps` replays.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {

Array leaf(Shape shape, DType type) {
  auto node = std::make_shared<Node>();
  node->shape = shape;
  node->dtype = type;
  node->materialized = true;
  return Array(node);
}

template <class T>
Array filled(Shape shape, DType type, const std::vector<T>& values) {
  auto* scheduler = default_scheduler();
  auto storage = scheduler->backend().allocate(values.size() * sizeof(T),
                                               backend::MemoryClass::kDevice);
  LSE_EXPECT(storage.ok());
  if (!storage.ok()) return {};
  auto buffer = storage.release();
  LSE_EXPECT_OK(scheduler->backend().copy(buffer, values.data(),
                                          values.size() * sizeof(T)));
  return Array::from_buffer(std::move(buffer), shape, type);
}

template <class T>
std::vector<T> read(Array value) {
  std::vector<T> data(value.shape().elem_count());
  LSE_EXPECT_OK(value.to_host(data.data(), data.size() * sizeof(T)));
  return data;
}

LSE_TEST(q4_gemm_shape_routes_prefill_rows_to_the_tiled_gemm) {
  // The graph takes the tiled GEMM only on a device that can run it (or with
  // no device at all), and slices K only when a device says how many
  // compute units it has. A host-only build keeps the original contraction.
  const backend::DeviceInfo* device = nullptr;
  if (Scheduler* scheduler = default_scheduler())
    device = &scheduler->backend().device_info();
  const bool tiled = device == nullptr || dispatch::q4_gemm_device(*device);
  for (const std::int64_t m : {16, 137, 1024}) {
    auto x = leaf({1, m, 5120}, DType::kF32);
    auto y = quant_linear(x, leaf({17408, 640}, DType::kU32),
                          leaf({17408, 80}, DType::kBF16),
                          leaf({17408, 80}, DType::kBF16), 4, 64);
    LSE_EXPECT((y.shape() == Shape{1, m, 17408}));
    if (!tiled) {
      LSE_EXPECT(y.node()->prim != nullptr &&
                 !y.node()->prim->name().starts_with("quant_linear.q4_gemm_f16"));
      continue;
    }
    NodePtr gemm = y.node();
    if (gemm->prim && gemm->prim->name() == "quant_linear.q4_gemm_f16.slice_sum.v1") {
      gemm = gemm->inputs[0];
      LSE_EXPECT(gemm->prim->name() == "quant_linear.q4_gemm_f16.slices.v1");
      LSE_EXPECT(gemm->shape.dim(0) == gemm->iattrs[2] && gemm->iattrs[2] > 1);
    } else {
      LSE_EXPECT(gemm->prim != nullptr &&
                 gemm->prim->name() == "quant_linear.q4_gemm_f16.v1");
    }
    LSE_EXPECT_EQ(gemm->inputs.size(), 5u);
    const auto& panel = gemm->inputs[4];
    LSE_EXPECT(panel->prim != nullptr &&
               panel->prim->name() == "quant_activation.f16_panel.v1");
    LSE_EXPECT(panel->dtype == DType::kF16);
    LSE_EXPECT((panel->shape == Shape{m, 5120}));
    // Two contractions over one activation share its panel.
    auto z = quant_linear(x, leaf({6144, 640}, DType::kU32),
                          leaf({6144, 80}, DType::kBF16),
                          leaf({6144, 80}, DType::kBF16), 4, 64);
    NodePtr zg = z.node()->inputs.size() == 1 ? z.node()->inputs[0] : z.node();
    LSE_EXPECT(zg->inputs.size() == 5u && zg->inputs[4].get() == panel.get());
  }
}

LSE_TEST(q4_gemm_emits_for_every_tile_and_width_on_a_described_device) {
  // No device: the kernel is emitted for a gfx1201 description, for each tile
  // the policy picks and both code widths, plain and K-sliced.
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  device.arch = "gfx1201";
  backend::apply_arch_defaults(device, amd);
  device.extension_id = backend::AmdDeviceInfo::kExtensionId;
  device.extension = &amd;
  for (const unsigned bits : {4u, 8u}) {
    for (const std::int64_t m : {16, 48, 64, 137, 272, 656, 1024}) {
      const std::int64_t n = 5120, k = 6144;
      auto x = leaf({1, m, k}, DType::kF32);
      auto panel_node = std::make_shared<Node>();
      panel_node->shape = Shape{m, k};
      panel_node->dtype = DType::kF16;
      panel_node->materialized = true;
      for (const std::int32_t slices : {1, 2}) {
        auto gemm = std::make_shared<Node>();
        gemm->set_kind(OpKind::kCustom);
        gemm->dtype = DType::kF32;
        gemm->inputs = {x.node(), leaf({n, k * bits / 32}, DType::kU32).node(),
                        leaf({n, k / 64}, DType::kBF16).node(),
                        leaf({n, k / 64}, DType::kBF16).node(), panel_node};
        gemm->iattrs[0] = static_cast<std::int32_t>(bits);
        gemm->iattrs[1] = 64;
        gemm->shape = Shape{1, m, n};
        gemm->prim = find_primitive("quant_linear.q4_gemm_f16.v1");
        if (slices > 1) {
          gemm->prim = find_primitive("quant_linear.q4_gemm_f16.slices.v1");
          gemm->shape = Shape{slices, 1, m, n};
          gemm->iattrs[2] = slices;
        }
        LSE_EXPECT(gemm->prim != nullptr);
        if (!gemm->prim) continue;
        gemm->fclass = gemm->prim->fusion_class();
        const NodePtr roots[]{gemm};
        const auto groups = Partitioner::partition(roots, &device);
        LSE_EXPECT_EQ(groups.size(), 1u);
        if (groups.size() != 1) continue;
        auto emitted = backend::LoomEmitter{}.emit(groups[0], device);
        if (!emitted.ok())
          std::fprintf(stderr, "q4 gemm emission M%lld bits %u slices %d: %s\n",
                       static_cast<long long>(m), bits, slices,
                       std::string(emitted.status().message()).c_str());
        LSE_EXPECT(emitted.ok());
        if (emitted.ok())
          LSE_EXPECT(emitted->source.find("vector.mma") != std::string::npos);
        // Q4G_TEST_COMPILE: also compile each kernel for the description and
        // print what the allocator gave it -- no device involved.
        if (emitted.ok() && std::getenv("Q4G_TEST_COMPILE")) {
          const backend::LoomcCompiler compiler;
          auto compiled = compiler.compile(emitted->source, device.arch);
          if (!compiled.ok()) {
            std::fprintf(stderr, "q4 gemm compile M%lld bits %u slices %d: %s\n",
                         static_cast<long long>(m), bits, slices,
                         std::string(compiled.status().message()).c_str());
          } else {
            for (const auto& r : compiled->resources)
              std::printf("q4 gemm M%lld bits %u slices %d: vgpr %u sgpr %u lds %u "
                          "vspill %u sspill %u\n",
                          static_cast<long long>(m), bits, slices,
                          r.vector_registers.value_or(0), r.scalar_registers.value_or(0),
                          r.workgroup_segment_bytes.value_or(0),
                          r.vector_spills.value_or(0), r.scalar_spills.value_or(0));
          }
          LSE_EXPECT(compiled.ok());
        }
      }
    }
  }
}

namespace {
// Operand registers per f16 fragment on each generation, from its own row.
std::uint32_t fragment_registers(math::MatrixTarget target) {
  for (const auto& row : math::matrix_core_table())
    if (row.target == target && row.acc == math::MatrixElem::kF32 &&
        row.operand == math::MatrixElem::kF16 && row.m == 16 && row.n == 16 &&
        row.k_step == 16 && row.chained == 1 && row.emittable())
      return dispatch::q4_gemm_fragment_registers(&row);
  return 0;
}
}  // namespace

LSE_TEST(q4_gemm_slices_fill_the_device_without_starving_a_slice) {
  // 64 compute units: a narrow M=32 projection over K=17408 is sliced, a
  // full M=1024 one is not, and every slice keeps at least the minimum steps.
  for (const auto target : {math::MatrixTarget::kRdna4, math::MatrixTarget::kRdna3}) {
    const auto regs = fragment_registers(target);
    LSE_EXPECT_EQ(dispatch::q4_gemm_slices(1024, 17408, 5120, 64, regs), 1u);
    LSE_EXPECT(dispatch::q4_gemm_slices(32, 5120, 17408, 64, regs) > 1u);
    for (const auto k : {5120ull, 6144ull, 17408ull})
      for (const auto m : {16ull, 32ull, 128ull, 1024ull}) {
        const auto s = dispatch::q4_gemm_slices(m, 5120, k, 64, regs);
        LSE_EXPECT((k / dispatch::kQ4GemmStepK) % s == 0);
        LSE_EXPECT(s == 1 || k / dispatch::kQ4GemmStepK / s >=
                                 dispatch::kQ4GemmMinSliceSteps);
      }
    LSE_EXPECT_EQ(dispatch::q4_gemm_slices(32, 5120, 17408, 0, regs), 1u);
  }
}

// The wave tile is sized to what the generation's fragments leave in a
// lane's registers: RDNA4 takes the 64x64 wave tiles on wide passes, RDNA3/3.5,
// whose f16 fragment is twice as wide, keeps the 48x32 ones. Below 768 rows
// both generations take the same tiles.
LSE_TEST(q4_gemm_tile_follows_the_generations_fragment_width) {
  const auto rdna4 = fragment_registers(math::MatrixTarget::kRdna4);
  const auto rdna3 = fragment_registers(math::MatrixTarget::kRdna3);
  LSE_EXPECT_EQ(rdna4, 4u);
  LSE_EXPECT_EQ(rdna3, 8u);
  for (const std::uint64_t m : {768ull, 1024ull, 4096ull}) {
    const auto t4 = dispatch::q4_gemm_tile(m, rdna4);
    const auto t3 = dispatch::q4_gemm_tile(m, rdna3);
    LSE_EXPECT((t4.bm == 256 && t4.wm == 4 && t4.wn == 2));
    LSE_EXPECT(dispatch::q4_gemm_wave48(t3));
    LSE_EXPECT(dispatch::q4_gemm_matrix_registers(t3, rdna3) <=
               dispatch::kQ4GemmMatrixRegisters);
  }
  for (const std::uint64_t m : {9ull, 16ull, 33ull, 64ull, 137ull, 512ull, 767ull}) {
    const auto t4 = dispatch::q4_gemm_tile(m, rdna4);
    const auto t3 = dispatch::q4_gemm_tile(m, rdna3);
    LSE_EXPECT((t4.bm == t3.bm && t4.bn == t3.bn && t4.wm == t3.wm && t4.wn == t3.wn));
  }
}

int gpu(std::size_t m, std::size_t n, std::size_t k, int reps) {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const std::size_t groups = k / 64;
  const int bits = std::getenv("Q4G_TEST_BITS") ? std::atoi(std::getenv("Q4G_TEST_BITS")) : 4;
  const std::size_t per_word = 32 / static_cast<std::size_t>(bits);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> x(m * k);
  for (auto& v : x) v = uni(rng) * 2.0f;
  if (const char* fill = std::getenv("Q4G_TEST_X")) {
    for (auto& v : x) v = static_cast<float>(std::atof(fill));
  }
  std::vector<std::uint32_t> w(n * k / per_word);
  for (auto& v : w) v = static_cast<std::uint32_t>(rng());
  if (const char* fill = std::getenv("Q4G_TEST_W")) {
    for (auto& v : w) v = static_cast<std::uint32_t>(std::strtoul(fill, nullptr, 16));
  }
  std::vector<bfloat16_t> s(n * groups), b(n * groups);
  for (std::size_t i = 0; i < s.size(); ++i) {
    const float span = bits == 8 ? 0.0001f : 0.01f;
    s[i] = bfloat16_t(span * 0.2f + span * std::abs(uni(rng)));
    b[i] = bfloat16_t(-static_cast<float>(1 << (bits - 1)) * static_cast<float>(s[i]) +
                      0.01f * uni(rng));
  }
  auto ax = filled(Shape{1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)},
                   DType::kF32, x);
  auto aw = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(k / per_word)},
                   DType::kU32, w);
  auto as = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(groups)},
                   DType::kBF16, s);
  auto ab = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(groups)},
                   DType::kBF16, b);
  auto y = quant_linear(ax, aw, as, ab, bits, 64);
  const NodePtr roots[]{y.node()};
  Program program;
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  LSE_EXPECT_OK(scheduler->drain());
  const auto trace = scheduler->last_trace();
  LSE_EXPECT_EQ(trace.host_groups, 0u);
  LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  const auto out = read<float>(y);

  // Oracle over a sample of outputs, in double, from the stored codes.
  double worst = 0, worst_rel = 0;
  std::size_t bad = 0;
  std::uniform_int_distribution<std::size_t> pick_m(0, m - 1), pick_n(0, n - 1);
  const std::size_t samples = 4096;
  for (std::size_t t = 0; t < samples; ++t) {
    const std::size_t row = t < 64 ? (t % m) : pick_m(rng);
    const std::size_t col = t < 64 ? (t * 997) % n : pick_n(rng);
    double acc = 0, mag = 0;
    for (std::size_t kk = 0; kk < k; ++kk) {
      const std::uint32_t word = w[col * (k / per_word) + kk / per_word];
      const double code = static_cast<double>(
          (word >> (bits * (kk % per_word))) & ((1u << bits) - 1u));
      const double weight =
          static_cast<double>(static_cast<float>(s[col * groups + kk / 64])) * code +
          static_cast<double>(static_cast<float>(b[col * groups + kk / 64]));
      const double term = static_cast<double>(x[row * k + kk]) * weight;
      acc += term;
      mag += term * term;
    }
    const double got = out[row * n + col];
    const double err = std::abs(got - acc);
    const double scale = std::sqrt(mag) + 1e-30;
    worst = std::max(worst, err);
    worst_rel = std::max(worst_rel, err / scale);
    if (!std::isfinite(got) || err / scale > 4e-3) ++bad;
  }
  LSE_EXPECT_EQ(bad, 0u);
  std::printf("q4_gemm M%zu N%zu K%zu primitive=%s worst_abs=%.6g "
              "worst_rel_to_rms=%.6g bad=%zu/%zu\n",
              m, n, k, std::string(y.node()->prim->name()).c_str(), worst,
              worst_rel, bad, samples);

  if (std::getenv("Q4G_TEST_SWIGLU")) {
    // The model's MLP: silu(gate) * up fused into the gate GEMM's stores.
    // Fresh nodes, so nothing is materialized and the partitioner fuses
    // silu and the product into the gate GEMM's stores.
    auto gate = quant_linear(ax, aw, as, ab, bits, 64);
    auto up = quant_linear(ax, aw, as, ab, bits, 64);
    auto act = silu(gate) * up;
    const NodePtr act_roots[]{act.node()};
    scheduler->reset_accumulated_trace();
    LSE_EXPECT_OK(scheduler->eval(act_roots, false));
    LSE_EXPECT_OK(scheduler->drain());
    const auto fused = read<float>(act);
    double worst_act = 0;
    for (std::size_t i = 0; i < fused.size(); i += 997) {
      const double g = out[i];
      const double want = g / (1.0 + std::exp(-g)) * g;
      worst_act = std::max(worst_act, std::abs(fused[i] - want) / (1e-3 + std::abs(want)));
    }
    LSE_EXPECT(worst_act < 1e-2);
    std::printf("q4_gemm swiglu epilogue worst_rel=%.6g\n", worst_act);
  }
  for (int warm = 0; warm < 3; ++warm) {
    program.reset_compute();
    LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  }
  LSE_EXPECT_OK(scheduler->drain());
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    program.reset_compute();
    LSE_EXPECT_OK(scheduler->eval(roots, false, &program));
  }
  LSE_EXPECT_OK(scheduler->drain());
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  const double tflops = 2.0 * m * n * k / (ms * 1e-3) / 1e12;
  std::printf("q4_gemm M%zu N%zu K%zu: %.4f ms/launch (wall, %d reps), %.2f TFLOPS\n",
              m, n, k, ms, reps, tflops);
  return lse::test::Registry::get().failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 5 && std::string_view(argv[1]) == "--gpu") {
    int rc = 0;
    const int reps = argc >= 6 ? std::atoi(argv[5]) : 20;
    // Comma-separated M lists sweep several row counts in one process.
    std::string ms = argv[2];
    std::size_t at = 0;
    while (at <= ms.size()) {
      const auto comma = ms.find(',', at);
      const auto token = ms.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
      rc |= gpu(std::strtoull(token.c_str(), nullptr, 10),
                std::strtoull(argv[3], nullptr, 10),
                std::strtoull(argv[4], nullptr, 10), reps);
      if (comma == std::string::npos) break;
      at = comma + 1;
    }
    return rc;
  }
  return lse::test::run_all();
}
