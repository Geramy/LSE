// The tiled 4-bit prefill GEMM against a double-precision oracle, on the GPU.
//
// Without arguments this checks only the graph-construction contract, which
// needs no device. `--gpu M N K [reps]` runs one contraction on the device,
// compares a sample of outputs against the oracle and reports the mean time
// of `reps` replays.
#include "harness.hpp"
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
  for (const std::int64_t m : {16, 137, 1024}) {
    // Two contractions over one activation share its panel.
    auto x = leaf({1, m, 5120}, DType::kF32);
    auto y = quant_linear(x, leaf({17408, 640}, DType::kU32),
                          leaf({17408, 80}, DType::kBF16),
                          leaf({17408, 80}, DType::kBF16), 4, 64);
    LSE_EXPECT(y.node()->prim != nullptr &&
               y.node()->prim->name() == "quant_linear.q4_gemm_f16.v1");
    LSE_EXPECT_EQ(y.node()->inputs.size(), 5u);
    const auto& panel = y.node()->inputs[4];
    LSE_EXPECT(panel->prim != nullptr &&
               panel->prim->name() == "quant_activation.f16_panel.v1");
    LSE_EXPECT(panel->dtype == DType::kF16);
    LSE_EXPECT((panel->shape == Shape{m, 5120}));
    auto z = quant_linear(x, leaf({6144, 640}, DType::kU32),
                          leaf({6144, 80}, DType::kBF16),
                          leaf({6144, 80}, DType::kBF16), 4, 64);
    LSE_EXPECT(z.node()->inputs.size() == 5u &&
               z.node()->inputs[4].get() == panel.get());
  }
}

int gpu(std::size_t m, std::size_t n, std::size_t k, int reps) {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  const std::size_t groups = k / 64;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> x(m * k);
  for (auto& v : x) v = uni(rng) * 2.0f;
  if (const char* fill = std::getenv("Q4G_TEST_X")) {
    for (auto& v : x) v = static_cast<float>(std::atof(fill));
  }
  std::vector<std::uint32_t> w(n * k / 8);
  for (auto& v : w) v = static_cast<std::uint32_t>(rng());
  if (const char* fill = std::getenv("Q4G_TEST_W")) {
    for (auto& v : w) v = static_cast<std::uint32_t>(std::strtoul(fill, nullptr, 16));
  }
  std::vector<bfloat16_t> s(n * groups), b(n * groups);
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = bfloat16_t(0.002f + 0.01f * std::abs(uni(rng)));
    b[i] = bfloat16_t(-8.0f * static_cast<float>(s[i]) + 0.01f * uni(rng));
  }
  auto ax = filled(Shape{1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(k)},
                   DType::kF32, x);
  auto aw = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(k / 8)},
                   DType::kU32, w);
  auto as = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(groups)},
                   DType::kBF16, s);
  auto ab = filled(Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(groups)},
                   DType::kBF16, b);
  auto y = quant_linear(ax, aw, as, ab, 4, 64);
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
      const std::uint32_t word = w[col * (k / 8) + kk / 8];
      const double code = static_cast<double>((word >> (4 * (kk % 8))) & 15u);
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
    auto gate = quant_linear(ax, aw, as, ab, 4, 64);
    auto up = quant_linear(ax, aw, as, ab, 4, 64);
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
