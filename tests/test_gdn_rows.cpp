// The GDN prefill scan against a double-precision recurrence, on the GPU.
//
// Without arguments this only checks that nothing here needs a device.
// `--gpu T H D` runs one scan of T timesteps over H heads of width D and
// compares every output and the final state with the reference.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

using namespace lse;
using namespace lse::graph;
namespace {

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

LSE_TEST(gdn_scan_emits_for_a_described_device) {
  // The model's prefill scan on a gfx1201 description, at a few lengths.
  // GDN_TEST_COMPILE also compiles each and prints what the allocator gave
  // it -- no device involved.
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  device.arch = "gfx1201";
  backend::apply_arch_defaults(device, amd);
  device.extension_id = backend::AmdDeviceInfo::kExtensionId;
  device.extension = &amd;
  auto leaf = [](Shape shape) {
    auto n = std::make_shared<Node>();
    n->shape = shape; n->dtype = DType::kF32; n->materialized = true;
    return Array(n);
  };
  for (const std::int64_t t : {16, 17, 37, 137, 1024}) {
    auto q = leaf({1, t, 48, 128}), k = leaf({1, t, 48, 128}), v = leaf({1, t, 48, 128});
    auto a = leaf({1, t, 48}), b = leaf({1, t, 48}), s0 = leaf({1, 48, 128, 128});
    Array s1;
    auto o = gated_delta_step(q, k, v, a, b, s0, &s1);
    const NodePtr roots[]{o.node(), s1.node()};
    for (const auto& group : Partitioner::partition(roots, &device)) {
      auto emitted = backend::LoomEmitter{}.emit(group, device);
      if (!emitted.ok())
        std::fprintf(stderr, "gdn emission T%lld: %s\n", static_cast<long long>(t),
                     std::string(emitted.status().message()).c_str());
      LSE_EXPECT(emitted.ok());
      if (!emitted.ok() || !std::getenv("GDN_TEST_COMPILE")) continue;
      const backend::LoomcCompiler compiler;
      auto compiled = compiler.compile(emitted->source, device.arch);
      if (!compiled.ok())
        std::fprintf(stderr, "gdn compile T%lld: %s\n", static_cast<long long>(t),
                     std::string(compiled.status().message()).c_str());
      LSE_EXPECT(compiled.ok());
      if (!compiled.ok()) continue;
      for (const auto& r : compiled->resources)
        std::printf("gdn T%lld wg %u x %u: vgpr %u sgpr %u lds %u vspill %u sspill %u\n",
                    static_cast<long long>(t), emitted->dims.workgroup_count[0],
                    emitted->dims.workgroup_size[0], r.vector_registers.value_or(0),
                    r.scalar_registers.value_or(0), r.workgroup_segment_bytes.value_or(0),
                    r.vector_spills.value_or(0), r.scalar_spills.value_or(0));
    }
  }
}

int gpu(std::size_t t, std::size_t heads, std::size_t d) {
  auto* scheduler = default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> q(t * heads * d), k(q.size()), v(q.size());
  for (auto& x : q) x = uni(rng);
  for (auto& x : v) x = uni(rng);
  // Unit keys, as the model's l2_normalize leaves them.
  for (std::size_t i = 0; i < t * heads; ++i) {
    double n = 0;
    for (std::size_t j = 0; j < d; ++j) {
      k[i * d + j] = uni(rng);
      n += k[i * d + j] * k[i * d + j];
    }
    for (std::size_t j = 0; j < d; ++j) k[i * d + j] /= static_cast<float>(std::sqrt(n));
  }
  std::vector<float> alpha(t * heads), beta(t * heads), s0(heads * d * d);
  for (auto& x : alpha) x = 0.9f + 0.1f * std::abs(uni(rng));
  for (auto& x : beta) x = 0.5f * std::abs(uni(rng));
  for (auto& x : s0) x = 0.1f * uni(rng);
  const auto T = static_cast<std::int64_t>(t), H = static_cast<std::int64_t>(heads),
             Dd = static_cast<std::int64_t>(d);
  auto aq = filled(Shape{1, T, H, Dd}, DType::kF32, q);
  auto ak = filled(Shape{1, T, H, Dd}, DType::kF32, k);
  auto av = filled(Shape{1, T, H, Dd}, DType::kF32, v);
  auto aa = filled(Shape{1, T, H}, DType::kF32, alpha);
  auto ab = filled(Shape{1, T, H}, DType::kF32, beta);
  auto as = filled(Shape{1, H, Dd, Dd}, DType::kF32, s0);
  Array s1;
  auto o = gated_delta_step(aq, ak, av, aa, ab, as, &s1);
  const NodePtr roots[]{o.node(), s1.node()};
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(roots, false));
  LSE_EXPECT_OK(scheduler->drain());
  LSE_EXPECT_EQ(scheduler->last_trace().host_groups, 0u);
  const auto out = read<float>(o);
  const auto state = read<float>(s1);
  // GDN_TEST_REPS more launches of the same scan, for a device profile to
  // time once the clocks have ramped.
  if (const char* reps = std::getenv("GDN_TEST_REPS")) {
    for (long r = std::strtol(reps, nullptr, 10); r > 0; --r) {
      Array again_state;
      auto again = gated_delta_step(aq, ak, av, aa, ab, as, &again_state);
      const NodePtr again_roots[]{again.node(), again_state.node()};
      LSE_EXPECT_OK(scheduler->eval(again_roots, false));
    }
    LSE_EXPECT_OK(scheduler->drain());
  }

  // Reference: each row of each head's state, in double.
  double worst_o = 0, worst_s = 0;
  std::vector<double> srow(d);
  for (std::size_t h = 0; h < heads; ++h) {
    for (std::size_t r = 0; r < d; ++r) {
      for (std::size_t j = 0; j < d; ++j) srow[j] = s0[(h * d + r) * d + j];
      for (std::size_t s = 0; s < t; ++s) {
        const std::size_t sc = s * heads + h, vec = sc * d;
        double sk = 0;
        for (std::size_t j = 0; j < d; ++j) {
          srow[j] *= alpha[sc];
          sk += srow[j] * k[vec + j];
        }
        const double delta = (v[vec + r] - sk) * beta[sc];
        double acc = 0;
        for (std::size_t j = 0; j < d; ++j) {
          srow[j] += delta * k[vec + j];
          acc += srow[j] * q[vec + j];
        }
        worst_o = std::max(worst_o, std::abs(acc - out[vec + r]));
      }
      for (std::size_t j = 0; j < d; ++j)
        worst_s = std::max(worst_s, std::abs(srow[j] - state[(h * d + r) * d + j]));
    }
  }
  LSE_EXPECT(worst_o < 1e-3 && worst_s < 1e-3);
  std::printf("gdn rows T%zu H%zu D%zu worst_out=%.3g worst_state=%.3g\n", t, heads,
              d, worst_o, worst_s);
  return lse::test::Registry::get().failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 5 && std::string_view(argv[1]) == "--gpu")
    return gpu(std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10),
               std::strtoull(argv[4], nullptr, 10));
  return lse::test::run_all();
}
