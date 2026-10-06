// Paged causal prefill attention against a double-precision reference, on the
// GPU.
//
// Without arguments this only checks that nothing here needs a device.
// `--gpu T OFFSET H KVH D` attends T queries that follow OFFSET cached keys,
// H query heads over KVH key/value heads of width D, with bf16 pages in a
// shuffled block table. Every slot past the live keys holds NaN, so a kernel
// that reads beyond the row shows it.
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

std::uint16_t to_bf16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if (std::isnan(f)) return 0x7fc0;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

LSE_TEST(flash_prefill_emits_for_a_described_device) {
  // The model's prefill shape on a gfx1201 description: 24 query heads over
  // 4 key/value heads of 256, bf16 pages. FLASH_TEST_COMPILE also compiles it
  // and prints what the allocator gave it -- no device involved.
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  device.arch = "gfx1201";
  backend::apply_arch_defaults(device, amd);
  device.extension_id = backend::AmdDeviceInfo::kExtensionId;
  device.extension = &amd;
  auto leaf = [](Shape shape, DType dtype) {
    auto n = std::make_shared<Node>();
    n->shape = shape; n->dtype = dtype; n->materialized = true;
    return Array(n);
  };
  auto q = leaf({1, 24, 1024, 256}, DType::kF32);
  auto k = leaf({256, 4, 16, 256}, DType::kBF16);
  auto v = leaf(k.shape(), DType::kBF16);
  auto output = sdpa_paged(q, k, v, 0.0625f, MaskKind::kCausal, 0,
                           leaf({kv::step_meta_elems(1)}, DType::kF32),
                           leaf({1, 256}, DType::kF32), 16, &device, kv::CacheDType::kBF16);
  const NodePtr roots[]{output.node()};
  for (const auto& group : Partitioner::partition(roots, &device)) {
    auto emitted = backend::LoomEmitter{}.emit(group, device);
    if (!emitted.ok())
      std::fprintf(stderr, "flash emission: %s\n", std::string(emitted.status().message()).c_str());
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok() || !std::getenv("FLASH_TEST_COMPILE")) continue;
    const backend::LoomcCompiler compiler;
    auto compiled = compiler.compile(emitted->source, device.arch);
    if (!compiled.ok())
      std::fprintf(stderr, "flash compile: %s\n", std::string(compiled.status().message()).c_str());
    LSE_EXPECT(compiled.ok());
    if (!compiled.ok()) continue;
    for (const auto& r : compiled->resources)
      std::printf("flash %s: vgpr %u sgpr %u lds %u vspill %u sspill %u\n", r.entry.c_str(),
                  r.vector_registers.value_or(0), r.scalar_registers.value_or(0),
                  r.workgroup_segment_bytes.value_or(0), r.vector_spills.value_or(0),
                  r.scalar_spills.value_or(0));
  }
}

int gpu(std::size_t t, std::size_t offset, std::size_t heads, std::size_t kvheads,
        std::size_t d) {
  auto* scheduler = default_scheduler();
  if (!scheduler || kvheads == 0 || heads % kvheads != 0) return 1;
  scheduler->set_mode(Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(Dialect::kLoom);
  constexpr std::size_t kBlock = 16;
  const std::size_t live = offset + t;
  // One spare block past the live keys, all NaN.
  const std::size_t blocks = live / kBlock + 2;
  std::mt19937 rng(91);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

  std::vector<float> q(heads * t * d);
  for (auto& x : q) x = uni(rng);
  const std::uint16_t nan = to_bf16(std::numeric_limits<float>::quiet_NaN());
  std::vector<std::uint16_t> kc(blocks * kvheads * kBlock * d, nan), vc(kc.size(), nan);
  // Logical block b lives in physical block order[b].
  std::vector<std::size_t> order(blocks);
  for (std::size_t b = 0; b < blocks; ++b) order[b] = b;
  std::shuffle(order.begin(), order.end(), rng);
  auto slot = [&](std::size_t key, std::size_t kh, std::size_t c) {
    return ((order[key / kBlock] * kvheads + kh) * kBlock + key % kBlock) * d + c;
  };
  for (std::size_t key = 0; key < live; ++key)
    for (std::size_t kh = 0; kh < kvheads; ++kh)
      for (std::size_t c = 0; c < d; ++c) {
        kc[slot(key, kh, c)] = to_bf16(uni(rng));
        vc[slot(key, kh, c)] = to_bf16(uni(rng));
      }
  std::vector<float> table(blocks);
  for (std::size_t b = 0; b < blocks; ++b) table[b] = static_cast<float>(order[b]);
  const std::vector<float> meta{static_cast<float>(offset), static_cast<float>(live), 1.0f,
                                static_cast<float>(offset), static_cast<float>(live)};

  const auto T = static_cast<std::int64_t>(t), H = static_cast<std::int64_t>(heads),
             KH = static_cast<std::int64_t>(kvheads), D = static_cast<std::int64_t>(d),
             B = static_cast<std::int64_t>(blocks);
  auto aq = filled(Shape{1, H, T, D}, DType::kF32, q);
  auto ak = filled(Shape{B, KH, static_cast<std::int64_t>(kBlock), D}, DType::kBF16, kc);
  auto av = filled(Shape{B, KH, static_cast<std::int64_t>(kBlock), D}, DType::kBF16, vc);
  auto am = filled(Shape{kv::step_meta_elems(1)}, DType::kF32, meta);
  auto at = filled(Shape{1, B}, DType::kF32, table);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const backend::DeviceInfo& device = scheduler->backend().device_info();
  auto run = [&] {
    return sdpa_paged(aq, ak, av, scale, MaskKind::kCausal, 0, am, at,
                      static_cast<int>(kBlock), &device, kv::CacheDType::kBF16);
  };
  auto o = run();
  const NodePtr roots[]{o.node()};
  scheduler->reset_accumulated_trace();
  LSE_EXPECT_OK(scheduler->eval(roots, false));
  LSE_EXPECT_OK(scheduler->drain());
  LSE_EXPECT_EQ(scheduler->last_trace().host_groups, 0u);
  std::vector<float> out(heads * t * d);
  LSE_EXPECT_OK(o.to_host(out.data(), out.size() * sizeof(float)));

  // Reference: the kernel's operands are bf16, so the queries are rounded the
  // same way; probabilities are exact here, which the tolerance covers.
  double worst = 0;
  std::size_t nonfinite = 0;
  std::vector<double> score(live), acc(d);
  // FLASH_TEST_HEADS checks only that many heads (a timing run need not wait
  // for the whole reference).
  std::size_t checked = heads;
  if (const char* limit = std::getenv("FLASH_TEST_HEADS"))
    checked = std::min(heads, static_cast<std::size_t>(std::strtoull(limit, nullptr, 10)));
  for (std::size_t h = 0; h < checked; ++h) {
    const std::size_t kh = h / (heads / kvheads);
    for (std::size_t i = 0; i < t; ++i) {
      const std::size_t visible = offset + i + 1;
      double top = -1e300;
      for (std::size_t key = 0; key < visible; ++key) {
        double dot = 0;
        for (std::size_t c = 0; c < d; ++c)
          dot += static_cast<double>(from_bf16(to_bf16(q[(h * t + i) * d + c]))) *
                 from_bf16(kc[slot(key, kh, c)]);
        score[key] = dot * scale;
        top = std::max(top, score[key]);
      }
      double denom = 0;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (std::size_t key = 0; key < visible; ++key) {
        const double p = std::exp(score[key] - top);
        denom += p;
        for (std::size_t c = 0; c < d; ++c) acc[c] += p * from_bf16(vc[slot(key, kh, c)]);
      }
      for (std::size_t c = 0; c < d; ++c) {
        const float got = out[(h * t + i) * d + c];
        if (!std::isfinite(got)) ++nonfinite;
        worst = std::max(worst, std::abs(static_cast<double>(got) - acc[c] / denom));
      }
    }
  }
  LSE_EXPECT_EQ(nonfinite, 0u);
  LSE_EXPECT(worst < 2e-2);
  std::printf("flash prefill T%zu offset%zu H%zu KVH%zu D%zu worst_abs=%.3g nonfinite=%zu\n",
              t, offset, heads, kvheads, d, worst, nonfinite);

  // FLASH_TEST_REPS more launches, for a device profile to time once the
  // clocks have ramped.
  if (const char* reps = std::getenv("FLASH_TEST_REPS")) {
    for (long r = std::strtol(reps, nullptr, 10); r > 0; --r) {
      auto again = run();
      const NodePtr again_roots[]{again.node()};
      LSE_EXPECT_OK(scheduler->eval(again_roots, false));
    }
    LSE_EXPECT_OK(scheduler->drain());
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 7 && std::string_view(argv[1]) == "--gpu")
    return gpu(std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10),
               std::strtoull(argv[4], nullptr, 10), std::strtoull(argv[5], nullptr, 10),
               std::strtoull(argv[6], nullptr, 10));
  return lse::test::run_all();
}
