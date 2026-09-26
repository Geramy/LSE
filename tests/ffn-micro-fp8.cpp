// cand-fp8-mma GPU microbenchmark fixture (task STEP 3). Isolated PP
// experiment; no production file is modified. No device is opened
// without --run.
//
// Measures, at M=512, the up shape (17408x5120), arms:
//   prod — the production staged-bf16 kernel (quant_linear.q6_wmma_bf16_reuse),
//          the 13.7 TFLOPS record,
//   fp8  — the fp8 WMMA kernel (quant_linear.q6_fp8_mma_prefill), enabled
//          with LSE_FFN_FP8_MMA=1 (set here).
//
// The fixture overrides the node's prim with the arm's primitive directly
// (A/B is primitive-vs-primitive at identical shapes), identical protocol to
// the cand-ffn-f32 fixture: 1 warmup + 8 measured retained-graph host
// eval+retire iterations; deterministic inputs, f64 oracle, per-iteration
// output hashes (bit-identical across iterations), JIT-free replay asserted
// on iterations 2..8.
//
// Metrics: ms (host eval+retire interval), rel-L2 vs the f64 oracle, TFLOPS
// = 2*M*N*K / ms, GB/s.
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/ir/args.hpp"
#include "lse/kernels/ffn_fp8_q6.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/place/devices.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace {
using namespace lse;
using namespace graph;
using lse::kernels::wmma_q6_linear_for;
void need(bool x, const std::string &s) {
  if (!x)
    throw std::runtime_error(std::string("need: ") + s);
}
void check(const Status &s, const char *why) {
  if (!s.ok())
    throw std::runtime_error(std::string(why) + ": " + s.to_string());
}
void retire(backend::IBackend &b) noexcept {
  try {
    if (b.synchronize().ok())
      return;
    std::fprintf(stderr, "Retirement unconfirmed; exiting before releasing owners\n");
  } catch (...) {
    std::fputs("Retirement threw; exiting before releasing owners\n", stderr);
  }
  std::_Exit(1);
}
constexpr int M = 512;
constexpr int GROUP = 64;
graph::DialectSourceTable g_sources;
struct Shape2 {
  int n, k;
  unsigned words() const { return unsigned(k) * 6 / 32; }
  unsigned groups() const { return unsigned(k) / GROUP; }
  unsigned tile_wg() const { return ((M + 63) / 64) * ((n + 63) / 64); }
};
std::uint16_t bf(float x) {
  auto b = std::bit_cast<std::uint32_t>(x);
  return std::uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
float widen(std::uint16_t x) {
  return std::bit_cast<float>(std::uint32_t(x) << 16);
}
struct Fixture {
  std::vector<float> x;
  std::vector<std::uint32_t> packed;
  std::vector<std::uint16_t> scale, bias;
  std::vector<double> oracle;
};
std::uint64_t fold(std::uint64_t h, const void *p, size_t n) {
  const auto *b = static_cast<const unsigned char *>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
unsigned code_of(const std::vector<std::uint32_t> &packed, int col, int at,
                 unsigned words) {
  unsigned v = 0;
  for (unsigned b = 0; b < 6; ++b) {
    unsigned bit = unsigned(at) * 6 + b;
    v |= ((packed[col * words + bit / 32] >> (bit % 32)) & 1u) << b;
  }
  return v;
}
Fixture make_fixture(const Shape2 &s, unsigned seed_shift) {
  Fixture c;
  const int N = s.n, K = s.k, GROUPS = s.groups(), WORDS = s.words();
  c.x.resize(size_t(M) * K);
  for (int i = 0; i < M * K; ++i) {
    float x = std::sin(i * .071f + seed_shift) * .83f +
              std::cos(i * .031f) * .23f;
    if (i % 137 == 0)
      x *= 32;
    c.x[i] = x;
  }
  c.packed.resize(size_t(N) * WORDS, 0);
  c.scale.resize(size_t(N) * GROUPS);
  c.bias.resize(size_t(N) * GROUPS);
  for (int col = 0; col < N; ++col) {
    for (int g = 0; g < GROUPS; ++g) {
      float sc = std::ldexp(1.00390625f + float((col + g) % 5) / 8,
                            -9 + (col + g) % 4);
      float b = -sc * (31.5f + float(g % 3) / 8);
      c.scale[col * GROUPS + g] = bf(sc);
      c.bias[col * GROUPS + g] = bf(b);
    }
    for (int j = 0; j < K; ++j) {
      unsigned q = unsigned((col * 17 + j * 13 + 7) % 64);
      for (unsigned b = 0; b < 6; ++b) {
        unsigned bit = unsigned(j) * 6 + b;
        c.packed[col * WORDS + bit / 32] |= ((q >> b) & 1u) << (bit % 32);
      }
      need(code_of(c.packed, col, j, WORDS) == q, "packed bit oracle");
    }
  }
  c.oracle.resize(size_t(M) * N);
  for (int i = 0; i < M; ++i)
    for (int col = 0; col < N; ++col) {
      double ref = 0;
      for (int g = 0; g < GROUPS; ++g) {
        double sc = widen(c.scale[col * GROUPS + g]);
        double b = widen(c.bias[col * GROUPS + g]);
        double chunk = 0;
        for (int j = g * GROUP; j < (g + 1) * GROUP; ++j) {
          double w = std::fma(double(code_of(c.packed, col, j, WORDS)), sc, b);
          chunk += double(c.x[i * K + j]) * w;
        }
        ref += chunk;
      }
      c.oracle[i * N + col] = ref;
    }
  return c;
}
constexpr size_t guard = 256;
struct Buffer {
  backend::DeviceBuffer base;
  Array array;
  std::vector<std::byte> original;
};
Buffer allocate(backend::IBackend &b, Shape shape, DType type,
                const void *data, size_t bytes) {
  Buffer g;
  g.original.assign(bytes + guard * 2, std::byte{0xa5});
  if (data)
    std::memcpy(g.original.data() + guard, data, bytes);
  else
    std::memset(g.original.data() + guard, 0xff, bytes);
  auto mem = b.allocate(g.original.size(), backend::MemoryClass::kDevice);
  check(mem.status(), "allocation");
  g.base = mem.release();
  check(b.copy_h2d(g.original.data(), g.base, g.original.size(), 0),
        "upload");
  auto view = g.base;
  view.offset += guard;
  view.size_bytes = bytes;
  g.array = Array::from_buffer(view, shape, type);
  return g;
}
std::vector<std::byte> read(backend::IBackend &b, const Buffer &g) {
  std::vector<std::byte> r(g.original.size());
  check(b.copy_d2h(g.base, r.data(), r.size(), 0), "readback");
  return r;
}
// arm: 1 prod (q6_wmma_bf16_reuse), 2 fp8 (q6_fp8_mma_prefill).
const char *arm_prim_name(int arm) {
  if (arm == 1)
    return "quant_linear.q6_wmma_bf16_reuse";
  return "quant_linear.q6_fp8_mma_prefill";
}
struct ArmResult {
  std::string name;
  std::uint64_t input_hash = 0, output_hash = 0;
  double mean_ms = 0;
  std::vector<double> per_iteration_ms;
  double relative_l2 = 0;
  double worst_abs_err = 0;
  std::vector<float> last_output;
};
ArmResult run_arm(backend::IBackend &b, Scheduler &s, const Fixture &f,
                  const Shape2 &sh, const char *name, int arm) {
  const int N = sh.n, K = sh.k;
  const unsigned WORDS = sh.words(), GROUPS = sh.groups();
  auto x = allocate(b, {M, K}, DType::kF32, f.x.data(), size_t(M) * K * 4);
  auto packed = allocate(b, {N, WORDS}, DType::kU32, f.packed.data(),
                         size_t(N) * WORDS * 4);
  auto scales = allocate(b, {N, GROUPS}, DType::kBF16, f.scale.data(),
                         size_t(N) * GROUPS * 2);
  auto biases = allocate(b, {N, GROUPS}, DType::kBF16, f.bias.data(),
                         size_t(N) * GROUPS * 2);
  auto output = allocate(b, Shape{M, N}, DType::kF32, nullptr,
                         size_t(M) * N * 4);
  auto y = quant_linear(x.array, packed.array, scales.array, biases.array, 6,
                        GROUP);
  const Primitive *mp;
  if (arm == 1) {
    KernelShapes probe;
    std::vector<Shape> shp;
    std::vector<DType> dts;
    for (const auto &in : y.node()->inputs) {
      shp.push_back(in->shape);
      dts.push_back(in->dtype);
    }
    probe.inputs = shp;
    probe.input_dtypes = dts;
    probe.output = y.node()->shape;
    probe.output_dtype = y.node()->dtype;
    probe.attrs = y.node()->attrs;
    probe.iattrs = y.node()->iattrs;
    probe.device = &b.device_info();
    probe.types = backend::loom_types();
    g_sources = backend::loom_sources();
    probe.intrinsics = &g_sources;
    const auto *prod = wmma_q6_linear_for(probe);
    if (!prod)
      throw std::runtime_error("prod arm declined (shape not in production records)");
    mp = prod;
  } else {
    mp = find_primitive(arm_prim_name(arm));
    need(mp != nullptr, std::string("arm primitive must be registered: ") +
                            arm_prim_name(arm));
  }
  y.node()->prim = mp;
  y.node()->fclass = mp->fusion_class();
  y.node()->buffer = output.array.node()->buffer;
  const NodePtr roots[] = {y.node()};
  // Grid-assert through the real emission for the fp8 arm: 128-of-2176,
  // HAS vector.mma, lds 32768.
  if (arm == 2) {
    backend::LoomEmitter emitter;
    bool found = false;
    for (const auto &g : Partitioner::partition(roots)) {
      if (g.nodes.size() != 1 || g.nodes[0] != y.node())
        continue;
      auto emitted = emitter.emit(g, b.device_info());
      check(emitted.status(), "arm emit");
      found = true;
      const std::string cfgkey = "kernel.launch.config workgroups("
                                 "%cfg_n0, %cfg_n1, %cfg_n2) "
                                 "workgroup_size(%cfg_wg0, %cfg_wg1, "
                                 "%cfg_wg2) : index";
      const size_t at = emitted->source.find(cfgkey);
      need(at != std::string::npos,
           "no self-indexed tile launch config in emitted source");
      const auto head = emitted->source.substr(0, at);
      std::vector<unsigned> consts;
      const std::string key = "index.constant ";
      for (size_t pos = 0;;) {
        const size_t p = head.find(key, pos);
        if (p == std::string::npos)
          break;
        size_t i = p + key.size();
        unsigned v = 0;
        while (i < head.size() && head[i] >= '0' && head[i] <= '9')
          v = v * 10u + (unsigned)(head[i++] - '0');
        consts.push_back(v);
        pos = i;
      }
      need(consts.size() >= 6, "tile launch header carries <6 constants");
      const unsigned *t = &consts[consts.size() - 6];
      const unsigned expected_grid = ((M + 63) / 64) * ((sh.n + 63) / 64);
      need(t[0] == 128 && t[1] == expected_grid && t[3] == 1 && t[5] == 1,
           std::string(name) + ": emitted launch " + std::to_string(t[0]) +
               " of " + std::to_string(t[1]) + "x" + std::to_string(t[3]) +
               "x" + std::to_string(t[5]) + "; expected 128 of " +
               std::to_string(expected_grid) + "x1x1");
      need(emitted->source.find("vector.mma") != std::string::npos,
           std::string(name) + ": emitted source has no vector.mma");
      need(emitted->lds_bytes == 32768u,
           std::string(name) + ": emitted lds " +
               std::to_string(emitted->lds_bytes) + " != expected 32768");
    }
    need(found, "anchor node not partitioned as a standalone group");
  }
  std::uint64_t input_hash = 1469598103934665603ull;
  input_hash = fold(input_hash, f.x.data(), f.x.size() * 4);
  input_hash = fold(input_hash, f.packed.data(), f.packed.size() * 4);
  input_hash = fold(input_hash, f.scale.data(), f.scale.size() * 2);
  input_hash = fold(input_hash, f.bias.data(), f.bias.size() * 2);
  ArmResult r;
  r.name = name;
  r.input_hash = input_hash;
  constexpr int warmups = 1, measured = 8;
  Program retained;
  std::uint64_t first_hash = 0;
  bool checked = false;
  for (int iteration = 0; iteration < warmups + measured; ++iteration) {
    if (iteration)
      retained.reset_compute();
    check(b.copy_h2d(output.original.data(), output.base,
                     output.original.size(), 0),
          "re-poison output");
    retire(b);
    const auto start = std::chrono::steady_clock::now();
    auto status = s.eval(roots, false, &retained);
    retire(b);
    need(status.ok(), "GPU eval: " + status.to_string());
    if (iteration >= warmups)
      r.per_iteration_ms.push_back(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count() *
          1000.0);
    const auto &trace = s.last_trace();
    if (iteration) {
      need(trace.replayed && trace.partition_passes == 0,
           "retained graph replay required");
      need(trace.spans.jit_compile.ns == 0, "no JIT on replay");
    }
    need(trace.device_groups == 1 && trace.kernels_launched == 1 &&
             !trace.host_groups && !trace.host_fallbacks,
         "one GPU kernel without fallback");
    const auto bytes = read(b, output);
    for (size_t i = 0; i < guard; ++i)
      need(bytes[i] == std::byte{0xa5} &&
               bytes[bytes.size() - 1 - i] == std::byte{0xa5},
           "output guard changed");
    std::uint64_t h = 1469598103934665603ull;
    h = fold(h, bytes.data() + guard, size_t(M) * N * 4);
    if (iteration == 0)
      first_hash = h;
    else if (first_hash != h) {
      std::fprintf(stderr, "DIVERGENCE arm=%s iteration=%d\n", name,
                   iteration);
      need(false, "repeat output hash changed (non-deterministic kernel)");
      first_hash = h;
    }
    r.output_hash = h;
    if (iteration == warmups + measured - 1) {
      checked = true;
      r.last_output.resize(size_t(M) * N);
      std::memcpy(r.last_output.data(), bytes.data() + guard,
                  r.last_output.size() * 4);
      double num = 0, den = 0, worst = 0;
      for (size_t i = 0; i < r.last_output.size(); ++i) {
        const double a = double(r.last_output[i]);
        const double ref = f.oracle[i];
        need(std::isfinite(a), std::string(name) + " non-finite output");
        worst = std::max(worst, std::abs(a - ref));
        num += (a - ref) * (a - ref);
        den += ref * ref;
      }
      r.relative_l2 = std::sqrt(num) / std::sqrt(den);
      r.worst_abs_err = worst;
    }
  }
  (void)checked;
  retire(b);
  double sum = 0;
  for (double v : r.per_iteration_ms)
    sum += v;
  r.mean_ms = sum / r.per_iteration_ms.size();
  return r;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2 || std::strcmp(argv[1], "--run")) {
    std::fputs("Usage: ffn-micro-fp8 --run\n"
               "Without a flag no device is opened; exits 0.\n", stderr);
    return argc == 1 ? 0 : 2;
  }
  try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("LSE_REQUIRE_DEVICE_KERNELS", "1", 1);
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    check(place::open_default_devices("hrx:0"), "open explicit GPU");
    auto *d = place::default_devices();
    need(d && d->size() == 1, "one device required");
    auto &b = d->device(d->primary());
    need(b.name() == "hrx" && b.device_info().arch == "gfx1201",
         "gfx1201 HRX required");
    auto *s = default_scheduler();
    need(s, "scheduler required");
    s->set_dialect(Dialect::kLoom);
    const Shape2 shapes[]{{17408, 5120}};
    const char *shape_name[] = {"up"};
    const auto f = make_fixture(shapes[0], 1u);
    std::fprintf(
        stdout,
        "micro shape=%s M=%d N=%d K=%d words=%u groups=%u input_hash "
        "x=%016llx packed=%016llx\n",
        shape_name[0], M, shapes[0].n, shapes[0].k, shapes[0].words(),
        shapes[0].groups(),
        static_cast<unsigned long long>(
            fold(1469598103934665603ull, f.x.data(), f.x.size() * 4)),
        static_cast<unsigned long long>(
            fold(1469598103934665603ull, f.packed.data(), f.packed.size() * 4)));
    std::vector<ArmResult> results;
    for (int arm = 1; arm <= 2; ++arm) {
      ArmResult r;
      bool have = true;
      try {
        r = run_arm(b, *s, f, shapes[0],
                    arm == 1 ? "prod" : "fp8", arm);
      } catch (const std::runtime_error &e) {
        if (std::string(e.what()).rfind("prod arm declined", 0) == 0) {
          have = false;
          std::printf("SKIP shape=%s arm=prod (not in production records)\n",
                      shape_name[0]);
        } else
          throw;
      }
      if (!have) continue;
      const double tflops =
          2.0 * double(M) * double(shapes[0].n) * double(shapes[0].k) /
          (r.mean_ms * 1e-3) / 1e12;
      const double bytes =
          double(M) * shapes[0].k * 4 + double(shapes[0].n) * shapes[0].k / 8 +
          double(shapes[0].n) * (shapes[0].k / 64) * 4 +
          double(M) * shapes[0].n * 4;
      const double gbps = bytes / (r.mean_ms * 1e-3) / 1e9;
      std::printf(
          "PASS shape=%s arm=%s M%d N%d K%d grid=%ux1x1 of %u "
          "warmups=1 measured=8 input_hash=%016llx output_hash=%016llx "
          "host_eval_retire_mean_ms=%.6f per_iteration_ms=[",
          shape_name[0], r.name.c_str(), M, shapes[0].n, shapes[0].k,
          128u, shapes[0].tile_wg(),
          static_cast<unsigned long long>(r.input_hash),
          static_cast<unsigned long long>(r.output_hash), r.mean_ms);
      for (size_t i = 0; i < r.per_iteration_ms.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", r.per_iteration_ms[i]);
      std::printf(
          "] relative_l2_vs_f64_oracle=%.9f worst_abs_err=%.6g "
          "tflops=%.3f gbps=%.2f\n",
          r.relative_l2, r.worst_abs_err, tflops, gbps);
      results.push_back(std::move(r));
    }
    // prod vs fp8 output comparison (fp8 expected ~0.03-0.05 rel-L2 vs
    // prod; fp8 rounds both operands to e4m3).
    if (results.size() == 2) {
      auto &prod = results[0], &fp8 = results[1];
      double num = 0, den = 0;
      for (size_t i = 0; i < fp8.last_output.size(); ++i) {
        const double a = double(fp8.last_output[i]);
        const double ref = double(prod.last_output[i]);
        num += (a - ref) * (a - ref);
        den += ref * ref;
      }
      std::printf("ACC shape=%s fp8 vs-prod-output rel-L2=%.9f\n",
                  shape_name[0], std::sqrt(num) / std::sqrt(den));
    }
    std::puts("PASS ffn micro fp8 complete");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
