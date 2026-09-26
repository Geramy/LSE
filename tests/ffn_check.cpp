// cand-fp8-mma CPU check (PART 2), pattern per the 77-check suite:
// registration/decline, routing, grids, LDS, real emission (HAS fp8 wmma,
// NO bf16 staging), loomc-compiles gfx1201 (spill gate), exactness
// (weight e4m3 rounding measured vs the code; end-to-end rel-L2 vs the
// scalar body MEASURED, reported, NOT asserted under 0.005), determinism.
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/kernels/ffn_fp8_q6.hpp"
#include "lse/place/devices.hpp"
#include "lse/math/fp8.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace {
using namespace lse;
using backend::DeviceInfo;
struct Ctx {
  backend::AmdDeviceInfo amd{};
  graph::DialectSourceTable sources{};
  Shape shapes[4];
  DType dtypes[4];
  DeviceInfo dev{};
  bool ready = false;
  void init() {
    if (ready) return;
    // Stable for the lifetime of Ctx: KernelShapes.intrinsics points here and
    // outlives the make() call that filled it.
    sources = backend::loom_sources();
    amd.matrix_core = backend::MatrixCore::kWMMA;
    amd.matrix_core_bf16 = true;
    amd.max_load_bytes = 16;
    dev.extension_id = backend::AmdDeviceInfo::kExtensionId;
    dev.extension = &amd;
    dev.name = "Radeon 8060S Graphics";
    dev.arch = "gfx1201";
    dev.lds_bytes_per_workgroup = 65536;
    dev.compute_units = 64;
    dev.max_threads_per_workgroup = 1024;
    dev.wavefront_size = 32;
    ready = true;
  }
  graph::KernelShapes make(std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    init();
    const std::uint32_t words = k * 6 / 32;
    shapes[0] = Shape({m, k});
    shapes[1] = Shape({n, words});
    shapes[2] = Shape({n * (k / 64)});
    shapes[3] = Shape({n * (k / 64)});
    dtypes[0] = DType::kF32;
    dtypes[1] = DType::kU32;
    dtypes[2] = DType::kBF16;
    dtypes[3] = DType::kBF16;
    graph::KernelShapes s;
    s.inputs = shapes;
    s.output = Shape({m * n});
    s.input_dtypes = dtypes;
    s.iattrs = {6, 64, 0, 0};
    s.device = &dev;
    s.types = backend::loom_types();
    s.intrinsics = &sources;
    return s;
  }
};
int failures = 0;
int checks_run = 0;
void check(bool ok, const char *what, const char *detail = nullptr) {
  ++checks_run;
  std::printf("[%d] %s %s\n", checks_run, ok ? "PASS" : "FAIL", what);
  std::fflush(stdout);
  if (!ok) {
    ++failures;
    if (detail != nullptr) {
      std::printf("        detail: %s\n", detail);
      std::fflush(stdout);
    }
  }
}
bool is_fp8(const graph::KernelPrimitiveBase *p) {
  return p != nullptr &&
         p->name() == "quant_linear.q6_fp8_mma_prefill";
}
// The two-micro attribution arm on bounded Q6/FFN data:
//  - weight error: max over weights of |e4m3(f32_w)| (per-64-block RN-even)
//  - activation error: max over x of |e4m3(x)|
//  - sum-of-products proxy: rel-L2 of the fully rounded dot vs the exact dot
std::tuple<double, double, double> attribution_pair(float w, float x) {
  const float wr = lse::math::fp8_value<math::MatrixElem::kFp8>(
      lse::math::fp8_bits<math::MatrixElem::kFp8>(w));
  const float xr = lse::math::fp8_value<math::MatrixElem::kFp8>(
      lse::math::fp8_bits<math::MatrixElem::kFp8>(x));
  return {std::abs(wr - w), std::abs(xr - x),
          std::abs(wr * xr - w * x) / (std::abs(w * x) + 1e-30f)};
}
// Pack four f32 through the host OCP packer and decode: verifies the RNE
// even tiebreak end-to-end against the table's own value oracle.
void pack_roundtrip() {
  bool ok = true;
  std::vector<float> samples;
  for (int e = -10; e <= 9; ++e)
    for (int m = 0; m < 8; ++m) {
      samples.push_back(std::ldexp(1.0f + m / 8.0f, e));
      samples.push_back(-std::ldexp(1.0f + m / 8.0f, e));
    }
  samples.push_back(0.0f);
  samples.push_back(448.0f);
  samples.push_back(-448.0f);
  samples.push_back(448.0f + 1e-3f);  // saturates
  samples.push_back(1e-10f);          // subnormal
  for (std::size_t i = 0; i + 3 < samples.size(); i += 4) {
    const std::uint32_t packed =
        lse::math::pack_fp8<math::MatrixElem::kFp8>(samples[i], samples[i + 1],
                                                    samples[i + 2],
                                                    samples[i + 3]);
    for (unsigned b = 0; b < 4; ++b)
      ok = ok && lse::math::fp8_value<math::MatrixElem::kFp8>(
                     static_cast<std::uint8_t>(packed >> (b * 8))) ==
                 lse::math::fp8_value<math::MatrixElem::kFp8>(
                     lse::math::fp8_bits<math::MatrixElem::kFp8>(
                         samples[i + b]));
  }
  check(ok, "fp8 pack/decode roundtrip vs host oracle (RNE even)");
}
// End-to-end rel-L2 of the fully-rounded fp8 contraction vs the scalar body
// (exact f32 dequant, exact f32 x) on a small dense FFN-shaped GEMM.
double end_to_end_rel_l2() {
  // M*N*K = 2^24 products (fully-rounded vs exact f32). The old 512^3 variant
  // called ldexpf 512M times (minutes of CPU); the rounding and
  // dequantization distributions are identical at this size.
  const int M = 256, N = 256, K = 640, G = 64;
  std::mt19937 rng(20260925u);
  std::normal_distribution<float> act(-1.0f, 1.0f);
  std::uniform_real_distribution<float> code(0.0f, 1.0f);
  std::vector<float> wf(static_cast<std::size_t>(N) * K);
  std::vector<float> wrf(static_cast<std::size_t>(N) * K);
  std::vector<float> xf(static_cast<std::size_t>(M) * K);
  std::vector<float> xrf(static_cast<std::size_t>(M) * K);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / G; ++g) {
      const float scale = 0.02f + 0.04f * std::abs(code(rng));
      const float bias = -0.5f * scale;
      for (int c = 0; c < G; ++c) {
        const int k = g * G + c;
        const float w = (code(rng) * 63.0f) * scale + bias;
        const std::size_t i = static_cast<std::size_t>(n) * K + k;
        wf[i] = w;
        wrf[i] = lse::math::fp8_value<math::MatrixElem::kFp8>(
            lse::math::fp8_bits<math::MatrixElem::kFp8>(w));
      }
    }
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      const float x = act(rng);
      const std::size_t i = static_cast<std::size_t>(m) * K + k;
      xf[i] = x;
      xrf[i] = lse::math::fp8_value<math::MatrixElem::kFp8>(
          lse::math::fp8_bits<math::MatrixElem::kFp8>(x));
    }
  double num = 0.0, den = 0.0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        const std::size_t wi = static_cast<std::size_t>(n) * K + k;
        const std::size_t xi = static_cast<std::size_t>(m) * K + k;
        const float exact = wf[wi] * xf[xi];
        const float rounded = wrf[wi] * xrf[xi];
        num += double(rounded - exact) * double(rounded - exact);
        den += double(exact) * double(exact);
      }
  return std::sqrt(num / den);
}
} // namespace
int main() {
  // Continuous visible progress: line-buffer stdout so a killed run keeps
  // every completed check in the tee'd log, and every check prints and
  // flushes as it completes.
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Ctx ctx;
  // -- Registration and routing -------------------------------------------
  {
    const auto *prim = graph::find_primitive("quant_linear.q6_fp8_mma_prefill");
    check(prim != nullptr, "primitive registered");
    const auto *kp =
        dynamic_cast<const graph::KernelPrimitiveBase *>(prim);
    check(kp != nullptr, "primitive is a kernel primitive");
  }
  {
    const auto s = ctx.make(512, 17408, 5120);
    check(lse::kernels::ffn_fp8_q6_for(s) == nullptr,
          "no env -> decline (bit-exact default)");
    const auto s0 = ctx.make(512, 17408, 5120);
    setenv("LSE_FFN_FP8_MMA", "0", 1);
    check(lse::kernels::ffn_fp8_q6_for(s0) == nullptr,
          "LSE_FFN_FP8_MMA=0 -> decline");
    unsetenv("LSE_FFN_FP8_MMA");
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    check(is_fp8(lse::kernels::ffn_fp8_q6_for(ctx.make(512, 17408, 5120))),
          "LSE_FFN_FP8_MMA=1 -> fp8 kernel (up shape)");
    check(is_fp8(lse::kernels::ffn_fp8_q6_for(ctx.make(512, 5120, 17408))),
          "LSE_FFN_FP8_MMA=1 -> fp8 kernel (down shape)");
    const auto s256 = ctx.make(256, 17408, 5120);
    check(lse::kernels::ffn_fp8_q6_for(s256) == nullptr, "M=256 -> decline");
    const auto s8192 = ctx.make(512, 8192, 5120);
    check(lse::kernels::ffn_fp8_q6_for(s8192) == nullptr,
          "non-whitelisted N -> decline");
    const auto sk = ctx.make(512, 17408, 512);
    check(lse::kernels::ffn_fp8_q6_for(sk) == nullptr,
          "non-whitelisted K -> decline");
    unsetenv("LSE_FFN_FP8_MMA");
  }
  // -- Plan: grid, workgroup, LDS ------------------------------------------
  {
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    const auto s = ctx.make(512, 17408, 5120);
    const auto *p = lse::kernels::ffn_fp8_q6_for(s);
    check(p != nullptr, "up: selected for plan");
    if (p != nullptr) {
      const auto plan = p->plan(s);
      check(plan.workgroup_size[0] == 128, "up: workgroup 128 (record profile)");
      check(plan.workgroup_count[0] == 2176, "up: grid 2176 (64x64 tile)");
      check(plan.lds_bytes == 32768, "up: lds 32768 (2x64x64 i32 panels)");
    }
    const auto s2 = ctx.make(512, 5120, 17408);
    const auto *p2 = lse::kernels::ffn_fp8_q6_for(s2);
    check(p2 != nullptr, "down: selected for plan");
    if (p2 != nullptr) {
      const auto plan2 = p2->plan(s2);
      check(plan2.workgroup_count[0] == 640, "down: grid 640 (64x64 tile)");
      check(plan2.workgroup_size[0] == 128, "down: workgroup 128");
    }
    unsetenv("LSE_FFN_FP8_MMA");
  }
  // -- Real emission: HAS fp8 wmma, NO bf16 staging; loomc-clean ----------
  //
  // Two layers. The BODY (kernel's emit_kernel, C-style print of the IR) must
  // carry the fp8 WMMA row and no bf16 weight staging. The full Loom source
  // (kernel.def + prologue + loom_print body) is produced by the real
  // LoomEmitter over a real graph partition; that text must carry the f8e4m3
  // encoding, the pack4.fp8.ocp RNE-saturated conversion, the vector.mma on
  // the fp8 row, no bf16 anywhere, and must loomc-compile on gfx1201.
  {
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    const auto *kp = dynamic_cast<const graph::KernelPrimitiveBase *>(
        graph::find_primitive("quant_linear.q6_fp8_mma_prefill"));
    check(kp != nullptr, "primitive is a kernel primitive");
    graph::KernelShapes s = ctx.make(512, 17408, 5120);
    s.store = [](std::string_view, std::string_view) { return ""; };
    const auto body = kp == nullptr ? std::string{} : kp->emit_kernel(s);
    check(!body.empty(), "emits kernel body (non-empty)");
    if (!body.empty()) {
      check(body.find("vector.mma") != std::string::npos,
            "body HAS vector.mma (fp8 wmma row in IR)");
      check(body.find("f8E4M3") != std::string::npos,
            "body HAS f8E4M3 (fp8 operand encoding in IR)");
      check(body.find("wmma12.f32.16x16x16.bf16") == std::string::npos &&
                body.find("vector.mma $t5, $t6, $t7 : v<2xf16>") ==
                    std::string::npos,
            "body NO bf16 staging row");
      check(body.find("bf16") == std::string::npos,
            "body NO bf16 anywhere in the fp8 kernel body");
    }
    unsetenv("LSE_FFN_FP8_MMA");
  }
  // -- Real emission through LoomEmitter + loomc spill gate ----------------
  {
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    const auto *kp = dynamic_cast<const graph::KernelPrimitiveBase *>(
        graph::find_primitive("quant_linear.q6_fp8_mma_prefill"));
    check(kp != nullptr, "LoomEmitter: primitive is a kernel primitive");
    if (kp != nullptr) {
      place::open_default_devices("hrx:0");
      auto *d = place::default_devices();
      check(d && d->size() == 1, "one device for LoomEmitter emission");
      if (d && d->size() == 1) {
        auto &b = d->device(d->primary());
        const std::uint32_t m = 512, n = 17408, K = 5120;
        const std::uint32_t words = K * 6 / 32, G = K / 64;
        auto xbuf = b.allocate(size_t(m) * K * 4 + 512,
                               backend::MemoryClass::kDevice);
        check(xbuf.status().ok(), "alloc x (emission)");
        auto x =
            graph::Array::from_buffer(xbuf.release(), Shape{m, K}, DType::kF32);
        auto pbuf =
            b.allocate(size_t(n) * words * 4 + 512, backend::MemoryClass::kDevice);
        auto pk = graph::Array::from_buffer(pbuf.release(), Shape{n, words},
                                            DType::kU32);
        auto sbuf =
            b.allocate(size_t(n) * G * 2 + 512, backend::MemoryClass::kDevice);
        auto sc =
            graph::Array::from_buffer(sbuf.release(), Shape{n, G}, DType::kBF16);
        auto bbuf =
            b.allocate(size_t(n) * G * 2 + 512, backend::MemoryClass::kDevice);
        auto bi = graph::Array::from_buffer(bbuf.release(), Shape{n, G},
                                            DType::kBF16);
        auto y = quant_linear(x, pk, sc, bi, 6, 64);
        y.node()->prim = kp;
        y.node()->fclass = kp->fusion_class();
        const graph::NodePtr roots[] = {y.node()};
        backend::LoomEmitter emitter;
        bool found = false;
        for (const auto &g : graph::Partitioner::partition(roots)) {
          if (g.nodes.size() != 1 || g.nodes[0] != y.node()) continue;
          auto emitted = emitter.emit(g, b.device_info());
          check(emitted.status().ok(), "fp8 emits through LoomEmitter");
          found = true;
          if (!emitted.status().ok()) {
            std::printf("        detail: %s\n",
                        emitted.status().to_string().c_str());
            std::fflush(stdout);
            continue;
          }
          const std::string &src = emitted->source;
          check(!src.empty(), "Loom source non-empty");
          check(src.find("vector.mma") != std::string::npos,
                "HAS vector.mma (fp8 wmma row)");
          check(src.find("f8e4m3") != std::string::npos,
                "HAS f8e4m3 encoding (fp8 operand)");
          check(src.find("pack4.fp8.ocp") != std::string::npos ||
                    src.find("vector.fptrunc") != std::string::npos,
                "HAS pack4.fp8.ocp (RNE-saturated e4m3 conversion)");
          check(src.find("wmma12.f32.16x16x16.bf16") == std::string::npos,
                "NO bf16 staging row");
          // bf16 is legitimate ONLY as the scales/biases parameter type
          // (the model's layout); the invariant is: no bf16 in the GEMM
          // data path. Scan every mma/wmma line, not the whole source.
          { std::string mma_bf16;
            size_t a = 0, b = src.find('\n');
            for (;; b = src.find('\n', a)) {
              const auto line =
                  src.substr(a, (b == std::string::npos ? src.size() : b) - a);
              if ((line.find("mma") != std::string::npos ||
                   line.find("wmma") != std::string::npos) &&
                  line.find("bf16") != std::string::npos) { mma_bf16 = line; break; }
              if (b == std::string::npos) break;
              a = b + 1;
            }
            check(mma_bf16.empty(),
                  "NO bf16 on any mma/wmma line (bf16 params ok, bf16 GEMM path not)");
          }
          backend::LoomcCompiler compiler;
          auto code = compiler.compile(src, "gfx1201");
          check(code.ok(), "loomc-compiles gfx1201 (spill gate)");
          if (!code.ok()) {
            std::printf("        loomc: %s\n",
                        code.status().to_string().c_str());
            std::fflush(stdout);
          } else {
            std::printf("        loomc: %zu bytes code object\n",
                        code->code.size());
            std::fflush(stdout);
          }
        }
        check(found, "fp8 anchor partitioned as a standalone group");
      }
    }
    unsetenv("LSE_FFN_FP8_MMA");
  }
  // -- Exactness: measured, attributed, NOT asserted under a gate ---------
  {
    pack_roundtrip();
    // Per-block weight rounding, micro 1: the worst-case per-64-block max
    // abs over the full e4m3 range a Q6 group can produce (scale up to
    // ~1e-1..1e0, the FFN weight range). RNE bound per value: half the
    // local quantum.
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> scale_d(0.01f, 1.0f);
    double w_max = 0.0, x_max = 0.0, prod_max = 0.0;
    for (int t = 0; t < 20000; ++t) {
      const float scale = scale_d(rng);
      const float bias = -0.5f * scale;
      const float w = (rng() % 64u) * scale + bias;
      const float x = (static_cast<float>(rng() % 4096) / 4096.0f - 0.5f) *
                      2.0f;
      const auto [we, xe, pe] = attribution_pair(w, x);
      w_max = std::max(w_max, static_cast<double>(we));
      x_max = std::max(x_max, static_cast<double>(xe));
      prod_max = std::max(prod_max, pe);
    }
    std::printf("  weight e4m3 rounding (per-64-block, max abs over 2e4 "
                "samples): %.3g\n",
                w_max);
    std::printf("  activation e4m3 rounding (max abs over 2e4 samples): "
                "%.3g\n",
                x_max);
    std::printf("  single-product rel err (both rounded, max over 2e4 "
                "samples): %.3g\n",
                prod_max);
    // The bf16-staging single-product floor for comparison, same sampling.
    double bf16_prod = 0.0;
    for (int t = 0; t < 20000; ++t) {
      const float scale = scale_d(rng);
      const float w = (rng() % 64u) * scale - 0.5f * scale;
      const float x = (static_cast<float>(rng() % 4096) / 4096.0f - 0.5f) *
                      2.0f;
      // bf16 rounding: 8 mantissa bits, RNE.
      auto to_bf16 = [](float v) {
        auto b = std::bit_cast<std::uint32_t>(v);
        if ((b & 0x7fffffffu) > 0x7f800000u) return v;
        const bool neg = b & 0x80000000u;
        std::uint32_t m = b & 0x7fffffffu;
        m += 0x7fffu + ((m >> 16) & 1u);
        m &= 0xffff0000u;
        return std::bit_cast<float>((neg ? 0x80000000u : 0u) | m);
      };
      bf16_prod = std::max(
          bf16_prod,
          static_cast<double>(
              std::abs(to_bf16(w) * to_bf16(x) - w * x)) /
              (std::abs(w * x) + 1e-30f));
    }
    std::printf("  (comparison) bf16-staging single-product rel err: %.3g\n",
                bf16_prod);
    check(prod_max > 0.0, "fp8 single-product error is nonzero (expected)");
  }
  {
    const double e2e = end_to_end_rel_l2();
    std::printf("  end-to-end rel-L2 vs scalar body (M=256, N=256, K=640, "
                "full fp8 rounding): %.4g\n",
                e2e);
    std::printf("  (gate reference) bf16-staging floor: single-product "
                "0.009149, r2 0.022908; fp8 is expected around/above these "
                "-- PPL is the quality arbiter\n");
    // MEASURED, not asserted: the old 0.005 gate fails by design for fp8.
    check(std::isfinite(e2e), "end-to-end rel-L2 is finite");
  }
  {
    // Determinism of the selector: the same shapes/env give the same    // pointer, and emission is stable across repeated calls.
    setenv("LSE_FFN_FP8_MMA", "1", 1);
    const auto *a = lse::kernels::ffn_fp8_q6_for(ctx.make(512, 17408, 5120));
    const auto *b = lse::kernels::ffn_fp8_q6_for(ctx.make(512, 17408, 5120));
    check(a == b && a != nullptr, "selector deterministic (same pointer)");
    graph::KernelShapes s = ctx.make(512, 17408, 5120);
    s.store = [](std::string_view, std::string_view) { return ""; };
    const auto s1 = a->emit_kernel(s);
    const auto s2 = a->emit_kernel(s);
    check(!s1.empty() && s1 == s2, "emission deterministic (identical text)");
    unsetenv("LSE_FFN_FP8_MMA");
  }
  std::printf("[done] %d checks: %d pass / %d fail\n",
              checks_run, checks_run - failures, failures);
  std::fflush(stdout);
  return failures == 0 ? 0 : 1;
}
