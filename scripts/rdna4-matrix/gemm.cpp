// GEMM microbenchmark driver: dense iu8 / fp8 / f16 versus 2:4-sparse SWMMAC, with an
// optional Q4-style group-64 rescale epilogue. Checks sampled outputs against a host
// reference before timing.
// usage: gemm <mode> <rescale> <M> <N> <K> [reps]
//   mode 0 iu8, 1 fp8, 2 f16, 3 iu8 2:4-sparse W, 4 fp8 2:4-sparse W
//   rescale 0 none, 1 full Q4 group epilogue, 2 weight-scale-only group epilogue
//   HSACO overrides the object (default gemm_<mode>_<rescale>.hsaco); TBN/TBM/TNT give its
//   tile and workgroup size when it was built with other WN/WM/TN/TM values.
#include "hk.hpp"
#include <algorithm>
#include <cmath>
#include <random>

static uint8_t e4m3_of_int(int v) {
  if (v == 0) return 0;
  uint8_t s = v < 0 ? 0x80 : 0; int a = std::abs(v); int e = 31 - __builtin_clz(a);
  return s | (uint8_t)((e + 7) << 3) | (uint8_t)(((a << 3) >> e) & 7);
}
static uint16_t f16_of_int(int v) {
  if (v == 0) return 0;
  uint16_t s = v < 0 ? 0x8000 : 0; int a = std::abs(v); int e = 31 - __builtin_clz(a);
  return s | (uint16_t)((e + 15) << 10) | (uint16_t)(((a << 10) >> e) & 0x3ff);
}

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: %s mode rescale M N K [reps]\n", argv[0]); return 1; }
  const int mode = atoi(argv[1]), rescale = atoi(argv[2]);
  const int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]), reps = argc > 6 ? atoi(argv[6]) : 10;
  const char* obj = getenv("HSACO"); const int TBN = getenv("TBN") ? atoi(getenv("TBN")) : 128, TBM = getenv("TBM") ? atoi(getenv("TBM")) : 128, TNT = getenv("TNT") ? atoi(getenv("TNT")) : 256;
  const bool sparse = mode == 3 || mode == 4, f16 = mode == 2, fp8 = mode == 1 || mode == 4;
  if (M % 128 || N % 128 || K % 64) { std::fprintf(stderr, "shape must be multiples of 128/128/64\n"); return 1; }
  char path[64]; std::snprintf(path, sizeof path, "gemm_%d_%d.hsaco", mode, rescale);
  if (obj) std::snprintf(path, sizeof path, "%s", obj);
  Hk hk; hk.init(path);
  std::mt19937 rng(1234);
  auto ri = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  const int eb = f16 ? 2 : 1, G = K / 64;
  // Logical dense values (small integers) for reference.
  std::vector<int8_t> Xv((size_t)M * K), Wv((size_t)N * K, 0);
  for (auto& v : Xv) v = (int8_t)ri(-7, 7);
  std::vector<uint8_t> Wbytes, Xbytes((size_t)M * K * eb);
  std::vector<uint16_t> WI;
  for (size_t i = 0; i < Xv.size(); ++i) {
    if (f16) ((uint16_t*)Xbytes.data())[i] = f16_of_int(Xv[i]);
    else Xbytes[i] = fp8 ? e4m3_of_int(Xv[i]) : (uint8_t)Xv[i];
  }
  if (!sparse) {
    Wbytes.resize((size_t)N * K * eb);
    for (size_t i = 0; i < Wv.size(); ++i) {
      Wv[i] = (int8_t)ri(-7, 7);
      if (f16) ((uint16_t*)Wbytes.data())[i] = f16_of_int(Wv[i]);
      else Wbytes[i] = fp8 ? e4m3_of_int(Wv[i]) : (uint8_t)Wv[i];
    }
  } else {
    Wbytes.resize((size_t)N * K / 2); WI.resize((size_t)N * K / 16);
    for (int n = 0; n < N; ++n)
      for (int blk = 0; blk < K / 16; ++blk) {      // one lane-half step: 16 dense K -> 8 stored values
        uint16_t idx = 0;
        for (int g = 0; g < 4; ++g) {
          int p0 = ri(0, 3), p1; do p1 = ri(0, 3); while (p1 == p0);
          if (p0 > p1) std::swap(p0, p1);
          int pos[2] = {p0, p1};
          for (int j = 0; j < 2; ++j) {
            int c = g * 2 + j, v = ri(1, 7) * (ri(0, 1) ? 1 : -1);
            idx |= (uint16_t)(pos[j] << (2 * c));
            Wbytes[(size_t)n * (K / 2) + blk * 8 + c] = fp8 ? e4m3_of_int(v) : (uint8_t)(int8_t)v;
            Wv[(size_t)n * K + blk * 16 + g * 4 + pos[j]] = (int8_t)v;
          }
        }
        WI[(size_t)n * (K / 16) + blk] = idx;
      }
  }
  std::vector<float> ws((size_t)N * G, 1.f), wb((size_t)N * G, 0.f), xs((size_t)M * G, 1.f), xu((size_t)M * G, 0.f);
  if (rescale) {
    std::uniform_real_distribution<float> U(0.5f, 1.5f), V(-1.f, 1.f);
    for (auto& v : ws) v = U(rng) * 0.01f; for (auto& v : wb) v = V(rng) * 0.01f;
    for (auto& v : xs) v = U(rng) * 0.1f;  for (auto& v : xu) v = V(rng);
  }
  void* dW = hk.alloc(Wbytes.size()); void* dX = hk.alloc(Xbytes.size()); void* dY = hk.alloc((size_t)M * N * 4);
  void* dI = hk.alloc(std::max<size_t>(WI.size() * 2, 64));
  void* dws = hk.alloc(ws.size() * 4); void* dwb = hk.alloc(wb.size() * 4); void* dxs = hk.alloc(xs.size() * 4); void* dxu = hk.alloc(xu.size() * 4);
  hk.upload(dW, Wbytes.data(), Wbytes.size()); hk.upload(dX, Xbytes.data(), Xbytes.size());
  if (sparse) hk.upload(dI, WI.data(), WI.size() * 2);
  hk.upload(dws, ws.data(), ws.size() * 4); hk.upload(dwb, wb.data(), wb.size() * 4);
  hk.upload(dxs, xs.data(), xs.size() * 4); hk.upload(dxu, xu.data(), xu.size() * 4);
  struct __attribute__((packed)) { void *W, *X, *Y, *I, *ws, *wb, *xs, *xu; int M, N, K; } args{dW, dX, dY, dI, dws, dwb, dxs, dxu, M, N, K};
  auto kern = hk.kernel("gemm");
  if (N % TBN || M % TBM) { std::fprintf(stderr, "shape not divisible by tile\n"); return 1; }
  const uint32_t gx = M / TBM, gy = N / TBN;  // M tiles fastest (see gemm.cl)
  hk.run(kern, &args, sizeof args, gx, gy, TNT);
  // Correctness on sampled outputs.
  std::vector<float> Y((size_t)M * N); hk.download(Y.data(), dY, Y.size() * 4);
  int bad = 0; double maxrel = 0;
  for (int t = 0; t < 256; ++t) {
    int m = ri(0, M - 1), n = ri(0, N - 1); double ref = 0;
    for (int g = 0; g < G; ++g) {
      long dot = 0; for (int k = g * 64; k < g * 64 + 64; ++k) dot += (long)Xv[(size_t)m * K + k] * Wv[(size_t)n * K + k];
      ref += rescale == 2 ? (double)ws[(size_t)n * G + g] * dot : rescale ? (double)ws[(size_t)n * G + g] * xs[(size_t)m * G + g] * dot + (double)wb[(size_t)n * G + g] * xu[(size_t)m * G + g] : (double)dot;
    }
    double got = Y[(size_t)m * N + n], rel = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
    maxrel = std::max(maxrel, rel); if (rel > 1e-3) ++bad;
  }
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) t.push_back(hk.run(kern, &args, sizeof args, gx, gy, TNT));
  // Re-validate after timing: clear Y, dispatch once more, re-check the same samples.
  if (getenv("RECHECK")) {
    std::vector<float> z(Y.size(), 0.f); hk.upload(dY, z.data(), z.size() * 4);
    double tr = hk.run(kern, &args, sizeof args, gx, gy, TNT);
    std::vector<float> Y2(Y.size()); hk.download(Y2.data(), dY, Y2.size() * 4);
    size_t diff = 0, zero = 0; for (size_t i = 0; i < Y.size(); ++i) { diff += Y2[i] != Y[i]; zero += Y2[i] == 0.f; }
    std::printf("recheck: %.3f ms, %zu of %zu outputs differ from first run, %zu zero\n", tr, diff, Y.size(), zero);
    for (int r = 0; r < reps; ++r) std::printf("  rep %d: %.3f ms\n", r, t[r]); std::printf("  wait wakeups without completion: %ld\n", hk.spurious);
  }
  std::sort(t.begin(), t.end());
  const double med = t[t.size() / 2], best = t[0];
  const double ops = 2.0 * M * N * (double)K;
  static const char* names[] = {"iu8", "fp8", "f16", "iu8-2:4", "fp8-2:4"};
  std::printf("%s ", obj ? obj : path); std::printf("%-8s rescale=%d M=%5d N=%5d K=%5d  check %s (%d/256 bad, max rel %.2e)  median %.3f ms best %.3f ms  %.1f dense-equiv TOPS (best %.1f)\n",
              names[mode], rescale, M, N, K, bad ? "FAIL" : "ok", bad, maxrel, med, best, ops / (med * 1e-3) / 1e12, ops / (best * 1e-3) / 1e12);
  hk.fini();
  return bad ? 4 : 0;
}
