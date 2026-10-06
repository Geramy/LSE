// Q4 prefill GEMM prototype check on real checkpoint data: activation pre-pass + GEMM,
// full-output error against the FP32 reference, and timings.
// usage: q4bench <qmode 0|1> <data dir> <tag up|down> <M> <N> <K> [reps]
#include "hk.hpp"
#include <algorithm>
#include <cmath>

template <class T> static std::vector<T> slurp(const std::string& p, size_t n) {
  std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
  if (!f.read((char*)v.data(), n * sizeof(T))) { std::fprintf(stderr, "cannot read %zu x %zu bytes from %s\n", n, sizeof(T), p.c_str()); std::exit(2); }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 7) { std::fprintf(stderr, "usage: %s qmode dir tag M N K [reps]\n", argv[0]); return 1; }
  const int qmode = atoi(argv[1]); std::string dir = argv[2], tag = argv[3];
  const int M = atoi(argv[4]), N = atoi(argv[5]), K = atoi(argv[6]), reps = argc > 7 ? atoi(argv[7]) : 7;
  const int G = K / 64;
  if (M % 128 || N % 128 || K % 64) { std::fprintf(stderr, "M, N multiple of 128 and K of 64\n"); return 1; }
  auto Wq = slurp<uint32_t>(dir + "/" + tag + ".wq.bin", (size_t)N * K / 8);
  auto Ws = slurp<uint16_t>(dir + "/" + tag + ".ws.bin", (size_t)N * G);
  auto Wb = slurp<uint16_t>(dir + "/" + tag + ".wb.bin", (size_t)N * G);
  auto X = slurp<float>(dir + "/" + tag + ".x.bin", (size_t)M * K);
  auto Yref = slurp<float>(dir + "/" + tag + ".y.bin", (size_t)M * N);
  char path[64]; std::snprintf(path, sizeof path, "q4_%d%s.hsaco", qmode, getenv("VARIANT") ? getenv("VARIANT") : "_bt1");
  Hk hk; hk.init(path);
  void* dWq = hk.alloc(Wq.size() * 4); void* dWs = hk.alloc(Ws.size() * 2); void* dWb = hk.alloc(Wb.size() * 2);
  void* dX = hk.alloc(X.size() * 4); void* dXq = hk.alloc((size_t)M * K); void* dY = hk.alloc((size_t)M * N * 4);
  void* dstep = hk.alloc((size_t)M * G * 4); void* dsu16 = hk.alloc((size_t)M * G * 2); void* dwb16 = hk.alloc((size_t)N * G * 2); void* dsum = hk.alloc((size_t)M * G * 4); void* dscale = hk.alloc((size_t)M * 4);
  hk.upload(dWq, Wq.data(), Wq.size() * 4); hk.upload(dWs, Ws.data(), Ws.size() * 2); hk.upload(dWb, Wb.data(), Wb.size() * 2);
  hk.upload(dX, X.data(), X.size() * 4);
  {  // load-time repack: bias bf16 -> f16 for the f16 WMMA bias tail
    std::vector<_Float16> b16(Wb.size());
    for (size_t i = 0; i < Wb.size(); ++i) { uint32_t u = (uint32_t)Wb[i] << 16; float f; std::memcpy(&f, &u, 4); b16[i] = (_Float16)f; }
    hk.upload(dwb16, b16.data(), b16.size() * 2);
  }
  auto kact = hk.kernel(qmode == 0 ? "act_int8_g64" : "act_fp8_row");
  auto kgemm = hk.kernel("q4gemm");
  struct __attribute__((packed)) { void *X, *Xq, *a, *b, *c; int M, K; } aargs{dX, dXq, qmode == 0 ? dstep : dscale, dsum, dsu16, M, K};
  struct __attribute__((packed)) { void *Wq, *Ws, *Wb, *Xq, *step, *sum, *scale, *Y, *wb16, *su16; int M, N, K; } gargs{
      dWq, dWs, dWb, dXq, qmode == 0 ? dstep : dsum, dsum, dscale, dY, dwb16, dsu16, M, N, K};
  const uint32_t act_groups = qmode == 0 ? (uint32_t)((M * G + 7) / 8) : (uint32_t)M;
  auto run_act = [&] { return hk.run(kact, &aargs, sizeof aargs, act_groups, 1, 256); };
  auto run_gemm = [&] { return hk.run(kgemm, &gargs, sizeof gargs, M / 128, N / 128, 256); };
  run_act(); run_gemm();
  std::vector<float> Y((size_t)M * N); hk.download(Y.data(), dY, Y.size() * 4);
  if (getenv("DUMPA")) {
    std::vector<uint8_t> xq((size_t)M * K); std::vector<float> st((size_t)M * G), su((size_t)M * G), sc(M);
    hk.download(xq.data(), dXq, xq.size()); hk.download(st.data(), dstep, st.size() * 4); hk.download(su.data(), dsum, su.size() * 4); hk.download(sc.data(), dscale, sc.size() * 4);
    std::string b = getenv("DUMPA");
    std::ofstream(b + ".xq", std::ios::binary).write((const char*)xq.data(), xq.size());
    std::ofstream(b + ".step", std::ios::binary).write((const char*)st.data(), st.size() * 4);
    std::ofstream(b + ".sum", std::ios::binary).write((const char*)su.data(), su.size() * 4);
    std::ofstream(b + ".scale", std::ios::binary).write((const char*)sc.data(), sc.size() * 4);
  }
  if (getenv("DUMPY")) { std::ofstream o(getenv("DUMPY"), std::ios::binary); o.write((const char*)Y.data(), Y.size() * 4); }
  double num = 0, den = 0, maxabs = 0; size_t nonfinite = 0;
  for (size_t i = 0; i < Y.size(); ++i) {
    if (!std::isfinite(Y[i])) { ++nonfinite; continue; }
    double d = (double)Y[i] - Yref[i]; num += d * d; den += (double)Yref[i] * Yref[i]; maxabs = std::max(maxabs, std::fabs(d));
  }
  std::vector<double> ta, tg;
  for (int r = 0; r < reps; ++r) { ta.push_back(run_act()); tg.push_back(run_gemm()); }
  // Repeat check: outputs must be unchanged by the timed replays.
  std::vector<float> Y2(Y.size()); hk.download(Y2.data(), dY, Y2.size() * 4);
  size_t changed = 0; for (size_t i = 0; i < Y.size(); ++i) changed += std::memcmp(&Y[i], &Y2[i], 4) != 0;
  std::sort(ta.begin(), ta.end()); std::sort(tg.begin(), tg.end());
  const double ops = 2.0 * M * N * (double)K, gm = tg[tg.size() / 2], am = ta[ta.size() / 2];
  std::printf("%-9s %-4s M=%5d N=%5d K=%5d  rel-L2 vs FP32 %.3e  max|err| %.3e  nonfinite %zu  replay-changed %zu  act %.3f ms  gemm %.3f ms (best %.3f)  %.1f TOPS  total %.3f ms\n",
              qmode == 0 ? "int8-g64" : "fp8-tok", tag.c_str(), M, N, K, std::sqrt(num / den), maxabs, nonfinite, changed, am, gm, tg[0],
              ops / (gm * 1e-3) / 1e12, am + gm);
  hk.fini();
  return (nonfinite || changed) ? 4 : 0;
}
