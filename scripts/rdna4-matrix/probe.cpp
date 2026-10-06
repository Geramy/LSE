// Determine the gfx12 WMMA/SWMMAC operand layouts on the device by testing hypotheses
// against a host reference.
#include "hk.hpp"
#include <cmath>
#include <random>

static uint8_t e4m3_of_int(int v) {  // exact for |v| <= 15
  if (v == 0) return 0;
  uint8_t s = v < 0 ? 0x80 : 0; int a = std::abs(v); int e = 31 - __builtin_clz(a);
  int mant = ((a << 3) >> e) & 7;  // a = 2^e * (1 + m/8)
  return s | (uint8_t)((e + 7) << 3) | (uint8_t)mant;
}
static uint16_t f16_of_int(int v) {
  if (v == 0) return 0;
  uint16_t s = v < 0 ? 0x8000 : 0; int a = std::abs(v); int e = 31 - __builtin_clz(a);
  int mant = ((a << 10) >> e) & 0x3ff; return s | (uint16_t)((e + 15) << 10) | (uint16_t)mant;
}

// K index for operand element j (0..cnt-1) on lane-half h under layout `lay`.
//   lay 0: contiguous  : K = h*cnt + j
//   lay 1: interleaved8: K = (j/8)*16 + h*8 + j%8   (for cnt=16) / generalises to 8-chunks
static int kpos(int lay, int h, int j, int cnt) {
  if (lay == 0) return h * cnt + j;
  return (j / 8) * 16 + h * 8 + (j % 8);
}

int main(int argc, char** argv) {
  Hk hk; hk.init(argc > 1 ? argv[1] : "probe.hsaco");
  std::printf("agent %s cus=%u\n", hk.name, hk.cus);
  std::mt19937 rng(7);
  auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  void* dA = hk.alloc(4096); void* dB = hk.alloc(4096); void* dI = hk.alloc(4096); void* dD = hk.alloc(4096);
  struct { void* a; void* b; void* i; void* d; } args{dA, dB, dI, dD};

  // ---------------- dense iu8 / fp8 WMMA 16x16x16: verify A/B lane layout (expect interleaved 8: lane half h -> K 8h..8h+7)
  for (int fp8 = 0; fp8 < 2; ++fp8) {
    int A[16][16], B[16][16];
    for (auto& r : A) for (int& v : r) v = rnd(-7, 7);
    for (auto& r : B) for (int& v : r) v = rnd(-7, 7);
    for (int lay = 0; lay < 2; ++lay) {
      uint8_t pa[32][8], pb[32][8];
      for (int l = 0; l < 32; ++l) for (int j = 0; j < 8; ++j) {
        int k = lay == 0 ? (l / 16) * 8 + j : (l / 16) * 8 + j;  // only one natural option for K=16: halves of 8
        if (lay == 1) k = j * 2 + (l / 16);                        // alternative: even/odd interleave
        int va = A[l % 16][k], vb = B[k][l % 16];
        pa[l][j] = fp8 ? e4m3_of_int(va) : (uint8_t)(int8_t)va;
        pb[l][j] = fp8 ? e4m3_of_int(vb) : (uint8_t)(int8_t)vb;
      }
      hk.upload(dA, pa, sizeof pa); hk.upload(dB, pb, sizeof pb);
      hk.run(hk.kernel(fp8 ? "probe_w_fp8" : "probe_w_iu8"), &args, sizeof args, 1, 1, 32);
      int32_t D[32][8]; hk.download(D, dD, sizeof D);
      int bad = 0;
      for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) {
        int m = (l / 16) * 8 + i, n = l % 16; long ref = 0; for (int k = 0; k < 16; ++k) ref += A[m][k] * B[k][n];
        double got = fp8 ? (double)((float*)&D[l][i])[0] : (double)D[l][i];
        if (std::fabs(got - ref) > 1e-3) ++bad;
      }
      std::printf("dense %s 16x16x16, K layout %s: %d/256 mismatches\n", fp8 ? "fp8" : "iu8", lay ? "even/odd" : "halves-of-8", bad);
    }
  }

  // ---------------- sparse 16x16x32: A 2:4 compressed (8 per lane), B dense 16 per lane, index 16 bits
  const char* names[3] = {"iu8", "fp8", "f16"};
  for (int ty = 0; ty < 3; ++ty) {
    int Ad[16][32], B[32][16];
    for (auto& r : B) for (int& v : r) v = rnd(-7, 7);
    // 2:4 pattern: in each group of 4 along K keep two random positions
    for (int m = 0; m < 16; ++m) for (int g = 0; g < 8; ++g) {
      int p0 = rnd(0, 3), p1; do p1 = rnd(0, 3); while (p1 == p0);
      for (int t = 0; t < 4; ++t) Ad[m][g * 4 + t] = (t == p0 || t == p1) ? rnd(1, 7) * (rnd(0, 1) ? 1 : -1) : 0;
    }
    for (int la = 0; la < 2; ++la) for (int lb = 0; lb < 2; ++lb) {
      uint8_t pa8[32][8] = {}, pb8[32][16] = {}; uint16_t pa16[32][8] = {}, pb16[32][16] = {}; uint32_t idx[32] = {};
      for (int l = 0; l < 32; ++l) {
        int h = l / 16, m = l % 16, n = l % 16;
        // A: the lane's 16 dense K positions (two 8-chunks under la=1), compress 2 of each 4
        int dense_k[16]; for (int j = 0; j < 16; ++j) dense_k[j] = kpos(la, h, j, 16);
        int c = 0;
        for (int g = 0; g < 4; ++g) {
          int kept = 0;
          for (int t = 0; t < 4 && kept < 2; ++t) {
            int k = dense_k[g * 4 + t];
            if (Ad[m][k] != 0) { int v = Ad[m][k];
              if (ty == 2) pa16[l][c] = f16_of_int(v); else pa8[l][c] = ty == 1 ? e4m3_of_int(v) : (uint8_t)(int8_t)v;
              idx[l] |= (uint32_t)t << (2 * c); ++c; ++kept; }
          }
          // Exactly two kept per group by construction.
        }
        for (int j = 0; j < 16; ++j) { int k = kpos(lb, h, j, 16); int v = B[k][n];
          if (ty == 2) pb16[l][j] = f16_of_int(v); else pb8[l][j] = ty == 1 ? e4m3_of_int(v) : (uint8_t)(int8_t)v; }
      }
      if (ty == 2) { hk.upload(dA, pa16, sizeof pa16); hk.upload(dB, pb16, sizeof pb16); }
      else { hk.upload(dA, pa8, sizeof pa8); hk.upload(dB, pb8, sizeof pb8); }
      hk.upload(dI, idx, sizeof idx);
      const char* kn = ty == 0 ? "probe_s_iu8" : ty == 1 ? "probe_s_fp8" : "probe_s_f16";
      hk.run(hk.kernel(kn), &args, sizeof args, 1, 1, 32);
      int32_t D[32][8]; hk.download(D, dD, sizeof D);
      int bad = 0;
      for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) {
        int m = (l / 16) * 8 + i, n = l % 16; long ref = 0; for (int k = 0; k < 32; ++k) ref += Ad[m][k] * B[k][n];
        double got = ty == 0 ? (double)D[l][i] : (double)((float*)&D[l][i])[0];
        if (std::fabs(got - ref) > 1e-3) ++bad;
      }
      std::printf("sparse %s 16x16x32: A-layout %s, B-layout %s: %d/256 mismatches\n", names[ty],
                  la ? "interleave8" : "contig16", lb ? "interleave8" : "contig16", bad);
    }
  }
  hk.fini();
  return 0;
}
