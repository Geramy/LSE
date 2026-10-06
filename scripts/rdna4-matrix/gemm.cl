// Parameterised tiled GEMM microbenchmark for gfx12: Y[M][N] = X[M][K] * W[N][K]^T,
// double-buffered LDS, M tiles fastest across the grid.
// W = matrix-core A operand (2:4-compressed in sparse modes), X = B operand.
// MODE: 0 iu8, 1 fp8, 2 f16, 3 iu8 2:4-sparse W, 4 fp8 2:4-sparse W
// RESCALE 1: per-(row, group-64) scale+bias and per-(token, group-64) step/sum epilogue.
// RESCALE 2: only the per-(row, group-64) weight scale inside the K loop.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef int i8 __attribute__((ext_vector_type(8)));

#ifndef MODE
#define MODE 0
#endif
#ifndef RESCALE
#define RESCALE 0
#endif
#ifndef WN
#define WN 2
#endif
#ifndef WM
#define WM 2
#endif
#ifndef TN
#define TN 4
#endif
#ifndef TM
#define TM 4
#endif
#ifndef BKB
#define BKB 128            // X bytes per row per step
#endif
#define NT (32 * WN * WM)
#define BN (WN * TN * 16)
#define BM (WM * TM * 16)
#define SPARSE (MODE == 3 || MODE == 4)
#define INTACC (MODE == 0 || MODE == 3)
#if INTACC
typedef i8 acc_t;
#else
typedef f8 acc_t;
#endif
#if MODE == 2
#define EB 2
#else
#define EB 1
#endif
#define KPER (BKB / EB)    // K elements per step
#if SPARSE
#define WB (BKB / 2)
#define IDXB (KPER / 8)    // index bytes per row per step (16 bits per 16 dense K)
#else
#define WB BKB
#define IDXB 0
#endif
#define XS (BKB + 16)
#define WS (WB + 16)
#define GPS (KPER / 64)    // groups per step (RESCALE)

#define LID ((int)__builtin_amdgcn_workitem_id_x())
static inline void wg_barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

static inline acc_t mma(acc_t c, const local uchar* wa, const local uchar* xb, uint idx) {
#if MODE == 0
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, *(const local i2*)wa, true, *(const local i2*)xb, c, false);
#elif MODE == 1
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(*(const local i2*)wa, *(const local i2*)xb, c);
#elif MODE == 2
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(*(const local h8*)wa, *(const local h8*)xb, c);
#elif MODE == 3
  return __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, *(const local i2*)wa, true, *(const local i4*)xb, c, (int)idx, false);
#else
  return __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(*(const local i2*)wa, *(const local i4*)xb, c, (int)idx);
#endif
}

#define XCH (BM * BKB / 16)
#define WCH (BN * WB / 16)
#define ICH (BN * IDXB / 8)
#define XPT ((XCH + NT - 1) / NT)
#define WPT ((WCH + NT - 1) / NT)
#define IPT ((ICH + NT - 1) / NT)

kernel __attribute__((reqd_work_group_size(NT, 1, 1)))
void gemm(global const uchar* W, global const uchar* X, global float* Y, global const uchar* WI,
          global const float* wscale, global const float* wbias, global const float* xstep, global const float* xsum,
          int M, int N, int K) {
  local uchar sW[2][BN * WS];
  local uchar sX[2][BM * XS];
#if SPARSE
  local uchar sI[2][BN * IDXB];
#endif
#if RESCALE
  local float sWs[2][GPS][BN], sWb[2][GPS][BN], sXs[2][GPS][BM], sXu[2][GPS][BM];
#endif
  const int tid = LID, w = tid >> 5, l = tid & 31, h = l >> 4, r = l & 15;
  const int wn = w % WN, wm = w / WN;
  // M tiles vary fastest so the WGs sharing one W tile run together and W streams from DRAM once.
  const int m0 = (int)__builtin_amdgcn_workgroup_id_x() * BM;
  const int n0 = (int)__builtin_amdgcn_workgroup_id_y() * BN;
  const size_t wrow = (size_t)K * EB / (SPARSE ? 2 : 1), xrow = (size_t)K * EB, irow = (size_t)K / 8;
  const int steps = K / KPER, groups = K / 64;
  global const uchar* Xb = X + (size_t)m0 * xrow;
  global const uchar* Wb = W + (size_t)n0 * wrow;

  acc_t acc[TN][TM];
#if RESCALE
  f8 accf[TN][TM];
#endif
  for (int a = 0; a < TN; ++a) for (int b = 0; b < TM; ++b) {
    acc[a][b] = (acc_t)(0);
#if RESCALE
    accf[a][b] = (f8)(0.0f);
#endif
  }
  uint4 px[XPT], pw[WPT];
#if SPARSE
  uint2 pi[IPT];
#endif
#if RESCALE
  float pscale[(2 * (KPER / 64) * (BN + BM) + NT - 1) / NT];
#endif
#define GLOAD(step) do { \
    _Pragma("unroll") for (int u = 0; u < XPT; ++u) { int c = tid + u * NT; if (c < XCH) { int row = c / (BKB / 16), col = (c % (BKB / 16)) * 16; \
      px[u] = *(global const uint4*)(Xb + row * xrow + (size_t)(step) * BKB + col); } } \
    _Pragma("unroll") for (int u = 0; u < WPT; ++u) { int c = tid + u * NT; if (c < WCH) { int row = c / (WB / 16), col = (c % (WB / 16)) * 16; \
      pw[u] = *(global const uint4*)(Wb + row * wrow + (size_t)(step) * WB + col); } } \
    ILOAD(step); SLOAD(step); } while (0)
#if SPARSE
#define ILOAD(step) _Pragma("unroll") for (int u = 0; u < IPT; ++u) { int c = tid + u * NT; if (c < ICH) { int row = c / (IDXB / 8), col = (c % (IDXB / 8)) * 8; \
      pi[u] = *(global const uint2*)(WI + (size_t)(n0 + row) * irow + (size_t)(step) * IDXB + col); } }
#else
#define ILOAD(step)
#endif
#if RESCALE
#define SC_N (2 * GPS * BN + 2 * GPS * BM)
#define SPT ((SC_N + NT - 1) / NT)
#define SLOAD(step) _Pragma("unroll") for (int u = 0; u < SPT; ++u) { int c = tid + u * NT; \
      if (c < GPS * BN) { pscale[u] = wscale[(size_t)(n0 + c % BN) * groups + (step) * GPS + c / BN]; } \
      else if (c < 2 * GPS * BN) { int d = c - GPS * BN; pscale[u] = wbias[(size_t)(n0 + d % BN) * groups + (step) * GPS + d / BN]; } \
      else if (c < 2 * GPS * BN + GPS * BM) { int d = c - 2 * GPS * BN; pscale[u] = xstep[(size_t)(m0 + d % BM) * groups + (step) * GPS + d / BM]; } \
      else if (c < SC_N) { int d = c - 2 * GPS * BN - GPS * BM; pscale[u] = xsum[(size_t)(m0 + d % BM) * groups + (step) * GPS + d / BM]; } }
#define SSTORE(buf) _Pragma("unroll") for (int u = 0; u < SPT; ++u) { int c = tid + u * NT; \
      if (c < GPS * BN) sWs[buf][c / BN][c % BN] = pscale[u]; \
      else if (c < 2 * GPS * BN) { int d = c - GPS * BN; sWb[buf][d / BN][d % BN] = pscale[u]; } \
      else if (c < 2 * GPS * BN + GPS * BM) { int d = c - 2 * GPS * BN; sXs[buf][d / BM][d % BM] = pscale[u]; } \
      else if (c < SC_N) { int d = c - 2 * GPS * BN - GPS * BM; sXu[buf][d / BM][d % BM] = pscale[u]; } }
#else
#define SLOAD(step)
#define SSTORE(buf)
#endif
#define LSTORE(buf) do { \
    _Pragma("unroll") for (int u = 0; u < XPT; ++u) { int c = tid + u * NT; if (c < XCH) { int row = c / (BKB / 16), col = (c % (BKB / 16)) * 16; \
      *(local uint4*)(sX[buf] + row * XS + col) = px[u]; } } \
    _Pragma("unroll") for (int u = 0; u < WPT; ++u) { int c = tid + u * NT; if (c < WCH) { int row = c / (WB / 16), col = (c % (WB / 16)) * 16; \
      *(local uint4*)(sW[buf] + row * WS + col) = pw[u]; } } \
    ISTORE(buf); SSTORE(buf); } while (0)
#if SPARSE
#define ISTORE(buf) _Pragma("unroll") for (int u = 0; u < IPT; ++u) { int c = tid + u * NT; if (c < ICH) { int row = c / (IDXB / 8), col = (c % (IDXB / 8)) * 8; \
      *(local uint2*)(sI[buf] + row * IDXB + col) = pi[u]; } }
#else
#define ISTORE(buf)
#endif

  GLOAD(0);
  LSTORE(0);
  wg_barrier();
  for (int s = 0; s < steps; ++s) {
    const int cur = s & 1;
    if (s + 1 < steps) GLOAD(s + 1);
    const local uchar* cw = sW[cur];
    const local uchar* cx = sX[cur];
    _Pragma("unroll") for (int g = 0; g < (RESCALE ? GPS : 1); ++g) {
#if RESCALE
      acc_t t[TN][TM];
      _Pragma("unroll") for (int a = 0; a < TN; ++a) _Pragma("unroll") for (int b = 0; b < TM; ++b) t[a][b] = (acc_t)(0);
#define ACC t
#define KLO (g * 64)
#define KHI (g * 64 + 64)
#else
#define ACC acc
#define KLO 0
#define KHI KPER
#endif
#if SPARSE
      _Pragma("unroll") for (int k = KLO; k < KHI; k += 32) {
        acc_t dummy;
        _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
          const int row = wn * TN * 16 + tn * 16 + r;
          const local uchar* wa = cw + row * WS + k / 2 + h * 8;
          uint idx = *(const local ushort*)(sI[cur] + row * IDXB + (k / 16 + h) * 2);
          _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
            const local uchar* xb = cx + (wm * TM * 16 + tm * 16 + r) * XS + k + h * 16;
            ACC[tn][tm] = mma(ACC[tn][tm], wa, xb, idx);
          }
        }
      }
#else
      _Pragma("unroll") for (int k = KLO; k < KHI; k += 16) {
        _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
          const local uchar* wa = cw + (wn * TN * 16 + tn * 16 + r) * WS + (k + h * 8) * EB;
          _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
            const local uchar* xb = cx + (wm * TM * 16 + tm * 16 + r) * XS + (k + h * 8) * EB;
            ACC[tn][tm] = mma(ACC[tn][tm], wa, xb, 0);
          }
        }
      }
#endif
#if RESCALE
      _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
        const int mm = wm * TM * 16 + tm * 16 + r;
        const float xs = sXs[cur][g][mm], xu = sXu[cur][g][mm];
        _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
          const int nb = wn * TN * 16 + tn * 16 + 8 * h;
          f8 ws = *(const local f8*)(&sWs[cur][g][nb]), wb = *(const local f8*)(&sWb[cur][g][nb]);
#if INTACC
          f8 d = __builtin_convertvector(t[tn][tm], f8);
#else
          f8 d = t[tn][tm];
#endif
#if RESCALE == 2
          // Per-token activation scale and the bias x row-sum term are applied once after
          // the K loop; only the per-group weight scale stays inside it.
          accf[tn][tm] = d * ws + accf[tn][tm];
#else
          accf[tn][tm] = d * (ws * xs) + (wb * xu + accf[tn][tm]);
#endif
        }
      }
#endif
    }
    if (s + 1 < steps) LSTORE(cur ^ 1);
    wg_barrier();
  }
  for (int tn = 0; tn < TN; ++tn) for (int tm = 0; tm < TM; ++tm) {
    const int m = m0 + wm * TM * 16 + tm * 16 + r;
    const int n = n0 + wn * TN * 16 + tn * 16 + 8 * h;
#if RESCALE
    f8 o = accf[tn][tm];
#elif INTACC
    f8 o = __builtin_convertvector(acc[tn][tm], f8);
#else
    f8 o = acc[tn][tm];
#endif
    *(global f8*)(Y + (size_t)m * N + n) = o;
  }
}
