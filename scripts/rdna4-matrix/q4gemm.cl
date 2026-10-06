// Q4 (MLX affine, group 64) prefill GEMM prototypes for gfx12.
//   Y[M][N] = X[M][K] * dequant(W)[N][K]^T,  dequant(W)[n][k] = s[n][g] * q[n][k] + b[n][g]
// QMODE 0 (current LSE semantics): INT8 activations per (token, group-64), u8 codes, iu8 WMMA,
//          full group epilogue: y += s*xstep*float(dot) + b*xsum.
// QMODE 1 (proposed): FP8 E4M3 activations with one scale per token, Q4 codes as exact E4M3
//          bytes, fp8 WMMA with f32 accumulate, light group epilogue: y' += s*dot + b*xsum/xscale,
//          y = xscale * y' once at the end.
// Weights stay 4-bit in memory; each WG expands its W tile to 8-bit codes once, in LDS.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i8 __attribute__((ext_vector_type(8)));

#ifndef QMODE
#define QMODE 1
#endif
#ifndef BIASTAIL
#define BIASTAIL 1   // bias x activation-sum term as an f16 WMMA tail after the K loop
#endif
typedef half h8 __attribute__((ext_vector_type(8)));
#define WN 4
#define WM 2
#define TN 2
#define TM 4
#define NT 256
#define BN (WN * TN * 16)
#define BM (WM * TM * 16)
#define BK 64                   // one quantisation group per step
#define XS (BK + 16)
#define WS (BK + 16)
#if QMODE == 0
typedef i8 acc_t;
#else
typedef f8 acc_t;
#endif

#define LID ((int)__builtin_amdgcn_workitem_id_x())
static inline void wg_barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
static inline float bf16f(ushort v) { return as_float((uint)v << 16); }

// 8 packed nibbles (k order, low nibble first) -> two words of 8-bit codes in k order.
static inline uint2 nibbles_to_bytes(uint w) {
  uint lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;  // lo: n0 n2 n4 n6, hi: n1 n3 n5 n7
  return (uint2)(__builtin_amdgcn_perm(hi, lo, 0x05010400u), __builtin_amdgcn_perm(hi, lo, 0x07030602u));
}
#if QMODE == 1
// Exact E4M3 encodings of the integers 0..15, by byte lookup.
static inline uint codes_to_e4m3(uint v) {
  const uint sel = v & 0x07070707u;
  const uint a = __builtin_amdgcn_perm(0x4E4C4A48u, 0x44403800u, sel);   // 0..7
  const uint b = __builtin_amdgcn_perm(0x57565554u, 0x53525150u, sel);   // 8..15
  const uint m = ((v >> 3) & 0x01010101u) * 0xFFu;
  return (a & ~m) | (b & m);
}
#endif

kernel __attribute__((reqd_work_group_size(NT, 1, 1)))
void q4gemm(global const uint* Wq, global const ushort* Ws, global const ushort* Wb,
            global const uchar* Xq, global const float* xstep, global const float* xsum,
            global const float* xscale, global float* Y,
            global const half* wb16, global const half* su16, int M, int N, int K) {
  local uchar sW[2][BN * WS] __attribute__((aligned(32)));
  local uchar sX[2][BM * XS] __attribute__((aligned(32)));
  local float sWs[2][BN] __attribute__((aligned(32))), sWb[2][BN] __attribute__((aligned(32))), sXs[2][BM] __attribute__((aligned(32))), sXu[2][BM] __attribute__((aligned(32)));
  const int tid = LID, w = tid >> 5, l = tid & 31, h = l >> 4, r = l & 15;
  const int wn = w % WN, wm = w / WN;
  const int m0 = (int)__builtin_amdgcn_workgroup_id_x() * BM;   // M fastest: W tile streams once
  const int n0 = (int)__builtin_amdgcn_workgroup_id_y() * BN;
  const int G = K / 64;

  f8 accf[TN][TM];
  for (int a = 0; a < TN; ++a) for (int b = 0; b < TM; ++b) accf[a][b] = (f8)(0.0f);
  uint4 pw, px[2]; float pscale;

#define GLOAD(g) do { \
    { int row = tid >> 1, hf = tid & 1; pw = *(global const uint4*)(Wq + (size_t)(n0 + row) * (K / 8) + (g) * 8 + hf * 4); } \
    for (int u = 0; u < 2; ++u) { int c = tid + u * NT; int row = c >> 2, col = (c & 3) * 16; \
      px[u] = *(global const uint4*)(Xq + (size_t)(m0 + row) * K + (size_t)(g) * BK + col); } \
    if (tid < BN) pscale = bf16f(Ws[(size_t)(n0 + tid) * G + (g)]); \
    else pscale = bf16f(Wb[(size_t)(n0 + tid - BN) * G + (g)]); \
  } while (0)
  // xstep/xsum are tiny; read them straight into LDS at store time.
#define LSTORE(buf, g) do { \
    { int row = tid >> 1, hf = tid & 1; uint4 o0, o1; uint2 t; \
      t = nibbles_to_bytes(pw.x); o0.x = t.x; o0.y = t.y; t = nibbles_to_bytes(pw.y); o0.z = t.x; o0.w = t.y; \
      t = nibbles_to_bytes(pw.z); o1.x = t.x; o1.y = t.y; t = nibbles_to_bytes(pw.w); o1.z = t.x; o1.w = t.y; \
      CONV(o0); CONV(o1); \
      *(local uint4*)(sW[buf] + row * WS + hf * 32) = o0; *(local uint4*)(sW[buf] + row * WS + hf * 32 + 16) = o1; } \
    for (int u = 0; u < 2; ++u) { int c = tid + u * NT; int row = c >> 2, col = (c & 3) * 16; \
      *(local uint4*)(sX[buf] + row * XS + col) = px[u]; } \
    if (tid < BN) sWs[buf][tid] = pscale; else sWb[buf][tid - BN] = pscale; \
    if (tid < BM) { sXs[buf][tid] = xstep[(size_t)(m0 + tid) * G + (g)]; sXu[buf][tid] = xsum[(size_t)(m0 + tid) * G + (g)]; } \
  } while (0)
#if QMODE == 1
#define CONV(o) do { o.x = codes_to_e4m3(o.x); o.y = codes_to_e4m3(o.y); o.z = codes_to_e4m3(o.z); o.w = codes_to_e4m3(o.w); } while (0)
#else
#define CONV(o) do {} while (0)
#endif

  GLOAD(0); LSTORE(0, 0); wg_barrier();
  for (int g = 0; g < G; ++g) {
    const int cur = g & 1;
    if (g + 1 < G) GLOAD(g + 1);
    acc_t t[TN][TM];
    _Pragma("unroll") for (int a = 0; a < TN; ++a) _Pragma("unroll") for (int b = 0; b < TM; ++b) t[a][b] = (acc_t)(0);
    _Pragma("unroll") for (int k = 0; k < BK; k += 16) {
      _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
        i2 wa = *(const local i2*)(sW[cur] + (wn * TN * 16 + tn * 16 + r) * WS + k + h * 8);
        _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
          i2 xb = *(const local i2*)(sX[cur] + (wm * TM * 16 + tm * 16 + r) * XS + k + h * 8);
#if QMODE == 0
          // A = unsigned Q4 codes, B = signed INT8 activations.
          t[tn][tm] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, wa, true, xb, t[tn][tm], false);
#else
          t[tn][tm] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wa, xb, t[tn][tm]);
#endif
        }
      }
    }
    _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
      const int mm = wm * TM * 16 + tm * 16 + r;
      const float xs = sXs[cur][mm], xu = sXu[cur][mm];
      _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
        const int nb = wn * TN * 16 + tn * 16 + 8 * h;
        f8 ws = *(const local f8*)(&sWs[cur][nb]), wb = *(const local f8*)(&sWb[cur][nb]);
#if BIASTAIL && QMODE == 0
        accf[tn][tm] = __builtin_convertvector(t[tn][tm], f8) * (ws * xs) + accf[tn][tm];
#elif BIASTAIL
        accf[tn][tm] = t[tn][tm] * ws + accf[tn][tm];
#elif QMODE == 0
        accf[tn][tm] = __builtin_convertvector(t[tn][tm], f8) * (ws * xs) + (wb * xu + accf[tn][tm]);
#else
        accf[tn][tm] = t[tn][tm] * ws + (wb * xu + accf[tn][tm]);
#endif
      }
    }
    if (g + 1 < G) LSTORE(cur ^ 1, g + 1);
    wg_barrier();
  }
#if BIASTAIL
  // sum_g b[n][g] * xsum[m][g]: a K = G matrix product on the f16 path, f32 accumulate.
  for (int gg = 0; gg < G; gg += 16) {
    _Pragma("unroll") for (int tn = 0; tn < TN; ++tn) {
      h8 a = *(global const h8*)(wb16 + (size_t)(n0 + wn * TN * 16 + tn * 16 + r) * G + gg + h * 8);
      _Pragma("unroll") for (int tm = 0; tm < TM; ++tm) {
        h8 b = *(global const h8*)(su16 + (size_t)(m0 + wm * TM * 16 + tm * 16 + r) * G + gg + h * 8);
        accf[tn][tm] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, accf[tn][tm]);
      }
    }
  }
#endif
  for (int tn = 0; tn < TN; ++tn) for (int tm = 0; tm < TM; ++tm) {
    const int m = m0 + wm * TM * 16 + tm * 16 + r;
    const int n = n0 + wn * TN * 16 + tn * 16 + 8 * h;
#if QMODE == 1
    f8 o = accf[tn][tm] * xscale[m];
#else
    f8 o = accf[tn][tm];
#endif
    *(global f8*)(Y + (size_t)m * N + n) = o;
  }
}

// ---------------------------------------------------------------- activation pre-passes
// INT8 per (token, group-64): one wave per (row, group); lane holds 2 values.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void act_int8_g64(global const float* X, global uchar* Xq, global float* xstep, global float* xsum, global half* su16, int M, int K) {
  const int G = K / 64;
  const int item = (int)__builtin_amdgcn_workgroup_id_x() * 8 + (LID >> 5);   // (row, group)
  if (item >= M * G) return;
  const int row = item / G, g = item % G, l = LID & 31;
  const float2 v = *(global const float2*)(X + (size_t)row * K + g * 64 + l * 2);
  float amax = fmax(fabs(v.x), fabs(v.y)), s = v.x + v.y;
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmax(amax, __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l ^ o) << 2, __builtin_bit_cast(int, amax))));
    s += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l ^ o) << 2, __builtin_bit_cast(int, s)));
  }
  const float inv = 127.0f / fmax(amax, 1e-30f);
  const int a = (int)rint(v.x * inv), b = (int)rint(v.y * inv);
  // The bias term uses the sum of the quantised values: its error then cancels against the
  // dot term's (both see the same dequantised weights); the raw sum roughly triples the error.
  int qs = a + b;
  for (int o = 16; o > 0; o >>= 1) qs += __builtin_amdgcn_ds_bpermute((l ^ o) << 2, qs);
  s = (float)qs * (amax / 127.0f);
  *(global ushort*)(Xq + (size_t)row * K + g * 64 + l * 2) = (ushort)((a & 255) | ((b & 255) << 8));
  if (l == 0) { xstep[(size_t)row * G + g] = amax / 127.0f; xsum[(size_t)row * G + g] = s; su16[(size_t)row * G + g] = (half)s; }
}

// FP8 E4M3 with one scale per token: one WG per row. xsum holds group sums divided by the
// token scale, so the bias term shares the accumulator that the token scale multiplies.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void act_fp8_row(global const float* X, global uchar* Xq, global float* xscale, global float* xsum, global half* su16, int M, int K) {
  local float red[8];
  const int row = (int)__builtin_amdgcn_workgroup_id_x(), tid = LID, l = tid & 31, w = tid >> 5;
  const int G = K / 64;
  global const float* x = X + (size_t)row * K;
  float amax = 0.0f;
  for (int k = tid * 4; k < K; k += 256 * 4) { float4 v = *(global const float4*)(x + k);
    amax = fmax(amax, fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w)))); }
  for (int o = 16; o > 0; o >>= 1)
    amax = fmax(amax, __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l ^ o) << 2, __builtin_bit_cast(int, amax))));
  if (l == 0) red[w] = amax;
  wg_barrier();
  amax = red[0]; for (int i = 1; i < 8; ++i) amax = fmax(amax, red[i]);
  const float scale = fmax(amax, 1e-30f) / 448.0f, inv = 1.0f / scale;
  if (tid == 0) xscale[row] = scale;
  // One wave per group of 64: lane holds 2 values.
  for (int g = w; g < G; g += 8) {
    const float2 v = *(global const float2*)(x + g * 64 + l * 2);
    float s = v.x + v.y;
    for (int o = 16; o > 0; o >>= 1) s += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l ^ o) << 2, __builtin_bit_cast(int, s)));
    const float a = __builtin_fminf(__builtin_fmaxf(v.x * inv, -448.0f), 448.0f), b = __builtin_fminf(__builtin_fmaxf(v.y * inv, -448.0f), 448.0f);
    const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    // Sum of the quantised values (see act_int8_g64), already in units of the token scale.
    s = __builtin_amdgcn_cvt_f32_fp8(pk, 0) + __builtin_amdgcn_cvt_f32_fp8(pk, 1);
    for (int o = 16; o > 0; o >>= 1) s += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l ^ o) << 2, __builtin_bit_cast(int, s)));
    *(global ushort*)(Xq + (size_t)row * K + g * 64 + l * 2) = (ushort)(pk & 0xFFFF);
    if (l == 0) { xsum[(size_t)row * G + g] = s; su16[(size_t)row * G + g] = (half)s; }
  }
}
