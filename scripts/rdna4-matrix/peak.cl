// Register-resident WMMA/SWMMAC throughput kernels for gfx12 (wave32).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Each wave runs CH independent accumulator chains for `iters` iterations.
typedef half h8 __attribute__((ext_vector_type(8)));
typedef half h16 __attribute__((ext_vector_type(16)));
typedef short s8 __attribute__((ext_vector_type(8)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef int i8 __attribute__((ext_vector_type(8)));
#define CH 8
#define LANE() ((int)__builtin_amdgcn_workitem_id_x())
#define SEED() (LANE() + 1)

#define PEAK(name, AT, BT, CT, ainit, binit, OP) \
kernel void name(global int* out, int iters, int flag) { \
  AT a = ainit; BT b = binit; CT c[CH]; \
  for (int j = 0; j < CH; ++j) c[j] = (CT)(0); \
  for (int i = 0; i < iters; ++i) { \
    _Pragma("unroll") for (int j = 0; j < CH; ++j) c[j] = OP; \
  } \
  if (flag) { CT s = c[0]; for (int j = 1; j < CH; ++j) s += c[j]; \
    global CT* o = (global CT*)out; o[__builtin_amdgcn_workgroup_id_x()*256 + LANE()] = s; } \
}
#define H8 ((h8)((half)SEED()*(half)0.001f))
#define H16 ((h16)((half)SEED()*(half)0.001f))
#define S8 ((s8)((short)SEED()))
#define S16 ((s16)((short)SEED()))
#define I1 (SEED()*0x01010101)
#define I2 ((i2)(SEED()*0x01010101))
#define I4 ((i4)(SEED()*0x01010101))

PEAK(w_f16_f32,  h8, h8, f8, H8, H8, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c[j]))
PEAK(w_bf16_f32, s8, s8, f8, S8, S8, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c[j]))
PEAK(w_f16_f16,  h8, h8, h8, H8, H8, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a,b,c[j]))
PEAK(w_bf16_bf16,s8, s8, s8, S8, S8, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(a,b,c[j]))
PEAK(w_fp8_fp8,  i2, i2, f8, I2, I2, __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c[j]))
PEAK(w_bf8_bf8,  i2, i2, f8, I2, I2, __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a,b,c[j]))
PEAK(w_fp8_bf8,  i2, i2, f8, I2, I2, __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(a,b,c[j]))
PEAK(w_iu8,      i2, i2, i8, I2, I2, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a,true,b,c[j],false))
PEAK(w_iu4_k16,  int, int, i8, I1, I1, __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true,a,true,b,c[j],false))
PEAK(w_iu4_k32,  i2, i2, i8, I2, I2, __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,a,true,b,c[j],false))
// Sparse: A is the 2:4-compressed operand, B is dense at 2x K; index VGPR.
PEAK(s_f16_f32,  h8, h16, f8, H8, H16, __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a,b,c[j],0x4E4E4E4E))
PEAK(s_bf16_f32, s8, s16, f8, S8, S16, __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a,b,c[j],0x4E4E4E4E))
PEAK(s_f16_f16,  h8, h16, h8, H8, H16, __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(a,b,c[j],0x4E4E4E4E))
PEAK(s_fp8_fp8,  i2, i4, f8, I2, I4, __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a,b,c[j],0x4E4E4E4E))
PEAK(s_bf8_bf8,  i2, i4, f8, I2, I4, __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(a,b,c[j],0x4E4E4E4E))
PEAK(s_iu8,      i2, i4, i8, I2, I4, __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true,a,true,b,c[j],0x4E4E4E4E,false))
PEAK(s_iu4_k32,  int, i2, i8, I1, I2, __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(true,a,true,b,c[j],0x4E4E4E4E,false))
PEAK(s_iu4_k64,  i2, i4, i8, I2, I4, __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true,a,true,b,c[j],0x4E4E4E4E,false))
