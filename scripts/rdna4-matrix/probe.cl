#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half h8 __attribute__((ext_vector_type(8)));
typedef half h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef int i8 __attribute__((ext_vector_type(8)));
#define L ((int)__builtin_amdgcn_workitem_id_x())
kernel void probe_w_iu8(global const int* A, global const int* B, global const int* I, global int* D) {
  i2 a = ((global const i2*)A)[L]; i2 b = ((global const i2*)B)[L]; i8 c = (i8)(0);
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false); ((global i8*)D)[L] = c; }
kernel void probe_s_iu8(global const int* A, global const int* B, global const int* I, global int* D) {
  i2 a = ((global const i2*)A)[L]; i4 b = ((global const i4*)B)[L]; i8 c = (i8)(0);
  c = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a, true, b, c, I[L], false); ((global i8*)D)[L] = c; }
kernel void probe_s_fp8(global const int* A, global const int* B, global const int* I, global float* D) {
  i2 a = ((global const i2*)A)[L]; i4 b = ((global const i4*)B)[L]; f8 c = (f8)(0);
  c = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c, I[L]); ((global f8*)D)[L] = c; }
kernel void probe_w_fp8(global const int* A, global const int* B, global const int* I, global float* D) {
  i2 a = ((global const i2*)A)[L]; i2 b = ((global const i2*)B)[L]; f8 c = (f8)(0);
  c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c); ((global f8*)D)[L] = c; }
kernel void probe_s_f16(global const half* A, global const half* B, global const int* I, global float* D) {
  h8 a = ((global const h8*)A)[L]; h16 b = ((global const h16*)B)[L]; f8 c = (f8)(0);
  c = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c, I[L]); ((global f8*)D)[L] = c; }
