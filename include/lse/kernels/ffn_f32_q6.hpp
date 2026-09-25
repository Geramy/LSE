#pragma once
#include "lse/graph/kernel_primitive.hpp"
namespace lse::kernels {
// The M=512 FFN GEMM f32 candidate selector (ffn_f32_q6.cpp). Declines to
// the production path when any gate refuses (device, wave32, M=512, shape
// whitelist, or the LSE_FFN_F32 / LSE_FFN_F32_SCALAR env gates).
const graph::KernelPrimitiveBase* ffn_f32_q6_for(const graph::KernelShapes& s);
// The f32 schedule sweep's two v1 candidates (ffn_f32_q6_v1.cpp). Both
// decline identically to v0 on every gate except the v1 env:
//   LSE_FFN_F32_V1=v1a   select v1a (DRAM-streamed weights, 2-stage x,
//                        32x64 tile, 128 threads, 16 accs/lane)
//   LSE_FFN_F32_V1=v1b   select v1b (LDS-both panels, 64x64 tile,
//                        2-stage x, 128 threads, 4 accs/lane)
// any other value (unset, 0, v0): decline to the v0 kernel.
const graph::KernelPrimitiveBase* ffn_f32_q6_v1a_for(const graph::KernelShapes& s);
const graph::KernelPrimitiveBase* ffn_f32_q6_v1b_for(const graph::KernelShapes& s);
}  // namespace lse::kernels
