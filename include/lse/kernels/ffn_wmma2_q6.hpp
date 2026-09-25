#pragma once
#include "lse/graph/kernel_primitive.hpp"
namespace lse::kernels {
// The M=512 FFN GEMM candidate selector (ffn_wmma2_q6.cpp). Build 199:
// residual-2 is the default for the M=512 FFN 17408 shapes; declines to the
// production path when any gate refuses (device, wave, shape, capability,
// LSE_FFN_WMMA2=0, LSE_FFN_WMMA2_SCALAR). LSE_FFN_WMMA2=single selects the
// single-product variant (Build 198 A/B only).
const graph::KernelPrimitiveBase* ffn_wmma2_q6_for(const graph::KernelShapes& s);
}
