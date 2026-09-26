// cand-fp8-mma: the M=512 FFN GEMM candidate on the fp8 packed WMMA.
//
// quant_linear.q6_fp8_mma_prefill -- the matrix-core path with 8-bit packed
// operands: v_wmma_f32_16x16x16_fp8_fp8 (the gfx1201 row
// `wmma12.f32.16x16x16.fp8_fp8`), f32 accumulate, 128-thread workgroup /
// 4 waves / 64x64 tile, the exact geometry of the 13.7 TFLOPS bf16 record
// (cand-ffn-wmma2 residual-2).
//
// Selection seam: LSE_FFN_FP8_MMA=1 selects this kernel for the M=512 FFN
// whitelist; anything else (unset, "0") declines to the wmma2 selector, so
// the default path is bit-exact against the bf16 record when the seam is off.
//
// Rounding contract (documented, measured separately in the checks):
//   weights:   Q6 code -> fp32 dequant (code*scale+bias, the exact scalar
//              body value) -> e4m3 with OCP SATFINITE clamp to 448.0 and
//              round-to-nearest-even (the device `pack4.fp8.ocp` builtin is
//              RNE; the host oracle math::fp8_bits is the same rule).
//              Conversion happens per 64-element group at the group's own
//              scale, i.e. the rounding unit is the per-64-block weight.
//   activations: x (f32) -> e4m3, same clamp/RNE, per element, packed four to
//              an i32 register; the f32 scale is the constant 1.0 (the
//              fp8 range covers the bounded FFN activation range once the
//              448 clamp is in; no per-K-group rescale).
//   accumulate: f32 in-core and across the whole K, never re-rounded until
//              the store.
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::kernels {

const graph::KernelPrimitiveBase* ffn_fp8_q6_for(const graph::KernelShapes& s);

}  // namespace lse::kernels
