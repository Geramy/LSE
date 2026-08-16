// HIP spellings for the built-in primitives.
//
// This file is the whole of what "add in HIP" means to the engine. A second
// backend supplies its own table under its own directory and shares nothing
// with this one; the primitives themselves carry no device text.
//
// `$0`, `$1` are input values (already widened to float). `$a0`..`$a3` are the
// node's float attrs, spliced as literals by the emitter.
#include "lse/backends/hrx/hip_sources.hpp"

#include <array>

namespace lse::backend {

namespace {

constexpr std::array<graph::PrimitiveSource, 55> kHipSources{{
    {"add", "$0 + $1"},
    {"sub", "$0 - $1"},
    {"mul", "$0 * $1"},
    {"div", "$0 / $1"},
    {"eq", "($0 == $1 ? 1.0f : 0.0f)"},
    {"ge", "($0 >= $1 ? 1.0f : 0.0f)"},

    {"neg", "-$0"},
    {"exp", "__expf($0)"},
    {"log", "__logf($0)"},
    {"sqrt", "sqrtf($0)"},
    {"rsqrt", "rsqrtf($0)"},
    {"sigmoid", "1.0f / (1.0f + __expf(-$0))"},
    {"silu", "$0 / (1.0f + __expf(-$0))"},
    {"tanh", "tanhf($0)"},
    {"relu", "fmaxf($0, 0.0f)"},

    // log1p(exp(x)); the branch keeps exp from overflowing past ~88.
    {"softplus", "($0 > 20.0f ? $0 : __logf(1.0f + __expf($0)))"},

    // tanh approximation, matching what the training stack uses.
    {"gelu",
     "0.5f * $0 * (1.0f + tanhf(0.7978845608f * ($0 + 0.044715f * $0 * $0 * $0)))"},

    {"clamp", "fminf(fmaxf($0, $a0), $a1)"},

    // Named so a kernel written against kir asks for the operation and gets
    // this target's spelling, rather than writing fmaf/fmaxf itself.
    {"fma", "fmaf($0, $1, $2)"},
    {"max", "fmaxf($0, $1)"},
    {"min", "fminf($0, $1)"},
    {"neg_inf", "-INFINITY"},
    {"abs", "fabsf($0)"},

    // roundf, not rintf: the block codecs must reproduce quant::QuantScheme's
    // host pack bit-for-bit, and that uses std::round (half away from zero).
    // rintf rounds half to even and would disagree on exact .5 quants.
    {"round", "roundf($0)"},

    // A narrow float seen as its bits, and back. Spelled with
    // __builtin_bit_cast rather than __half_as_ushort so it needs no HIP fp16
    // header and holds on any AMD target; the conversion itself is the
    // hardware's round-to-nearest-even, which is what float16_t::from_float
    // implements. A bf16 wire format would be two more rows here.
    {"bits.f16",
     "((unsigned int)__builtin_bit_cast(unsigned short, (_Float16)($0)))"},
    {"value.f16",
     "((float)__builtin_bit_cast(_Float16, (unsigned short)($0)))"},

    // Matrix core. Spelling only — the widths, the K step and the lane
    // mappings are the descriptor's, in lse::math's table, and a kernel reads
    // them from there. Which row a kernel asks for follows the storage type of
    // its operands and the device's generation, so the same operation appears
    // once per generation here rather than once in a kernel.
    //
    // gfx11 wave32: A and B 16-wide fragments, C/D an 8-wide f32 accumulator.
    {"wmma.f32.16x16x16.f16",
     "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32($0, $1, $2)"},
    {"wmma.f32.16x16x16.bf16",
     "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32($0, $1, $2)"},

    // Narrow-accumulate forms. The trailing false is opsel: which half of the
    // 16-wide accumulator the 8 results land in.
    {"wmma.f16.16x16x16.f16",
     "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32($0, $1, $2, false)"},
    {"wmma.bf16.16x16x16.bf16",
     "__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32($0, $1, $2, false)"},

    // Integer operands, i32 accumulator. The leading trues say both operands
    // are signed and the trailing false disables output clamping; a kernel
    // never sees those, which is the point of the row carrying the spelling.
    {"wmma.i32.16x16x16.iu8",
     "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, $0, true, $1, $2, false)"},
    {"wmma.i32.16x16x16.iu4",
     "__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, $0, true, $1, $2, false)"},

    // gfx12 (RDNA4) wave32: same operations, a suffixed builtin and half the
    // operand width. Present so a gfx1201 bring-up is a measurement rather
    // than a port; the descriptor still refuses to emit an unmeasured layout.
    {"wmma12.f32.16x16x16.f16",
     "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12($0, $1, $2)"},
    {"wmma12.f32.16x16x16.bf16",
     "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12($0, $1, $2)"},
    {"wmma12.f16.16x16x16.f16",
     "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12($0, $1, $2)"},
    {"wmma12.bf16.16x16x16.bf16",
     "__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12($0, $1, $2)"},
    {"wmma12.i32.16x16x16.iu8",
     "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, $0, true, $1, $2, false)"},
    {"wmma12.i32.16x16x32.iu4",
     "__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, $0, true, $1, $2, false)"},
    {"wmma12.f32.16x16x16.fp8_fp8",
     "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12($0, $1, $2)"},
    {"wmma12.f32.16x16x16.bf8_bf8",
     "__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12($0, $1, $2)"},

    // CDNA3 MFMA, wave64. A different instruction family, not a wider WMMA:
    // 64 lanes cooperate and the trailing 0,0,0 are cbsz/abid/blgp.
    {"mfma.f32.16x16x16.f16",
     "__builtin_amdgcn_mfma_f32_16x16x16f16($0, $1, $2, 0, 0, 0)"},
    {"mfma.f32.32x32x8.f16",
     "__builtin_amdgcn_mfma_f32_32x32x8f16($0, $1, $2, 0, 0, 0)"},
    {"mfma.f32.16x16x16.bf16",
     "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k($0, $1, $2, 0, 0, 0)"},
    {"mfma.f32.32x32x8.bf16",
     "__builtin_amdgcn_mfma_f32_32x32x8bf16_1k($0, $1, $2, 0, 0, 0)"},
    {"mfma.i32.16x16x32.i8",
     "__builtin_amdgcn_mfma_i32_16x16x32_i8($0, $1, $2, 0, 0, 0)"},
    {"mfma.f32.16x16x32.fp8_fp8",
     "__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8($0, $1, $2, 0, 0, 0)"},
    {"mfma.f32.32x32x16.fp8_fp8",
     "__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8($0, $1, $2, 0, 0, 0)"},

    {"thread.local_id", "threadIdx.x"},
    {"thread.workgroup_id.x", "blockIdx.x"},
    {"thread.workgroup_id.y", "blockIdx.y"},
    {"thread.workgroup_size", "blockDim.x"},
    {"thread.grid_dim.x", "gridDim.x"},
    {"wave.shfl_xor", "__shfl_xor($0, (int)($1))"},
    {"barrier", "__syncthreads()"},
    {"shared", "__shared__"},
}};

}  // namespace

graph::DialectSourceTable hip_sources() noexcept {
  return graph::DialectSourceTable(kHipSources);
}

}  // namespace lse::backend
