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

constexpr std::array<graph::PrimitiveSource, 31> kHipSources{{
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

    // Matrix core. This is the gfx11 wave32 form: A and B are 16-wide f16
    // fragments, C/D an 8-wide f32 accumulator. RDNA4 spells the same operation
    // with a different builtin and a narrower operand, so it belongs in a table
    // keyed by target rather than in the kernel that calls it.
    {"wmma.f32.16x16x16.f16",
     "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32($0, $1, $2)"},

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
