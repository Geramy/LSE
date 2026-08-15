// Element names every backend can speak. Not host storage — that is
// float16_t / bfloat16_t in dtype.hpp — and not a graph concept. A kernel
// names `f16` the way it names `float`; the backend table spells it.
#pragma once

namespace lse {

struct f16 {};
struct bf16 {};

template <typename T, int N>
struct vec {};

using f32 = float;

}  // namespace lse
