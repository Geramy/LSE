// Gated activation families. Model-agnostic.
#pragma once

#include "lse/graph/graph.hpp"

namespace lse::ops {

using graph::Array;

// down(silu(gate(x)) * up(x)). lemonseed names these w1/w3/w2; Qwen3.6 names
// them gate_proj/up_proj/down_proj. Same math.
Array swiglu(const Array& x, const Array& w_gate, const Array& w_up,
             const Array& w_down);

// Same shape with GELU in place of SiLU.
Array geglu(const Array& x, const Array& w_gate, const Array& w_up,
            const Array& w_down);

}  // namespace lse::ops
