#include "lse/ops/activation.hpp"

#include "lse/graph/ops.hpp"

namespace lse::ops {

namespace {
Array gated(const Array& x, const Array& w_gate, const Array& w_up,
            const Array& w_down, Array (*act)(const Array&)) {
  Array g = graph::linear(x, w_gate);
  Array u = graph::linear(x, w_up);
  return graph::linear(act(g) * u, w_down);
}
}  // namespace

Array swiglu(const Array& x, const Array& w_gate, const Array& w_up,
             const Array& w_down) {
  return gated(x, w_gate, w_up, w_down, graph::silu);
}

Array geglu(const Array& x, const Array& w_gate, const Array& w_up,
            const Array& w_down) {
  return gated(x, w_gate, w_up, w_down, graph::gelu);
}

}  // namespace lse::ops
