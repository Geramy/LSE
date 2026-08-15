// Built-in elementwise ops, defined as ordinary primitives.
//
// There is no privileged path: `add` and a user-registered primitive are the
// same kind of object, fuse by the same rule, and emit source the same way.
// Adding an op is one struct plus one registration line.
//
// Only the name and the host reference live here. Device source is a backend's
// business — the HIP spellings are in backends/hrx/hip_sources.cpp, keyed by
// the same names.
#include <cmath>

#include "lse/graph/codegen.hpp"
#include "lse/graph/primitive.hpp"

namespace lse::graph::builtin {

#define LSE_UNARY(Type, name_, expr_)                                \
  struct Type final : ElementwisePrimitive<Type, 1> {                \
    static constexpr std::string_view kName = name_;                 \
    static float apply(float x) { return (expr_); }                  \
  };                                                                 \
  LSE_REGISTER_PRIMITIVE(Type)

#define LSE_BINARY(Type, name_, expr_)                               \
  struct Type final : ElementwisePrimitive<Type, 2> {                \
    static constexpr std::string_view kName = name_;                 \
    static float apply(float a, float b) { return (expr_); }         \
  };                                                                 \
  LSE_REGISTER_PRIMITIVE(Type)

LSE_BINARY(Add, "add", a + b);
LSE_BINARY(Sub, "sub", a - b);
LSE_BINARY(Mul, "mul", a * b);
LSE_BINARY(Div, "div", a / b);
LSE_BINARY(Eq, "eq", a == b ? 1.0f : 0.0f);
LSE_BINARY(Ge, "ge", a >= b ? 1.0f : 0.0f);

LSE_UNARY(Neg, "neg", -x);
LSE_UNARY(Exp, "exp", std::exp(x));
LSE_UNARY(Log, "log", std::log(x));
LSE_UNARY(Sqrt, "sqrt", std::sqrt(x));
LSE_UNARY(Rsqrt, "rsqrt", 1.0f / std::sqrt(x));
LSE_UNARY(Sigmoid, "sigmoid",
          1.0f / (1.0f + std::exp(-x)));
LSE_UNARY(SiLU, "silu", x / (1.0f + std::exp(-x)));
LSE_UNARY(Tanh, "tanh", std::tanh(x));
LSE_UNARY(ReLU, "relu", x > 0.0f ? x : 0.0f);

// log1p(exp(x)); the branch keeps exp from overflowing past ~88.
LSE_UNARY(Softplus, "softplus",
          x > 20.0f ? x : std::log1p(std::exp(x)));

// tanh approximation, matching what the training stack uses.
LSE_UNARY(GELU, "gelu",
          0.5f * x * (1.0f + std::tanh(0.7978845608f *
                                       (x + 0.044715f * x * x * x))));

#undef LSE_UNARY
#undef LSE_BINARY

// Clamp reads its bounds from the node's attrs rather than the input list, so
// it overrides eval_cpu instead of using the arity-based default.
struct Clamp final : ElementwisePrimitive<Clamp, 1> {
  static constexpr std::string_view kName = "clamp";
  static float apply(float x) { return x; }

  void eval_cpu(std::span<const float* const> inputs, float* out,
                std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const float lo = attrs[0];
    const float hi = attrs[1];
    for (std::size_t i = 0; i < count; ++i) {
      const float v = inputs[0][i];
      out[i] = v < lo ? lo : (v > hi ? hi : v);
    }
  }
};
LSE_REGISTER_PRIMITIVE(Clamp);

}  // namespace lse::graph::builtin
