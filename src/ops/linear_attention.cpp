#include "lse/ops/linear_attention.hpp"

#include "lse/graph/ops.hpp"

namespace lse::ops {

namespace {

// beta off the 0/1 rails: a saturated beta drives the delta-rule gain up until
// the recurrent state overflows to inf, then NaN.
constexpr float kBetaFloor = 1e-4f;

// Runs the depthwise causal conv over x with `tail` standing in for the zero
// pad, leaving `tail` holding the last kernel-1 inputs for the next call —
// no concat of tail ++ x and no slices, which is what keeps the decode-path
// copy count down. A null or empty tail is the stateless full-sequence path.
Array conv_stream(const Array& x, const Array& weight, const Array& bias,
                  bool has_bias, Array* tail) {
  const Array b = has_bias
                      ? bias
                      : Array::zeros(Shape{weight.shape().dim(0)}, x.dtype());
  if (tail == nullptr || !tail->valid()) return graph::causal_conv1d(x, weight, b);
  const Array prev = *tail;
  *tail = graph::conv_tail(prev, x);
  return graph::causal_conv1d(x, weight, b, prev);
}

}  // namespace

Result<Array> gated_delta_net(const Array& x, const GatedDeltaNetWeights& w,
                              const GatedDeltaNetSpec& spec,
                              GatedDeltaNetState* state) {
  if (spec.key_heads <= 0 || spec.value_heads <= 0 ||
      spec.value_heads % spec.key_heads != 0) {
    return LSE_ERROR(kInvalidArgument,
                     "value_heads must be a positive multiple of key_heads");
  }
  // The delta rule carries S[i, j] with i over the value dim and j over the key
  // dim, and graph::gated_delta_step allocates it square.
  if (spec.key_head_dim != spec.value_head_dim || spec.key_head_dim <= 0) {
    return LSE_ERROR(kInvalidArgument, "key and value head dims must match");
  }

  const Shape& sx = x.shape();
  const std::int64_t batch = sx.dim(0);
  const std::int64_t seq = sx.dim(1);
  const auto kh = static_cast<std::int64_t>(spec.key_heads);
  const auto vh = static_cast<std::int64_t>(spec.value_heads);
  const auto kd = static_cast<std::int64_t>(spec.key_head_dim);
  const auto vd = static_cast<std::int64_t>(spec.value_head_dim);
  const std::int64_t key_width = kh * kd;
  const std::int64_t value_width = vh * vd;

  Array q_raw, k_raw, v_raw;
  if (spec.layout == ProjLayout::kFusedQKV) {
    Array qkv = conv_stream(graph::linear(x, w.in_proj_qkv), w.conv_w, w.conv_b,
                            spec.conv_bias,
                            state != nullptr ? &state->conv_qkv : nullptr);
    q_raw = graph::slice(qkv, -1, 0, key_width);
    k_raw = graph::slice(qkv, -1, key_width, 2 * key_width);
    v_raw = graph::slice(qkv, -1, 2 * key_width, 2 * key_width + value_width);
  } else {
    q_raw = conv_stream(graph::linear(x, w.in_proj_q), w.conv_q_w, w.conv_q_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_q : nullptr);
    k_raw = conv_stream(graph::linear(x, w.in_proj_k), w.conv_k_w, w.conv_k_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_k : nullptr);
    v_raw = conv_stream(graph::linear(x, w.in_proj_v), w.conv_v_w, w.conv_v_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_v : nullptr);
  }

  const Shape key_shape{batch, seq, kh, kd};
  Array q = graph::l2_normalize(graph::reshape(q_raw, key_shape), spec.eps);
  Array k = graph::l2_normalize(graph::reshape(k_raw, key_shape), spec.eps);
  Array v = graph::reshape(v_raw, Shape{batch, seq, vh, vd});

  // alpha = exp(-softplus(exp(A_log)) * softplus(a + dt_bias)) — the decay
  // applied to the recurrent state each step.
  Array a = graph::linear(x, w.in_proj_a);
  // Widened once, at the weight boundary: the decay multiplies the recurrent
  // state on every step, so its error compounds without bound, and the
  // delta-rule solve this feeds NaNs outright in bf16. `a + dt_bias` is a
  // binary op and already promotes to f32; the exp chain would not.
  Array decay =
      graph::softplus(graph::exp(graph::cast(w.a_log, DType::kF32)));
  Array alpha =
      graph::exp(graph::neg(decay * graph::softplus(a + w.dt_bias)));
  Array beta = graph::clamp(graph::sigmoid(graph::linear(x, w.in_proj_b)),
                            kBetaFloor, 1.0f - kBetaFloor);

  // Key heads are shared across value heads, GQA-style.
  const auto ratio = static_cast<int>(vh / kh);
  if (ratio > 1) {
    q = graph::repeat(q, ratio, 2);
    k = graph::repeat(k, ratio, 2);
    if (!spec.decay_per_value_head) {
      alpha = graph::repeat(alpha, ratio, -1);
      beta = graph::repeat(beta, ratio, -1);
    }
  }

  Array s_in = state != nullptr && state->recurrent.valid()
                   ? state->recurrent
                   : Array::zeros(Shape{batch, vh, vd, vd}, DType::kF32);

  Array s_out;
  Array o = graph::gated_delta_step(q, k, v, alpha, beta, s_in,
                                    state != nullptr ? &s_out : nullptr);
  if (state != nullptr) state->recurrent = s_out;

  o = graph::rms_norm(o, w.norm, spec.norm_eps, spec.zero_centered_norm);
  o = graph::reshape(o, Shape{batch, seq, value_width});

  Array gate = graph::sigmoid(graph::linear(x, w.gate_proj));
  return graph::linear(o * gate, w.out_proj);
}

}  // namespace lse::ops
