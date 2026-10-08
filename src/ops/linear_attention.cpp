#include "lse/ops/linear_attention.hpp"

#include <utility>

#include "lse/graph/ops.hpp"
#include "lse/graph/graph.hpp"

namespace lse::ops {

namespace {

// Widest row count (a speculative verify pass) that fuses q/k preparation.
constexpr std::int64_t kFusedPrepRows = 8;

// beta off the 0/1 rails: a saturated beta drives the delta-rule gain up until
// the recurrent state overflows to inf, then NaN.
constexpr float kBetaFloor = 1e-4f;

Array raw_rate(const Array& a_log, DecayRate decay) {
  Array rate = graph::exp(graph::cast(a_log, DType::kF32));
  return decay == DecayRate::kSoftplusExpALog ? graph::softplus(rate) : rate;
}

Status validate_rate_shape(const Array& rate, const GatedDeltaNetSpec& spec) {
  const auto heads = spec.decay_per_value_head ? spec.value_heads : spec.key_heads;
  if (!rate.valid() || heads <= 0 || rate.shape() != Shape{heads})
    return LSE_ERROR(kInvalidArgument, "GDN rate must have one element per decay head");
  return OkStatus();
}

// Runs the depthwise causal conv over x with `tail` standing in for the zero
// pad, leaving `tail` holding the last kernel-1 inputs for the next call —
// no concat of tail ++ x and no slices, which is what keeps the decode-path
// copy count down. A null or empty tail is the stateless full-sequence path.
Array conv_stream(const Array& x, const Array& weight, const Array& bias,
                  bool has_bias, Array* tail, const Array& tail_rows = {}) {
  // weight.dtype(), not x's: the conv kernel reads filter and bias through one
  // element type, so a bias in the activation's dtype fails its argument bind.
  // On HRX that surfaces as a kernel with an empty body and an output of exactly
  // zero, not as an error — Qwen3.5 is the only model with conv_bias == false,
  // and it produced zeros from block 0 until this matched.
  const Array b = has_bias
                      ? bias
                      : Array::zeros(Shape{weight.shape().dim(0)}, weight.dtype());
  if (tail == nullptr || !tail->valid()) return graph::causal_conv1d(x, weight, b);
  const Array prev = *tail;
  if (tail_rows.valid()) {
    // A padded pass: the tail is the last kernel-1 real rows, wherever the
    // pass's last real token falls.
    const Shape& sx = x.shape();
    const Array flat = graph::reshape(x, Shape{sx.dim(0) * sx.dim(1), sx.dim(2)});
    *tail = graph::reshape(graph::gather_rows(flat, tail_rows), prev.shape());
  } else {
    *tail = graph::conv_tail(prev, x);
  }
  return graph::causal_conv1d(x, weight, b, prev);
}

}  // namespace

Status prepare_gated_delta_rate(GatedDeltaNetWeights& weights,
                                const GatedDeltaNetSpec& spec) {
  LSE_RETURN_IF_ERROR(validate_rate_shape(weights.a_log, spec));
  if (spec.decay != DecayRate::kExpALog &&
      spec.decay != DecayRate::kSoftplusExpALog)
    return LSE_ERROR(kInvalidArgument, "unknown GDN decay rule");
  Array rate = raw_rate(weights.a_log, spec.decay);
  LSE_RETURN_IF_ERROR(rate.materialize());
  const graph::Node& source = *rate.node();
  if (!source.materialized || !source.buffer.valid() ||
      source.buffer.size_bytes < dtype_storage_bytes(source.dtype, source.element_count()))
    return LSE_ERROR(kInternal, "GDN rate preparation did not produce a buffer");
  Array leaf = Array::from_buffer(source.buffer, source.shape, source.dtype);
  leaf.node()->member = source.member;
  leaf.node()->host_mirror = source.host_mirror;
  leaf.node()->host_dirty = source.host_dirty;
  leaf.node()->device_dirty = source.device_dirty;
  weights.prepared_rate = std::move(leaf);
  weights.prepared_rate_decay = spec.decay;
  return OkStatus();
}

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

  if (w.prepared_rate.valid()) {
    LSE_RETURN_IF_ERROR(validate_rate_shape(w.prepared_rate, spec));
    const graph::Node& rate = *w.prepared_rate.node();
    if (w.prepared_rate_decay != spec.decay || rate.dtype != DType::kF32 ||
        rate.kind != graph::OpKind::kBuffer || !rate.materialized ||
        !rate.buffer.valid() || rate.buffer.size_bytes <
            dtype_storage_bytes(DType::kF32, rate.element_count()))
      return LSE_ERROR(kInvalidArgument, "GDN prepared rate does not match its decay rule");
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

  Array q_raw, k_raw, v_raw, qkv, qkv_act;
  // A decode row takes SiLU over the whole conv output, where it rides the
  // conv kernel's epilogue, and slices q, k and v out of that: one row's
  // slices are contiguous windows, so they cost nothing, where SiLU after the
  // slices was three more launches. The values are the same SiLU of the same
  // conv outputs.
  bool silu_first = false;
  const bool tree = state != nullptr && state->tree_depth.valid();
  if (tree && (spec.layout != ProjLayout::kFusedQKV || batch != 1 || !state->conv_qkv.valid() ||
               !state->tree_ancestors.valid() || !state->tree_path.valid() ||
               state->step_mask.valid()))
    return LSE_ERROR(kUnimplemented, "a tree verify pass needs one sequence, a fused q/k/v "
                     "projection and its conv tail");
  if (tree) {
    // Each row convolves its own ancestors; the carried tail is the top path's.
    const Array input = graph::linear(x, w.in_proj_qkv);
    const Array bias = spec.conv_bias ? w.conv_b
        : Array::zeros(Shape{w.conv_w.shape().dim(0)}, w.conv_w.dtype());
    const Array prev = state->conv_qkv;
    qkv = graph::causal_conv1d_tree(input, w.conv_w, bias, prev, state->tree_ancestors);
    state->conv_qkv = graph::conv_tail_rows(prev, input, state->tree_path);
    state->tree_conv_input = input;
    qkv_act = qkv;
    q_raw = graph::slice(qkv, -1, 0, key_width);
    k_raw = graph::slice(qkv, -1, key_width, 2 * key_width);
    v_raw = graph::slice(qkv, -1, 2 * key_width, 2 * key_width + value_width);
  } else if (spec.layout == ProjLayout::kFusedQKV) {
    qkv = conv_stream(graph::linear(x, w.in_proj_qkv), w.conv_w, w.conv_b,
                      spec.conv_bias,
                      state != nullptr ? &state->conv_qkv : nullptr,
                      state != nullptr ? state->tail_rows : Array{});
    silu_first = spec.conv_activation && batch == 1 && seq == 1;
    qkv_act = silu_first ? graph::silu(qkv) : qkv;
    const Array& act = qkv_act;
    q_raw = graph::slice(act, -1, 0, key_width);
    k_raw = graph::slice(act, -1, key_width, 2 * key_width);
    v_raw = graph::slice(act, -1, 2 * key_width, 2 * key_width + value_width);
  } else {
    q_raw = conv_stream(graph::linear(x, w.in_proj_q), w.conv_q_w, w.conv_q_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_q : nullptr,
                        state != nullptr ? state->tail_rows : Array{});
    k_raw = conv_stream(graph::linear(x, w.in_proj_k), w.conv_k_w, w.conv_k_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_k : nullptr,
                        state != nullptr ? state->tail_rows : Array{});
    v_raw = conv_stream(graph::linear(x, w.in_proj_v), w.conv_v_w, w.conv_v_b,
                        spec.conv_bias,
                        state != nullptr ? &state->conv_v : nullptr,
                        state != nullptr ? state->tail_rows : Array{});
  }

  // A speculative verify's few rows take SiLU and the head norm of q and k
  // straight from the conv output in one dispatch each (gdn.silu_l2norm.v1),
  // computing what the separate slice/silu and l2_normalize kernels did. One
  // row (wave-level norm) and long prefill chunks keep the separate ops.
  const bool fused_prep = qkv.valid() && spec.conv_activation && batch == 1 &&
                          seq > 1 && seq <= kFusedPrepRows;
  auto silu_l2norm = [&](std::int64_t offset, float scale) -> Result<Array> {
    LSE_ASSIGN_OR(Array out, graph::custom("gdn.silu_l2norm.v1", {qkv},
        {static_cast<float>(offset), spec.eps, scale, 0.0f}));
    out.node()->shape = Shape{batch, seq, kh, kd};
    return out;
  };
  Array q, k;
  if (fused_prep) {
    LSE_ASSIGN_OR(q, silu_l2norm(0, spec.query_scale));
    LSE_ASSIGN_OR(k, silu_l2norm(key_width, 1.0f));
  }
  // A decode row's q and k head norms (and q's scale) in one launch, on a
  // wave32 device whose heads are 128 wide: what the two l2_normalize
  // launches and the multiply computed, value for value.
  bool qk_fused = false;
  if (silu_first && kd == 128) {
    if (graph::Scheduler* scheduler = graph::default_scheduler()) {
      const auto member = graph::preferred_member();
      qk_fused = member < scheduler->devices().size() &&
                 scheduler->devices().device(member).device_info().wavefront_size == 32;
    }
  }
  if (qk_fused) {
    LSE_ASSIGN_OR(Array qk, graph::custom("gdn.qk_l2norm.wave32.v1",
        {graph::slice(qkv_act, -1, 0, 2 * key_width)},
        {spec.eps, spec.query_scale, 0.0f, 0.0f}));
    qk.node()->shape = Shape{batch, seq, 2 * kh, kd};
    qk.node()->iattrs[0] = spec.query_scale != 1.0f ? static_cast<std::int32_t>(kh) : 0;
    q = graph::slice(qk, 2, 0, kh);
    k = graph::slice(qk, 2, kh, 2 * kh);
  }
  if (spec.conv_activation && !silu_first) {
    if (!fused_prep) {
      q_raw = graph::silu(q_raw);
      k_raw = graph::silu(k_raw);
    }
    v_raw = graph::silu(v_raw);
  }

  const Shape key_shape{batch, seq, kh, kd};
  if (!fused_prep && !qk_fused) {
    q = graph::l2_normalize(graph::reshape(q_raw, key_shape), spec.eps);
    k = graph::l2_normalize(graph::reshape(k_raw, key_shape), spec.eps);
    if (spec.query_scale != 1.0f) {
      q = q * Array::full(Shape{1}, DType::kF32, spec.query_scale);
    }
  }
  Array v = graph::reshape(v_raw, Shape{batch, seq, vh, vd});

  // alpha = exp(-rate * softplus(a + dt_bias)) — the decay applied to the
  // recurrent state each step. The rate is where the two models part company.
  Array a = graph::linear(x, w.in_proj_a);
  // Decay stays FP32: its rounding error compounds in the recurrent state.
  const Array rate = w.prepared_rate.valid() ? w.prepared_rate
                                            : raw_rate(w.a_log, spec.decay);
  Array alpha =
      graph::exp(graph::neg(rate * graph::softplus(a + w.dt_bias)));
  Array beta = graph::clamp(graph::sigmoid(graph::linear(x, w.in_proj_b)),
                            kBetaFloor, 1.0f - kBetaFloor);

  // Key heads are shared across value heads, GQA-style.
  const auto ratio = static_cast<int>(vh / kh);
  if (ratio > 1) {
    // q and k stay at the key heads: the scan maps each value head to its
    // key head, which saves materializing two repeated copies per layer.
    if (!spec.decay_per_value_head) {
      alpha = graph::repeat(alpha, ratio, -1);
      beta = graph::repeat(beta, ratio, -1);
    }
  }

  if (state != nullptr && state->step_mask.valid()) {
    alpha = alpha * state->step_mask + state->step_unmask;
    beta = beta * state->step_mask;
  }

  Array s_in = state != nullptr && state->recurrent.valid()
                   ? state->recurrent
                   : Array::zeros(Shape{batch, vh, vd, vd}, DType::kF32);

  Array s_out, o;
  if (tree) {
    // Every row's output from its parent's state; the carried state is the
    // top path's, and the commit replays whichever path was accepted.
    o = graph::gated_delta_tree(q, k, v, alpha, beta, s_in, state->tree_depth);
    s_out = graph::gated_delta_path(k, v, alpha, beta, s_in, state->tree_path);
    state->tree_k = k;
    state->tree_v = v;
    state->tree_alpha = alpha;
    state->tree_beta = beta;
  } else {
    o = graph::gated_delta_step(q, k, v, alpha, beta, s_in,
                                state != nullptr ? &s_out : nullptr);
  }
  if (state != nullptr) state->recurrent = s_out;

  o = graph::rms_norm(o, w.norm, spec.norm_eps, spec.zero_centered_norm);
  o = graph::reshape(o, Shape{batch, seq, value_width});

  Array gate_lin = graph::linear(x, w.gate_proj);
  Array gate = spec.gate == GateActivation::kSiLU ? graph::silu(gate_lin)
                                                  : graph::sigmoid(gate_lin);
  return graph::linear(o * gate, w.out_proj);
}

}  // namespace lse::ops
