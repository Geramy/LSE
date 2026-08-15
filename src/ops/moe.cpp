#include "lse/ops/moe.hpp"

// store_element: Array exposes to_host for reads but has no host-write path, so
// gathering a token row and scattering the weighted result back go through the
// interpreter's element accessors.
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/activation.hpp"

namespace lse::ops {

namespace {

// Routing decisions are host values by nature — top-k over materialized
// probabilities — so they enter the graph as literal buffers.
Result<Array> host_vector(const std::vector<float>& values, Shape shape) {
  Array a = Array::zeros(shape, DType::kF32);
  LSE_RETURN_IF_ERROR(a.eval());
  for (std::size_t i = 0; i < values.size(); ++i) {
    graph::interpreter::store_element(*a.node(), i, values[i]);
  }
  return a;
}

std::int64_t row_width(const Array& x) {
  const Shape& s = x.shape();
  return s.dim(s.rank() - 1);
}

Array last_slot(const Array& a, std::int64_t s) {
  return graph::slice(a, -1, s, s + 1);
}

}  // namespace

Array expert_slice(const Array& stacked, std::int64_t expert) {
  const Shape& s = stacked.shape();
  Array one = graph::slice(stacked, 0, expert, expert + 1);
  return graph::reshape(one, Shape{s.dim(1), s.dim(2)});
}

Result<std::vector<std::vector<ExpertChoice>>> route_topk(
    const Array& probs, const RouteConfig& cfg) {
  if (cfg.num_experts <= 0 || cfg.num_active <= 0 ||
      cfg.num_active > cfg.num_experts) {
    return LSE_ERROR(kInvalidArgument, "invalid routing configuration");
  }
  if (probs.dtype() != DType::kF32) {
    return LSE_ERROR(kInvalidArgument, "route_topk needs f32 probabilities");
  }
  const Shape& ps = probs.shape();
  if (ps.rank() == 0 || ps.dim(ps.rank() - 1) != cfg.num_experts) {
    return LSE_ERROR(kInvalidArgument, "route_topk last axis must be num_experts");
  }

  Array idx;
  Array vals = graph::topk(probs, cfg.num_active, -1, &idx);
  const auto keep = static_cast<std::size_t>(cfg.num_active);
  std::vector<float> host_v(vals.shape().elem_count());
  std::vector<float> host_i(idx.shape().elem_count());
  LSE_RETURN_IF_ERROR(
      vals.to_host(host_v.data(), host_v.size() * sizeof(float)));
  LSE_RETURN_IF_ERROR(
      idx.to_host(host_i.data(), host_i.size() * sizeof(float)));

  const std::size_t rows = host_v.size() / keep;
  std::vector<std::vector<ExpertChoice>> out(rows);

  for (std::size_t r = 0; r < rows; ++r) {
    const float top = host_v[r * keep];
    float total = 0.0f;
    std::vector<float> w(keep);
    for (std::size_t s = 0; s < keep; ++s) {
      w[s] = host_v[r * keep + s];
      if (cfg.score_band < 1.0f && w[s] < (1.0f - cfg.score_band) * top) {
        w[s] = 0.0f;
      }
      total += w[s];
    }
    total += 1e-9f;

    for (std::size_t s = 0; s < keep; ++s) {
      if (w[s] == 0.0f) continue;
      out[r].push_back(ExpertChoice{static_cast<std::int32_t>(host_i[r * keep + s]),
                                    w[s] / total});
    }
  }
  return out;
}

Result<Array> dispatch_combine(
    const Array& x, const std::vector<std::vector<ExpertChoice>>& routing,
    const ExpertWeights& stacked) {
  const std::int64_t width = row_width(x);
  const std::size_t rows =
      x.shape().elem_count() / static_cast<std::size_t>(width);
  if (routing.size() != rows) {
    return LSE_ERROR(kInvalidArgument, "routing has ",
                     std::to_string(routing.size()), " rows, input has ",
                     std::to_string(rows));
  }

  // Grouped by expert, not by token: each expert then runs one batched swiglu
  // over all the rows it was given. Iterating tokens instead would build a
  // one-row graph per (token, expert) pair and materialize each on its own,
  // which defeats fusion and makes cost scale with sequence length.
  const auto num_experts = static_cast<std::size_t>(stacked.gate.shape().dim(0));
  std::vector<std::vector<float>> expert_rows(num_experts);
  std::vector<std::vector<float>> expert_weights(num_experts);
  for (std::size_t r = 0; r < rows; ++r) {
    for (const ExpertChoice& choice : routing[r]) {
      const auto e = static_cast<std::size_t>(choice.expert);
      if (e >= num_experts) {
        return LSE_ERROR(kOutOfRange, "expert ", std::to_string(choice.expert),
                         " is outside ", std::to_string(num_experts));
      }
      expert_rows[e].push_back(static_cast<float>(r));
      expert_weights[e].push_back(choice.weight);
    }
  }

  Array flat = graph::reshape(x, Shape{static_cast<std::int64_t>(rows), width});
  Array out = Array::zeros(flat.shape(), DType::kF32);
  LSE_RETURN_IF_ERROR(out.eval());

  for (std::size_t e = 0; e < num_experts; ++e) {
    const auto n = static_cast<std::int64_t>(expert_rows[e].size());
    if (n == 0) continue;

    LSE_ASSIGN_OR(Array rows_idx,
                  host_vector(expert_rows[e], Shape{n}));
    LSE_ASSIGN_OR(Array gate_w, host_vector(expert_weights[e], Shape{n, 1}));

    Array tokens = graph::gather_rows(flat, rows_idx);
    Array y = swiglu(tokens, expert_slice(stacked.gate, static_cast<std::int32_t>(e)),
                     expert_slice(stacked.up, static_cast<std::int32_t>(e)),
                     expert_slice(stacked.down, static_cast<std::int32_t>(e)));
    out = graph::scatter_add_rows(out, rows_idx, y * gate_w);
  }

  return graph::reshape(out, x.shape());
}

Result<Array> routed_experts(const Array& x, const Array& router,
                             const Array& router_bias,
                             const ExpertWeights& stacked,
                             const RouteConfig& cfg) {
  if (cfg.num_experts <= 0 || cfg.num_active <= 0 ||
      cfg.num_active > cfg.num_experts) {
    return LSE_ERROR(kInvalidArgument, "invalid routing configuration");
  }
  Array logits = graph::linear(x, router);
  if (router_bias.valid()) logits = logits + router_bias;
  Array idx;
  Array w = graph::topk(graph::softmax(logits, -1), cfg.num_active, -1, &idx,
                        cfg.score_band);

  Array out;
  for (std::int32_t s = 0; s < cfg.num_active; ++s) {
    Array g = graph::linear_indexed(x, stacked.gate, idx, s);
    Array u = graph::linear_indexed(x, stacked.up, idx, s);
    Array y = graph::linear_indexed(graph::silu(g) * u, stacked.down, idx, s);
    y = y * last_slot(w, s);
    out = s == 0 ? y : out + y;
  }
  return out;
}

Array shared_experts(const Array& x, const std::vector<ExpertWeights>& experts) {
  if (experts.empty()) return Array{};
  Array out = swiglu(x, experts[0].gate, experts[0].up, experts[0].down);
  for (std::size_t i = 1; i < experts.size(); ++i) {
    out = out + swiglu(x, experts[i].gate, experts[i].up, experts[i].down);
  }
  return out;
}

Array depth_gate(const Array& x, const Array& weight, const Array& bias,
                 const Array& value) {
  Array logits = graph::linear(x, weight);
  if (bias.valid()) logits = logits + bias;
  return value * graph::sigmoid(logits);
}

}  // namespace lse::ops
