#include "lse/model/dflash2.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "lse/dispatch/dflash2.hpp"
#include "lse/model/dflash2_convert.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/graph/program.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/qwen3_5_common.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/norm.hpp"
#include "lse/ops/rope.hpp"

namespace lse::model {
namespace {
using graph::Array;

class DraftProfile {
 public:
  explicit DraftProfile(const char* phase) : phase_(phase) {
    static const bool enabled = std::getenv("LSE_TIME_STEPS") != nullptr;
    enabled_ = enabled;
    if (enabled_) start_ = std::chrono::steady_clock::now();
  }
  ~DraftProfile() {
    if (!enabled_) return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_).count();
    std::fprintf(stderr, "[dflash2-spans] phase=%s elapsed_ns=%lld\n",
                 phase_, static_cast<long long>(ns));
  }
 private:
  const char* phase_;
  bool enabled_ = false;
  std::chrono::steady_clock::time_point start_;
};

Array slot_view(const Array& owner, Shape shape, std::size_t begin) {
  auto buffer = owner.node()->buffer;
  buffer.offset += begin * sizeof(float);
  buffer.size_bytes = shape.elem_count() * sizeof(float);
  return Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
}

Result<Array> slot(Shape shape) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return LSE_ERROR(kInternal, "DFlash2 needs a scheduler");
  const backend::ScopedAllocationSite site(backend::AllocationSite::kDraft);
  auto buffer = scheduler->backend().allocate(
      dtype_storage_bytes(DType::kF32, shape.elem_count()), backend::MemoryClass::kDevice);
  if (!buffer.ok()) return buffer.status();
  return Array::from_buffer(buffer.release(), std::move(shape), DType::kF32);
}
Status poke(Array& array, std::span<const float> data) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler || !array.valid() || data.size() != array.shape().elem_count())
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 input slot");
  LSE_RETURN_IF_ERROR(scheduler->drain());
  LSE_RETURN_IF_ERROR(scheduler->backend().copy(array.node()->buffer, data.data(), data.size_bytes()));
  array.node()->materialized = true;
  array.node()->host_dirty = false;
  array.node()->device_dirty = true;
  return OkStatus();
}
Status copy_slot(Array& destination, Array source) {
  if (!source.valid() || source.dtype() != DType::kF32 ||
      source.shape() != destination.shape())
    return LSE_ERROR(kInvalidArgument, "DFlash2 input shape or dtype mismatch");
  LSE_RETURN_IF_ERROR(source.materialize());
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return LSE_ERROR(kInternal, "DFlash2 needs a scheduler");
  LSE_RETURN_IF_ERROR(scheduler->drain());
  LSE_RETURN_IF_ERROR(scheduler->backend().copy(
      destination.node()->buffer, source.node()->buffer, source.shape().elem_count() * sizeof(float)));
  destination.node()->materialized = true;
  destination.node()->device_dirty = true;
  destination.node()->host_dirty = false;
  return OkStatus();
}
void accept_cache_write(Array& cache, const Array& written) {
  auto& destination = *cache.node();
  const auto& source = *written.node();
  destination.buffer = source.buffer;
  destination.host_mirror = source.host_mirror;
  destination.host_dirty = source.host_dirty;
  destination.device_dirty = source.device_dirty;
  destination.materialized = true;
}
Status evaluate(graph::Program& program, const std::vector<graph::NodePtr>& roots) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return LSE_ERROR(kInternal, "DFlash2 needs a scheduler");
  program.reset_compute();
  return scheduler->eval(roots, false, &program);
}
std::vector<std::int64_t> rope_order(std::int32_t heads, std::int32_t dim) {
  std::vector<std::int64_t> order;
  order.reserve(static_cast<std::size_t>(heads) * static_cast<std::size_t>(dim));
  for (std::int32_t h = 0; h < heads; ++h)
    for (std::int32_t d = 0; d < dim; ++d)
      order.push_back(static_cast<std::int64_t>(h) * dim + d / 2 + (d % 2) * (dim / 2));
  return order;
}
Array norm(const Array& x, const Array& weight, float epsilon) {
  return ops::rms_norm(x, weight, epsilon);
}
}

Result<DFlash2Config> DFlash2Config::from_json_string(const std::string& text) {
  DFlash2Config c;
  try {
    const auto j = nlohmann::json::parse(text);
    const auto architectures = j.at("architectures").get<std::vector<std::string>>();
    if (std::find(architectures.begin(), architectures.end(), "DFlash2DraftModel") == architectures.end())
      return LSE_ERROR(kInvalidArgument, "checkpoint is not a DFlash2DraftModel");
    if (j.at("is_causal").get<bool>())
      return LSE_ERROR(kUnimplemented, "causal DFlash2 draft attention is unsupported");
    c.hidden_size = j.at("hidden_size"); c.vocab_size = j.at("vocab_size");
    c.num_layers = j.at("num_hidden_layers"); c.target_num_layers = j.at("num_target_layers");
    c.q_heads = j.at("num_attention_heads"); c.kv_heads = j.at("num_key_value_heads");
    c.head_dim = j.at("head_dim"); c.intermediate_size = j.at("intermediate_size");
    c.sliding_window = j.at("sliding_window"); c.rms_eps = j.at("rms_norm_eps");
    const auto& rope = j.at("rope_parameters");
    if (rope.at("rope_type") != "default")
      return LSE_ERROR(kUnimplemented, "DFlash2 requires default RoPE scaling");
    c.rope_theta = rope.at("rope_theta");
    const auto layers = j.at("layer_types").get<std::vector<std::string>>();
    if (layers.size() != static_cast<std::size_t>(c.num_layers) ||
        std::any_of(layers.begin(), layers.end(), [](const auto& s) { return s != "sliding_attention"; }))
      return LSE_ERROR(kUnimplemented, "DFlash2 requires sliding attention in every draft layer");
    const auto& d = j.at("dflash_config");
    const auto block = d.at("block_size").get<std::int64_t>();
    const auto mask = d.at("mask_token_id").get<std::int64_t>();
    if (block < 2 || block > 8 || mask < 0 || mask >= c.vocab_size)
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 block size or mask token");
    c.block_size = static_cast<std::uint32_t>(block); c.mask_token = static_cast<std::uint32_t>(mask);
    c.target_layers = d.at("target_layer_ids").get<std::vector<std::int32_t>>();
    c.conv_group_size = d.at("conv_group_size");
    if (d.at("conv_kernel_size").get<std::int32_t>() != 2)
      return LSE_ERROR(kUnimplemented, "DFlash2 requires two-tap dynamic convolution");
    c.selector_rank = d.at("selector_rank"); c.selector_top_k = d.at("selector_top_k");
    if (d.value("input_embedding_scale", 1.0f) != 1.0f ||
        d.value("output_multiplier", 1.0f) != 1.0f ||
        d.value("final_logit_softcapping", 0.0f) != 0.0f)
      return LSE_ERROR(kUnimplemented, "DFlash2 embedding or logit scaling is unsupported");
  } catch (const nlohmann::json::exception& e) {
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 config: ", e.what());
  }
  LSE_ASSIGN_OR(c.quantization, quant::GroupAffineMap::from_config_json(text));
  return c;
}
Status DFlash2Config::validate(const Config& target) const {
  if (hidden_size != target.hidden_size || vocab_size != target.vocab_size ||
      target_num_layers != target.num_layers)
    return LSE_ERROR(kInvalidArgument, "DFlash2 checkpoint does not match target geometry");
  if (hidden_size <= 0 || vocab_size <= 0 || num_layers <= 0 ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads ||
      head_dim <= 0 || head_dim % 2 || intermediate_size <= 0 ||
      sliding_window < 2 || conv_group_size <= 0 || hidden_size % conv_group_size ||
      selector_rank <= 0 || selector_top_k < 1 || selector_top_k > vocab_size ||
      block_size < 2 || block_size > 8 || mask_token >= static_cast<std::uint32_t>(vocab_size) ||
      !std::isfinite(rms_eps) || rms_eps <= 0 || !std::isfinite(rope_theta) || rope_theta <= 0)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 dimensions");
  if (target_layers.empty()) return LSE_ERROR(kInvalidArgument, "DFlash2 has no target feature taps");
  std::int32_t previous = -1;
  for (auto layer : target_layers) {
    if (layer <= previous || layer < 0 || layer >= target_num_layers)
      return LSE_ERROR(kInvalidArgument, "invalid DFlash2 target feature tap order");
    previous = layer;
  }
  return OkStatus();
}

Result<std::vector<std::uint32_t>> dflash2_select_path(
    std::span<const float> scores, std::span<const std::uint32_t> candidates,
    std::uint32_t positions, std::uint32_t top_k) {
  if (positions == 0 || top_k == 0 ||
      candidates.size() != static_cast<std::size_t>(positions) * top_k ||
      scores.size() != static_cast<std::size_t>(positions) * top_k * top_k)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 candidate lattice");
  std::vector<std::uint32_t> path;
  path.reserve(positions);
  std::uint32_t predecessor = 0;
  for (std::uint32_t p = 0; p < positions; ++p) {
    const auto row = scores.subspan((static_cast<std::size_t>(p) * top_k + predecessor) * top_k, top_k);
    std::uint32_t winner = 0;
    for (std::uint32_t k = 0; k < top_k; ++k) {
      if (!std::isfinite(row[k])) return LSE_ERROR(kInvalidArgument, "nonfinite DFlash2 selector score");
      if (row[k] > row[winner] || (row[k] == row[winner] &&
          candidates[static_cast<std::size_t>(p) * top_k + k] <
          candidates[static_cast<std::size_t>(p) * top_k + winner])) winner = k;
    }
    path.push_back(candidates[static_cast<std::size_t>(p) * top_k + winner]);
    predecessor = winner;
  }
  return path;
}

Result<DFlash2Proposal> dflash2_sample_path(
    std::span<const float> scores, std::span<const std::uint32_t> candidates,
    std::uint32_t positions, std::uint32_t top_k, std::uint32_t vocab_size,
    float temperature, runtime::SpeculativeSampler& sampler) {
  if (positions == 0 || top_k == 0 || vocab_size == 0 ||
      !std::isfinite(temperature) || temperature <= 0 ||
      candidates.size() != static_cast<std::size_t>(positions) * top_k ||
      scores.size() != static_cast<std::size_t>(positions) * top_k * top_k)
    return LSE_ERROR(kInvalidArgument, "invalid sampled DFlash2 lattice");
  DFlash2Proposal result;
  result.tokens.reserve(positions);
  result.conditionals.reserve(positions);
  std::uint32_t predecessor = 0;
  for (std::uint32_t p = 0; p < positions; ++p) {
    const auto row = scores.subspan((static_cast<std::size_t>(p) * top_k + predecessor) * top_k, top_k);
    const auto ids = candidates.subspan(static_cast<std::size_t>(p) * top_k, top_k);
    runtime::DiscreteDistribution q;
    q.ids.assign(ids.begin(), ids.end());
    q.probabilities.resize(top_k);
    double maximum = -INFINITY;
    for (std::uint32_t k = 0; k < top_k; ++k) {
      if (!std::isfinite(row[k]) || ids[k] >= vocab_size ||
          std::find(ids.begin(), ids.begin() + k, ids[k]) != ids.begin() + k)
        return LSE_ERROR(kInvalidArgument, "invalid sampled DFlash2 candidate");
      maximum = std::max(maximum, static_cast<double>(row[k]));
    }
    double total = 0;
    for (std::uint32_t k = 0; k < top_k; ++k) {
      q.probabilities[k] = std::exp((static_cast<double>(row[k]) - maximum) / temperature);
      total += q.probabilities[k];
    }
    for (auto& probability : q.probabilities) probability /= total;
    LSE_ASSIGN_OR(auto token, sampler.sample_proposal(q));
    predecessor = static_cast<std::uint32_t>(std::find(ids.begin(), ids.end(), token) - ids.begin());
    result.tokens.push_back(token);
    result.conditionals.push_back(std::move(q));
  }
  return result;
}

Array dflash2_convolve(const Array& hidden, const Array& dynamic,
                       const Array& base, std::int32_t group_size) {
  if (!hidden.valid() || !dynamic.valid() || !base.valid() || hidden.shape().rank() != 3 ||
      dynamic.shape().rank() != 4 || base.shape().rank() != 2 || group_size <= 0)
    return {};
  const auto b = hidden.shape().dim(0), t = hidden.shape().dim(1), d = hidden.shape().dim(2);
  if (d % group_size || dynamic.shape() != Shape{b, t, 2, d / group_size} || base.shape() != Shape{2, d}) return {};
  if (hidden.dtype() != DType::kF32 || dynamic.dtype() != DType::kF32 ||
      base.dtype() != DType::kF32) return {};
  auto result = graph::custom("dflash2.convolve2.v1", {hidden, dynamic, base},
      {static_cast<float>(t), static_cast<float>(d), static_cast<float>(group_size), 0.0f});
  return result.ok() ? result.release() : Array{};
}

struct DFlash2Module::Impl {
  DFlash2Config config;
  HybridLM* target = nullptr;
  std::int32_t capacity = 0, position = 0, live = 0, max_position = 0;
  Array fc, hidden_norm, final_norm, predecessor, successor, selector_projection;
  ops::RopeTables rope;
  struct Conv { Array base, projection; };
  struct Layer {
    Array input_norm, post_norm, q, k, v, o, q_norm, k_norm, gate, up, down;
    Conv attention_conv, mlp_conv;
    Array keys, values;
  };
  std::vector<Layer> layers;
  struct ContextPass {
    Array features, offset;
    std::vector<Array> keys, values;
    graph::Program program;
    std::vector<graph::NodePtr> roots;
  };
  struct DraftPass {
    Array inputs, tokens, anchor, offset, meta, indices, scores, path;
    std::vector<float> input_image;
    graph::Program program;
    std::vector<graph::NodePtr> roots;
  };
  std::map<std::int64_t, ContextPass> contexts;
  std::map<std::pair<std::uint32_t, bool>, DraftPass> drafts;

  Status load(WeightBinder& binder);
  Result<ContextPass> context_pass(std::int64_t rows);
  Result<DraftPass> draft_pass(std::uint32_t head_rows, std::uint32_t proposals, bool sampled);
  Result<DraftPass*> run_draft(std::uint32_t anchor, std::int32_t first, std::uint32_t proposals, bool sampled);
  Result<Array> attention(Array x, Layer& layer, const DraftPass& pass);
  std::pair<Array, Array> prepare(Array x, const Conv& conv) const;
};

Status DFlash2Module::Impl::load(WeightBinder& binder) {
  const auto d = static_cast<std::int64_t>(config.hidden_size);
  const auto feature_width = d * static_cast<std::int64_t>(config.target_layers.size());
  const auto require = [&](Array& out, const std::string& name, Shape shape) -> Status {
    LSE_ASSIGN_OR(out, binder.require(name));
    return qwen3_5::expect_shape(out, name, shape);
  };
  LSE_RETURN_IF_ERROR(require(fc, "fc.weight", Shape{d, feature_width}));
  LSE_RETURN_IF_ERROR(require(hidden_norm, "hidden_norm.weight", Shape{d}));
  LSE_RETURN_IF_ERROR(require(final_norm, "norm.weight", Shape{d}));
  LSE_RETURN_IF_ERROR(require(selector_projection, "candidate_selector.hidden_projection.weight", Shape{config.selector_rank, d}));
  for (auto item : {std::pair(&predecessor, "candidate_selector.predecessor_codebook"),
                    std::pair(&successor, "candidate_selector.successor_codebook")}) {
    std::string name = item.second;
    if (binder.weights().find(name + ".weight")) name += ".weight";
    LSE_RETURN_IF_ERROR(require(*item.first, name, Shape{config.vocab_size, config.selector_rank}));
  }
  const auto q_rows = rope_order(config.q_heads, config.head_dim);
  const auto k_rows = rope_order(config.kv_heads, config.head_dim);
  const auto norm_rows = rope_order(1, config.head_dim);
  layers.resize(static_cast<std::size_t>(config.num_layers));
  for (std::size_t i = 0; i < layers.size(); ++i) {
    Layer& l = layers[i]; const auto p = "layers." + std::to_string(i);
    LSE_RETURN_IF_ERROR(require(l.input_norm, p + ".input_layernorm.weight", Shape{d}));
    LSE_RETURN_IF_ERROR(require(l.post_norm, p + ".post_attention_layernorm.weight", Shape{d}));
    LSE_ASSIGN_OR(l.q, binder.require_rows(p + ".self_attn.q_proj.weight", q_rows, Shape{config.q_heads * config.head_dim, d}));
    LSE_ASSIGN_OR(l.k, binder.require_rows(p + ".self_attn.k_proj.weight", k_rows, Shape{config.kv_heads * config.head_dim, d}));
    LSE_ASSIGN_OR(l.q_norm, binder.require_rows(p + ".self_attn.q_norm.weight", norm_rows, Shape{config.head_dim}));
    LSE_ASSIGN_OR(l.k_norm, binder.require_rows(p + ".self_attn.k_norm.weight", norm_rows, Shape{config.head_dim}));
    LSE_RETURN_IF_ERROR(require(l.v, p + ".self_attn.v_proj.weight", Shape{config.kv_heads * config.head_dim, d}));
    LSE_RETURN_IF_ERROR(require(l.o, p + ".self_attn.o_proj.weight", Shape{d, config.q_heads * config.head_dim}));
    LSE_RETURN_IF_ERROR(require(l.gate, p + ".mlp.gate_proj.weight", Shape{config.intermediate_size, d}));
    LSE_RETURN_IF_ERROR(require(l.up, p + ".mlp.up_proj.weight", Shape{config.intermediate_size, d}));
    LSE_RETURN_IF_ERROR(require(l.down, p + ".mlp.down_proj.weight", Shape{d, config.intermediate_size}));
    for (auto item : {std::pair(&l.attention_conv, ".attention_conv"), std::pair(&l.mlp_conv, ".mlp_conv")}) {
      LSE_RETURN_IF_ERROR(require(item.first->base, p + item.second + ".base_kernel", Shape{2, 2, d}));
      item.first->base = graph::cast(item.first->base, DType::kF32);
      LSE_RETURN_IF_ERROR(item.first->base.materialize());
      LSE_RETURN_IF_ERROR(require(item.first->projection, p + item.second + ".kernel_projection.weight", Shape{4 * (d / config.conv_group_size), d}));
    }
    LSE_ASSIGN_OR(l.keys, slot(Shape{1, config.kv_heads, capacity, config.head_dim}));
    LSE_ASSIGN_OR(l.values, slot(l.keys.shape()));
    const std::vector<float> zero(l.keys.shape().elem_count(), 0.0f);
    LSE_RETURN_IF_ERROR(poke(l.keys, zero));
    LSE_RETURN_IF_ERROR(poke(l.values, zero));
  }
  LSE_ASSIGN_OR(rope, ops::build_rope(config.head_dim, max_position + static_cast<std::int32_t>(config.block_size) - 1, config.rope_theta));
  const auto unused = binder.unclaimed();
  if (!unused.empty()) return LSE_ERROR(kInvalidArgument, "DFlash2 left tensors unclaimed: ", unused.front());
  return OkStatus();
}

Result<DFlash2Module::Impl::ContextPass> DFlash2Module::Impl::context_pass(std::int64_t rows) {
  ContextPass p;
  LSE_ASSIGN_OR(p.features, slot(Shape{1, rows, static_cast<std::int64_t>(config.hidden_size) * static_cast<std::int64_t>(config.target_layers.size())}));
  LSE_ASSIGN_OR(p.offset, slot(Shape{1}));
  Array context = norm(graph::linear(p.features, fc), hidden_norm, config.rms_eps);
  for (Layer& l : layers) {
    Array k = ops::split_heads(graph::linear(context, l.k), config.kv_heads, config.head_dim);
    k = norm(k, l.k_norm, config.rms_eps);
    LSE_ASSIGN_OR(k, ops::apply_rope(k, rope, p.offset));
    Array v = ops::split_heads(graph::linear(context, l.v), config.kv_heads, config.head_dim);
    const std::array<float,4> geometry{static_cast<float>(config.kv_heads), static_cast<float>(capacity),
        static_cast<float>(config.head_dim), static_cast<float>(rows)};
    LSE_ASSIGN_OR(Array keys, graph::custom("dflash2.cache_write", {l.keys, k, p.offset}, geometry));
    LSE_ASSIGN_OR(Array values, graph::custom("dflash2.cache_write", {l.values, v, p.offset}, geometry));
    p.keys.push_back(std::move(keys));
    p.values.push_back(std::move(values));
    p.roots.push_back(p.keys.back().node()); p.roots.push_back(p.values.back().node());
  }
  return p;
}
std::pair<Array, Array> DFlash2Module::Impl::prepare(Array x, const Conv& conv) const {
  const auto t = x.shape().dim(1);
  const auto groups = config.hidden_size / config.conv_group_size;
  Array dynamic = graph::reshape(graph::linear(x, conv.projection), Shape{1, t, 2, 2, groups});
  Array before = graph::reshape(graph::slice(dynamic, 2, 0, 1), Shape{1, t, 2, groups});
  Array after = graph::reshape(graph::slice(dynamic, 2, 1, 2), Shape{1, t, 2, groups});
  Array base = graph::reshape(graph::slice(conv.base, 0, 0, 1), Shape{2, config.hidden_size});
  return {dflash2_convolve(x, before, base, config.conv_group_size), after};
}
Result<Array> DFlash2Module::Impl::attention(Array x, Layer& l, const DraftPass& p) {
  Array q = ops::split_heads(graph::linear(x, l.q), config.q_heads, config.head_dim);
  Array k = ops::split_heads(graph::linear(x, l.k), config.kv_heads, config.head_dim);
  Array v = ops::split_heads(graph::linear(x, l.v), config.kv_heads, config.head_dim);
  q = norm(q, l.q_norm, config.rms_eps); k = norm(k, l.k_norm, config.rms_eps);
  LSE_ASSIGN_OR(q, ops::apply_rope(q, rope, p.offset));
  LSE_ASSIGN_OR(k, ops::apply_rope(k, rope, p.offset));
  LSE_ASSIGN_OR(Array partial, graph::custom("dflash2.ring_partial256.v1", {q, l.keys, l.values, k, v, p.meta},
      {static_cast<float>(config.q_heads), static_cast<float>(config.kv_heads), static_cast<float>(config.head_dim), static_cast<float>(config.sliding_window)}));
  LSE_ASSIGN_OR(Array attended, graph::custom("dflash2.ring_merge256.v1", {partial},
      {static_cast<float>(config.q_heads), static_cast<float>(q.shape().dim(2)), static_cast<float>(config.head_dim), static_cast<float>(partial.shape().dim(2))}));
  return graph::linear(ops::merge_heads(attended), l.o);
}
Result<DFlash2Module::Impl::DraftPass> DFlash2Module::Impl::draft_pass(
    std::uint32_t head_rows, std::uint32_t proposals, bool sampled) {
  DraftPass p;
  const auto rows = static_cast<std::int64_t>(config.block_size);
  LSE_ASSIGN_OR(p.inputs, slot(Shape{rows + 5}));
  p.tokens = slot_view(p.inputs, Shape{1, rows}, 0);
  p.anchor = slot_view(p.inputs, Shape{1, 1}, rows);
  p.offset = slot_view(p.inputs, Shape{1}, rows + 1);
  p.meta = slot_view(p.inputs, Shape{3}, rows + 2);
  p.input_image.resize(static_cast<std::size_t>(rows) + 5);
  LSE_ASSIGN_OR(Array x, target->embed(p.tokens));
  for (Layer& l : layers) {
    auto [prepared, after] = prepare(norm(x, l.input_norm, config.rms_eps), l.attention_conv);
    LSE_ASSIGN_OR(Array attended, attention(prepared, l, p));
    Array base = graph::reshape(graph::slice(l.attention_conv.base, 0, 1, 2), Shape{2, config.hidden_size});
    x = graph::add(x, dflash2_convolve(attended, after, base, config.conv_group_size));
    auto [mlp_input, mlp_after] = prepare(norm(x, l.post_norm, config.rms_eps), l.mlp_conv);
    Array mlp = graph::linear(graph::mul(graph::silu(graph::linear(mlp_input, l.gate)), graph::linear(mlp_input, l.up)), l.down);
    base = graph::reshape(graph::slice(l.mlp_conv.base, 0, 1, 2), Shape{2, config.hidden_size});
    x = graph::add(x, dflash2_convolve(mlp, mlp_after, base, config.conv_group_size));
  }
  x = norm(graph::slice(x, 1, 1, static_cast<std::int64_t>(head_rows) + 1),
           final_norm, config.rms_eps);
  LSE_ASSIGN_OR(Array logits, target->lm_head(x));
  if (head_rows != proposals) {
    x = graph::slice(x, 1, 0, proposals);
    logits = graph::slice(logits, 1, 0, proposals);
  }
  Array unary = graph::topk(logits, config.selector_top_k, -1, &p.indices);
  if (!unary.valid() || !p.indices.valid()) return LSE_ERROR(kInternal, "DFlash2 top-k failed");
  const auto rank = static_cast<std::int64_t>(config.selector_rank), top = static_cast<std::int64_t>(config.selector_top_k);
  const auto embed = [](const Array& table, const Array& ids) { return graph::embedding(table, ids); };
  Array pred = embed(predecessor, p.indices), succ = embed(successor, p.indices);
  Array anchor_embedding = graph::repeat(graph::reshape(embed(predecessor, p.anchor), Shape{1, 1, 1, rank}), config.selector_top_k, 2);
  pred = proposals == 1 ? anchor_embedding : graph::concat({anchor_embedding, graph::slice(pred, 1, 0, proposals - 1)}, 1);
  Array gate = graph::reshape(graph::linear(x, selector_projection), Shape{1, proposals, rank});
  LSE_ASSIGN_OR(p.scores, graph::custom("dflash2.selector", {pred, gate, succ, unary},
      {static_cast<float>(proposals), static_cast<float>(top), static_cast<float>(top), static_cast<float>(rank)}));
  if (sampled) {
    p.roots = {p.scores.node(), p.indices.node()};
  } else {
    LSE_ASSIGN_OR(p.path, graph::custom("dflash2.selector_walk.v1", {p.scores, p.indices},
        {static_cast<float>(proposals), static_cast<float>(top), static_cast<float>(config.vocab_size), 0.0f}));
    p.roots = {p.path.node()};
  }
  return p;
}

DFlash2Module::DFlash2Module(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DFlash2Module::~DFlash2Module() = default;
Result<std::unique_ptr<DFlash2Module>> DFlash2Module::open(const std::string& path, const Config& parent, HybridLM& model) {
  // A BF16 source checkpoint is converted to Q8 once and cached.
  LSE_ASSIGN_OR(const auto paths, prepare_dflash2_checkpoint(path));
  std::ifstream in(paths.config); std::ostringstream text; text << in.rdbuf();
  if (!in) return LSE_ERROR(kNotFound, "DFlash2 config not readable at ", paths.config);
  LSE_ASSIGN_OR(auto config, DFlash2Config::from_json_string(text.str()));
  LSE_RETURN_IF_ERROR(config.validate(parent));
  auto impl = std::make_unique<Impl>(); impl->config = std::move(config); impl->target = &model;
  impl->capacity = impl->config.sliding_window - 1 + static_cast<std::int32_t>(impl->config.block_size);
  impl->max_position = parent.kv_capacity();
  LSE_ASSIGN_OR(SafeTensors weights, paths.weights.ends_with(".index.json") ? SafeTensors::open_sharded(paths.weights) : SafeTensors::open(paths.weights));
  WeightBinder binder(weights, &impl->config.quantization);
  const graph::ScopedSplitScheme unsplit(graph::SplitScheme::kNone);
  LSE_RETURN_IF_ERROR(impl->load(binder));
  return std::unique_ptr<DFlash2Module>(new DFlash2Module(std::move(impl)));
}
std::span<const std::int32_t> DFlash2Module::target_layers() const noexcept { return impl_->config.target_layers; }
std::uint32_t DFlash2Module::block_size() const noexcept { return impl_->config.block_size; }
std::int32_t DFlash2Module::context_position() const noexcept { return impl_->position; }
void DFlash2Module::reset() {
  impl_->position = 0;
  impl_->live = 0;
  impl_->contexts.clear();
}
void DFlash2Module::release_programs() {
  if (auto* scheduler = graph::default_scheduler()) (void)scheduler->drain();
  reset();
  impl_->drafts.clear();
}
Status DFlash2Module::retire_prefill() {
  const auto wide = [this](const auto& entry) {
    return entry.first > static_cast<std::int64_t>(impl_->config.block_size);
  };
  if (std::none_of(impl_->contexts.begin(), impl_->contexts.end(), wide)) return OkStatus();
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return LSE_ERROR(kInternal, "DFlash2 needs a scheduler");
  LSE_RETURN_IF_ERROR(scheduler->drain());
  std::erase_if(impl_->contexts, wide);
  return OkStatus();
}
Status DFlash2Module::rewind(std::int32_t position) {
  if (position < 0 || position > impl_->position) return LSE_ERROR(kInvalidArgument, "invalid DFlash2 rewind position");
  const auto drop = impl_->position - position;
  if (drop == 0) return OkStatus();
  if (drop > static_cast<std::int32_t>(impl_->config.block_size) || drop > impl_->live)
    return LSE_ERROR(kInvalidArgument, "DFlash2 rewind exceeds retained speculative context");
  impl_->position = position; impl_->live -= drop;
  return OkStatus();
}
Status DFlash2Module::append_context(const Array& features, std::int32_t first) {
  if (!features.valid() || features.dtype() != DType::kF32 || features.shape().rank() != 3 ||
      features.shape().dim(0) != 1 || features.shape().dim(1) <= 0 ||
      features.shape().dim(2) != static_cast<std::int64_t>(impl_->config.hidden_size) * static_cast<std::int64_t>(impl_->config.target_layers.size()) || first < 0)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 target features");
  auto rows = features.shape().dim(1);
  if (static_cast<std::int64_t>(first) + rows > impl_->max_position)
    return LSE_ERROR(kInvalidArgument, "DFlash2 context exceeds target KV capacity");
  if (first < impl_->position) LSE_RETURN_IF_ERROR(rewind(first));
  if (first != impl_->position && impl_->live != 0) return LSE_ERROR(kInvalidArgument, "DFlash2 context has a gap");
  Array input = features;
  if (rows > impl_->capacity) {
    input = graph::slice(features, 1, rows - impl_->capacity, rows);
    first += static_cast<std::int32_t>(rows - impl_->capacity); rows = impl_->capacity;
  }
  // Keep one wide prefill program; narrow verifier widths replay during decode.
  const auto obsolete = [&](const auto& entry) {
    return entry.first > static_cast<std::int64_t>(impl_->config.block_size) &&
           entry.first != rows;
  };
  if (std::any_of(impl_->contexts.begin(), impl_->contexts.end(), obsolete)) {
    auto* scheduler = graph::default_scheduler();
    if (!scheduler) return LSE_ERROR(kInternal, "DFlash2 needs a scheduler");
    LSE_RETURN_IF_ERROR(scheduler->drain());
    std::erase_if(impl_->contexts, obsolete);
  }
  auto it = impl_->contexts.find(rows);
  if (it == impl_->contexts.end()) {
    LSE_ASSIGN_OR(auto pass, impl_->context_pass(rows));
    it = impl_->contexts.emplace(rows, std::move(pass)).first;
  }
  auto& pass = it->second;
  {
    DraftProfile profile("context-copy");
    LSE_RETURN_IF_ERROR(copy_slot(pass.features, input));
  }
  {
    DraftProfile profile("context-offset-upload");
    const float offset = static_cast<float>(first);
    LSE_RETURN_IF_ERROR(poke(pass.offset, std::span(&offset, 1)));
  }
  {
    DraftProfile profile("context-submit");
    LSE_RETURN_IF_ERROR(evaluate(pass.program, pass.roots));
  }
  for (std::size_t i = 0; i < impl_->layers.size(); ++i) {
    accept_cache_write(impl_->layers[i].keys, pass.keys[i]);
    accept_cache_write(impl_->layers[i].values, pass.values[i]);
  }
  impl_->position = first + static_cast<std::int32_t>(rows);
  impl_->live = std::min(impl_->capacity, impl_->live + static_cast<std::int32_t>(rows));
  return OkStatus();
}
Result<DFlash2Module::Impl::DraftPass*> DFlash2Module::Impl::run_draft(std::uint32_t anchor, std::int32_t first, std::uint32_t proposals, bool sampled) {
  if (anchor >= static_cast<std::uint32_t>(config.vocab_size) || first != position ||
      live <= 0 || proposals == 0 || proposals >= config.block_size ||
      static_cast<std::int64_t>(first) + proposals >= max_position)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 draft request");
  const auto output = dispatch::dflash2_output_plan(config.block_size, proposals);
  const auto key = std::pair(output.selector_positions, sampled);
  auto it = drafts.find(key);
  if (it == drafts.end()) {
    LSE_ASSIGN_OR(auto pass, draft_pass(output.head_rows, output.selector_positions, sampled));
    it = drafts.emplace(key, std::move(pass)).first;
  }
  auto& pass = it->second;
  const auto block = config.block_size;
  auto& image = pass.input_image;
  std::fill_n(image.begin(), block, static_cast<float>(config.mask_token));
  image[0] = image[block] = static_cast<float>(anchor);
  image[block + 1] = static_cast<float>(first);
  image[block + 2] = static_cast<float>(live);
  image[block + 3] = static_cast<float>(capacity);
  image[block + 4] = static_cast<float>(position % capacity);
  {
    DraftProfile profile("input-upload");
    LSE_RETURN_IF_ERROR(poke(pass.inputs, image));
    for (auto* view : {&pass.tokens, &pass.anchor, &pass.offset, &pass.meta}) {
      view->node()->materialized = true;
      view->node()->host_dirty = false;
      view->node()->device_dirty = true;
    }
  }
  {
    DraftProfile profile("draft-submit");
    LSE_RETURN_IF_ERROR(evaluate(pass.program, pass.roots));
  }
  return &pass;
}

Result<std::vector<std::uint32_t>> DFlash2Module::draft(
    std::uint32_t anchor, std::int32_t first, std::uint32_t proposals) {
  LSE_ASSIGN_OR(auto ready, impl_->run_draft(anchor, first, proposals, false));
  auto& pass = *ready;
  const auto output = dispatch::dflash2_output_plan(impl_->config.block_size, proposals);
  std::vector<std::uint32_t> path(output.selector_positions + 1);
  {
    DraftProfile profile("path-readback");
    LSE_RETURN_IF_ERROR(pass.path.to_host(path.data(), path.size() * sizeof(std::uint32_t)));
  }
  const auto validation = path.back();
  if (validation & 2u) return LSE_ERROR(kInternal, "invalid DFlash2 candidate token");
  if (validation & 1u) return LSE_ERROR(kInvalidArgument, "nonfinite DFlash2 selector score");
  if (validation != 0) return LSE_ERROR(kInternal, "invalid DFlash2 selector validation");
  path.resize(proposals);
  return path;
}
Result<DFlash2Proposal> DFlash2Module::draft_sampled(
    std::uint32_t anchor, std::int32_t first, std::uint32_t proposals,
    float temperature, runtime::SpeculativeSampler& sampler) {
  LSE_ASSIGN_OR(auto ready, impl_->run_draft(anchor, first, proposals, true));
  const auto positions = impl_->config.block_size - 1;
  const auto top = static_cast<std::uint32_t>(impl_->config.selector_top_k);
  std::vector<float> scores(static_cast<std::size_t>(positions) * top * top);
  std::vector<float> raw_ids(static_cast<std::size_t>(positions) * top);
  {
    DraftProfile profile("lattice-readback");
    LSE_RETURN_IF_ERROR(ready->scores.to_host(scores.data(), scores.size() * sizeof(float)));
    LSE_RETURN_IF_ERROR(ready->indices.to_host(raw_ids.data(), raw_ids.size() * sizeof(float)));
  }
  std::vector<std::uint32_t> ids(raw_ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (!std::isfinite(raw_ids[i]) || raw_ids[i] < 0 ||
        raw_ids[i] >= impl_->config.vocab_size || std::floor(raw_ids[i]) != raw_ids[i])
      return LSE_ERROR(kInternal, "invalid DFlash2 candidate token");
    ids[i] = static_cast<std::uint32_t>(raw_ids[i]);
  }
  return dflash2_sample_path(std::span(scores).first(static_cast<std::size_t>(proposals) * top * top),
      std::span(ids).first(static_cast<std::size_t>(proposals) * top), proposals,
      top, impl_->config.vocab_size, temperature, sampler);
}

}  // namespace lse::model
