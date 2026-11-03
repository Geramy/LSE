// LFM3: Liquid AI's conv/attention hybrid, read off
// LiquidAI/LFM3-1.7B-MLX-4bit and LiquidAI/LFM3-Omni-2B-MLX-4bit and
// cross-checked against transformers/models/lfm3/modeling_lfm3.py.
//
// A "conv" layer runs a gated depthwise width-7 convolution: in_proj emits
// B|C|x, the stream convolved is B*x, the result is gated by C and read out
// through out_proj. An attention layer is llama-style with per-head q/k
// RMSNorm and a per-head sigmoid output gate — gate_proj is [q_heads,
// hidden], expanded to per-channel rows at load because merging heads gives
// each of a head's channels the same gate value either way. Both attention
// kinds carry their own rotary profile (theta and rotary width), and every
// block is a sandwich: post_operator_norm and post_ffn_norm sit inside the
// residual adds, so they live inside the mixer and FFN impls.
//
// Text only: Omni checkpoints carry audio, vision and depth towers beside
// this stack; the spec's refusal list names them and reports the cost.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/qwen3_5_common.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/activation.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/norm.hpp"
#include "lse/ops/rope.hpp"

namespace lse::model {

namespace {

constexpr std::string_view kBlockPrefix = "language_model.model.layers";

// Same convention as qwen3_5: the engine's rotary pairs adjacent channels
// (2j, 2j+1) while HF's rotate_half pairs d with d + rope_dim/2, so the
// projection and head-norm rows are permuted once at load. The dot products
// attention reads are unchanged because q and k share the permutation.
std::int64_t rope_source(std::int64_t d, std::int64_t rope_dim) {
  if (d >= rope_dim) return d;
  const std::int64_t j = d / 2;
  return d % 2 == 0 ? j : j + rope_dim / 2;
}

class Lfm3Attention final : public IMixer {
 public:
  static constexpr std::string_view kName = "lfm3.attention";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".self_attn";
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const auto qh = static_cast<std::int64_t>(c.attn_q_heads);
    const auto kvh = static_cast<std::int64_t>(c.attn_kv_heads);
    const auto hd = static_cast<std::int64_t>(c.attn_head_dim);
    const bool sliding = c.is_sliding_attention(ctx.layer_index);
    const auto rd = static_cast<std::int64_t>(
        sliding ? c.sliding_rope_dim : c.rope_dim);
    const float theta = sliding ? c.sliding_rope_theta : c.rope_theta;

    std::vector<std::int64_t> q_rows;
    q_rows.reserve(static_cast<std::size_t>(qh * hd));
    for (std::int64_t h = 0; h < qh; ++h)
      for (std::int64_t d = 0; d < hd; ++d)
        q_rows.push_back(h * hd + rope_source(d, rd));
    LSE_ASSIGN_OR(w_.q_proj, b.require_rows(p + ".q_proj.weight", q_rows,
                                          Shape{qh * hd, hidden}));

    std::vector<std::int64_t> k_rows;
    k_rows.reserve(static_cast<std::size_t>(kvh * hd));
    for (std::int64_t h = 0; h < kvh; ++h)
      for (std::int64_t d = 0; d < hd; ++d)
        k_rows.push_back(h * hd + rope_source(d, rd));
    LSE_ASSIGN_OR(w_.k_proj, b.require_rows(p + ".k_proj.weight", k_rows,
                                          Shape{kvh * hd, hidden}));

    LSE_ASSIGN_OR(w_.v_proj, b.require(p + ".v_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(
        w_.v_proj, p + ".v_proj.weight", Shape{kvh * hd, hidden}));
    LSE_ASSIGN_OR(w_.o_proj, b.require(p + ".o_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(
        w_.o_proj, p + ".o_proj.weight", Shape{hidden, qh * hd}));

    // The HF gate is one scalar per head ([q_heads, hidden]). Expanding each
    // head row across its head_dim channels at load turns it into the
    // per-channel gate GatedAttention multiplies in — identical math, since
    // merged channel h*hd+d belongs to head h for every d.
    std::vector<std::int64_t> g_rows;
    g_rows.reserve(static_cast<std::size_t>(qh * hd));
    for (std::int64_t h = 0; h < qh; ++h)
      for (std::int64_t d = 0; d < hd; ++d) g_rows.push_back(h);
    LSE_ASSIGN_OR(w_.g_proj, b.require_rows(p + ".gate_proj.weight", g_rows,
                                          Shape{qh * hd, hidden}));

    // The head norms run before the rotation, so their channel order follows
    // the same projection reordering.
    std::vector<std::int64_t> norm_rows;
    norm_rows.reserve(static_cast<std::size_t>(hd));
    for (std::int64_t d = 0; d < hd; ++d)
      norm_rows.push_back(rope_source(d, rd));
    LSE_ASSIGN_OR(w_.q_norm,
                  b.require_rows(p + ".q_norm.weight", norm_rows, Shape{hd}));
    LSE_ASSIGN_OR(w_.k_norm,
                  b.require_rows(p + ".k_norm.weight", norm_rows, Shape{hd}));

    spec_.q_heads = static_cast<std::int32_t>(qh);
    spec_.kv_heads = static_cast<std::int32_t>(kvh);
    spec_.head_dim = c.attn_head_dim;
    spec_.gate = ops::GateSource::kSeparateProj;
    spec_.mask = sliding ? graph::MaskKind::kSlidingWindow
                         : graph::MaskKind::kCausal;
    spec_.window = sliding ? static_cast<std::int32_t>(c.sliding_window) : 0;
    spec_.norm_eps = c.rms_eps;
    spec_.zero_centered_norm = false;
    spec_.kv_length = c.kv_capacity();
    spec_.kv_cache_dtype = c.kv_cache_dtype;
    spec_.sparse_attention = c.sparse_attention;
    LSE_ASSIGN_OR(rope_, ops::build_rope(static_cast<std::int32_t>(rd), c.kv_capacity(), theta));

    auto post = b.optional(std::string(prefix) +
                           ".post_operator_norm.weight");
    if (post.ok()) post_norm_ = post.release();
    return OkStatus();
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    LSE_ASSIGN_OR(Array y, attend(x, state, ctx.attention_phase));
    if (post_norm_.valid())
      y = ops::rms_norm(y, post_norm_, ctx.config->rms_eps);
    return y;
  }

 private:
  Result<Array> attend(const Array& x, MixerState* state,
                       ops::AttentionExecutionPhase phase) {
    if (state == nullptr) {
      return ops::gated_attention(x, w_, spec_, rope_, 0, nullptr, phase);
    }
    MixerState& st = *state;
    ops::AttentionCache cache;
    cache.keys = st.key_cache;
    cache.values = st.value_cache;
    cache.table = st.paged.table;
    cache.meta = st.kv_meta;
    cache.paged = &st.paged;
    cache.capacity = spec_.kv_length;
    cache.used = st.position;
    cache.tree_positions = st.tree_positions;
    LSE_ASSIGN_OR(Array y, ops::gated_attention(x, w_, spec_, rope_,
                                                st.position, &cache, phase));
    st.key_cache = cache.keys;
    st.value_cache = cache.values;
    st.tree_keys = cache.tree_keys;
    st.tree_values = cache.tree_values;
    return y;
  }

  ops::GatedAttentionWeights w_;
  ops::GatedAttentionSpec spec_;
  ops::RopeTables rope_;
  Array post_norm_;
};

class Lfm3Conv final : public IMixer {
 public:
  static constexpr std::string_view kName = "lfm3.conv";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".conv";
    hidden_ = static_cast<std::int64_t>(c.hidden_size);
    const auto kernel = static_cast<std::int64_t>(c.gdn_conv_kernel);

    LSE_ASSIGN_OR(in_proj_, b.require(p + ".in_proj.weight"));
    LSE_RETURN_IF_ERROR(
        qwen3_5::expect_shape(in_proj_, p + ".in_proj.weight",
                              Shape{3 * hidden_, hidden_}));
    // MLX stores the depthwise filter [channels, kernel, 1]; the trailing
    // axis is the group-count of 1, so require_as flattens it to [C, K].
    LSE_ASSIGN_OR(conv_w_,
                  b.require_as(p + ".conv.weight", Shape{hidden_, kernel}));
    LSE_ASSIGN_OR(out_proj_, b.require(p + ".out_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(
        out_proj_, p + ".out_proj.weight", Shape{hidden_, hidden_}));

    // conv_bias=True checkpoints (LFM2 style) bias all three projections —
    // none of the published LFM3 checkpoints use it, so it fails here rather
    // than silently dropping a term.
    for (const char* name : {".in_proj.bias", ".conv.bias", ".out_proj.bias"}) {
      auto bias = b.optional(p + name);
      if (bias.ok()) {
        return LSE_ERROR(kUnimplemented, "'", std::string(prefix),
                         "' sets conv_bias; lfm3 convs are read bias-free");
      }
    }
    // The conv kernel reads weight and bias through one element type —
    // the zero pad has to carry the filter's dtype or the bind refuses.
    bias_ = Array::zeros(Shape{hidden_}, conv_w_.dtype());

    auto post = b.optional(std::string(prefix) +
                           ".post_operator_norm.weight");
    if (post.ok()) post_norm_ = post.release();
    return OkStatus();
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    const Array bcx = graph::linear(x, in_proj_);
    const Array gate_b = graph::slice(bcx, -1, 0, hidden_);
    const Array gate_c = graph::slice(bcx, -1, hidden_, 2 * hidden_);
    const Array xv = graph::slice(bcx, -1, 2 * hidden_, 3 * hidden_);
    // The convolved stream; state tails are FP32 so the mix happens in F32.
    const Array h = graph::cast(graph::mul(gate_b, xv), DType::kF32);

    Array y;
    if (state != nullptr && state->gdn_conv_qkv.valid()) {
      const Array prev = state->gdn_conv_qkv;
      if (state->pad_tail_rows.valid()) {
        // A padded pass: the tail is the last kernel-1 real rows, wherever
        // the pass's last real token falls — never the pad columns.
        const Shape& sx = h.shape();
        const Array flat = graph::reshape(h, Shape{sx.dim(0) * sx.dim(1), sx.dim(2)});
        state->gdn_conv_qkv = graph::reshape(
            graph::gather_rows(flat, state->pad_tail_rows), prev.shape());
      } else {
        state->gdn_conv_qkv = graph::conv_tail(prev, h);
      }
      y = graph::causal_conv1d(h, conv_w_, bias_, prev);
    } else {
      y = graph::causal_conv1d(h, conv_w_, bias_);
    }
    if (y.dtype() != gate_c.dtype()) y = graph::cast(y, gate_c.dtype());
    y = graph::mul(gate_c, y);
    y = graph::linear(y, out_proj_);
    if (post_norm_.valid())
      y = ops::rms_norm(y, post_norm_, ctx.config->rms_eps);
    return y;
  }

 private:
  std::int64_t hidden_ = 0;
  Array in_proj_, conv_w_, out_proj_, bias_, post_norm_;
};

class Lfm3Mlp final : public IFeedForward {
 public:
  static constexpr std::string_view kName = "lfm3.mlp";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".feed_forward";
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const auto inter = static_cast<std::int64_t>(c.mlp_intermediate);

    LSE_ASSIGN_OR(w1_, b.require(p + ".w1.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(w1_, p + ".w1.weight",
                                              Shape{inter, hidden}));
    LSE_ASSIGN_OR(w3_, b.require(p + ".w3.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(w3_, p + ".w3.weight",
                                              Shape{inter, hidden}));
    LSE_ASSIGN_OR(w2_, b.require(p + ".w2.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(w2_, p + ".w2.weight",
                                              Shape{hidden, inter}));

    auto post = b.optional(std::string(prefix) + ".post_ffn_norm.weight");
    if (post.ok()) post_norm_ = post.release();
    return OkStatus();
  }

  Result<Array> forward(const Array& x, Array* aux_loss,
                        const LayerContext& ctx) override {
    (void)aux_loss;
    Array y = ops::swiglu(x, w1_, w3_, w2_);
    if (post_norm_.valid())
      y = ops::rms_norm(y, post_norm_, ctx.config->rms_eps);
    return y;
  }

 private:
  Array w1_, w3_, w2_, post_norm_;
};

HybridBlockSpec block_spec() {
  return HybridBlockSpec{".operator_norm.weight", ".ffn_norm.weight"};
}

// The conv tail rides the GDN channel-width slot: every conv layer gets a
// [B, kernel-1, hidden] window of its own conv input. gdn_state stays zero —
// there is no recurrent state anywhere in the stack.
HybridLMSpec lm_spec(const Config& config) {
  HybridLMSpec spec;
  spec.embed_name = "language_model.model.embed_tokens.weight";
  // Despite the name this is the model's final norm — modeling_lfm3 applies
  // it to the last hidden state, not to the embedding output.
  spec.final_norm_name = "language_model.model.embedding_norm.weight";
  spec.block_prefix = std::string(kBlockPrefix);
  spec.lm_head_name =
      config.tie_word_embeddings ? "" : "language_model.lm_head.weight";
  spec.zero_centered_norm = false;
  spec.gdn_conv_width = config.hidden_size;
  spec.gdn_state_heads = 0;
  spec.gdn_state_dim = 0;
  // Omni checkpoints carry non-text towers beside this stack; refusing them
  // is what keeps text decode honest about what it cannot encode.
  spec.refused.push_back(
      {"vision_tower.", "this build decodes text only; the Lfm3Omni vision "
                       "tower is not implemented, so image tokens cannot be "
                       "encoded"});
  spec.refused.push_back(
      {"audio_tower.", "this build decodes text only; the Lfm3Omni audio "
                       "tower is not implemented, so audio tokens cannot be "
                       "encoded"});
  spec.refused.push_back(
      {"audio_projector.", "this build decodes text only; the audio "
                           "projector feeds the refused audio tower"});
  spec.refused.push_back(
      {"depth_decoder.", "this build decodes text only; the depth decoder "
                         "head is not implemented"});
  spec.refused.push_back(
      {"multi_modal_projector.", "this build decodes text only; the "
                                 "multimodal projector feeds refused towers"});
  spec.refused.push_back(
      {"codebook_embedding.", "this build decodes text only; the audio "
                              "codebook table is not implemented"});
  return spec;
}

// Every LFM3 layer 0 is a conv layer carrying conv.conv.weight, and the
// model-level norm is embedding_norm — together they separate this stack
// from qwen3_5 (linear_attn.*, norm.weight) and lemonseed (blocks.N.mixer.*)
// regardless of how similar the layer schedules look.
bool looks_like_lfm3(const Config& config, const SafeTensors& weights) {
  (void)config;
  const std::string block = std::string(kBlockPrefix) + ".0.conv";
  return weights.find(block + ".conv.weight") != nullptr &&
         weights.find(block + ".in_proj.weight") != nullptr &&
         weights.find(block + ".out_proj.weight") != nullptr &&
         weights.find("language_model.model.embedding_norm.weight") !=
             nullptr;
}

std::unique_ptr<HybridLM> make_lfm3(const Config& config) {
  return std::make_unique<HybridLM>(
      config, lm_spec(config),
      [config](std::int32_t layer) -> Result<std::unique_ptr<HybridBlock>> {
        std::unique_ptr<IMixer> mixer =
            config.is_attention_layer(layer)
                ? std::unique_ptr<IMixer>(std::make_unique<Lfm3Attention>())
                : std::unique_ptr<IMixer>(std::make_unique<Lfm3Conv>());
        return std::make_unique<HybridBlock>(
            std::move(mixer), std::make_unique<Lfm3Mlp>(),
            /*zero_centered_norm=*/false, /*mod=*/nullptr, block_spec());
      });
}

}  // namespace

}  // namespace lse::model

LSE_REGISTER_MODEL(::lse::model::ModelArch{
    "lfm3", 0, &::lse::model::looks_like_lfm3,
    [](const ::lse::model::Config& c) { return ::lse::model::make_lfm3(c); }})
