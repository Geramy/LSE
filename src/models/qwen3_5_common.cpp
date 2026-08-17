// Qwen3.5 mixers: checkpoint names -> weight structs -> op specs.
//
// No math lives here. Every algorithm is in lse::ops and is shared with the
// other model kernels; this file only says which tensor plays which role and
// which parameters Qwen picked.
//
// Transcribed from transformers/models/qwen3_5/modeling_qwen3_5.py and checked
// against the tensor shapes in Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B and
// Qwen/Qwen3.5-35B-A3B. It is not a plain transformer: every layer whose index
// is not 3 mod 4 is a Gated DeltaNet, the same skeleton lemonseed uses. The
// pieces that differ from lemonseed and are easily got wrong:
//   - the GDN decay rate is exp(A_log), with no softplus around it
//   - the GDN output gate and the post-conv activation are SiLU, not sigmoid
//   - the GDN head norm is plain RMSNorm; every other norm is zero-centered
//   - attention has no separate gate weight: q_proj is twice as wide and each
//     head's second head_dim is its gate
//   - RoPE is HF rotate_half over a quarter of a 256-wide head
#include "lse/model/qwen3_5_common.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "lse/ops/attention.hpp"
#include "lse/ops/linear_attention.hpp"
#include "lse/ops/rope.hpp"

namespace lse::model::qwen3_5 {

using graph::Array;

namespace {

Result<ops::RopeTables> shared_rope(const Config& c) {
  const std::int32_t max_seq = c.kv_capacity();
  struct Key {
    std::int32_t dim = 0;
    std::int32_t max_seq = 0;
    float theta = 0.0f;
  };
  static Key have;
  static ops::RopeTables tables;
  if (tables.cos.valid() && have.dim == c.rope_dim && have.max_seq == max_seq &&
      have.theta == c.rope_theta) {
    return tables;
  }
  LSE_ASSIGN_OR(tables, ops::build_rope(c.rope_dim, max_seq, c.rope_theta));
  have = {c.rope_dim, max_seq, c.rope_theta};
  return tables;
}

// Where channel `d` of an engine-convention head vector comes from in an HF
// one. HF rotates with rotate_half, pairing channel i with i + rope_dim/2;
// graph::rope rotates adjacent pairs (2j, 2j+1). Reordering the projection
// rows once at load makes the two agree, and since q and k are reordered
// identically their dot product — the only thing attention reads — is
// unchanged. Channels at or past rope_dim are not rotated and stay put.
std::int64_t rope_source(std::int64_t d, std::int64_t rope_dim) {
  if (d >= rope_dim) return d;
  const std::int64_t j = d / 2;
  return d % 2 == 0 ? j : j + rope_dim / 2;
}

// Rows of a [.., kernel] conv weight, unchanged: Qwen stores conv1d.weight as
// [channels, 1, kernel] and graph::causal_conv1d wants [channels, kernel].
std::vector<std::int64_t> identity_rows(std::int64_t n) {
  std::vector<std::int64_t> order(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  return order;
}

class Qwen35Attention final : public IMixer {
 public:
  static constexpr std::string_view kName = "qwen3_5.attention";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".self_attn";
    const auto qh = static_cast<std::int64_t>(c.attn_q_heads);
    const auto kvh = static_cast<std::int64_t>(c.attn_kv_heads);
    const auto hd = static_cast<std::int64_t>(c.attn_head_dim);
    const auto rd = static_cast<std::int64_t>(c.rope_dim);
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);

    // q_proj is [q_heads * 2 * head_dim, hidden] and HF splits it per head:
    // head h owns rows [h*2*hd, (h+1)*2*hd), of which the first head_dim is q
    // and the second is the output gate. Gathering all the q rows first and
    // all the gate rows after turns that into the contiguous upper-half gate
    // GateSource::kFusedInQProj already expects, and the q half carries the
    // RoPE reordering at the same time.
    std::vector<std::int64_t> q_rows;
    q_rows.reserve(static_cast<std::size_t>(2 * qh * hd));
    for (std::int64_t h = 0; h < qh; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) {
        q_rows.push_back(h * 2 * hd + rope_source(d, rd));
      }
    }
    for (std::int64_t h = 0; h < qh; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) q_rows.push_back(h * 2 * hd + hd + d);
    }
    LSE_ASSIGN_OR(w_.q_proj, b.require_rows(p + ".q_proj.weight", q_rows,
                                            Shape{2 * qh * hd, hidden}));

    std::vector<std::int64_t> k_rows;
    k_rows.reserve(static_cast<std::size_t>(kvh * hd));
    for (std::int64_t h = 0; h < kvh; ++h) {
      for (std::int64_t d = 0; d < hd; ++d) {
        k_rows.push_back(h * hd + rope_source(d, rd));
      }
    }
    LSE_ASSIGN_OR(w_.k_proj, b.require_rows(p + ".k_proj.weight", k_rows,
                                            Shape{kvh * hd, hidden}));

    // The head norms are applied before the rotation, so they follow the same
    // reordering; RMSNorm's scaling is permutation-invariant, its weight is not.
    std::vector<std::int64_t> norm_rows;
    norm_rows.reserve(static_cast<std::size_t>(hd));
    for (std::int64_t d = 0; d < hd; ++d) norm_rows.push_back(rope_source(d, rd));
    LSE_ASSIGN_OR(w_.q_norm,
                  b.require_rows(p + ".q_norm.weight", norm_rows, Shape{hd}));
    LSE_ASSIGN_OR(w_.k_norm,
                  b.require_rows(p + ".k_norm.weight", norm_rows, Shape{hd}));

    LSE_ASSIGN_OR(w_.v_proj, b.require(p + ".v_proj.weight"));
    LSE_RETURN_IF_ERROR(
        expect_shape(w_.v_proj, p + ".v_proj.weight", Shape{kvh * hd, hidden}));
    LSE_ASSIGN_OR(w_.o_proj, b.require(p + ".o_proj.weight"));
    LSE_RETURN_IF_ERROR(
        expect_shape(w_.o_proj, p + ".o_proj.weight", Shape{hidden, qh * hd}));

    spec_.q_heads = c.attn_q_heads;
    spec_.kv_heads = c.attn_kv_heads;
    spec_.head_dim = c.attn_head_dim;
    spec_.gate = ops::GateSource::kFusedInQProj;
    // Qwen has no sliding window: every full-attention layer is global.
    spec_.mask = graph::MaskKind::kCausal;
    spec_.window = 0;
    spec_.norm_eps = c.rms_eps;
    spec_.zero_centered_norm = true;
    spec_.kv_length = c.kv_capacity();
    LSE_ASSIGN_OR(rope_, shared_rope(c));
    return OkStatus();
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    (void)ctx;
    if (state == nullptr) return ops::gated_attention(x, w_, spec_, rope_, 0);

    ops::AttentionCache cache;
    cache.keys = state->key_cache;
    cache.values = state->value_cache;
    cache.pos = state->kv_pos;
    cache.capacity = spec_.kv_length;
    cache.used = state->position;
    LSE_ASSIGN_OR(Array y, ops::gated_attention(x, w_, spec_, rope_,
                                                state->position, &cache));
    state->key_cache = cache.keys;
    state->value_cache = cache.values;
    if (cache.pos.valid()) state->kv_pos = cache.pos;
    return y;
  }

 private:
  ops::GatedAttentionWeights w_;
  ops::GatedAttentionSpec spec_;
  ops::RopeTables rope_;
};

class Qwen35GatedDeltaNet final : public IMixer {
 public:
  static constexpr std::string_view kName = "qwen3_5.gdn";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".linear_attn";
    const auto kh = static_cast<std::int64_t>(c.gdn_qk_heads);
    const auto vh = static_cast<std::int64_t>(c.gdn_v_heads);
    const auto hd = static_cast<std::int64_t>(c.gdn_head_dim);
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const std::int64_t conv_dim = 2 * kh * hd + vh * hd;

    LSE_ASSIGN_OR(w_.in_proj_qkv, b.require(p + ".in_proj_qkv.weight"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.in_proj_qkv, p + ".in_proj_qkv.weight",
                                     Shape{conv_dim, hidden}));
    LSE_ASSIGN_OR(w_.gate_proj, b.require(p + ".in_proj_z.weight"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.gate_proj, p + ".in_proj_z.weight",
                                     Shape{vh * hd, hidden}));
    LSE_ASSIGN_OR(w_.in_proj_a, b.require(p + ".in_proj_a.weight"));
    LSE_RETURN_IF_ERROR(
        expect_shape(w_.in_proj_a, p + ".in_proj_a.weight", Shape{vh, hidden}));
    LSE_ASSIGN_OR(w_.in_proj_b, b.require(p + ".in_proj_b.weight"));
    LSE_RETURN_IF_ERROR(
        expect_shape(w_.in_proj_b, p + ".in_proj_b.weight", Shape{vh, hidden}));

    // [conv_dim, 1, kernel] in the checkpoint, [conv_dim, kernel] in the graph.
    const auto kernel = static_cast<std::int64_t>(c.gdn_conv_kernel);
    LSE_ASSIGN_OR(w_.conv_w,
                  b.require_rows(p + ".conv1d.weight", identity_rows(conv_dim),
                                 Shape{conv_dim, kernel}));

    LSE_ASSIGN_OR(w_.a_log, b.require(p + ".A_log"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.a_log, p + ".A_log", Shape{vh}));
    LSE_ASSIGN_OR(w_.dt_bias, b.require(p + ".dt_bias"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.dt_bias, p + ".dt_bias", Shape{vh}));
    LSE_ASSIGN_OR(w_.norm, b.require(p + ".norm.weight"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.norm, p + ".norm.weight", Shape{hd}));
    LSE_ASSIGN_OR(w_.out_proj, b.require(p + ".out_proj.weight"));
    LSE_RETURN_IF_ERROR(expect_shape(w_.out_proj, p + ".out_proj.weight",
                                     Shape{hidden, vh * hd}));

    spec_.key_heads = c.gdn_qk_heads;
    spec_.value_heads = c.gdn_v_heads;
    spec_.key_head_dim = spec_.value_head_dim = c.gdn_head_dim;
    spec_.conv_bias = false;
    spec_.decay_per_value_head = true;
    spec_.decay = ops::DecayRate::kExpALog;
    spec_.gate = ops::GateActivation::kSiLU;
    spec_.conv_activation = true;
    spec_.query_scale = 1.0f / std::sqrt(static_cast<float>(c.gdn_head_dim));
    spec_.layout = ops::ProjLayout::kFusedQKV;
    spec_.norm_eps = c.rms_eps;
    // Qwen3_5RMSNormGated is weight * x, not (1 + weight) * x. The rest of the
    // model's norms are zero-centered; this one is not.
    spec_.zero_centered_norm = false;
    return OkStatus();
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    (void)ctx;
    ops::GatedDeltaNetState carried;
    if (state != nullptr) {
      carried.recurrent = state->gdn_state;
      carried.conv_qkv = state->gdn_conv_qkv;
    }
    LSE_ASSIGN_OR(Array y,
                  ops::gated_delta_net(x, w_, spec_,
                                       state != nullptr ? &carried : nullptr));
    if (state != nullptr) {
      state->gdn_state = carried.recurrent;
      state->gdn_conv_qkv = carried.conv_qkv;
    }
    return y;
  }

 private:
  ops::GatedDeltaNetWeights w_;
  ops::GatedDeltaNetSpec spec_;
};

}  // namespace

std::unique_ptr<IMixer> make_attention() {
  return std::make_unique<Qwen35Attention>();
}

std::unique_ptr<IMixer> make_gdn() {
  return std::make_unique<Qwen35GatedDeltaNet>();
}

HybridBlockSpec block_spec() {
  return HybridBlockSpec{".input_layernorm.weight",
                         ".post_attention_layernorm.weight"};
}

HybridLMSpec lm_spec(const Config& config) {
  HybridLMSpec spec;
  spec.embed_name = "model.language_model.embed_tokens.weight";
  spec.final_norm_name = "model.language_model.norm.weight";
  spec.block_prefix = std::string(kBlockPrefix);
  spec.lm_head_name = config.tie_word_embeddings ? "" : "lm_head.weight";
  spec.zero_centered_norm = true;
  // The image tower and the speculative-decoding head are in the same file and
  // are not part of text generation. Naming them here is what makes every
  // other unclaimed tensor a load error.
  spec.ignored_prefixes = {"model.visual.", "mtp."};
  spec.gdn_state_heads = config.gdn_v_heads;
  spec.gdn_state_dim = config.gdn_head_dim;
  spec.gdn_conv_width =
      (2 * config.gdn_qk_heads + config.gdn_v_heads) * config.gdn_head_dim;
  return spec;
}

Status expect_shape(const Array& a, std::string_view name, Shape want) {
  if (a.shape() == want) return OkStatus();
  return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                   a.shape().to_string(), " but the config implies ",
                   want.to_string());
}

}  // namespace lse::model::qwen3_5
