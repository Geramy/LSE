// lemonseed model kernel: checkpoint names -> weight structs -> op specs.
//
// No math lives here. Every algorithm is in lse::ops and is shared with the
// other model kernels; this file only says which tensor plays which role and
// which parameters lemonseed picked.
//
// Transcribed from reference/lemonseed/lemonseed/layers/*.py @ 6fb97de, not
// inferred from weight names. The pieces most easily got wrong:
//   - RMSNorm is zero-centered: scale = 1 + weight
//   - MoD gates only the ROUTED experts; the shared expert is ungated
//   - inference bypasses MoD top-k entirely (it is non-causal) and applies the
//     per-token sigmoid to every position
//   - the expert band renormalizes over kept experts only
//   - the GDN runs v through the same qk_heads * head_dim width as q/k; the
//     config's gdn_v_heads is unused by the reference layer
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "lse/model/lemonseed.hpp"
#include "lse/model/registry.hpp"
#include "lse/ops/attention.hpp"
#include "lse/ops/linear_attention.hpp"
#include "lse/ops/moe.hpp"
#include "lse/ops/rope.hpp"

namespace lse::model {

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

}  // namespace

// --- MoE ---------------------------------------------------------------------

class LemonseedMoE final : public IFeedForward {
 public:
  std::string_view name() const noexcept override { return "lemonseed.moe"; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const std::string p(prefix);
    LSE_ASSIGN_OR(router_, b.require(p + ".moe.router.weight"));
    LSE_ASSIGN_OR(stacked_.gate, b.require(p + ".moe.w_gate"));
    LSE_ASSIGN_OR(stacked_.up, b.require(p + ".moe.w_up"));
    LSE_ASSIGN_OR(stacked_.down, b.require(p + ".moe.w_down"));

    // DeepSeek-style aux-loss-free balancing. The checkpoint carries a trained
    // expert_bias, but the reference only applies it when
    // LEMONSEED_MOE_BIAS_BALANCE is set — default off. Applying it
    // unconditionally reroutes every token (this bias has absmax ~1.7).
    auto bias = b.optional(p + ".moe.expert_bias");
    if (bias.ok() && ctx.config->moe_bias_balance) expert_bias_ = bias.release();

    for (std::int32_t i = 0; i < ctx.config->num_shared_experts; ++i) {
      const std::string sp = p + ".moe.shared." + std::to_string(i);
      LSE_ASSIGN_OR(Array w1, b.require(sp + ".w1.weight"));
      LSE_ASSIGN_OR(Array w2, b.require(sp + ".w2.weight"));
      LSE_ASSIGN_OR(Array w3, b.require(sp + ".w3.weight"));
      shared_.push_back({w1, w3, w2});
    }

    route_.num_experts = ctx.config->num_experts;
    route_.num_active = ctx.config->num_active_experts;
    route_.score_band = ctx.config->expert_score_band;
    return OkStatus();
  }

  // Returns routed-expert output only. The shared expert is deliberately
  // separate: MoD must not gate it.
  Result<Array> forward(const Array& x, Array* aux_loss,
                        const LayerContext& ctx) override {
    (void)aux_loss;
    (void)ctx;
    return routed_experts(x);
  }

  Result<Array> ungated(const Array& x) const override {
    return ops::shared_experts(x, shared_);
  }

  Result<Array> shared_expert(const Array& x) const {
    if (shared_.empty()) return LSE_ERROR(kNotFound, "no shared expert");
    return ops::shared_experts(x, shared_);
  }

  Result<Array> routed_experts(const Array& x) const {
    return ops::routed_experts(x, router_, expert_bias_, stacked_, route_);
  }

 private:
  Array router_;
  ops::ExpertWeights stacked_;
  Array expert_bias_;
  std::vector<ops::ExpertWeights> shared_;
  ops::RouteConfig route_;
};

// --- Gated GQA attention ------------------------------------------------------

class GatedAttention final : public IMixer {
 public:
  static constexpr std::string_view kName = "lemonseed.attention";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const std::string p = std::string(prefix) + ".mixer";
    LSE_ASSIGN_OR(w_.q_proj, b.require(p + ".q_proj.weight"));
    LSE_ASSIGN_OR(w_.k_proj, b.require(p + ".k_proj.weight"));
    LSE_ASSIGN_OR(w_.v_proj, b.require(p + ".v_proj.weight"));
    LSE_ASSIGN_OR(w_.o_proj, b.require(p + ".o_proj.weight"));
    LSE_ASSIGN_OR(w_.g_proj, b.require(p + ".g_proj.weight"));
    LSE_ASSIGN_OR(w_.q_norm, b.require(p + ".q_norm.weight"));
    LSE_ASSIGN_OR(w_.k_norm, b.require(p + ".k_norm.weight"));

    const Config& c = *ctx.config;
    spec_.q_heads = c.attn_q_heads;
    spec_.kv_heads = c.attn_kv_heads;
    spec_.head_dim = c.attn_head_dim;
    spec_.gate = ops::GateSource::kSeparateProj;
    spec_.mask = c.is_global_attention(ctx.layer_index) || c.sliding_window <= 0
                     ? graph::MaskKind::kCausal
                     : graph::MaskKind::kSlidingWindow;
    spec_.window = c.sliding_window;
    spec_.norm_eps = c.rms_eps;
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
    LSE_ASSIGN_OR(Array y,
                  ops::gated_attention(x, w_, spec_, rope_, state->position,
                                       &cache));
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

// --- Gated DeltaNet -----------------------------------------------------------

class GatedDeltaNet final : public IMixer {
 public:
  static constexpr std::string_view kName = "lemonseed.gdn";
  std::string_view name() const noexcept override { return kName; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const std::string p = std::string(prefix) + ".mixer";
    LSE_ASSIGN_OR(w_.in_proj_q, b.require(p + ".in_proj_q.weight"));
    LSE_ASSIGN_OR(w_.in_proj_k, b.require(p + ".in_proj_k.weight"));
    LSE_ASSIGN_OR(w_.in_proj_v, b.require(p + ".in_proj_v.weight"));
    LSE_ASSIGN_OR(w_.in_proj_a, b.require(p + ".in_proj_a.weight"));
    LSE_ASSIGN_OR(w_.in_proj_b, b.require(p + ".in_proj_b.weight"));
    LSE_ASSIGN_OR(w_.conv_q_w, b.require(p + ".conv_q.weight"));
    LSE_ASSIGN_OR(w_.conv_q_b, b.require(p + ".conv_q.bias"));
    LSE_ASSIGN_OR(w_.conv_k_w, b.require(p + ".conv_k.weight"));
    LSE_ASSIGN_OR(w_.conv_k_b, b.require(p + ".conv_k.bias"));
    LSE_ASSIGN_OR(w_.conv_v_w, b.require(p + ".conv_v.weight"));
    LSE_ASSIGN_OR(w_.conv_v_b, b.require(p + ".conv_v.bias"));
    LSE_ASSIGN_OR(w_.a_log, b.require(p + ".A_log"));
    LSE_ASSIGN_OR(w_.dt_bias, b.require(p + ".dt_bias"));
    LSE_ASSIGN_OR(w_.norm, b.require(p + ".norm.weight"));
    LSE_ASSIGN_OR(w_.gate_proj, b.require(p + ".out_gate.weight"));
    LSE_ASSIGN_OR(w_.out_proj, b.require(p + ".out_proj.weight"));

    const Config& c = *ctx.config;
    spec_.key_heads = spec_.value_heads = c.gdn_qk_heads;
    spec_.key_head_dim = spec_.value_head_dim = c.gdn_head_dim;
    spec_.norm_eps = c.rms_eps;
    return OkStatus();
  }

  Result<Array> forward(const Array& x, MixerState* state,
                        const LayerContext& ctx) override {
    (void)ctx;
    ops::GatedDeltaNetState carried;
    if (state != nullptr) {
      carried.recurrent = state->gdn_state;
      carried.conv_q = state->gdn_conv_q;
      carried.conv_k = state->gdn_conv_k;
      carried.conv_v = state->gdn_conv_v;
      carried.conv_qkv = state->gdn_conv_qkv;
    }
    LSE_ASSIGN_OR(Array y, ops::gated_delta_net(x, w_, spec_,
                                                state != nullptr ? &carried
                                                                 : nullptr));
    if (state != nullptr) {
      state->gdn_state = carried.recurrent;
      state->gdn_conv_q = carried.conv_q;
      state->gdn_conv_k = carried.conv_k;
      state->gdn_conv_v = carried.conv_v;
      state->gdn_conv_qkv = carried.conv_qkv;
    }
    return y;
  }

 private:
  ops::GatedDeltaNetWeights w_;
  ops::GatedDeltaNetSpec spec_;
};

// --- MoD ---------------------------------------------------------------------

class MixtureOfDepths final : public IModGate {
 public:
  Status load(WeightBinder& b, std::string_view prefix) override {
    const std::string p(prefix);
    LSE_ASSIGN_OR(weight_, b.require(p + ".mod.router.weight"));
    auto bias = b.optional(p + ".mod.router.bias");
    if (bias.ok()) bias_ = bias.release();
    return OkStatus();
  }

  Result<Array> gate_all(const Array& h, const Array& routed) const override {
    return ops::depth_gate(h, weight_, bias_, routed);
  }

 private:
  Array weight_;
  Array bias_;
};

std::unique_ptr<IFeedForward> make_lemonseed_moe() {
  return std::make_unique<LemonseedMoE>();
}

std::unique_ptr<IMixer> make_lemonseed_attention() {
  return std::make_unique<GatedAttention>();
}

std::unique_ptr<IMixer> make_lemonseed_gdn() {
  return std::make_unique<GatedDeltaNet>();
}

std::unique_ptr<IModGate> make_lemonseed_mod() {
  return std::make_unique<MixtureOfDepths>();
}

std::unique_ptr<HybridLM> make_lemonseed(const Config& config) {
  HybridLMSpec spec;  // lemonseed uses the default names and a tied head.
  return std::make_unique<HybridLM>(
      config, spec,
      [&config](std::int32_t layer) -> Result<std::unique_ptr<HybridBlock>> {
        auto mixer = config.is_attention_layer(layer) ? make_lemonseed_attention()
                                                      : make_lemonseed_gdn();
        return std::make_unique<HybridBlock>(std::move(mixer),
                                             make_lemonseed_moe(),
                                             /*zero_centered_norm=*/true,
                                             make_lemonseed_mod());
      });
}

Result<Array> lemonseed_moe_routed(const IFeedForward& moe, const Array& x) {
  return static_cast<const LemonseedMoE&>(moe).routed_experts(x);
}

Result<Array> lemonseed_moe_shared(const IFeedForward& moe, const Array& x) {
  return static_cast<const LemonseedMoE&>(moe).shared_expert(x);
}

// What makes a checkpoint lemonseed rather than any other GDN/MoE hybrid: the
// Mixture-of-Depths router, which Qwen3-Next and Qwen3.6 do not have, together
// with the stacked-expert MoE and the split (not fused) GDN projections.
namespace {

bool looks_like_lemonseed(const Config& config, const SafeTensors& weights) {
  (void)config;
  return weights.find("blocks.0.mod.router.weight") != nullptr &&
         weights.find("blocks.0.moe.w_gate") != nullptr &&
         weights.find("blocks.0.mixer.in_proj_q.weight") != nullptr;
}

}  // namespace

}  // namespace lse::model

LSE_REGISTER_MODEL(::lse::model::ModelArch{
    "lemonseed", 0, ::lse::model::looks_like_lemonseed,
    [](const ::lse::model::Config& c) { return ::lse::model::make_lemonseed(c); }})
