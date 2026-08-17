// Qwen3.5 MoE: the shared Qwen3.5 mixers with a routed feed-forward.
//
// Read off Qwen/Qwen3.5-35B-A3B and transcribed from
// transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py. The routing shape is
// 256 experts, top-8, expert width 512 — lemonseed's 8-experts-top-2 does not
// generalise, so nothing here is inherited from it.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lse/model/registry.hpp"
#include "lse/ops/moe.hpp"
#include "lse/model/qwen3_5_common.hpp"

namespace lse::model {

namespace {

class Qwen35MoE final : public IFeedForward {
 public:
  std::string_view name() const noexcept override { return "qwen3_5.moe"; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".mlp";
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const auto experts = static_cast<std::int64_t>(c.num_experts);
    const auto inter = static_cast<std::int64_t>(c.expert_intermediate);
    const auto shared_inter =
        static_cast<std::int64_t>(c.shared_expert_intermediate);

    LSE_ASSIGN_OR(router_, b.require(p + ".gate.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(router_, p + ".gate.weight",
                                              Shape{experts, hidden}));

    // gate_up_proj is one [E, 2*inter, hidden] tensor whose per-expert output
    // rows are gate then up, which HF splits with chunk(2, -1) on every token.
    // Splitting it into the two [E, out, in] matrices ops::routed_experts wants
    // costs one pass over the tensor here instead of a slice per forward. The
    // staging buffer is the size of the fused tensor, so this peaks at roughly
    // 1.5x its footprint per layer.
    const std::string gu = p + ".experts.gate_up_proj";
    std::vector<std::int64_t> gate_rows;
    std::vector<std::int64_t> up_rows;
    gate_rows.reserve(static_cast<std::size_t>(experts * inter));
    up_rows.reserve(static_cast<std::size_t>(experts * inter));
    for (std::int64_t e = 0; e < experts; ++e) {
      for (std::int64_t i = 0; i < inter; ++i) {
        gate_rows.push_back(e * 2 * inter + i);
        up_rows.push_back(e * 2 * inter + inter + i);
      }
    }
    LSE_ASSIGN_OR(stacked_.gate,
                  b.require_rows(gu, gate_rows, Shape{experts, inter, hidden}));
    LSE_ASSIGN_OR(stacked_.up,
                  b.require_rows(gu, up_rows, Shape{experts, inter, hidden}));
    LSE_ASSIGN_OR(stacked_.down, b.require(p + ".experts.down_proj"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(stacked_.down,
                                              p + ".experts.down_proj",
                                              Shape{experts, hidden, inter}));

    const std::string sp = p + ".shared_expert";
    ops::ExpertWeights shared;
    LSE_ASSIGN_OR(shared.gate, b.require(sp + ".gate_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(shared.gate,
                                              sp + ".gate_proj.weight",
                                              Shape{shared_inter, hidden}));
    LSE_ASSIGN_OR(shared.up, b.require(sp + ".up_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(shared.up, sp + ".up_proj.weight",
                                              Shape{shared_inter, hidden}));
    LSE_ASSIGN_OR(shared.down, b.require(sp + ".down_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(shared.down,
                                              sp + ".down_proj.weight",
                                              Shape{hidden, shared_inter}));
    shared_.push_back(shared);

    LSE_ASSIGN_OR(shared_gate_, b.require(p + ".shared_expert_gate.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(
        shared_gate_, p + ".shared_expert_gate.weight", Shape{1, hidden}));

    route_.num_experts = c.num_experts;
    route_.num_active = c.num_active_experts;
    // Plain top-k over softmaxed logits, no band and no router bias, but the
    // kept weights are rescaled to sum to 1 — with 8 of 256 experts they
    // otherwise carry only a fraction of the probability mass.
    route_.score_band = 1.0f;
    route_.renormalize = true;
    return OkStatus();
  }

  // Routed experts only. The shared expert is deliberately separate so the
  // block adds it outside anything that gates the routed path.
  Result<Array> forward(const Array& x, Array* aux_loss,
                        const LayerContext& ctx) override {
    (void)aux_loss;
    (void)ctx;
    return ops::routed_experts(x, router_, Array{}, stacked_, route_);
  }

  Result<Array> ungated(const Array& x) const override {
    return ops::depth_gate(x, shared_gate_, Array{},
                           ops::shared_experts(x, shared_));
  }

 private:
  Array router_;
  ops::ExpertWeights stacked_;
  std::vector<ops::ExpertWeights> shared_;
  Array shared_gate_;
  ops::RouteConfig route_;
};

// A Qwen3.5 checkpoint that stacks routed experts. The same tensor that this
// requires present, looks_like_qwen3_5_dense requires absent, so the two are
// disjoint by construction rather than by priority.
bool looks_like_qwen3_5_moe(const Config& config, const SafeTensors& weights) {
  (void)config;
  return weights.find(qwen3_5::kGdnMarker) != nullptr &&
         weights.find(qwen3_5::kMoeMarker) != nullptr &&
         weights.find("model.language_model.layers.0.mlp.gate.weight") != nullptr;
}

std::unique_ptr<HybridLM> make_qwen3_5_moe(const Config& config) {
  return std::make_unique<HybridLM>(
      config, qwen3_5::lm_spec(config),
      [config](std::int32_t layer) -> Result<std::unique_ptr<HybridBlock>> {
        auto mixer = config.is_attention_layer(layer) ? qwen3_5::make_attention()
                                                      : qwen3_5::make_gdn();
        return std::make_unique<HybridBlock>(
            std::move(mixer), std::make_unique<Qwen35MoE>(),
            /*zero_centered_norm=*/true, /*mod=*/nullptr, qwen3_5::block_spec());
      });
}

}  // namespace

}  // namespace lse::model

LSE_REGISTER_MODEL(::lse::model::ModelArch{
    "qwen3.5-moe", 0, ::lse::model::looks_like_qwen3_5_moe,
    [](const ::lse::model::Config& c) {
      return ::lse::model::make_qwen3_5_moe(c);
    }})
