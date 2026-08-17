// Qwen3.5 dense: the shared Qwen3.5 mixers with a plain SwiGLU feed-forward.
//
// Read off Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-4B. Despite the name, "dense"
// describes the feed-forward only — the stack is the same 3x Gated DeltaNet,
// 1x gated attention hybrid the MoE variant uses.
#include <memory>
#include <string>

#include "lse/model/registry.hpp"
#include "lse/ops/activation.hpp"
#include "lse/model/qwen3_5_common.hpp"

namespace lse::model {

namespace {

class Qwen35MLP final : public IFeedForward {
 public:
  std::string_view name() const noexcept override { return "qwen3_5.mlp"; }

  Status load(WeightBinder& b, std::string_view prefix,
              const LayerContext& ctx) override {
    const Config& c = *ctx.config;
    const std::string p = std::string(prefix) + ".mlp";
    const auto hidden = static_cast<std::int64_t>(c.hidden_size);
    const auto inter = static_cast<std::int64_t>(c.mlp_intermediate);

    LSE_ASSIGN_OR(gate_, b.require(p + ".gate_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(gate_, p + ".gate_proj.weight",
                                              Shape{inter, hidden}));
    LSE_ASSIGN_OR(up_, b.require(p + ".up_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(up_, p + ".up_proj.weight",
                                              Shape{inter, hidden}));
    LSE_ASSIGN_OR(down_, b.require(p + ".down_proj.weight"));
    LSE_RETURN_IF_ERROR(qwen3_5::expect_shape(down_, p + ".down_proj.weight",
                                              Shape{hidden, inter}));
    return OkStatus();
  }

  Result<Array> forward(const Array& x, Array* aux_loss,
                        const LayerContext& ctx) override {
    (void)aux_loss;
    (void)ctx;
    return ops::swiglu(x, gate_, up_, down_);
  }

 private:
  Array gate_, up_, down_;
};

// A Qwen3.5 checkpoint with no stacked expert tensor. The two markers are
// jointly required and the MoE marker is required absent, so this and
// looks_like_qwen3_5_moe can never both hold: one tensor decides, and it
// decides in opposite directions. lemonseed names its blocks `blocks.N.mixer.*`
// and carries none of these three, so it cannot reach either test.
bool looks_like_qwen3_5_dense(const Config& config, const SafeTensors& weights) {
  (void)config;
  return weights.find(qwen3_5::kGdnMarker) != nullptr &&
         weights.find(qwen3_5::kMoeMarker) == nullptr &&
         weights.find("language_model.model.layers.0.mlp.gate_proj.weight") !=
             nullptr;
}

std::unique_ptr<HybridLM> make_qwen3_5(const Config& config) {
  return std::make_unique<HybridLM>(
      config, qwen3_5::lm_spec(config),
      [config](std::int32_t layer) -> Result<std::unique_ptr<HybridBlock>> {
        auto mixer = config.is_attention_layer(layer) ? qwen3_5::make_attention()
                                                      : qwen3_5::make_gdn();
        return std::make_unique<HybridBlock>(
            std::move(mixer), std::make_unique<Qwen35MLP>(),
            /*zero_centered_norm=*/false, /*mod=*/nullptr, qwen3_5::block_spec());
      });
}

}  // namespace

}  // namespace lse::model

LSE_REGISTER_MODEL(::lse::model::ModelArch{
    "qwen3.5", 0, ::lse::model::looks_like_qwen3_5_dense,
    [](const ::lse::model::Config& c) { return ::lse::model::make_qwen3_5(c); }})
