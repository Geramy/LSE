// Qwen3.5 dense: the shared Qwen3.5 mixers with a plain SwiGLU feed-forward.
//
// Read off Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-4B. Despite the name, "dense"
// describes the feed-forward only — the stack is the same 3x Gated DeltaNet,
// 1x gated attention hybrid the MoE variant uses.
#include <memory>
#include <string>

#include "lse/model/registry.hpp"
#include "lse/model/qwen3_5_common.hpp"

namespace lse::model {

namespace {

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
            std::move(mixer), qwen3_5::make_mlp(),
            /*zero_centered_norm=*/false, /*mod=*/nullptr, qwen3_5::block_spec());
      });
}

}  // namespace

}  // namespace lse::model

LSE_REGISTER_MODEL(::lse::model::ModelArch{
    "qwen3.5", 0, ::lse::model::looks_like_qwen3_5_dense,
    [](const ::lse::model::Config& c) { return ::lse::model::make_qwen3_5(c); }})
