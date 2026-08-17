// Pieces every Qwen3.5 checkpoint shares. Dense and MoE differ only in the FFN
// and in whether the head is tied, so the mixers, the tensor prefix and the
// stack-level spec live here and each kernel adds its own feed-forward.
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/weights.hpp"

namespace lse::model::qwen3_5 {

// Qwen3.5 is a multimodal wrapper, so the decoder sits under `language_model`
// and the vision tower and MTP head share the file. Both released dense repos
// and the MoE repo use exactly these names.
inline constexpr std::string_view kBlockPrefix = "model.language_model.layers";
inline constexpr std::string_view kLayer0 = "model.language_model.layers.0.";

// The linear-attention projection every Qwen3.5 layer 0 carries, dense or MoE.
// Layer 0 is always a linear_attention layer: full_attention_interval is 4 and
// the rule is (i + 1) % interval, so the first full-attention layer is 3.
inline constexpr std::string_view kGdnMarker =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";

// The tensor that separates the two: only the MoE stacks routed experts.
inline constexpr std::string_view kMoeMarker =
    "model.language_model.layers.0.mlp.experts.gate_up_proj";

std::unique_ptr<IMixer> make_attention();
std::unique_ptr<IMixer> make_gdn();

HybridBlockSpec block_spec();

// Everything above the blocks. `head` is the lm_head tensor name, empty when
// the checkpoint ties it to the embedding.
HybridLMSpec lm_spec(const Config& config);

// Fails naming the tensor when its shape is not what the config implies. The
// config is authoritative for counts the tensors cannot state (top-k, the
// layer schedule); the tensors are authoritative for widths. Disagreement is
// an error either way, never a silent reinterpretation.
Status expect_shape(const graph::Array& a, std::string_view name, Shape want);

}  // namespace lse::model::qwen3_5
