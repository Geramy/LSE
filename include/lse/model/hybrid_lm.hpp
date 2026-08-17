// The stack: embed -> N x HybridBlock -> final norm -> LM head.
//
// Model-independent. Everything that differs between lemonseed and Qwen3.6 is
// either in the blocks a BlockFactory hands back or in HybridLMSpec, so a model
// kernel declares its tensor names and its per-layer mixer choice and nothing
// else.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/program.hpp"
#include "lse/model/config.hpp"
#include "lse/model/layer.hpp"

namespace lse::model {

using graph::Array;

struct HybridLMSpec {
  std::string embed_name = "embed.weight";
  std::string final_norm_name = "final_norm.weight";
  std::string block_prefix = "blocks";
  // Empty means the head reuses the embedding table.
  std::string lm_head_name;
  bool zero_centered_norm = true;

  // Checkpoint tensors that are deliberately not part of the decoder stack —
  // Qwen3.5 ships a vision tower and a multi-token-prediction head in the same
  // file. Everything else must be claimed by some layer: an unclaimed tensor is
  // a builder that misread the checkpoint, and it fails load rather than
  // quietly producing a model with pieces missing.
  std::vector<std::string> ignored_prefixes;

  // Recurrent state for the linear-attention layers, [B, heads, dim, dim].
  // Zero derives it from the Config the way lemonseed's mixer does, where the
  // value heads run at the key width.
  std::int32_t gdn_state_heads = 0;
  std::int32_t gdn_state_dim = 0;
  // Channel width of a single fused q|k|v conv tail. Zero means the model
  // convolves the three streams separately, which is what a split projection
  // layout needs.
  std::int32_t gdn_conv_width = 0;
};

class HybridLM {
 public:
  using BlockFactory =
      std::function<Result<std::unique_ptr<HybridBlock>>(std::int32_t layer)>;

  HybridLM(Config config, HybridLMSpec spec, BlockFactory factory)
      : config_(std::move(config)),
        spec_(std::move(spec)),
        factory_(std::move(factory)) {}

  Status load(WeightBinder& binder);

  // Token ids [B,T] (held as f32, matching how the graph carries indices).
  Result<Array> embed(const Array& tokens) const;

  // Post-final-norm hidden states [B,T,D]. `states` may be null for a stateless
  // prefill; otherwise it must hold one entry per layer and is updated in
  // place. `trace`, when non-null, is filled with each block's output — that is
  // what lets a mismatch be localized to a layer instead of just to the logits.
  Result<Array> hidden(const Array& tokens, std::vector<MixerState>* states,
                       Array* aux_loss, std::vector<Array>* trace = nullptr);

  // [.., D] -> [.., vocab]. Applied to only the positions a caller needs: at
  // long context the full [B,T,vocab] tensor does not fit.
  [[nodiscard]] Result<Array> lm_head(const Array& hidden_states) const;

  [[nodiscard]] std::vector<MixerState> make_states() const {
    return std::vector<MixerState>(blocks_.size());
  }

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] std::size_t num_layers() const noexcept { return blocks_.size(); }
  [[nodiscard]] HybridBlock& block(std::size_t i) noexcept { return *blocks_[i]; }

 private:
  Config config_;
  HybridLMSpec spec_;
  BlockFactory factory_;
  std::vector<std::unique_ptr<HybridBlock>> blocks_;
  Array embed_weight_;
  Array final_norm_weight_;
  Array lm_head_weight_;

  struct ForwardCache {
    graph::Program program;
    Array tokens;
    Array hidden;
    Array pos;
    const void* states = nullptr;
    std::int64_t seq = -1;
    std::vector<graph::Node*> kv_leaves;
  };
  ForwardCache cache_;
};

}  // namespace lse::model
