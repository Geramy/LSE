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

// Padded batch buckets.
//
// graph::FusionGroup::signature() mixes every dimension of every node, and the
// batch axis is dim(0) of essentially every activation — so a free batch axis
// multiplies the shape set the JIT has to compile by the number of distinct
// batch sizes, and a miss is not a slow path: the group lands on the host
// interpreter at roughly 0.5 tok/s against 100. Rows in a pass are therefore
// padded up to a rung, and the rung — never the true row count — is what reaches
// the graph.
//
// Powers of two, matching the prefill chunk ladder, and chosen so a family
// change happens *at* a rung rather than inside one: MatmulKernel::specialize()
// leaves the LDS GEMV path at m >= 16, so 16 is a rung and the choice is a
// function of the bucket, which is in the key, rather than of the true batch,
// which is not.
//
// The ladder stops at 32. A batch that fits no bucket is an error naming the
// size, not a silent widening: padding 33 rows to 64 would double the work of
// every row in the pass, and the caller that assembled 33 rows is the one that
// knows how to split them.
inline constexpr std::int32_t kBatchRungs[] = {1, 2, 4, 8, 16, 32};

Result<std::int32_t> batch_bucket(std::int32_t rows);

struct HybridLMSpec {
  std::string embed_name = "embed.weight";
  std::string final_norm_name = "final_norm.weight";
  std::string block_prefix = "blocks";
  // Empty means the head reuses the embedding table.
  std::string lm_head_name;
  bool zero_centered_norm = true;

  // Checkpoint tensors this architecture deliberately does not read — Qwen3.5
  // ships a vision tower in the same file. Everything else must be claimed by
  // some layer: an unclaimed tensor is a builder that misread the checkpoint,
  // and it fails load rather than quietly producing a model with pieces
  // missing.
  //
  // A refusal is reported at load with its count, its size and its reason. Not
  // reading part of a checkpoint is a limitation of the build, and a silent
  // skip is exactly how a half-loaded model comes to look like a working one.
  struct Refusal {
    std::string prefix;
    std::string reason;
  };
  std::vector<Refusal> refused;

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
  //
  // `rows` is how many of tokens' leading rows carry a real sequence. The batch
  // axis is padded up to a bucket so the bucket — never the true row count — is
  // what the JIT keys on; the rows past `rows` run the same kernels on their own
  // pad blocks and their output is discarded. 0 means every row is real.
  Result<Array> hidden(const Array& tokens, std::vector<MixerState>* states,
                       Array* aux_loss, std::vector<Array>* trace = nullptr,
                       std::int32_t rows = 0);

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
    Array meta;
    const void* states = nullptr;
    std::int64_t seq = -1;
    std::vector<graph::Node*> kv_leaves;
  };
  ForwardCache cache_;
};

}  // namespace lse::model
