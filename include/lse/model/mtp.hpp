// The multi-token-prediction module that ships beside a Qwen3.5/3.8 checkpoint.
//
// Not a model, and the architecture registry is right to refuse it standalone:
// it has no embedding table, no LM head and one layer. It hangs off a loaded
// HybridLM and borrows both, computing
//
//   draft = lm_head(norm(layer0(fc(cat(norm_e(embed(t)), norm_h(h))))))
//
// where `h` is the decoder's hidden state for the position *before* the token
// `t`. So a row that carries token t_i and hidden h_{i-1} sits at absolute
// position i and proposes t_{i+1}. Getting that pairing wrong still produces
// fluent drafts and a much lower acceptance rate, which is the failure mode
// worth naming.
//
// The module proposes only. Nothing it returns can change what the engine
// emits — the decoder verifies every proposal — so its cache being warm is a
// speed question, never a correctness one.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/program.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"

namespace lse::model {

class MtpModule {
 public:
  // `path` is a checkpoint directory, a .safetensors file or an HF repo id,
  // resolved the way --model is. `model` must already be loaded: the module
  // reads its embedding table and head, so it outlives nothing and owns
  // neither.
  static Result<std::unique_ptr<MtpModule>> open(const std::string& path,
                                                 const Config& parent,
                                                 HybridLM& model);

  // Where `path` would be looked for when nobody named one: `<model>/mtp`, then
  // the repo id with `-MTP` folded into it. Empty when none of them resolves.
  static std::string find_beside(const std::string& model_name);

  // One pass of T rows starting at absolute position `first`. `hidden` is
  // T * hidden_size floats — the decoder's hidden state for each row's
  // preceding position — and `tokens` the token id sitting at each row's own
  // position. Returns the argmax of the last row: the proposal for
  // `first + T`. Every row is written to the module's own KV.
  Result<std::uint32_t> draft(std::span<const float> hidden,
                              std::span<const std::uint32_t> tokens,
                              std::int32_t first);

  // Drops the module's KV. The decoder's session and this must be cleared
  // together or the module drafts against another conversation's prefix.
  void reset();

  [[nodiscard]] std::int32_t position() const noexcept { return position_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  Status build(WeightBinder& binder);
  Result<graph::Array> record(std::int64_t rows);

  std::string path_;
  // The parent's config with the module's own one-layer stack in it: the
  // mixers read layer counts and widths off this, and a layer index that did
  // not answer is_attention_layer() would build a Gated DeltaNet against
  // self_attn tensors.
  Config config_;
  HybridLM* model_ = nullptr;
  std::unique_ptr<HybridBlock> block_;
  graph::Array fc_, pre_norm_hidden_, pre_norm_embedding_, final_norm_;

  MixerState state_;
  std::int32_t position_ = 0;

  struct Pass {
    graph::Program program;
    graph::Array hidden;   // [1, T, D] leaf, poked
    graph::Array tokens;   // [1, T] leaf, poked
    graph::Array meta;     // kv::step_meta_elems(1) leaf, poked
    graph::Array pick;
    std::int64_t rows = 0;
    graph::Node* keys = nullptr;
    graph::Node* values = nullptr;
  };
  Pass pass_;
};

}  // namespace lse::model
