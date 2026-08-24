// The stack: embed -> N x HybridBlock -> final norm -> LM head.
//
// Model-independent. Everything that differs between lemonseed and Qwen3.6 is
// either in the blocks a BlockFactory hands back or in HybridLMSpec, so a model
// kernel declares its tensor names and its per-layer mixer choice and nothing
// else.
#pragma once

#include <array>

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

// Where each row of a batched pass sits.
//
// `first[r]` is the absolute position of row r's first query token; a negative
// entry means row r holds no sequence this step. One entry per row of the
// padded bucket, so the vector's own length says which bucket the caller built
// for and a mismatch is caught rather than silently reinterpreted.
//
// This is the whole difference between a batch and a wide single sequence. A
// pass with one shared position gives every row but one the wrong causal mask,
// the wrong softmax bound and the wrong RoPE angle — three separate wrong
// answers, all of which still produce fluent text.
struct StepRows {
  std::vector<std::int32_t> first;

  [[nodiscard]] std::int32_t bucket() const noexcept {
    return static_cast<std::int32_t>(first.size());
  }
  // Rows up to and including the last live one. The kernels take it as a prefix
  // bound and skip everything past it; a hole below it is still a row, and it
  // is made harmless by its own zero length, not by this.
  [[nodiscard]] std::int32_t live_prefix() const noexcept {
    std::int32_t last = 0;
    for (std::size_t r = 0; r < first.size(); ++r) {
      if (first[r] >= 0) last = static_cast<std::int32_t>(r) + 1;
    }
    return last;
  }
};

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
  // `rows` says where each row of the pass sits and which rows hold a sequence
  // at all. The batch axis is padded up to a bucket so the bucket — never the
  // true row count — is what the JIT keys on; a row holding no sequence runs the
  // same kernels on its own pad blocks and answers zero. Null means one
  // sequence spread over every row, all at the state's current position, which
  // is what a single-session decode and every prefill pass want.
  //
  // `replaces_previous` says this pass stands in for the one that just ran
  // rather than following it: the carried state on the input nodes is still
  // what that pass started from, so it is held where it is instead of being
  // folded forward. That is the whole of a speculative rollback for the
  // recurrent layers — a rejected proposal's Gated DeltaNet state is never
  // committed, because the pass that produced it is overwritten. The caller
  // must put the mixer positions back first; see MixerState::position.
  Result<Array> hidden(const Array& tokens, std::vector<MixerState>* states,
                       Array* aux_loss, std::vector<Array>* trace = nullptr,
                       const StepRows* rows = nullptr,
                       bool replaces_previous = false);

  // Puts every attention layer's write cursor back to `position`. The paged
  // pool is overwritten in place by the pass that follows, so this plus a
  // `replaces_previous` pass is what un-does a speculative step.
  void rewind(std::vector<MixerState>& states, std::int32_t position) const;

  // [.., D] -> [.., vocab]. Applied to only the positions a caller needs: at
  // long context the full [B,T,vocab] tensor does not fit.
  [[nodiscard]] Result<Array> lm_head(const Array& hidden_states) const;

  [[nodiscard]] std::vector<MixerState> make_states() const {
    return std::vector<MixerState>(blocks_.size() * state_shards());
  }

  // The program the last hidden() built or replayed. Exposed so a test can
  // assert the carry-ownership invariant: a state the next pass reads must not
  // share bytes with anything this pass writes.
  [[nodiscard]] const graph::Program& retained_program() const noexcept {
    for (const ForwardCache& c : caches_) {
      if (c.t_key == 1) return c.program;
    }
    static const graph::Program kNone;
    return kNone;
  }

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] std::size_t num_layers() const noexcept { return blocks_.size(); }

  // How many states each layer keeps. A tensor split shards the KV pool and the
  // recurrent state along the head axis with the weights that produce them, so
  // each member carries its own; every other scheme keeps one.
  [[nodiscard]] std::size_t state_shards() const noexcept;

  // How many MixerStates a session has to carry: one per layer per shard.
  [[nodiscard]] std::size_t state_slots() const noexcept {
    return blocks_.size() * state_shards();
  }

  // Which device holds layer `i`. Contiguous blocks, so a run of layers stays
  // put and the activation crosses once at each boundary instead of at every
  // layer. Load and forward both ask this, and they have to agree: weights
  // placed one way and a KV pool placed another means every attention group
  // is pulled back to whoever holds the pool.
  [[nodiscard]] std::size_t member_for_layer(std::int32_t i) const noexcept;
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
    // The pass shape this slot serves: the token count of its retained
    // program. -1 is an empty slot.
    std::int64_t t_key = -1;
    // Chain identity. pass_id names the build that retained this slot;
    // prev_pass names the pass that ran immediately before it. A replay whose
    // carry-ins are another pass's out-nodes is only coherent when that exact
    // pass has just run again.
    std::uint64_t pass_id = 0;
    std::uint64_t prev_pass = 0;
    // The state arrays as this pass's retain stamped them. A replay leaves the
    // MixerStates pointing at whatever the last BUILD stamped, and a later
    // rebuild would read that other chain's nodes; restoring these on every
    // replay keeps the handoff coherent for whichever pass comes next.
    struct StateStamp {
      Array gdn, cq, ck, cv, cqkv, keys, values, meta;
    };
    std::vector<StateStamp> state_stamp;
  };
  // One slot per pass shape, so decode retention (T=1) stops evicting the
  // prefill program (T=N) — with one slot every server request rebuilt its
  // prefill graph because the first decode had overwritten it. Four covers
  // decode, an MTP verify width, and two prompt shapes; eviction is
  // round-robin among full slots.
  std::array<ForwardCache, 8> caches_;
  // Chain bookkeeping across passes, build or replay. Folding a slot's
  // carries is only correct when that same pass ran immediately before; at a
  // ladder handoff the in-node IS the previous chunk's out node and already
  // holds the fresh state, so a fold would swap stale bytes in.
  std::uint64_t pass_counter_ = 0;
  std::uint64_t last_pass_id_ = 0;
  std::size_t next_cache_ = 0;
  [[nodiscard]] ForwardCache& cache_slot(std::int64_t t) {
    for (ForwardCache& c : caches_) {
      if (c.t_key == t) return c;
    }
    for (ForwardCache& c : caches_) {
      if (c.t_key < 0) {
        c.t_key = t;
        return c;
      }
    }
    ForwardCache& c = caches_[next_cache_++ % caches_.size()];
    c = ForwardCache{};
    c.t_key = t;
    return c;
  }
  // Groups the last pass could not put on the device. A pass that ran anywhere
  // else cannot be replaced: the carried state travels differently there, and
  // the pass replacing it would start from what it is meant to discard.
  std::uint32_t last_pass_host_groups_ = 0;
};

}  // namespace lse::model
