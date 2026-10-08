// Prefill, then one token at a time.
//
// The hybrid part is that a block's state is not one shape: an attention layer
// carries a KV pair allocated at the engine length, a GDN layer carries a
// fixed-size recurrent matrix.
// MixerState holds whichever the block needs and the generator never has to
// know which kind a given layer is.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/program.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/runtime/draft_tree.hpp"
#include "lse/runtime/draft_width.hpp"
#include "lse/runtime/perplexity.hpp"
#include "lse/runtime/sampler.hpp"
#include "lse/runtime/prefill_batch.hpp"
#include "lse/runtime/session.hpp"

namespace lse::runtime {

using model::HybridLM;

inline constexpr std::uint32_t kDefaultMtpDepth = 3;
inline constexpr std::uint32_t kMaxMtpDepth = 7;

[[nodiscard]] constexpr std::uint32_t dflash2_verify_depth(
    std::uint32_t draft_block_size) noexcept {
  return draft_block_size >= 2 && draft_block_size <= 8 ? draft_block_size - 1 : 0;
}

[[nodiscard]] constexpr bool valid_mtp_depth(std::uint32_t depth) noexcept {
  return depth >= 1 && depth <= kMaxMtpDepth;
}

[[nodiscard]] constexpr std::uint32_t mtp_verify_rows(
    std::uint32_t depth, std::uint64_t remaining_tokens,
    std::int64_t remaining_positions) noexcept {
  if (!valid_mtp_depth(depth) || remaining_tokens == 0 || remaining_positions <= 0)
    return 0;
  std::uint64_t rows = depth + 1;
  if (rows > remaining_tokens) rows = remaining_tokens;
  if (rows > static_cast<std::uint64_t>(remaining_positions))
    rows = static_cast<std::uint64_t>(remaining_positions);
  return static_cast<std::uint32_t>(rows);
}

// No limit on generated tokens: generation runs until a stop token, the
// token callback, or a full context.
inline constexpr std::int32_t kNoTokenLimit = std::numeric_limits<std::int32_t>::max();

struct GenerationLimits {
  // At most this many generated tokens. kNoTokenLimit (the default) imposes
  // none; 0 runs the prompt and generates nothing. Whatever the limit, the
  // prompt plus the generated tokens never exceed the engine's KV length
  // (model.config().kv_capacity()): there generation stops, as kContextFull.
  std::int32_t max_tokens = kNoTokenLimit;
  // Generation stops on any of these. Empty means run to a limit.
  std::vector<std::uint32_t> stop_tokens;
  std::uint32_t mtp_depth = kDefaultMtpDepth;
  // With an adaptive MTP policy (Generator::use_mtp), a sampled request
  // chains as deep as the policy picks; false holds it to mtp_depth.
  bool adaptive_mtp = true;
};

// Why a generation ended.
enum class StopReason : std::uint8_t {
  kNone,         // not run, or prefill only
  kStopToken,    // the model produced a stop token
  kMaxTokens,    // limits.max_tokens generated
  kContextFull,  // the prompt and the generated tokens fill the KV length
  kCallback,     // the token callback asked to stop (stop string, cancel)
};
[[nodiscard]] constexpr const char* to_string(StopReason r) noexcept {
  switch (r) {
    case StopReason::kNone: return "none";
    case StopReason::kStopToken: return "stop_token";
    case StopReason::kMaxTokens: return "max_tokens";
    case StopReason::kContextFull: return "context_full";
    case StopReason::kCallback: return "callback";
  }
  return "none";
}

struct GenerationStats {
  std::int32_t prompt_tokens = 0;
  std::int32_t generated_tokens = 0;
  StopReason stop_reason = StopReason::kNone;
  // Tokens the session's context holds at the end (the whole prompt plus the
  // generated tokens) and the most it can hold.
  std::int32_t context_tokens = 0;
  std::int32_t context_length = 0;
  std::uint64_t prefill_ns = 0;
  // Wall time after the first token callback, including later model steps,
  // sampling and callbacks. Zero when no post-prefill step was attempted.
  std::uint64_t decode_ns = 0;
  std::uint32_t device_groups = 0;
  std::uint32_t host_groups = 0;
  std::uint32_t kernels_launched = 0;
  std::uint32_t phase_groups = 0;
  std::uint32_t phase_ideal_launches = 0;
  std::uint32_t views_aliased = 0;
  std::uint32_t host_fallbacks = 0;
  // Execution streams: how many the device offers, how many the scheduler put
  // work on, what the cross-stream ordering cost, and the length of the
  // longest dependency chain against the groups it was drawn from.
  std::uint32_t streams_available = 1;
  std::uint32_t streams_used = 1;
  std::uint32_t stream_waits = 0;
  std::uint32_t peer_migrations = 0;
  std::uint64_t peer_bytes = 0;
  std::uint32_t stream_chain = 0;

  // Share of dispatched groups that did not have to wait for the one before
  // them. 0 when every group is on the critical path — which is the honest
  // answer when a step is a straight line, and the number to watch when the
  // partitioner starts producing wider steps.
  [[nodiscard]] double spread() const noexcept {
    if (device_groups == 0 || stream_chain == 0) return 0.0;
    if (stream_chain >= device_groups) return 0.0;
    return 1.0 - static_cast<double>(stream_chain) /
                     static_cast<double>(device_groups);
  }
  std::uint64_t partition_ns = 0;
  std::uint64_t emit_ns = 0;
  std::uint64_t launch_ns = 0;
  std::uint64_t sync_ns = 0;
  std::uint64_t jit_memory_hits = 0;
  std::uint64_t jit_disk_hits = 0;
  std::uint64_t jit_compiles = 0;
  std::uint64_t jit_compile_ns = 0;
  // One verifier pass can compare several draft proposals. spec_proposed
  // counts the proposals the passes carried, spec_tested those the target
  // checked (a pass stops checking at its first rejection), spec_accepted
  // those it kept. spec_plain_steps are passes that carried none.
  std::uint32_t spec_steps = 0;
  std::uint32_t spec_accepted = 0;
  std::uint32_t spec_tested = 0;
  std::uint32_t spec_proposed = 0;
  std::uint32_t spec_plain_steps = 0;
  // Steps that verified a draft tree, and the rows those passes carried.
  std::uint32_t tree_steps = 0;
  std::uint32_t tree_rows = 0;
  std::array<std::uint32_t, 7> spec_tested_by_position{}, spec_accepted_by_position{};
  std::array<double, 7> spec_overlap_sum{}, spec_candidate_mass_sum{}, spec_deterministic_mass_sum{};
  std::uint32_t mtp_depth = 0;
  std::uint32_t dflash2_depth = 0;
  // The verify width followed DraftWidthPolicy rather than a fixed width.
  bool spec_adaptive = false;
  // Where a speculative step's time goes: the decoder passes that verify a
  // proposal (two of them when it is rejected) against the module pass that
  // made it. The second is what speculation costs whether or not it pays.
  std::uint64_t spec_verify_ns = 0;
  std::uint64_t spec_draft_ns = 0;
  std::uint32_t spec_verify_passes = 0;

  [[nodiscard]] double acceptance_rate() const noexcept {
    if (spec_tested == 0) return 0.0;
    return static_cast<double>(spec_accepted) / static_cast<double>(spec_tested);
  }
  // Rows per verify pass: the anchor plus the proposals it carried.
  [[nodiscard]] double mean_verify_width() const noexcept {
    if (spec_steps == 0) return 0.0;
    return static_cast<double>(spec_proposed + spec_steps) / static_cast<double>(spec_steps);
  }

  // The first generated token is sampled from prefill logits. Only later
  // emitted tokens belong to the decode interval, including speculative ones.
  [[nodiscard]] std::int32_t decoded_tokens() const noexcept {
    return generated_tokens > 0 ? generated_tokens - 1 : 0;
  }

  [[nodiscard]] double prompt_tokens_per_second() const noexcept {
    if (prefill_ns == 0 || prompt_tokens <= 0) return 0.0;
    return static_cast<double>(prompt_tokens) * 1e9 /
           static_cast<double>(prefill_ns);
  }

  [[nodiscard]] double decode_tokens_per_second() const noexcept {
    if (decode_ns == 0 || decoded_tokens() == 0) return 0.0;
    return static_cast<double>(decoded_tokens()) * 1e9 /
           static_cast<double>(decode_ns);
  }
};

class Generator {
 public:
  Generator(HybridLM& model, SamplingParams params, PrefillBatch prefill = {})
      : model_(model), sampler_(params), prefill_batch_(prefill) {}

  // Called with each token as it is produced. Returning false stops early,
  // which is how a server cancels a stream mid-flight.
  using TokenCallback = std::function<bool(std::uint32_t)>;

  // `prompt` is already tokenized. Returns the generated ids, not including
  // the prompt.
  //
  // The session owns the cache: passing the same one again continues that
  // conversation, and only the tokens past session.position() are fed through
  // the model. Passing a fresh session is a cold start.
  Result<std::vector<std::uint32_t>> generate(
      Session& session, const std::vector<std::uint32_t>& prompt,
      const GenerationLimits& limits, const TokenCallback& on_token = {});

  // Convenience for a one-shot run: owns a private session internally.
  Result<std::vector<std::uint32_t>> generate(
      const std::vector<std::uint32_t>& prompt, const GenerationLimits& limits,
      const TokenCallback& on_token = {});

  [[nodiscard]] const GenerationStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const std::vector<std::string>& host_group_reasons() const
      noexcept {
    return host_reasons_;
  }
  [[nodiscard]] Sampler& sampler() noexcept { return sampler_; }

  // Proposals are checked by the decoder; limits.mtp_depth selects the chain
  // length. The module and session caches are reset together by generate().
  // With `widths`, each step of a sampled request chains as deep as that
  // policy chooses (up to kMaxMtpDepth) and verifies the prefix it keeps;
  // without, and for greedy requests, limits.mtp_depth every step.
  void use_mtp(model::MtpModule& mtp, DraftWidthPolicy* widths = nullptr) {
    mtp_ = &mtp; dflash2_ = nullptr; widths_ = widths;
    mtp.set_scored(widths != nullptr);
  }
  // With `widths`, each step of a sampled request verifies the prefix of the
  // draft's block that policy chooses (and may skip the draft); without, and
  // for greedy requests, the whole block.
  // With `tree`, a step may verify a draft tree built from the draft's whole
  // candidate lattice (runtime::DraftTree) instead of a chain; with `widths`
  // the policy picks the tree's size or a chain from measured costs.
  void use_dflash2(model::DFlash2Module& draft, DraftWidthPolicy* widths = nullptr,
                   bool tree = false) noexcept {
    dflash2_ = &draft; mtp_ = nullptr; widths_ = widths; tree_ = tree;
  }

  // One forward pass of a prompt: `width` rows, the first `valid` of them
  // real tokens and the rest padding past the last one.
  struct PassPlan {
    std::size_t width = 0;
    std::size_t valid = 0;
  };
  // The passes a prompt of `tokens` runs as under this generator's prefill
  // batch, in order.
  [[nodiscard]] std::vector<PassPlan> prefill_passes(std::size_t tokens) const;

  // Makes every kernel a request can launch resident on the device before the
  // first request arrives (Scheduler::prepare): the target pass of each pass
  // shape prefill_passes can produce, the full-width pass at each KV pool
  // doubling, and with a DFlash2 draft every verify width, its heads and
  // prefix commits, and the draft's context and draft passes. Nothing runs;
  // it is what a server does at load.
  Status prepare_kernels();

  // Scores tokens[first_target ..], each given the tokens before it: the
  // negative log-likelihood in nats and, on request, this run's top_k ids with
  // their log-probabilities and its log-probability of each probe id
  // (probe_ids holds probe_k ids per scored token). `session` must be empty;
  // `tokens` runs through the same prefill passes a prompt of that length takes
  // (prefill_passes, the prompt attention phase, the session's KV storage), and
  // the LM head is then applied to the scored rows of each pass, kScoreRows at
  // a time. The log-softmax statistics are reduced on the device
  // (logits.lse_pick.v1) and the top k is selected there (topk_pairs, which
  // needs a vocabulary of at least 4096). No draft module may be attached:
  // scoring is the target model alone.
  static constexpr std::size_t kScoreRows = 128;
  Result<TokenScores> score(Session& session, std::span<const std::uint32_t> tokens,
                            std::size_t first_target, std::size_t top_k = 0,
                            std::span<const std::uint32_t> probe_ids = {},
                            std::size_t probe_k = 0);

  // Last position of a [.., T, D] hidden state, reshaped to [.., D].
  // `valid` rows of the sequence axis are real (0: all of them).
  static Result<graph::Array> last_hidden(const graph::Array& hidden,
                                          std::int64_t valid = 0);

 private:
  // Runs `tokens` from the session's current position and returns the logits
  // for the last one.
  Result<std::vector<float>> step(Session& session,
                                  const std::vector<std::uint32_t>& tokens);

  // One decode token through the retained head program, returning its root:
  // the logit row, or the argmax index when `greedy`. Built once against the
  // forward cache's hidden root, then replayed — this is what keeps the
  // per-token lm_head off the partitioner.
  Result<graph::Array> decode_head(Session& session, std::uint32_t token,
                                   bool greedy);
  // Device-greedy decode step: forward + argmax on device, 4 bytes back.
  Result<std::uint32_t> greedy_step(Session& session, std::uint32_t token);
  Status poke_decode_ids(std::uint32_t token);

  HybridLM& model_;
  Sampler sampler_;
  PrefillBatch prefill_batch_;
  std::unique_ptr<Session> owned_;
  GenerationStats stats_;
  std::vector<std::string> host_reasons_;

  // The lm_head subgraph, retained like HybridLM's forward program so decode
  // token 2 onward replays instead of re-partitioning. `compute` is exactly
  // the head's own nodes: clearing only those leaves the forward graph's
  // materialized flags alone.
  struct DecodeHead {
    graph::Program program;
    graph::Array hidden;
    graph::Array logits;
    graph::Array pick;
    // Sampling decided by a top-k with no history penalty reads each row's
    // best logits from the device instead of whole rows.
    graph::Array top;
    std::vector<graph::NodePtr> compute;
    bool greedy = false;
    std::uint32_t top_k = 0;
  };
  DecodeHead head_;
  // Persistent [1,1] token slot for decode; poked on the host and uploaded by
  // whichever program consumes it, never re-allocated per token.
  graph::Array decode_ids_;

  // The current verifier width retains its head across passes.
  struct SpecHead {
    graph::Program program;
    graph::Array hidden;
    graph::Array logits;
    graph::Array pick;
    // Sampling decided by a top-k with no history penalty reads each row's
    // best logits from the device instead of whole rows.
    graph::Array top;
    std::vector<graph::NodePtr> compute;
    bool greedy = false;
    std::uint32_t top_k = 0;
  };
  SpecHead spec_;
  graph::Array spec_ids_;
  // Preserve recorded heads and token slots when the verifier width changes.
  std::unordered_map<std::size_t, SpecHead> spec_by_m_;
  std::unordered_map<std::size_t, graph::Array> spec_ids_by_m_;
  // Logits reach the host only when the sampler needs more than an argmax,
  // and then a row at a time as the acceptance walk asks (spec_logit_rows):
  // rows [0, spec_rows_ready_) of spec_logits_ hold the last pass's values.
  std::vector<float> spec_logits_;
  std::size_t spec_rows_ready_ = 0;
  Status spec_logit_rows(std::size_t rows);
  // With spec_.top: every row's (value, index) candidates of the last pass,
  // read in one copy, and row i's split into spec_top_values_/spec_top_ids_.
  std::vector<float> spec_top_;
  bool spec_top_ready_ = false;
  std::vector<float> spec_top_values_;
  std::vector<std::uint32_t> spec_top_ids_;
  Status spec_top_row(std::size_t row);
  // The verify pass is a launch burst from its drained start to the drained
  // readback of its answers (backend::IBackend::begin_launch_burst).
  backend::IBackend* burst_ = nullptr;
  Status begin_verify_burst();
  void end_verify_burst() noexcept;
  // The decoder's hidden state for the last prompt position, which is the
  // module's input for the first proposal of the generation.
  graph::Array prefill_tail_;

  model::MtpModule* mtp_ = nullptr;
  model::DFlash2Module* dflash2_ = nullptr;
  DraftWidthPolicy* widths_ = nullptr;
  bool tree_ = false;
  // Appends the draft context for a tree path's rows of the last verify pass
  // (row 0 at position `first`), copied together into tree_rows_.
  Status append_draft_path(std::span<const std::uint32_t> path, std::int32_t first);
  graph::Array tree_rows_;
  graph::Array spec_features_;
  Status append_draft_context(std::size_t rows, std::int32_t first);
  // The prompt's last pass's target features, for the DFlash2 context. The
  // first token needs only the target's logits, so this pass is appended
  // after that token is out (flush_draft_context), not before it.
  graph::Array pending_context_;
  std::int32_t pending_context_first_ = 0;
  std::int64_t pending_context_rows_ = 0;
  Status flush_draft_context();

  // Runs row tokens at the session cursor and leaves the
  // per-row answers readable: the greedy picks in spec_, the raw logits in
  // spec_logits_ otherwise, and every row's hidden on the device in spec_.hidden.
  // `replaces_previous` says this pass stands in for the one that just ran
  // rather than following it, which is how a rejected proposal is undone: the
  // decoder's carried state is re-derived from the same starting point with
  // the corrected tokens in place.
  Status verify(Session& session, std::span<const std::uint32_t> rows,
                bool replaces_previous, const model::TreeLayout* tree = nullptr);
  // spec_by_m_ keys: a pass's width, plus this for a tree pass.
  static constexpr std::size_t kTreeHeadKey = std::size_t{1} << 20;
  std::size_t spec_key_ = 0;
  Status mtp_prefill_chunk(const graph::Array& hidden,
                           std::span<const std::uint32_t> tokens,
                           std::int32_t first, graph::Array* carry);
  Result<std::vector<std::uint32_t>> speculate(
      Session& session, std::vector<float>& prefill_logits,
      const GenerationLimits& limits, const TokenCallback& on_token);
};

}  // namespace lse::runtime
