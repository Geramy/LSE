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
  // One verifier pass can compare several draft proposals.
  std::uint32_t spec_steps = 0;
  std::uint32_t spec_accepted = 0;
  std::uint32_t spec_tested = 0;
  std::array<std::uint32_t, 7> spec_tested_by_position{}, spec_accepted_by_position{};
  std::array<double, 7> spec_overlap_sum{}, spec_candidate_mass_sum{}, spec_deterministic_mass_sum{};
  std::uint32_t mtp_depth = 0;
  std::uint32_t dflash2_depth = 0;
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
  void use_mtp(model::MtpModule& mtp) noexcept { mtp_ = &mtp; dflash2_ = nullptr; }
  void use_dflash2(model::DFlash2Module& draft) noexcept {
    dflash2_ = &draft; mtp_ = nullptr;
  }

  // Last position of a [.., T, D] hidden state, reshaped to [.., D].
  static Result<graph::Array> last_hidden(const graph::Array& hidden);

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
    std::vector<graph::NodePtr> compute;
    bool greedy = false;
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
    std::vector<graph::NodePtr> compute;
    bool greedy = false;
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
  // The decoder's hidden state for the last prompt position, which is the
  // module's input for the first proposal of the generation.
  graph::Array prefill_tail_;

  model::MtpModule* mtp_ = nullptr;
  model::DFlash2Module* dflash2_ = nullptr;
  graph::Array spec_features_;
  Status append_draft_context(std::size_t rows, std::int32_t first);

  // Runs row tokens at the session cursor and leaves the
  // per-row answers readable: the greedy picks in spec_, the raw logits in
  // spec_logits_ otherwise, and every row's hidden on the device in spec_.hidden.
  // `replaces_previous` says this pass stands in for the one that just ran
  // rather than following it, which is how a rejected proposal is undone: the
  // decoder's carried state is re-derived from the same starting point with
  // the corrected tokens in place.
  Status verify(Session& session, std::span<const std::uint32_t> rows,
                bool replaces_previous);
  Status mtp_prefill_chunk(const graph::Array& hidden,
                           std::span<const std::uint32_t> tokens,
                           std::int32_t first, graph::Array* carry);
  Result<std::vector<std::uint32_t>> speculate(
      Session& session, std::vector<float>& prefill_logits,
      const GenerationLimits& limits, const TokenCallback& on_token);
};

}  // namespace lse::runtime
