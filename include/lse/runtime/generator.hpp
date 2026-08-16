// Prefill, then one token at a time.
//
// The hybrid part is that a block's state is not one shape: an attention layer
// carries a KV pair allocated at the engine length, a GDN layer carries a
// fixed-size recurrent matrix.
// MixerState holds whichever the block needs and the generator never has to
// know which kind a given layer is.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/program.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/runtime/sampler.hpp"
#include "lse/runtime/session.hpp"

namespace lse::runtime {

using model::HybridLM;

struct GenerationLimits {
  std::int32_t max_tokens = 256;
  // Generation stops on any of these. Empty means run to max_tokens.
  std::vector<std::uint32_t> stop_tokens;
};

struct GenerationStats {
  std::int32_t prompt_tokens = 0;
  std::int32_t generated_tokens = 0;
  std::uint64_t prefill_ns = 0;
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

  [[nodiscard]] double decode_tokens_per_second() const noexcept {
    if (decode_ns == 0 || generated_tokens == 0) return 0.0;
    return static_cast<double>(generated_tokens) * 1e9 /
           static_cast<double>(decode_ns);
  }
};

class Generator {
 public:
  Generator(HybridLM& model, SamplingParams params)
      : model_(model), sampler_(params) {}

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
};

}  // namespace lse::runtime
