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

  HybridLM& model_;
  Sampler sampler_;
  std::unique_ptr<Session> owned_;
  GenerationStats stats_;
  std::vector<std::string> host_reasons_;
};

}  // namespace lse::runtime
