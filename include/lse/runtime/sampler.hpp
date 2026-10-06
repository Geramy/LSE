// Turning a logit row into the next token.
//
// The transforms are applied in the order the reference stacks use — penalties,
// then temperature, then the truncation filters — because each one changes what
// the next sees. Reordering top-k and top-p in particular gives different
// distributions for the same settings.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::runtime {

struct SamplingParams {
  // <= 0 is greedy: the argmax, with every other setting ignored.
  float temperature = 0.8f;
  // 0 disables. Keeps the k highest-probability tokens.
  std::int32_t top_k = 0;
  // 1.0 disables. Keeps the smallest prefix whose mass reaches p.
  float top_p = 1.0f;
  // 0 disables. After top-k and top-p, drops tokens whose probability (at the
  // sampling temperature) is below min_p times the most likely token's.
  float min_p = 0.0f;
  // 1.0 disables. Divides the logit of any token already seen (>1 discourages).
  float repetition_penalty = 1.0f;
  // 0 disables. Subtracted once from the logit of every distinct token in the
  // penalty window (>0 discourages), as OpenAI's presence_penalty.
  float presence_penalty = 0.0f;
  // How far back the penalties look. 0 means the whole context.
  std::int32_t repetition_window = 64;
  std::uint64_t seed = 0;

  // Every setting leaves the argmax as the answer: the device can pick it.
  [[nodiscard]] bool greedy_argmax() const noexcept {
    return temperature <= 0.0f && repetition_penalty == 1.0f && presence_penalty == 0.0f;
  }
};

struct DiscreteDistribution {
  std::vector<std::uint32_t> ids;
  std::vector<double> probabilities;
  [[nodiscard]] double probability(std::uint32_t token) const noexcept;
};

struct ProposalVerification {
  std::uint32_t token = 0;
  bool accepted = false;
};

[[nodiscard]] Result<std::uint32_t> sample_distribution(
    const DiscreteDistribution&, double uniform);
[[nodiscard]] Result<ProposalVerification> verify_proposal(
    const DiscreteDistribution& target, const DiscreteDistribution& proposal,
    std::uint32_t drafted_token, double acceptance_uniform,
    double residual_uniform);

class SpeculativeSampler {
 public:
  explicit SpeculativeSampler(std::uint64_t seed) noexcept { reseed(seed); }
  void reseed(std::uint64_t seed) noexcept;
  [[nodiscard]] Result<std::uint32_t> sample_proposal(const DiscreteDistribution&);
  [[nodiscard]] Result<std::uint32_t> sample_target(const DiscreteDistribution&);
  [[nodiscard]] Result<ProposalVerification> verify(
      const DiscreteDistribution& target, const DiscreteDistribution& proposal,
      std::uint32_t drafted_token);
 private:
  std::uint64_t draft_state_ = 0, accept_state_ = 0, residual_state_ = 0;
};

class Sampler {
 public:
  explicit Sampler(SamplingParams params) noexcept;

  // `history` is the tokens generated so far, used by the repetition penalty.
  // `logits` is modified in place — the caller owns a scratch row, not the
  // model's output buffer.
  [[nodiscard]] std::uint32_t sample(std::span<float> logits,
                                     std::span<const std::uint32_t> history);

  // Applies the same transforms as sample() without consuming its RNG.
  [[nodiscard]] Result<DiscreteDistribution> distribution(
      std::span<float> logits, std::span<const std::uint32_t> history);

  // A row's `candidates` best logits decide the draw exactly when top-k keeps
  // no more than that and no history penalty rewrites a logit: the softmax,
  // top-p and min-p only ever look at the kept head of the sorted row.
  [[nodiscard]] bool decided_by_top(std::size_t candidates, std::size_t vocab) const noexcept;
  // sample() and distribution() over a row's best logits, best first (ties
  // by lower id), as a device top-k returns them. `values[0]` is a NaN or
  // +inf whenever the full row holds one, which is what makes the checks
  // distribution() runs over the row decidable here.
  [[nodiscard]] Result<std::uint32_t> sample_top(std::span<const float> values,
                                                 std::span<const std::uint32_t> ids);
  [[nodiscard]] Result<DiscreteDistribution> distribution_top(
      std::span<const float> values, std::span<const std::uint32_t> ids);

  [[nodiscard]] const SamplingParams& params() const noexcept { return params_; }
  void reseed(std::uint64_t seed) noexcept { state_ = mix_seed(seed); }

 private:
  struct PreparedWeights {
    std::size_t keep = 0;
    double total = 0;
    bool indexed = false, point_mass = false;
    std::uint32_t token = 0;
  };
  [[nodiscard]] PreparedWeights prepare_weights(
      std::span<float> logits, std::span<const std::uint32_t> history);
  // Softmax, top-p and min-p over sorted_[0, keep), the kept logits in
  // descending order with their ids in order_.
  [[nodiscard]] PreparedWeights weigh_sorted(std::size_t keep);
  [[nodiscard]] Result<PreparedWeights> prepare_top(std::span<const float> values,
                                                    std::span<const std::uint32_t> ids);
  [[nodiscard]] std::uint32_t draw_prepared(const PreparedWeights& weights);
  [[nodiscard]] Result<DiscreteDistribution> distribution_of(const PreparedWeights& weights);
  [[nodiscard]] static std::uint64_t mix_seed(std::uint64_t seed) noexcept;
  [[nodiscard]] float next_uniform() noexcept;

  SamplingParams params_;
  std::uint64_t state_;
  // Reused across calls so a decode step does not allocate.
  std::vector<std::uint32_t> order_;
  std::vector<float> probs_;
  std::vector<float> sorted_;
};

// Highest logit, ties going to the lowest index.
[[nodiscard]] std::uint32_t argmax(std::span<const float> logits) noexcept;

}  // namespace lse::runtime
