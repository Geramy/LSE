// Turning a logit row into the next token.
//
// The transforms are applied in the order the reference stacks use — penalties,
// then temperature, then the truncation filters — because each one changes what
// the next sees. Reordering top-k and top-p in particular gives different
// distributions for the same settings.
#pragma once

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
  // 1.0 disables. Divides the logit of any token already seen (>1 discourages).
  float repetition_penalty = 1.0f;
  // How far back the penalty looks. 0 means the whole context.
  std::int32_t repetition_window = 64;
  std::uint64_t seed = 0;
};

class Sampler {
 public:
  explicit Sampler(SamplingParams params) noexcept;

  // `history` is the tokens generated so far, used by the repetition penalty.
  // `logits` is modified in place — the caller owns a scratch row, not the
  // model's output buffer.
  [[nodiscard]] std::uint32_t sample(std::span<float> logits,
                                     std::span<const std::uint32_t> history);

  [[nodiscard]] const SamplingParams& params() const noexcept { return params_; }
  void reseed(std::uint64_t seed) noexcept { state_ = mix_seed(seed); }

 private:
  [[nodiscard]] static std::uint64_t mix_seed(std::uint64_t seed) noexcept;
  [[nodiscard]] float next_uniform() noexcept;

  SamplingParams params_;
  std::uint64_t state_;
  // Reused across calls so a decode step does not allocate.
  std::vector<std::uint32_t> order_;
  std::vector<float> probs_;
};

// Highest logit, ties going to the lowest index.
[[nodiscard]] std::uint32_t argmax(std::span<const float> logits) noexcept;

}  // namespace lse::runtime
