#include "lse/runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lse::runtime {

namespace {

// splitmix64. Self-contained and reproducible across platforms, which
// std::mt19937 with a distribution is not.
std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  state += 0x9e3779b97f4a7c15ull;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

}  // namespace

std::uint32_t argmax(std::span<const float> logits) noexcept {
  if (logits.empty()) return 0;
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::uint32_t>(best);
}

std::uint64_t Sampler::mix_seed(std::uint64_t seed) noexcept {
  // A zero seed must not give a degenerate stream.
  std::uint64_t s = seed == 0 ? 0x853c49e6748fea9bull : seed;
  return splitmix64(s);
}

Sampler::Sampler(SamplingParams params) noexcept
    : params_(params), state_(mix_seed(params.seed)) {}

float Sampler::next_uniform() noexcept {
  // 24 bits into [0,1): the mantissa of a float, so every value is exact.
  const std::uint64_t bits = splitmix64(state_) >> 40;
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

std::uint32_t Sampler::sample(std::span<float> logits,
                              std::span<const std::uint32_t> history) {
  if (logits.empty()) return 0;

  if (params_.repetition_penalty != 1.0f && !history.empty()) {
    const std::size_t window =
        params_.repetition_window > 0
            ? std::min(history.size(),
                       static_cast<std::size_t>(params_.repetition_window))
            : history.size();
    for (std::size_t i = history.size() - window; i < history.size(); ++i) {
      const std::size_t id = history[i];
      if (id >= logits.size()) continue;
      // Sign-aware: dividing a negative logit would *raise* it.
      logits[id] = logits[id] > 0.0f ? logits[id] / params_.repetition_penalty
                                     : logits[id] * params_.repetition_penalty;
    }
  }

  if (params_.temperature <= 0.0f) {
    return argmax(logits);
  }

  order_.resize(logits.size());
  for (std::size_t i = 0; i < order_.size(); ++i) {
    order_[i] = static_cast<std::uint32_t>(i);
  }

  std::size_t keep = order_.size();
  if (params_.top_k > 0 && static_cast<std::size_t>(params_.top_k) < keep) {
    keep = static_cast<std::size_t>(params_.top_k);
    std::partial_sort(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(keep),
                      order_.end(), [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
  } else {
    std::sort(order_.begin(), order_.end(),
              [&](std::uint32_t a, std::uint32_t b) {
                return logits[a] > logits[b];
              });
  }

  // Softmax over the survivors only, shifted by the max for stability.
  const float max_logit = logits[order_[0]];
  const float inv_t = 1.0f / params_.temperature;
  probs_.resize(keep);
  double total = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    const float p = std::exp((logits[order_[i]] - max_logit) * inv_t);
    probs_[i] = p;
    total += static_cast<double>(p);
  }
  if (total <= 0.0) return order_[0];

  if (params_.top_p < 1.0f && params_.top_p > 0.0f) {
    double cumulative = 0.0;
    std::size_t cut = 0;
    for (; cut < keep; ++cut) {
      cumulative += static_cast<double>(probs_[cut]) / total;
      // Include the token that crosses the threshold, so top_p never empties.
      if (cumulative >= static_cast<double>(params_.top_p)) {
        ++cut;
        break;
      }
    }
    if (cut > 0 && cut < keep) {
      keep = cut;
      total = 0.0;
      for (std::size_t i = 0; i < keep; ++i) total += static_cast<double>(probs_[i]);
    }
  }

  const auto target = static_cast<double>(next_uniform()) * total;
  double running = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    running += static_cast<double>(probs_[i]);
    if (running >= target) return order_[i];
  }
  return order_[keep - 1];
}

}  // namespace lse::runtime
