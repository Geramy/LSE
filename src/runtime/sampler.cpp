#include "lse/runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_set>

namespace lse::runtime {

namespace {

class SamplingProfile {
 public:
  explicit SamplingProfile(std::size_t logits) : logits_(logits) {
    static const bool enabled = std::getenv("LSE_TIME_STEPS") != nullptr;
    enabled_ = enabled;
    if (enabled_) start_ = std::chrono::steady_clock::now();
  }
  ~SamplingProfile() {
    if (!enabled_) return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_).count();
    std::fprintf(stderr, "[sampling-spans] logits=%zu elapsed_ns=%lld\n",
                 logits_, static_cast<long long>(ns));
  }
 private:
  bool enabled_ = false;
  std::size_t logits_;
  std::chrono::steady_clock::time_point start_;
};

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

Sampler::PreparedWeights Sampler::prepare_weights(
    std::span<float> logits, std::span<const std::uint32_t> history) {
  if (logits.empty()) return {0, 0, false, true, 0};

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

  if (params_.presence_penalty != 0.0f && !history.empty()) {
    const std::size_t window =
        params_.repetition_window > 0
            ? std::min(history.size(),
                       static_cast<std::size_t>(params_.repetition_window))
            : history.size();
    std::unordered_set<std::uint32_t> seen;
    for (std::size_t i = history.size() - window; i < history.size(); ++i) {
      const std::uint32_t id = history[i];
      if (id >= logits.size() || !seen.insert(id).second) continue;
      logits[id] -= params_.presence_penalty;
    }
  }

  if (params_.temperature <= 0.0f) {
    return {1, 1, false, true, argmax(logits)};
  }

  const bool unfiltered =
      (params_.top_k <= 0 ||
       static_cast<std::size_t>(params_.top_k) >= logits.size()) &&
      !(params_.top_p > 0.0f && params_.top_p < 1.0f) && !(params_.min_p > 0.0f);
  if (unfiltered && std::isfinite(params_.temperature)) {
    const std::uint32_t best = argmax(logits);
    const float max_logit = logits[best];
    if (std::isfinite(max_logit)) {
      const float inv_t = 1.0f / params_.temperature;
      probs_.resize(logits.size());
      double total = 0.0;
      bool finite = true;
      for (std::size_t i = 0; i < logits.size(); ++i) {
        const float p = std::exp((logits[i] - max_logit) * inv_t);
        probs_[i] = p;
        finite = finite && std::isfinite(p);
        total += static_cast<double>(p);
      }
      if (finite && total > 0.0) {
        return {logits.size(), total, false, false, best};
      }
    }
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
  if (total <= 0.0) return {1, 1, true, true, order_[0]};

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

  if (params_.min_p > 0.0f) {
    // probs_ is in descending order; the first entry is the most likely token.
    const double floor = static_cast<double>(params_.min_p) * static_cast<double>(probs_[0]);
    std::size_t cut = 1;
    while (cut < keep && static_cast<double>(probs_[cut]) >= floor) ++cut;
    if (cut < keep) {
      keep = cut;
      total = 0.0;
      for (std::size_t i = 0; i < keep; ++i) total += static_cast<double>(probs_[i]);
    }
  }

  return {keep, total, true, false, order_[0]};
}

std::uint32_t Sampler::sample(std::span<float> logits,
                              std::span<const std::uint32_t> history) {
  const SamplingProfile profile(logits.size());
  const auto weights = prepare_weights(logits, history);
  if (weights.point_mass) return weights.token;
  const double target = static_cast<double>(next_uniform()) * weights.total;
  double running = 0;
  std::uint32_t last = weights.token;
  for (std::size_t i = 0; i < weights.keep; ++i) {
    const float p = probs_[i];
    if (!weights.indexed && p == 0.0f) continue;
    last = weights.indexed ? order_[i] : static_cast<std::uint32_t>(i);
    running += static_cast<double>(p);
    if (running >= target) return last;
  }
  return last;
}

Result<DiscreteDistribution> Sampler::distribution(
    std::span<float> logits, std::span<const std::uint32_t> history) {
  const SamplingProfile profile(logits.size());
  if (logits.empty() || logits.size() > UINT32_MAX ||
      !std::isfinite(params_.temperature) || !std::isfinite(params_.top_p) ||
      !std::isfinite(params_.repetition_penalty) || params_.repetition_penalty <= 0 ||
      !std::isfinite(params_.min_p) || !std::isfinite(params_.presence_penalty))
    return LSE_ERROR(kInvalidArgument, "invalid target sampling inputs");
  bool positive_mass = false;
  for (const float logit : logits) {
    if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity())
      return LSE_ERROR(kInvalidArgument, "nonfinite target logit");
    positive_mass |= std::isfinite(logit);
  }
  if (!positive_mass) return LSE_ERROR(kInvalidArgument, "target has no probability mass");
  const auto weights = prepare_weights(logits, history);
  for (const float logit : logits)
    if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity())
      return LSE_ERROR(kInvalidArgument, "nonfinite transformed target logit");
  DiscreteDistribution out;
  if (weights.point_mass) {
    out.ids = {weights.token}; out.probabilities = {1.0};
    return out;
  }
  if (!std::isfinite(weights.total) || weights.total <= 0)
    return LSE_ERROR(kInvalidArgument, "invalid target probability mass");
  out.ids.reserve(weights.keep); out.probabilities.reserve(weights.keep);
  for (std::size_t i = 0; i < weights.keep; ++i) {
    const double p = static_cast<double>(probs_[i]);
    if (!std::isfinite(p) || p < 0)
      return LSE_ERROR(kInvalidArgument, "invalid target probability");
    out.ids.push_back(weights.indexed ? order_[i] : static_cast<std::uint32_t>(i));
    out.probabilities.push_back(p / weights.total);
  }
  return out;
}

double DiscreteDistribution::probability(std::uint32_t token) const noexcept {
  for (std::size_t i = 0; i < ids.size() && i < probabilities.size(); ++i)
    if (ids[i] == token) return probabilities[i];
  return 0;
}

namespace {
Result<double> distribution_mass(const DiscreteDistribution& d) {
  if (d.ids.empty() || d.ids.size() != d.probabilities.size())
    return LSE_ERROR(kInvalidArgument, "invalid discrete distribution shape");
  std::unordered_set<std::uint32_t> seen;
  seen.reserve(d.ids.size());
  double total = 0;
  for (std::size_t i = 0; i < d.ids.size(); ++i) {
    const double p = d.probabilities[i];
    if (!std::isfinite(p) || p < 0 || !seen.insert(d.ids[i]).second)
      return LSE_ERROR(kInvalidArgument, "invalid discrete probability or duplicate token");
    total += p;
  }
  if (!std::isfinite(total) || std::abs(total - 1.0) > 1e-6)
    return LSE_ERROR(kInvalidArgument, "discrete distribution is not normalized");
  return total;
}
bool valid_uniform(double u) { return std::isfinite(u) && u >= 0 && u < 1; }
std::uint32_t draw(const DiscreteDistribution& d, double total, double uniform) {
  const double target = uniform * total;
  double running = 0;
  std::uint32_t last = 0;
  for (std::size_t i = 0; i < d.ids.size(); ++i) {
    const double p = d.probabilities[i];
    if (p <= 0) continue;
    last = d.ids[i]; running += p;
    if (target < running) return last;
  }
  return last;
}
double speculative_uniform(std::uint64_t& state) noexcept {
  return (static_cast<double>(splitmix64(state) >> 12) + 0.5) * 0x1p-52;
}
}  // namespace

Result<std::uint32_t> sample_distribution(const DiscreteDistribution& d, double uniform) {
  if (!valid_uniform(uniform)) return LSE_ERROR(kInvalidArgument, "invalid sampling uniform");
  LSE_ASSIGN_OR(const double total, distribution_mass(d));
  return draw(d, total, uniform);
}

Result<ProposalVerification> verify_proposal(
    const DiscreteDistribution& target, const DiscreteDistribution& proposal,
    std::uint32_t drafted_token, double acceptance_uniform, double residual_uniform) {
  if (!valid_uniform(acceptance_uniform) || !valid_uniform(residual_uniform))
    return LSE_ERROR(kInvalidArgument, "invalid rejection sampling uniform");
  LSE_ASSIGN_OR(const double pt, distribution_mass(target));
  LSE_ASSIGN_OR(const double qt, distribution_mass(proposal));
  const double q = proposal.probability(drafted_token) / qt;
  if (!(q > 0)) return LSE_ERROR(kInvalidArgument, "drafted token has zero proposal probability");
  const double p = target.probability(drafted_token) / pt;
  if (p >= q || acceptance_uniform < p / q)
    return ProposalVerification{drafted_token, true};
  DiscreteDistribution residual;
  residual.ids = target.ids;
  residual.probabilities.reserve(target.ids.size());
  double total = 0;
  for (std::size_t i = 0; i < target.ids.size(); ++i) {
    const double r = std::max(target.probabilities[i] / pt -
                             proposal.probability(target.ids[i]) / qt, 0.0);
    residual.probabilities.push_back(r); total += r;
  }
  if (!std::isfinite(total) || total <= 0)
    return LSE_ERROR(kInternal, "rejected proposal has no residual mass");
  return ProposalVerification{draw(residual, total, residual_uniform), false};
}

void SpeculativeSampler::reseed(std::uint64_t seed) noexcept {
  draft_state_ = seed ^ 0x243f6a8885a308d3ull;
  accept_state_ = seed ^ 0x13198a2e03707344ull;
  residual_state_ = seed ^ 0xa4093822299f31d0ull;
  (void)splitmix64(draft_state_); (void)splitmix64(accept_state_);
  (void)splitmix64(residual_state_);
}
Result<std::uint32_t> SpeculativeSampler::sample_proposal(const DiscreteDistribution& d) {
  return sample_distribution(d, speculative_uniform(draft_state_));
}
Result<std::uint32_t> SpeculativeSampler::sample_target(const DiscreteDistribution& d) {
  return sample_distribution(d, speculative_uniform(residual_state_));
}
Result<ProposalVerification> SpeculativeSampler::verify(
    const DiscreteDistribution& target, const DiscreteDistribution& proposal,
    std::uint32_t drafted_token) {
  const double accept = speculative_uniform(accept_state_);
  const double residual = speculative_uniform(residual_state_);
  return verify_proposal(target, proposal, drafted_token, accept, residual);
}

}  // namespace lse::runtime
