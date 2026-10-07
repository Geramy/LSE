#include "lse/runtime/draft_width.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lse::runtime {

namespace {
// Pseudo-observations behind every calibration estimate: a bin starts at its
// own confidence (a draft that says 0.9 is right 9 times in 10), a position
// at the acceptance the essay benchmark measured, and data takes over fast.
constexpr double kPriorWeight = 2.0;
constexpr double kPositionPrior = 0.6;
// Old observations fade so the table follows the text being written (code
// is accepted at confidences prose is not): an observation weighs half as
// much after 512 more checked proposals, about one 640-token request.
constexpr double kCalibrationDecay = 1.0 - 0.6931471805599453 / 512.0;
// Cost tracking. The level follows the context quickly; a width's offset
// (what its extra rows cost) changes slowly.
constexpr double kBaseRate = 1.0 / 16.0;
constexpr double kOffsetRate = 1.0 / 32.0;
constexpr double kDraftRate = 1.0 / 16.0;
constexpr double kRateRate = 1.0 / 32.0;
// Slow on purpose: one weak draft says little about the next one, so drafting
// stops only when drafts have been losing for a stretch.
constexpr double kValueRate = 1.0 / 16.0;
// One stalled step (a host hiccup, a clock change) moves an estimate by at
// most this share of it.
constexpr double kOutlierClamp = 0.25;

double clamp_step(double residual, double scale) {
  // Scales below a millisecond clamp as one millisecond does.
  const double bound = kOutlierClamp * std::max(scale, 1e6);
  return std::clamp(residual, -bound, bound);
}
}  // namespace

std::size_t DraftWidthPolicy::bin_of(double confidence) noexcept {
  if (!(confidence >= 0.0)) confidence = 0.0;
  const double decades = -std::log10(std::max(1.0 - confidence, 1e-12));
  return std::min<std::size_t>(kBins - 1, static_cast<std::size_t>(decades * kBinsPerDecade));
}

double DraftWidthPolicy::acceptance(double confidence) const noexcept {
  const std::size_t bin = bin_of(confidence);
  const double middle = 1.0 - std::pow(10.0, -(static_cast<double>(bin) + 0.5) / kBinsPerDecade);
  const Bin& b = bins_[bin];
  return (b.accepted + kPriorWeight * middle) / (b.tested + kPriorWeight);
}

double DraftWidthPolicy::position_mean(std::uint32_t position) const noexcept {
  if (position >= kMaxProposals) return kPositionPrior;
  const Bin& b = positions_[position];
  return (b.accepted + kPriorWeight * kPositionPrior) / (b.tested + kPriorWeight);
}

double DraftWidthPolicy::verify_ns(std::uint32_t rows) const noexcept {
  if (rows == 0 || rows > kMaxRows || samples_[rows] == 0) return 0.0;
  return std::max(0.0, base_ + offset_[rows]);
}

double DraftWidthPolicy::rate() const noexcept {
  return rate_ns_ > 0.0 ? rate_tokens_ / rate_ns_ : 0.0;
}

std::uint32_t DraftWidthPolicy::exploring() const noexcept {
  for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows)
    if (samples_[rows] < kExploreSamples) return rows;
  for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows)
    if (steps_ - last_seen_[rows] > kRefreshSteps) return rows;
  return 0;
}

std::uint32_t DraftWidthPolicy::proposals(std::span<const double> confidence,
                                          std::span<double> estimates) {
  const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(confidence.size(), kMaxProposals));
  std::array<double, kMaxProposals> a{};
  for (std::uint32_t j = 0; j < n; ++j) {
    a[j] = acceptance(confidence[j]);
    if (j < estimates.size()) estimates[j] = a[j];
  }
  if (const std::uint32_t rows = exploring(); rows != 0) return std::min(n, rows - 1);
  // Verify proposal j + 1 when some prefix ending at or after it adds more
  // expected tokens per added nanosecond than the long-run rate: the step's
  // tokens minus rate * time grows. Positions after j enter at their mean
  // acceptance, not at their own confidence, which keeps this a stopping
  // rule (see the header).
  const double rate_now = rate();
  std::uint32_t k = 0;
  double reach = 1.0;  // chance every verified proposal so far is accepted
  for (std::uint32_t j = 0; j < n; ++j) {
    const double here = reach * a[j];
    double best = -std::numeric_limits<double>::infinity();
    double gain = 0.0, chain = here;
    const double now = verify_ns(k + 1);
    for (std::uint32_t last = j; last < n; ++last) {
      if (last > j) chain *= position_mean(last);
      gain += chain;
      const double cost = verify_ns(last + 2) - now;
      best = std::max(best, cost <= 0.0 ? std::numeric_limits<double>::infinity() : gain / cost);
    }
    if (best < rate_now) break;
    k = j + 1;
    reach = here;
  }
  return k;
}

bool DraftWidthPolicy::draft_next() {
  if (const std::uint32_t rows = exploring(); rows != 0) return rows > 1;
  if (!draft_value_known_ || draft_value_ >= 0.0) {
    skipped_ = 0;
    return true;
  }
  if (skipped_ >= kProbeAfter) {
    skipped_ = 0;
    return true;
  }
  ++skipped_;
  return false;
}

void DraftWidthPolicy::observe_acceptance(double confidence, std::uint32_t position,
                                          bool accepted) {
  for (Bin& b : bins_) { b.accepted *= kCalibrationDecay; b.tested *= kCalibrationDecay; }
  for (Bin& b : positions_) { b.accepted *= kCalibrationDecay; b.tested *= kCalibrationDecay; }
  Bin& b = bins_[bin_of(confidence)];
  b.tested += 1.0;
  b.accepted += accepted ? 1.0 : 0.0;
  if (position < kMaxProposals) {
    positions_[position].tested += 1.0;
    positions_[position].accepted += accepted ? 1.0 : 0.0;
  }
}

void DraftWidthPolicy::observe_verify(std::uint32_t rows, std::uint64_t ns) {
  if (rows == 0 || rows > kMaxRows) return;
  ++steps_;
  const auto t = static_cast<double>(ns);
  bool any = false;
  for (std::uint32_t m = 1; m <= kMaxRows; ++m) any = any || samples_[m] != 0;
  if (!any) {
    base_ = t;
    offset_[rows] = 0.0;
  } else if (samples_[rows] == 0) {
    offset_[rows] = t - base_;
  } else {
    const double predicted = base_ + offset_[rows];
    base_ += kBaseRate * clamp_step(t - predicted, predicted);
    // While a width is being explored its offset is a plain average.
    const double rate = samples_[rows] < kExploreSamples
                            ? 1.0 / static_cast<double>(samples_[rows] + 1) : kOffsetRate;
    offset_[rows] += rate * clamp_step(t - base_ - offset_[rows], predicted);
  }
  ++samples_[rows];
  last_seen_[rows] = steps_;
}

void DraftWidthPolicy::observe_draft(std::uint64_t ns, std::uint32_t depth) {
  if (depth > kMaxProposals) return;
  const auto t = static_cast<double>(ns);
  double& d = draft_ns_[depth];
  if (draft_samples_[depth]++ == 0) d = t;
  else d += kDraftRate * clamp_step(t - d, d);
}

double DraftWidthPolicy::draft_ns(std::uint32_t depth) const noexcept {
  return depth <= kMaxProposals ? draft_ns_[depth] : 0.0;
}

std::uint32_t DraftWidthPolicy::depth(std::uint32_t max) {
  max = std::min(max, kMaxProposals);
  if (const std::uint32_t rows = exploring(); rows != 0) return std::min(max, rows - 1);
  const double rate_now = rate();
  std::uint32_t best = 0;
  double best_value = -std::numeric_limits<double>::infinity();
  double gain = 0.0, chain = 1.0;
  for (std::uint32_t d = 0; d <= max; ++d) {
    if (d > 0) {
      chain *= position_mean(d - 1);
      gain += chain;
    }
    const double value = gain - rate_now * (draft_ns(d) + verify_ns(d + 1));
    if (value > best_value) {
      best_value = value;
      best = d;
    }
  }
  return best;
}

void DraftWidthPolicy::observe_step(std::uint32_t tokens, std::uint64_t ns) {
  if (rate_ns_ == 0.0) {
    rate_tokens_ = tokens;
    rate_ns_ = static_cast<double>(ns);
    return;
  }
  rate_tokens_ += kRateRate * (static_cast<double>(tokens) - rate_tokens_);
  rate_ns_ += kRateRate * (static_cast<double>(ns) - rate_ns_);
}

void DraftWidthPolicy::observe_draft_value(std::span<const double> estimates,
                                           std::uint32_t proposals) {
  double gain = 0.0, chain = 1.0;
  for (std::uint32_t j = 0; j < proposals && j < estimates.size(); ++j) {
    chain *= estimates[j];
    gain += chain;
  }
  const double cost = verify_ns(proposals + 1) - verify_ns(1) + draft_ns(kMaxProposals);
  const double value = gain - rate() * cost;
  if (!draft_value_known_) {
    draft_value_ = value;
    draft_value_known_ = true;
  } else {
    draft_value_ += kValueRate * (value - draft_value_);
  }
}

}  // namespace lse::runtime
