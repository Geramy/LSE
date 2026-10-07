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
// Old observations fade so the table follows the text being written: after
// about this many checked proposals an observation weighs half as much.
constexpr double kCalibrationDecay = 1.0 - 1.0 / 4096.0;
// Cost tracking. The level follows the context quickly; a width's offset
// (what its extra rows cost) changes slowly.
constexpr double kBaseRate = 1.0 / 16.0;
constexpr double kOffsetRate = 1.0 / 32.0;
constexpr double kDraftRate = 1.0 / 16.0;
constexpr double kRateRate = 1.0 / 32.0;
constexpr double kValueRate = 1.0 / 4.0;
// One stalled step (a host hiccup, a clock change) moves an estimate by at
// most this share of it.
constexpr double kOutlierClamp = 0.25;

double clamp_step(double residual, double scale) {
  const double bound = kOutlierClamp * std::max(scale, 1.0);
  return std::clamp(residual, -bound, bound);
}
}  // namespace

double DraftWidthPolicy::acceptance(double confidence, bool sampled) const noexcept {
  if (!(confidence >= 0.0)) confidence = 0.0;
  const auto bin = std::min<std::size_t>(kBins - 1, static_cast<std::size_t>(confidence * kBins));
  const double middle = (static_cast<double>(bin) + 0.5) / kBins;
  const Bin& b = calibration_[sampled ? 1 : 0].bins[bin];
  return (b.accepted + kPriorWeight * middle) / (b.tested + kPriorWeight);
}

double DraftWidthPolicy::position_mean(std::uint32_t position, bool sampled) const noexcept {
  if (position >= kMaxProposals) return kPositionPrior;
  const Bin& b = calibration_[sampled ? 1 : 0].positions[position];
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

std::uint32_t DraftWidthPolicy::proposals(std::span<const double> confidence, bool sampled,
                                          std::span<double> estimates) {
  const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(confidence.size(), kMaxProposals));
  std::array<double, kMaxProposals> a{};
  for (std::uint32_t j = 0; j < n; ++j) {
    a[j] = acceptance(confidence[j], sampled);
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
      if (last > j) chain *= position_mean(last, sampled);
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
                                          bool accepted, bool sampled) {
  Calibration& c = calibration_[sampled ? 1 : 0];
  for (Bin& b : c.bins) { b.accepted *= kCalibrationDecay; b.tested *= kCalibrationDecay; }
  for (Bin& b : c.positions) { b.accepted *= kCalibrationDecay; b.tested *= kCalibrationDecay; }
  if (!(confidence >= 0.0)) confidence = 0.0;
  Bin& b = c.bins[std::min<std::size_t>(kBins - 1, static_cast<std::size_t>(confidence * kBins))];
  b.tested += 1.0;
  b.accepted += accepted ? 1.0 : 0.0;
  if (position < kMaxProposals) {
    c.positions[position].tested += 1.0;
    c.positions[position].accepted += accepted ? 1.0 : 0.0;
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

void DraftWidthPolicy::observe_draft(std::uint64_t ns) {
  const auto t = static_cast<double>(ns);
  if (draft_samples_++ == 0) draft_ns_ = t;
  else draft_ns_ += kDraftRate * clamp_step(t - draft_ns_, draft_ns_);
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
  const double cost = verify_ns(proposals + 1) - verify_ns(1) + draft_ns_;
  const double value = gain - rate() * cost;
  if (!draft_value_known_) {
    draft_value_ = value;
    draft_value_known_ = true;
  } else {
    draft_value_ += kValueRate * (value - draft_value_);
  }
}

}  // namespace lse::runtime
