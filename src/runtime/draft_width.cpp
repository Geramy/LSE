#include "lse/runtime/draft_width.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
// Each position's mean acceptance decides how deep an MTP chain goes before
// any proposal exists, so it has to follow a change of text within a request
// (code to prose): half weight after 64 more checks, about 20 steps.
constexpr double kPositionDecay = 1.0 - 0.6931471805599453 / 64.0;
constexpr double kBiasRate = 0.1;
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
  const double p = std::clamp((b.accepted + kPriorWeight * middle) / (b.tested + kPriorWeight),
                              1e-4, 1.0 - 1e-4);
  if (bias_ == 0.0) return p;
  return 1.0 / (1.0 + std::exp(-(std::log(p / (1.0 - p)) + bias_)));
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
  if (steps_ < kWarmupSteps) return 0;
  std::uint32_t fewest = 0;
  for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows)
    if (samples_[rows] < kExploreSamples && (fewest == 0 || samples_[rows] < samples_[fewest]))
      fewest = rows;
  return fewest;
}

std::uint32_t DraftWidthPolicy::refresh(std::uint32_t max, bool chained) const noexcept {
  // Decided from history alone (position means, costs), never from this
  // step's proposals, so a refresh is a stopping rule like any other choice.
  max = std::min(max, kMaxProposals);
  const double rate_now = rate();
  std::array<double, kMaxRows> value{};
  double best = -std::numeric_limits<double>::infinity(), gain = 0.0, chain = 1.0;
  for (std::uint32_t d = 0; d <= max; ++d) {
    if (d > 0) {
      chain *= position_mean(d - 1);
      gain += chain;
    }
    value[d] = gain - rate_now * ((chained ? draft_ns(d) : 0.0) + verify_ns(d + 1));
    best = std::max(best, value[d]);
  }
  std::uint32_t stalest = 0;
  for (std::uint32_t rows = 1; rows <= max + 1; ++rows) {
    const std::uint64_t age = steps_ - last_seen_[rows];
    if (age <= kRefreshSteps || (best - value[rows - 1] > kRefreshLoss && age <= kHardRefreshSteps))
      continue;
    if (stalest == 0 || last_seen_[rows] < last_seen_[stalest]) stalest = rows;
  }
  return stalest;
}

std::uint32_t DraftWidthPolicy::proposals(std::span<const double> confidence,
                                          std::span<double> estimates) {
  const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(confidence.size(), kMaxProposals));
  std::array<double, kMaxProposals> a{};
  for (std::uint32_t j = 0; j < n; ++j) {
    a[j] = acceptance(confidence[j]);
    if (j < estimates.size()) estimates[j] = a[j];
  }
  if (steps_ < kWarmupSteps) return n;
  if (const std::uint32_t rows = exploring(); rows != 0) return std::min(n, rows - 1);
  if (const std::uint32_t rows = refresh(n, false); rows != 0) return rows - 1;
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

std::string DraftWidthPolicy::describe() const {
  std::string out = "verify_ms=";
  char item[32];
  for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows) {
    std::snprintf(item, sizeof item, "%s%.1f", rows == 1 ? "" : ",", verify_ns(rows) / 1e6);
    out += item;
  }
  out += " draft_ms=";
  for (std::uint32_t d = 0; d <= kMaxProposals; ++d) {
    std::snprintf(item, sizeof item, "%s%.1f", d == 0 ? "" : ",", draft_ns(d) / 1e6);
    out += item;
  }
  out += " position_mean=";
  for (std::uint32_t p = 0; p < kMaxProposals; ++p) {
    std::snprintf(item, sizeof item, "%s%.2f", p == 0 ? "" : ",", position_mean(p));
    out += item;
  }
  out += " tree_ms=";
  for (std::size_t r = 0; r < kTreeRows.size(); ++r) {
    std::snprintf(item, sizeof item, "%s%u:%.1f", r == 0 ? "" : ",", kTreeRows[r],
                  tree_verify_ns(kTreeRows[r]) / 1e6);
    out += item;
  }
  out += " candidate=";
  for (const double q : {0.05, 0.2, 0.5, 0.8, 0.95}) {
    std::snprintf(item, sizeof item, "%s%.2f", q == 0.05 ? "" : ",", candidate(q));
    out += item;
  }
  std::snprintf(item, sizeof item, " bias=%.2f", bias_);
  out += item;
  std::snprintf(item, sizeof item, " rate_tps=%.1f", rate() * 1e9);
  out += item;
  return out;
}

bool DraftWidthPolicy::draft_next(bool tree) {
  if (tree) {
    // Trees take no chain widths: a plain step is the only one to measure.
    if (steps_ >= kWarmupSteps && samples_[1] < kExploreSamples) return false;
    if (tree_exploring() != 0) return true;
  } else if (const std::uint32_t rows = exploring(); rows != 0) {
    return rows > 1;
  }
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
  // One step of online logistic regression on the bias: it settles where the
  // estimates are right on average, and at zero once the bins are.
  const double predicted = acceptance(confidence);
  bias_ = std::clamp(bias_ + kBiasRate * ((accepted ? 1.0 : 0.0) - predicted), -4.0, 4.0);
  for (Bin& b : bins_) { b.accepted *= kCalibrationDecay; b.tested *= kCalibrationDecay; }
  for (Bin& b : positions_) { b.accepted *= kPositionDecay; b.tested *= kPositionDecay; }
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
  // Warm-up steps count toward the warm-up and price nothing.
  if (steps_++ < kWarmupSteps) return;
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
    // A width's first observations average; one seen rarely (a refresh
    // every kRefreshSteps) keeps moving fast, so a sample taken while the
    // clocks ramped does not price it for good.
    const double rate = std::max(kOffsetRate, 1.0 / static_cast<double>(samples_[rows] + 1));
    offset_[rows] += rate * clamp_step(t - base_ - offset_[rows], predicted);
  }
  ++samples_[rows];
  last_seen_[rows] = steps_;
}

void DraftWidthPolicy::observe_draft(std::uint64_t ns, std::uint32_t depth) {
  if (depth > kMaxProposals || steps_ <= kWarmupSteps) return;
  const auto t = static_cast<double>(ns);
  double& d = draft_ns_[depth];
  if (draft_samples_[depth]++ == 0) d = t;
  else d += kDraftRate * clamp_step(t - d, d);
}

double DraftWidthPolicy::draft_ns(std::uint32_t depth) const noexcept {
  return depth <= kMaxProposals ? draft_ns_[depth] : 0.0;
}

std::uint32_t DraftWidthPolicy::depth(std::uint32_t max, std::uint32_t initial) {
  max = std::min(max, kMaxProposals);
  if (steps_ < kWarmupSteps) return std::min(max, initial);
  if (const std::uint32_t rows = exploring(); rows != 0) return std::min(max, rows - 1);
  if (const std::uint32_t rows = refresh(max, true); rows != 0) return rows - 1;
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

double DraftWidthPolicy::candidate(double q) const noexcept {
  if (!(q >= 0.0)) q = 0.0;
  q = std::min(q, 1.0);
  const auto bin = std::min<std::size_t>(kCandidateBins - 1,
      static_cast<std::size_t>(q * static_cast<double>(kCandidateBins)));
  const Bin& b = candidates_[bin];
  return std::clamp((b.accepted + kPriorWeight * q) / (b.tested + kPriorWeight), 0.0, 1.0);
}

void DraftWidthPolicy::observe_candidate(double q, bool answered) {
  // Sixteen candidates per reached row: the memory is sixteen times as many
  // observations as a chain's (half weight after about 8k).
  constexpr double kDecay = 1.0 - 0.6931471805599453 / 8192.0;
  if (!(q >= 0.0)) q = 0.0;
  for (Bin& b : candidates_) { b.accepted *= kDecay; b.tested *= kDecay; }
  Bin& b = candidates_[std::min<std::size_t>(kCandidateBins - 1,
      static_cast<std::size_t>(std::min(q, 1.0) * static_cast<double>(kCandidateBins)))];
  b.tested += 1.0;
  b.accepted += answered ? 1.0 : 0.0;
}

std::size_t DraftWidthPolicy::tree_rung(std::uint32_t rows) noexcept {
  for (std::size_t i = 0; i < kTreeRows.size(); ++i)
    if (kTreeRows[i] == rows) return i;
  return kTreeRows.size();
}

double DraftWidthPolicy::tree_verify_ns(std::uint32_t rows) const noexcept {
  const std::size_t r = tree_rung(rows);
  if (r >= kTreeRows.size() || tree_samples_[r] == 0) return 0.0;
  return std::max(0.0, base_ + tree_offset_[r]);
}

std::uint32_t DraftWidthPolicy::tree_exploring() const noexcept {
  if (steps_ < kWarmupSteps) return 0;
  std::size_t fewest = kTreeRows.size();
  for (std::size_t r = 0; r < kTreeRows.size(); ++r)
    if (tree_samples_[r] < kExploreSamples &&
        (fewest == kTreeRows.size() || tree_samples_[r] < tree_samples_[fewest]))
      fewest = r;
  return fewest < kTreeRows.size() ? kTreeRows[fewest] : 0;
}

std::uint32_t DraftWidthPolicy::tree_nodes(std::span<const double> value_prefix) {
  if (value_prefix.size() < 2) return 0;
  const auto most = static_cast<std::uint32_t>(value_prefix.size() - 1);
  const auto fit = [&](std::uint32_t rows) { return std::min(most, rows - 1); };
  // Warming up: the rung a chain would take.
  if (steps_ < kWarmupSteps) return fit(kTreeRows[1]);
  if (const std::uint32_t rows = tree_exploring(); rows != 0) return fit(rows);
  // A rung unmeasured for long enough is measured again, whatever it costs.
  for (std::size_t r = 0; r < kTreeRows.size(); ++r)
    if (steps_ - tree_last_seen_[r] > kHardRefreshSteps) return fit(kTreeRows[r]);
  const double draft = draft_ns(kMaxProposals);
  std::uint32_t best = fit(kTreeRows[0]);
  double best_rate = -1.0;
  for (const std::uint32_t rows : kTreeRows) {
    const std::uint32_t nodes = fit(rows);
    const double cost = draft + tree_verify_ns(rows);
    if (cost <= 0.0) continue;
    const double rate = (1.0 + value_prefix[nodes]) / cost;
    if (rate > best_rate) {
      best_rate = rate;
      best = nodes;
    }
    if (nodes == most) break;
  }
  return best;
}

void DraftWidthPolicy::observe_tree_verify(std::uint32_t rows, std::uint64_t ns) {
  const std::size_t r = tree_rung(rows);
  if (r >= kTreeRows.size()) return;
  if (steps_++ < kWarmupSteps) return;
  const auto t = static_cast<double>(ns);
  bool any = false;
  for (std::uint32_t m = 1; m <= kMaxRows; ++m) any = any || samples_[m] != 0;
  for (const auto s : tree_samples_) any = any || s != 0;
  if (!any) {
    base_ = t;
    tree_offset_[r] = 0.0;
  } else if (tree_samples_[r] == 0) {
    tree_offset_[r] = t - base_;
  } else {
    const double predicted = base_ + tree_offset_[r];
    base_ += kBaseRate * clamp_step(t - predicted, predicted);
    const double rate = std::max(kOffsetRate, 1.0 / static_cast<double>(tree_samples_[r] + 1));
    tree_offset_[r] += rate * clamp_step(t - base_ - tree_offset_[r], predicted);
  }
  ++tree_samples_[r];
  tree_last_seen_[r] = steps_;
}

void DraftWidthPolicy::observe_tree_value(double gain, std::uint32_t rows) {
  const double cost = tree_verify_ns(rows) - verify_ns(1) + draft_ns(kMaxProposals);
  const double value = gain - rate() * cost;
  if (!draft_value_known_) {
    draft_value_ = value;
    draft_value_known_ = true;
  } else {
    draft_value_ += kValueRate * (value - draft_value_);
  }
}

}  // namespace lse::runtime
