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
// (code to prose): half weight after 20 more checks of that position, about
// 20 steps for the first one. A position fades only when it is checked
// itself: one that a shallow chain stops checking keeps what it last showed
// instead of sliding back to the prior, which would price every deeper chain
// at the prior and keep the chain shallow for good.
constexpr double kPositionDecay = 1.0 - 0.6931471805599453 / 20.0;
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
  // Only a rise is clamped. A stall can make one step slow, but nothing makes
  // a step faster than its work, so a fall says the estimate was high (it
  // may hold a step that compiled) and is taken whole. Scales below a
  // millisecond clamp as one millisecond does.
  const double bound = kOutlierClamp * std::max(scale, 1e6);
  return std::min(residual, bound);
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
  // A position starts at the one before it (the first at kPositionPrior), so
  // one never checked is priced like the deepest position that was and the
  // chain reaches it to find out.
  double mean = kPositionPrior;
  for (std::uint32_t p = 0; p <= position && p < kMaxProposals; ++p) {
    const Bin& b = positions_[p];
    mean = (b.accepted + kPriorWeight * mean) / (b.tested + kPriorWeight);
  }
  return mean;
}

std::size_t DraftWidthPolicy::context_bucket(std::int64_t tokens) noexcept {
  std::size_t bucket = 0;
  for (std::int64_t at = 2048; tokens >= at && bucket + 1 < kContextBuckets; at *= 2) ++bucket;
  return bucket;
}

template <std::size_t N>
void DraftWidthPolicy::seed_from(Ladder<N>& ladder, const Ladder<N>& from) {
  ladder = from;
  if (!ladder.live) return;
  ladder.seeded = true;
  ladder.seed = ladder.offset;
  ladder.seed_known = ladder.known;
  ladder.samples.fill(0);
  ladder.entered.fill(false);
}

template <std::size_t N>
bool DraftWidthPolicy::complete(const Ladder<N>& ladder, std::size_t first) noexcept {
  if (!ladder.live) return false;
  for (std::size_t i = first; i < N; ++i)
    if (!ladder.known[i]) return false;
  return true;
}

void DraftWidthPolicy::set_context(std::int64_t tokens) noexcept {
  const std::size_t bucket = context_bucket(tokens);
  if (bucket == bucket_) return;
  bucket_ = bucket;
  Costs& here = costs_[bucket];
  // Nearest other bucket whose ladder `pick` selects passes `usable`, the
  // shorter context on a tie.
  const auto nearest = [&](auto pick, auto usable) -> const Costs* {
    for (std::size_t d = 1; d < kContextBuckets; ++d) {
      if (bucket >= d && usable(pick(costs_[bucket - d]))) return &costs_[bucket - d];
      if (bucket + d < kContextBuckets && usable(pick(costs_[bucket + d])))
        return &costs_[bucket + d];
    }
    return nullptr;
  };
  if (here.verify.live || here.tree.live) {
    // A bucket left before its own measurements were finished (a process's
    // first request is mostly warm-up) would explore every width again on
    // its return, a forced width a step. While another bucket holds every
    // width, it starts from that one instead and measures only what that
    // one lacks, as a newly reached bucket does.
    const auto renew = [&](auto pick, std::size_t first) {
      auto& ladder = pick(here);
      if (!ladder.live || ladder.seeded || complete(ladder, first)) return;
      const auto usable = [&](const auto& l) { return complete(l, first); };
      if (const Costs* from = nearest(pick, usable)) seed_from(ladder, pick(*from));
    };
    renew([](auto& c) -> auto& { return c.verify; }, 1);
    renew([](auto& c) -> auto& { return c.tree; }, 0);
    return;
  }
  const Costs* from = nearest([](const Costs& c) -> const Costs& { return c; },
                              [](const Costs& c) { return c.verify.live || c.tree.live; });
  if (from == nullptr) return;
  here = *from;
  seed_from(here.verify, from->verify);
  seed_from(here.tree, from->tree);
  here.draft_seed = here.draft;
  here.draft_samples.fill(0);
  here.draft_entered.fill(false);
  here.draft_seeded = true;
}

template <std::size_t N>
bool DraftWidthPolicy::needs_sample(const Ladder<N>& ladder, std::size_t i) noexcept {
  // A bucket started from another one measures only what that one never had.
  return ladder.seeded ? !ladder.known[i] : ladder.samples[i] < kExploreSamples;
}

template <std::size_t N, class Rows>
void DraftWidthPolicy::follow_growth(Ladder<N>& ladder, Rows rows_of) {
  // How much each width measured here grew over its carried cost, fitted
  // linear in rows; widths not measured here take their carried cost plus
  // that growth.
  double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (ladder.samples[i] == 0 || !ladder.seed_known[i]) continue;
    const double x = rows_of(i), y = ladder.offset[i] - ladder.seed[i];
    n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  if (n == 0) return;
  const double spread = n * sxx - sx * sx;
  const double slope = n >= 2 && spread > 0 ? (n * sxy - sx * sy) / spread : 0.0;
  const double intercept = (sy - slope * sx) / n;
  for (std::size_t i = 0; i < N; ++i)
    if (ladder.samples[i] == 0 && ladder.seed_known[i])
      ladder.offset[i] = ladder.seed[i] + intercept + slope * rows_of(i);
}

template <std::size_t N, class Rows>
void DraftWidthPolicy::observe_cost(Ladder<N>& ladder, std::size_t i, double t, Rows rows_of) {
  bool any = false;
  for (const auto count : ladder.samples) any = any || count != 0;
  if (ladder.seeded && !ladder.entered[i]) {
    ladder.entered[i] = true;
    ladder.last_seen[i] = steps_;
    return;
  }
  if (ladder.seeded && ladder.samples[i] == 0) {
    // The first measurement of this width in a bucket started from another
    // one: the first moves the level, each later one sets its own offset.
    if (!any && ladder.known[i]) ladder.base = t - ladder.offset[i];
    else ladder.offset[i] = t - ladder.base;
  } else if (!any) {
    ladder.base = t;
    ladder.offset[i] = 0.0;
  } else if (ladder.samples[i] == 0) {
    ladder.offset[i] = t - ladder.base;
  } else if (ladder.samples[i] < kTrustedSamples && t < ladder.base + ladder.offset[i]) {
    // While a cost rests on few samples, a lower one replaces it: early
    // samples err high (a step that compiled, clocks still ramping), so the
    // estimate is their minimum until the width is trusted.
    ladder.offset[i] = t - ladder.base;
  } else {
    const double predicted = ladder.base + ladder.offset[i];
    ladder.base += kBaseRate * clamp_step(t - predicted, predicted);
    // A width's first observations average; one seen rarely (a refresh
    // every kRefreshSteps) keeps moving fast, so a sample taken while the
    // clocks ramped does not price it for good.
    const double rate = std::max(kOffsetRate, 1.0 / static_cast<double>(ladder.samples[i] + 1));
    ladder.offset[i] += rate * clamp_step(t - ladder.base - ladder.offset[i], predicted);
  }
  ladder.live = true;
  ladder.known[i] = true;
  ++ladder.samples[i];
  ladder.last_seen[i] = steps_;
  if (ladder.seeded) follow_growth(ladder, rows_of);
}

double DraftWidthPolicy::verify_ns(std::uint32_t rows) const noexcept {
  const auto& ladder = costs().verify;
  if (rows == 0 || rows > kMaxRows || !ladder.known[rows]) return 0.0;
  return std::max(0.0, ladder.base + ladder.offset[rows]);
}

double DraftWidthPolicy::rate() const noexcept {
  return rate_ns_ > 0.0 ? rate_tokens_ / rate_ns_ : 0.0;
}

std::uint32_t DraftWidthPolicy::exploring() const noexcept {
  if (steps_ < kWarmupSteps) return 0;
  const auto& ladder = costs().verify;
  std::uint32_t fewest = 0;
  for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows)
    if (needs_sample(ladder, rows) &&
        (fewest == 0 || ladder.samples[rows] < ladder.samples[fewest]))
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
  const auto& last_seen = costs().verify.last_seen;
  std::uint32_t stalest = 0;
  for (std::uint32_t rows = 1; rows <= max + 1; ++rows) {
    const std::uint64_t age = steps_ - last_seen[rows];
    if (age <= kRefreshSteps || (best - value[rows - 1] > kRefreshLoss && age <= kHardRefreshSteps))
      continue;
    if (stalest == 0 || last_seen[rows] < last_seen[stalest]) stalest = rows;
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
                  tree_step_ns(kTreeRows[r]) / 1e6);
    out += item;
  }
  out += " candidate=";
  for (const double q : {0.05, 0.2, 0.5, 0.8, 0.95}) {
    std::snprintf(item, sizeof item, "%s%.2f", q == 0.05 ? "" : ",", candidate(q));
    out += item;
  }
  std::snprintf(item, sizeof item, " ctx_bucket=%zu", bucket_);
  out += item;
  std::snprintf(item, sizeof item, " bias=%.2f", bias_);
  out += item;
  std::snprintf(item, sizeof item, " rate_tps=%.1f", rate() * 1e9);
  out += item;
  return out;
}

bool DraftWidthPolicy::draft_next(bool tree) {
  if (tree) {
    // Trees take no chain widths: a plain step is the only one to measure.
    if (steps_ >= kWarmupSteps && needs_sample(costs().verify, 1)) return false;
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
  Bin& b = bins_[bin_of(confidence)];
  b.tested += 1.0;
  b.accepted += accepted ? 1.0 : 0.0;
  if (position < kMaxProposals) {
    Bin& at = positions_[position];
    at.accepted = at.accepted * kPositionDecay + (accepted ? 1.0 : 0.0);
    at.tested = at.tested * kPositionDecay + 1.0;
  }
}

void DraftWidthPolicy::observe_verify(std::uint32_t rows, std::uint64_t ns) {
  if (rows == 0 || rows > kMaxRows) return;
  // Warm-up steps count toward the warm-up and price nothing.
  if (steps_++ < kWarmupSteps) return;
  observe_cost(costs().verify, rows, static_cast<double>(ns),
               [](std::size_t i) { return static_cast<double>(i); });
}

void DraftWidthPolicy::observe_draft(std::uint64_t ns, std::uint32_t depth) {
  if (depth > kMaxProposals || steps_ <= kWarmupSteps) return;
  const auto t = static_cast<double>(ns);
  Costs& c = costs();
  if (c.draft_seeded && !c.draft_entered[depth]) {
    c.draft_entered[depth] = true;
    return;
  }
  double& d = c.draft[depth];
  if (c.draft_samples[depth]++ == 0) {
    d = t;
    // A bucket started from another one: depths not yet measured here scale
    // as this one did (an MTP chain's passes all attend over the context).
    if (c.draft_seeded && c.draft_seed[depth] > 0.0) {
      const double scale = t / c.draft_seed[depth];
      for (std::uint32_t e = 0; e <= kMaxProposals; ++e)
        if (c.draft_samples[e] == 0) c.draft[e] = c.draft_seed[e] * scale;
    }
  } else if (c.draft_samples[depth] <= kTrustedSamples && t < d) {
    // A lower early sample replaces the estimate, as a width's does.
    d = t;
  } else {
    d += kDraftRate * clamp_step(t - d, d);
  }
}

double DraftWidthPolicy::draft_ns(std::uint32_t depth) const noexcept {
  return depth <= kMaxProposals ? costs().draft[depth] : 0.0;
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
  // A position holding less evidence than chains reaching it would give
  // (one behind weak positions is not expected to hold much) may be priced
  // low by a few unlucky checks. Priced instead at the mean of the last
  // position that holds enough, the depth worth most is probed when it
  // beats the chosen one; on a device where even that does not pay for the
  // deeper passes there is no probe.
  double reach = 1.0, trusted = kPositionPrior, hoped_chain = 1.0, hoped_gain = 0.0;
  double probe_value = best_value;
  std::uint32_t probe = best;
  bool doubtful = false;
  for (std::uint32_t d = 1; d <= max; ++d) {
    const std::uint32_t p = d - 1;
    const double mean = position_mean(p);
    const bool shaky = positions_[p].tested < kPositionTrusted * reach;
    if (!shaky) trusted = mean;
    doubtful = doubtful || shaky;
    reach *= mean;
    hoped_chain *= shaky ? std::max(mean, trusted) : mean;
    hoped_gain += hoped_chain;
    const double value = hoped_gain - rate_now * (draft_ns(d) + verify_ns(d + 1));
    if (doubtful && d > best && value > probe_value) {
      probe_value = value;
      probe = d;
    }
  }
  if (probe != best && steps_ - position_probe_ >= kPositionProbeSteps) {
    position_probe_ = steps_;
    return probe;
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

double DraftWidthPolicy::tree_step_ns(std::uint32_t rows) const noexcept {
  const std::size_t r = tree_rung(rows);
  const auto& ladder = costs().tree;
  if (r >= kTreeRows.size() || !ladder.known[r]) return 0.0;
  return std::max(0.0, ladder.base + ladder.offset[r]);
}

std::uint32_t DraftWidthPolicy::tree_exploring() const noexcept {
  if (steps_ < kWarmupSteps) return 0;
  const auto& ladder = costs().tree;
  std::size_t fewest = kTreeRows.size();
  for (std::size_t r = 0; r < kTreeRows.size(); ++r)
    if (needs_sample(ladder, r) &&
        (fewest == kTreeRows.size() || ladder.samples[r] < ladder.samples[fewest]))
      fewest = r;
  return fewest < kTreeRows.size() ? kTreeRows[fewest] : 0;
}

std::uint32_t DraftWidthPolicy::tree_nodes(std::span<const double> value_prefix,
                                           std::span<const double> chain_prefix) {
  if (value_prefix.size() < 2) return 0;
  const auto most = static_cast<std::uint32_t>(value_prefix.size() - 1);
  const auto fit = [&](std::uint32_t rows) { return std::min(most, rows - 1); };
  const auto chain_rows = static_cast<std::uint32_t>(
      std::min<std::size_t>(chain_prefix.size(), kMaxRows));
  const bool chain = chain_rows >= 2;
  // Warming up: the rung a chain would take.
  if (steps_ < kWarmupSteps) return fit(kTreeRows[1]);
  if (const std::uint32_t rows = tree_exploring(); rows != 0) return fit(rows);
  if (chain && exploring() != 0) return 0;
  // A rung unmeasured for long enough is measured again, whatever it costs.
  for (std::size_t r = 0; r < kTreeRows.size(); ++r)
    if (steps_ - costs().tree.last_seen[r] > kHardRefreshSteps) return fit(kTreeRows[r]);
  const double draft = draft_ns(kMaxProposals);
  std::uint32_t best = fit(kTreeRows[0]);
  double best_rate = -1.0;
  for (const std::uint32_t rows : kTreeRows) {
    const std::uint32_t nodes = fit(rows);
    const double cost = tree_step_ns(rows);
    if (cost <= 0.0) continue;
    const double rate = (1.0 + value_prefix[nodes]) / cost;
    if (rate > best_rate) {
      best_rate = rate;
      best = nodes;
    }
    if (nodes == most) break;
  }
  // The best chain prefix, priced as chain steps are.
  for (std::uint32_t rows = 2; chain && rows <= chain_rows; ++rows) {
    const double cost = draft + verify_ns(rows);
    if (cost > 0.0 && verify_ns(rows) > 0.0 && (1.0 + chain_prefix[rows - 1]) / cost > best_rate)
      return 0;
  }
  return best;
}

void DraftWidthPolicy::observe_tree_step(std::uint32_t rows, std::uint64_t ns) {
  const std::size_t r = tree_rung(rows);
  if (r >= kTreeRows.size()) return;
  if (steps_++ < kWarmupSteps) return;
  // A level of its own that follows the context, and per-rung offsets, as
  // the chain widths have.
  observe_cost(costs().tree, r, static_cast<double>(ns),
               [](std::size_t i) { return static_cast<double>(kTreeRows[i]); });
}

void DraftWidthPolicy::observe_tree_value(double gain, std::uint32_t rows) {
  const double cost = tree_step_ns(rows) - verify_ns(1);
  const double value = gain - rate() * cost;
  if (!draft_value_known_) {
    draft_value_ = value;
    draft_value_known_ = true;
  } else {
    draft_value_ += kValueRate * (value - draft_value_);
  }
}

}  // namespace lse::runtime
