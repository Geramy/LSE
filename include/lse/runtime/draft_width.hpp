// How many of a DFlash2 draft's proposals the target verifies, step by step.
//
// A DFlash2 draft always proposes a whole block (seven tokens after the
// anchor), but a verify pass costs more the more rows it carries, and a
// proposal past the first likely rejection rarely pays for its row. The
// policy verifies the prefix that maximizes expected tokens per unit of time:
//
//   tokens(k) = 1 + sum_{j<=k} prod_{i<=j} a_i
//   time(k)   = draft + verify(k + 1)
//
// where a_i is the estimated chance that proposal i is accepted given the ones
// before it were. a_i comes from the draft's own conditional distribution at
// position i (its largest probability), mapped through a calibration table
// that learns, from every proposal the target checks, how often proposals of
// that confidence are accepted. verify(m) and draft are wall times measured on
// this device for this model, so each GPU tunes itself; the first steps of a
// process visit every verify width to measure it.
//
// The choice of k is a stopping rule: whether proposal j is verified depends
// only on the draft's distributions for positions 1..j and on the drafted
// tokens before j, never on token j itself or anything after it. Under that
// rule the proposal at every verified position is still a sample of the
// draft's conditional there, so rejection sampling keeps the target's
// distribution exactly, and greedy verification is unchanged.
//
// When even the best prefix is expected to lose to a plain decode step (one
// row, no draft), the next steps skip the draft. The draft module keeps no
// state between drafts beyond its context, and every verify pass, plain or
// not, appends its verified rows to that context, so skipping is safe; a
// skipped stretch ends with a probing draft that refreshes the estimate.
//
// One policy serves one engine (one model on one device) across requests,
// and is used by one generation at a time.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace lse::runtime {

class DraftWidthPolicy {
 public:
  // Verify rows: the anchor and up to seven proposals.
  static constexpr std::uint32_t kMaxRows = 8;
  static constexpr std::uint32_t kMaxProposals = kMaxRows - 1;
  // Calibration bins over the draft's top probability, [0, 1].
  static constexpr std::size_t kBins = 20;
  // Observations of every verify width before the policy trusts its costs.
  static constexpr std::uint32_t kExploreSamples = 2;
  // Steps after which a width's cost is measured again.
  static constexpr std::uint64_t kRefreshSteps = 4096;
  // Skipped (draft-free) steps before a probing draft.
  static constexpr std::uint32_t kProbeAfter = 8;

  // Proposals to verify, 0..confidence.size(): confidence[j] is the draft's
  // top conditional probability at proposal j + 1. `sampled` selects the
  // calibration table (sampled drafts and greedy drafts read their
  // distributions at different temperatures). Fills `estimates` (when not
  // empty) with the calibrated acceptance of each position.
  [[nodiscard]] std::uint32_t proposals(std::span<const double> confidence, bool sampled,
                                        std::span<double> estimates = {});

  // Whether the next step drafts at all. False means a plain step: one row,
  // no proposals, no draft.
  [[nodiscard]] bool draft_next();

  // What the target said about a proposal of this confidence.
  void observe_acceptance(double confidence, std::uint32_t position, bool accepted,
                          bool sampled);
  // Wall time of a verify pass of `rows` rows, from its submission to the end
  // of its acceptance walk (prefix commit and context append included).
  void observe_verify(std::uint32_t rows, std::uint64_t ns);
  void observe_draft(std::uint64_t ns);
  // A whole step: tokens it emitted and its wall time, draft included.
  void observe_step(std::uint32_t tokens, std::uint64_t ns);
  // After a draft was verified with `estimates` for its chosen prefix of
  // `proposals`: the expected value of having drafted, used to decide
  // whether drafting pays (draft_next).
  void observe_draft_value(std::span<const double> estimates, std::uint32_t proposals);

  // Estimated wall time of a verify pass of `rows` rows (0 until measured).
  [[nodiscard]] double verify_ns(std::uint32_t rows) const noexcept;
  [[nodiscard]] double draft_ns() const noexcept { return draft_ns_; }
  // Long-run tokens per nanosecond.
  [[nodiscard]] double rate() const noexcept;
  // Calibrated acceptance of a proposal of this confidence.
  [[nodiscard]] double acceptance(double confidence, bool sampled) const noexcept;
  // The width the next step must take to measure its cost, or 0 when every
  // width is measured and fresh.
  [[nodiscard]] std::uint32_t exploring() const noexcept;

 private:
  struct Bin {
    double accepted = 0, tested = 0;
  };
  struct Calibration {
    std::array<Bin, kBins> bins{};
    std::array<Bin, kMaxProposals> positions{};
  };
  [[nodiscard]] double position_mean(std::uint32_t position, bool sampled) const noexcept;

  std::array<Calibration, 2> calibration_{};  // [greedy, sampled]
  // verify(m) = base_ + offset_[m]: one level that follows the context, and
  // per-width offsets that hold what the extra rows cost.
  double base_ = 0;
  std::array<double, kMaxRows + 1> offset_{};
  std::array<std::uint32_t, kMaxRows + 1> samples_{};
  std::array<std::uint64_t, kMaxRows + 1> last_seen_{};
  double draft_ns_ = 0;
  std::uint32_t draft_samples_ = 0;
  double rate_tokens_ = 0, rate_ns_ = 0;
  std::uint64_t steps_ = 0;
  // Expected tokens gained per drafted step minus what the draft and the
  // wider pass cost at the long-run rate, and how long drafting has been off.
  double draft_value_ = 0;
  bool draft_value_known_ = false;
  std::uint32_t skipped_ = 0;
};

}  // namespace lse::runtime
