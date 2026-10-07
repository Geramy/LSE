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
// distribution exactly. (DSpark, arXiv 2607.05147, schedules verify prefixes
// the same way, from a learned confidence head and a profiled width curve.)
//
// Sampled requests use the policy; a greedy request verifies the whole block
// (Generator::speculate says why).
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
  // Calibration bins over the draft's top probability c, a quarter decade of
  // 1 - c each: [0, 0.44), [0.44, 0.68), ... [0.9, 0.944), ... up to
  // 1 - 1e-6. Acceptance changes most between c = 0.9 and c = 0.9999.
  static constexpr std::size_t kBins = 24;
  static constexpr double kBinsPerDecade = 4.0;
  [[nodiscard]] static std::size_t bin_of(double confidence) noexcept;
  // Observations of every verify width before the policy trusts its costs.
  static constexpr std::uint32_t kExploreSamples = 2;
  // Steps after which a width's cost is measured again.
  static constexpr std::uint64_t kRefreshSteps = 4096;
  // Skipped (draft-free) steps before a probing draft.
  static constexpr std::uint32_t kProbeAfter = 4;

  // Proposals to verify, 0..confidence.size(): confidence[j] is the draft's
  // top conditional probability at proposal j + 1. Fills `estimates` (when
  // not empty) with the calibrated acceptance of each position.
  [[nodiscard]] std::uint32_t proposals(std::span<const double> confidence,
                                        std::span<double> estimates = {});

  // Whether the next step drafts at all. False means a plain step: one row,
  // no proposals, no draft.
  [[nodiscard]] bool draft_next();

  // What the target said about a proposal of this confidence.
  void observe_acceptance(double confidence, std::uint32_t position, bool accepted);
  // Wall time of a verify pass of `rows` rows, from its submission to the end
  // of its acceptance walk (prefix commit and context append included).
  void observe_verify(std::uint32_t rows, std::uint64_t ns);
  // A draft of `depth` proposals. A DFlash2 draft costs the same at any
  // depth (it always evaluates the whole block) and is recorded at the full
  // depth; an MTP chain costs a pass per proposal, and depth 0 is the
  // catch-up pass alone.
  void observe_draft(std::uint64_t ns, std::uint32_t depth = kMaxProposals);
  // A whole step: tokens it emitted and its wall time, draft included.
  void observe_step(std::uint32_t tokens, std::uint64_t ns);
  // After a draft was verified with `estimates` for its chosen prefix of
  // `proposals`: the expected value of having drafted, used to decide
  // whether drafting pays (draft_next).
  void observe_draft_value(std::span<const double> estimates, std::uint32_t proposals);

  // Estimated wall time of a verify pass of `rows` rows (0 until measured).
  [[nodiscard]] double verify_ns(std::uint32_t rows) const noexcept;
  [[nodiscard]] double draft_ns(std::uint32_t depth = kMaxProposals) const noexcept;
  // For a chained draft (MTP): how many proposals to draft before any of
  // them exists, 0..max, maximizing expected tokens per second from each
  // position's mean acceptance and the measured cost of each depth and verify
  // width. proposals() then trims the verified prefix by confidence.
  [[nodiscard]] std::uint32_t depth(std::uint32_t max);
  // Long-run tokens per nanosecond.
  [[nodiscard]] double rate() const noexcept;
  // Calibrated acceptance of a proposal of this confidence.
  [[nodiscard]] double acceptance(double confidence) const noexcept;
  // The width the next step must take to measure its cost, or 0 when every
  // width is measured and fresh.
  [[nodiscard]] std::uint32_t exploring() const noexcept;

 private:
  struct Bin {
    double accepted = 0, tested = 0;
  };
  [[nodiscard]] double position_mean(std::uint32_t position) const noexcept;

  // Acceptance by confidence, and by position (the mean the lookahead uses).
  std::array<Bin, kBins> bins_{};
  std::array<Bin, kMaxProposals> positions_{};
  // verify(m) = base_ + offset_[m]: one level that follows the context, and
  // per-width offsets that hold what the extra rows cost.
  double base_ = 0;
  std::array<double, kMaxRows + 1> offset_{};
  std::array<std::uint32_t, kMaxRows + 1> samples_{};
  std::array<std::uint64_t, kMaxRows + 1> last_seen_{};
  std::array<double, kMaxRows> draft_ns_{};
  std::array<std::uint32_t, kMaxRows> draft_samples_{};
  double rate_tokens_ = 0, rate_ns_ = 0;
  std::uint64_t steps_ = 0;
  // Expected tokens gained per drafted step minus what the draft and the
  // wider pass cost at the long-run rate, and how long drafting has been off.
  double draft_value_ = 0;
  bool draft_value_known_ = false;
  std::uint32_t skipped_ = 0;
};

}  // namespace lse::runtime
