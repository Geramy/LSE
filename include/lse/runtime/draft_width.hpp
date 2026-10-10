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
#include <string>

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
  // Observations of every verify width before the policy trusts its costs,
  // taken round-robin so a clock ramp at the start of a process does not
  // favor whichever widths come last.
  static constexpr std::uint32_t kExploreSamples = 2;
  // Steps after which a width's cost is measured again, on a step where that
  // costs at most kRefreshLoss expected tokens by the position means (on
  // code at the full block a narrower width never qualifies; it would throw
  // away most of a step).
  static constexpr std::uint64_t kRefreshSteps = 512;
  static constexpr double kRefreshLoss = 0.1;
  // ... and whatever it costs after this many, so an estimate that came out
  // high (and so is never cheap) cannot keep its width out of use for good.
  static constexpr std::uint64_t kHardRefreshSteps = 2048;
  // Until a cost rests on kTrustedSamples measurements in its context bucket,
  // a lower sample replaces it (early samples err high: a step whose wall
  // time was mostly a first compile, clocks still ramping). It is not
  // measured again any sooner than other costs: forced measurements of
  // narrow widths throw away most of a step, and at long contexts, where a
  // request is a few dozen steps, they cost more than they could find.
  static constexpr std::uint32_t kTrustedSamples = kExploreSamples;
  // Measured steps before the first measurements of every width: the GPU's
  // clocks ramp up over a process's first steps, and a width measured then
  // would be priced high. Until then the full width is used.
  static constexpr std::uint64_t kWarmupSteps = 32;
  // A position's mean resting on fewer than kPositionTrusted checks, scaled
  // by the chance a chain reaches it (a chain shallower than it never does),
  // is doubtful: when pricing it at the last trusted position's mean makes
  // a deeper chain worth more than the chosen one, that chain is drafted at
  // most every kPositionProbeSteps steps until the evidence is in. A few
  // unlucky early checks cannot price the deeper chains out for good, and a
  // device where deeper passes cannot pay is not probed.
  static constexpr double kPositionTrusted = 8.0;
  static constexpr std::uint64_t kPositionProbeSteps = 16;
  // Skipped (draft-free) steps before a probing draft.
  static constexpr std::uint32_t kProbeAfter = 4;

  // Proposals to verify, 0..confidence.size(): confidence[j] is the draft's
  // top conditional probability at proposal j + 1. Fills `estimates` (when
  // not empty) with the calibrated acceptance of each position.
  [[nodiscard]] std::uint32_t proposals(std::span<const double> confidence,
                                        std::span<double> estimates = {});

  // Whether the next step drafts at all. False means a plain step: one row,
  // no proposals, no draft.
  // `tree`: the drafts are trees (observe_tree_step measures them).
  [[nodiscard]] bool draft_next(bool tree = false);

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
  // A step left unpriced (it compiled, loaded or tried kernels): it still
  // counts toward the warm-up, whose steps are not priced either, so a cold
  // first request does not push the first measurements back.
  void observe_unpriced_step() noexcept {
    if (steps_ < kWarmupSteps) ++steps_;
  }
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
  // `initial` is the depth used while the policy warms up (kWarmupSteps).
  [[nodiscard]] std::uint32_t depth(std::uint32_t max, std::uint32_t initial);
  // Long-run tokens per nanosecond.
  [[nodiscard]] double rate() const noexcept;
  // Calibrated acceptance of a proposal of this confidence.
  [[nodiscard]] double acceptance(double confidence) const noexcept;
  // The width the next step must take for its first measurements, or 0
  // (also 0 while warming up).
  [[nodiscard]] std::uint32_t exploring() const noexcept;
  // A stale width (unmeasured for kRefreshSteps) cheap enough to measure on
  // this step, or 0. `chained`: the draft costs a pass per proposal (MTP).
  [[nodiscard]] std::uint32_t refresh(std::uint32_t max, bool chained) const noexcept;
  // Everything the policy has learned, on one line (for LSE_DEBUG traces).
  [[nodiscard]] std::string describe() const;

  // The KV length the next pass attends over. Every cost is kept per context
  // bucket (a doubling of the context each): a verify pass's attention reads
  // the whole cache once per row tile, so the extra rows of a wide pass cost
  // more the longer the context, and costs measured at 2K priced a 64K pass's
  // 31-row tree as if it cost what it did there. A bucket first entered
  // starts from the nearest measured one; its first measured width moves its
  // level, each width measured there takes its own cost at once, and widths
  // not yet measured there follow the growth the measured ones show, linear
  // in rows. Without a call every cost sits in the first bucket.
  void set_context(std::int64_t tokens) noexcept;
  static constexpr std::size_t kContextBuckets = 9;
  // 0 below 2K tokens, then one per doubling: 2K, 4K, ... 128K, 256K and up.
  [[nodiscard]] static std::size_t context_bucket(std::int64_t tokens) noexcept;

  // Draft trees (runtime::DraftTree). A tree pass's rows come in rungs, the
  // widths whose costs the policy measures; a tree of b nodes takes b + 1 rows.
  // 15 and 31 rather than 16 and 32 rows: draft-tree verify widths that no
  // prompt pass takes, with one and two 16-row tiles of kernels of their own
  // (the 16-row matrix panel, kQuantVerifyRows).
  static constexpr std::array<std::uint32_t, 4> kTreeRows{4, 8, 15, 31};
  static constexpr std::uint32_t kMaxTreeNodes = 30;
  // The chance the target answers a candidate the draft gives conditional
  // probability q, learned from every candidate of every row a tree walk
  // reaches (the draft's own probability until it has data).
  [[nodiscard]] double candidate(double q) const noexcept;
  void observe_candidate(double q, bool answered);
  // Nodes for the next tree, given its expansion's value prefix
  // (value_prefix[b]: expected accepted nodes of the best b-node tree): the
  // rung that maximizes expected tokens per second, b + 1 <= the rung's rows.
  // Decided from the draft's distributions alone, never from any token the
  // target has not yet answered, so the walk stays exact.
  //
  // `chain_prefix[k]` is the expected accepted proposals of the first k of
  // the draft's top path verified as a chain (k + 1 rows). 0 means a chain
  // is expected to decode faster than any tree, or a chain width still has
  // to be measured: the step then drafts a chain and proposals() picks its
  // width as it does without trees.
  [[nodiscard]] std::uint32_t tree_nodes(std::span<const double> value_prefix,
                                         std::span<const double> chain_prefix = {});
  // A tree step's whole wall time, draft and commit included: a tree's
  // commit runs on the device while the next draft waits for it, so the two
  // are priced together rather than split.
  void observe_tree_step(std::uint32_t rows, std::uint64_t ns);
  [[nodiscard]] double tree_step_ns(std::uint32_t rows) const noexcept;
  // After a tree of `rows` rows with expected accepted nodes `gain`: what
  // drafting was worth (draft_next).
  void observe_tree_value(double gain, std::uint32_t rows);

 private:
  struct Bin {
    double accepted = 0, tested = 0;
  };
  [[nodiscard]] double position_mean(std::uint32_t position) const noexcept;

  // Acceptance by confidence, and by position (the mean the lookahead uses).
  // bias_ is a log-odds correction learned online with a short memory, so a
  // change of text (prose to code) moves every estimate within a few steps
  // while the bins catch up.
  std::array<Bin, kBins> bins_{};
  double bias_ = 0;
  std::array<Bin, kMaxProposals> positions_{};
  std::uint64_t position_probe_ = 0;  // step of the last full-depth probe
  // cost(i) = base + offset[i]: one level that follows the context within a
  // bucket, and per-width (or per-rung) offsets that hold what the extra
  // rows cost.
  template <std::size_t N>
  struct Ladder {
    bool live = false;    // holds a cost, measured here or carried over
    bool seeded = false;  // started from another bucket's costs
    double base = 0;
    std::array<double, N> offset{}, seed{};
    std::array<bool, N> known{}, seed_known{};
    // A width's first pass in a bucket started from another one is not
    // priced (it pays for whatever that bucket's first passes load).
    std::array<bool, N> entered{};
    std::array<std::uint32_t, N> samples{};  // measured in this bucket
    std::array<std::uint64_t, N> last_seen{};
  };
  struct Costs {
    Ladder<kMaxRows + 1> verify;          // by rows
    Ladder<kTreeRows.size()> tree;        // by rung
    std::array<double, kMaxRows> draft{}, draft_seed{};
    std::array<std::uint32_t, kMaxRows> draft_samples{};
    std::array<bool, kMaxRows> draft_entered{};
    bool draft_seeded = false;
  };
  template <std::size_t N, class Rows>
  void observe_cost(Ladder<N>& ladder, std::size_t i, double t, Rows rows_of);
  template <std::size_t N, class Rows>
  static void follow_growth(Ladder<N>& ladder, Rows rows_of);
  template <std::size_t N>
  [[nodiscard]] static bool needs_sample(const Ladder<N>& ladder, std::size_t i) noexcept;
  [[nodiscard]] Costs& costs() noexcept { return costs_[bucket_]; }
  [[nodiscard]] const Costs& costs() const noexcept { return costs_[bucket_]; }
  std::array<Costs, kContextBuckets> costs_{};
  std::size_t bucket_ = 0;
  double rate_tokens_ = 0, rate_ns_ = 0;
  std::uint64_t steps_ = 0;
  // Expected tokens gained per drafted step minus what the draft and the
  // wider pass cost at the long-run rate, and how long drafting has been off.
  double draft_value_ = 0;
  bool draft_value_known_ = false;
  std::uint32_t skipped_ = 0;

  [[nodiscard]] static std::size_t tree_rung(std::uint32_t rows) noexcept;
  [[nodiscard]] std::uint32_t tree_exploring() const noexcept;
  static constexpr std::size_t kCandidateBins = 20;
  std::array<Bin, kCandidateBins> candidates_{};
};

// Whether DFlash2 steps may verify draft trees. `setting` is
// lse_config::dflash2_tree: positive always, zero never, negative (the
// default) whenever the adaptive policy runs, since it prices every tree rung
// against every chain prefix from this device's measured step costs and
// takes a chain wherever wide passes cost more than they return.
[[nodiscard]] constexpr bool draft_trees_enabled(int setting, bool adaptive) noexcept {
  return setting > 0 || (setting < 0 && adaptive);
}

}  // namespace lse::runtime
