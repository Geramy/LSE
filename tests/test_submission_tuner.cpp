#include "harness.hpp"
#include "lse/backends/hrx/submission_tuner.hpp"
#include <limits>
using lse::backend::SubmissionTuner;

static void train(SubmissionTuner::State& s, const std::array<double, 4>& times) {
  for (unsigned attempt = 0; attempt < 40 && !s.done; ++attempt) {
    const auto interval = SubmissionTuner::next(s);
    const auto it = std::find(SubmissionTuner::candidates.begin(), SubmissionTuner::candidates.end(), interval);
    SubmissionTuner::observe(s, 123, times[static_cast<std::size_t>(it - SubmissionTuner::candidates.begin())], true);
  }
}
LSE_TEST(submission_tuner_selects_measured_win_and_caches_it) {
  SubmissionTuner tuner;
  auto* s = tuner.find(42);
  train(*s, {200, 140, 139, 180});
  LSE_EXPECT(s->done && s->selected);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 64u);
  LSE_EXPECT_EQ(s->samples, 20u);
  const auto attempts = s->attempts;
  SubmissionTuner::observe(*s, 123, 9000, true);
  LSE_EXPECT_EQ(s->attempts, attempts);
  LSE_EXPECT(tuner.find(42) == s);
}
LSE_TEST(submission_tuner_preserves_baseline_without_five_percent_win) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  train(*s, {200, 194, 197, 199});
  LSE_EXPECT(s->done && !s->selected);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 16u);
}
LSE_TEST(submission_tuner_can_select_deferred_only_when_measurably_best) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  train(*s, {200, 150, 140, 100});
  LSE_EXPECT(s->done && s->selected);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 0u);
}
LSE_TEST(submission_tuner_excludes_jit_and_warmup_and_bounds_attempts) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  for (unsigned i = 0; i < 5; ++i) SubmissionTuner::observe(*s, 123, 100, false);
  LSE_EXPECT_EQ(s->samples, 0u);
  for (unsigned i = 0; i < 80; ++i) SubmissionTuner::observe(*s, 123, 100, false);
  LSE_EXPECT(s->done && !s->selected);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 16u);
}
LSE_TEST(submission_tuner_invalidates_changed_workload_shape) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  train(*s, {200, 140, 139, 180});
  SubmissionTuner::observe(*s, 456, 100, true);
  LSE_EXPECT(!s->done && !s->selected);
  LSE_EXPECT_EQ(s->samples, 0u);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 16u);
}
LSE_TEST(submission_tuner_rejects_noisy_evidence) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  for (unsigned i = 0; i < 40 && !s->done; ++i) {
    const auto round = s->samples / 4;
    SubmissionTuner::observe(*s, 123, (round % 3 == 0 ? 100 : (round % 3 == 1 ? 200 : 300)), true);
  }
  LSE_EXPECT(s->done && !s->selected);
}
LSE_TEST(submission_tuner_bounds_workload_cache_and_preserves_override) {
  SubmissionTuner tuner;
  for (unsigned i = 0; i < 16; ++i) LSE_EXPECT(tuner.find(i) != nullptr);
  LSE_EXPECT(tuner.find(17) == nullptr);
  using lse::backend::automatic_submission_policy;
  LSE_EXPECT(automatic_submission_policy(nullptr, nullptr));
  LSE_EXPECT(!automatic_submission_policy(nullptr, "0"));
  for (auto value : {"", "0", "64", "invalid"}) LSE_EXPECT(!automatic_submission_policy(value, nullptr));
}
LSE_TEST(submission_tuner_noisy_alternative_does_not_veto_stable_improvement) {
  SubmissionTuner tuner;
  auto* s = tuner.find(1);
  for (unsigned attempt = 0; attempt < 40 && !s->done; ++attempt) {
    const auto interval = SubmissionTuner::next(*s);
    const double elapsed = interval == 16 ? 200 : interval == 64 ? 140 :
        interval == 256 ? 139 : (s->samples / 4 % 3 == 0 ? 100 :
                                s->samples / 4 % 3 == 1 ? 200 : 300);
    SubmissionTuner::observe(*s, 123, elapsed, true);
  }
  LSE_EXPECT(s->done && s->selected);
  LSE_EXPECT_EQ(SubmissionTuner::next(*s), 64u);
}
LSE_TEST_MAIN()
