#include "harness.hpp"
#include "../src/graph/dispatch_profile.hpp"

using lse::graph::detail::DispatchProfile;
using lse::graph::detail::DispatchProfileMode;
using lse::graph::detail::dispatch_profile_mode;

LSE_TEST(dispatch_profile_modes_are_explicit) {
  LSE_EXPECT(dispatch_profile_mode(nullptr) == DispatchProfileMode::kOff);
  LSE_EXPECT(dispatch_profile_mode("off") == DispatchProfileMode::kOff);
  LSE_EXPECT(dispatch_profile_mode("submit") == DispatchProfileMode::kSubmit);
  LSE_EXPECT(dispatch_profile_mode("serial") == DispatchProfileMode::kSerial);
  LSE_EXPECT(dispatch_profile_mode("1") == DispatchProfileMode::kInvalid);
}

LSE_TEST(dispatch_profile_separates_shapes_failures_and_clock_domains) {
  DispatchProfile p;
  p.record("q6 M1", 10, 100, 7, true);
  p.record("q6 M1", 30, 300, 3, false);
  p.record("q6 M16", 80, 90, 5, true);
  const auto submit = p.rows(false);
  LSE_EXPECT_EQ(submit.size(), 2u);
  LSE_EXPECT(submit[0].first == "q6 M16");
  const auto serial = p.rows(true);
  LSE_EXPECT(serial[0].first == "q6 M1");
  const auto& s = serial[0].second;
  LSE_EXPECT_EQ(s.count, 2u);
  LSE_EXPECT_EQ(s.failures, 1u);
  LSE_EXPECT_EQ(s.submit_ns, 40u);
  LSE_EXPECT_EQ(s.completion_ns, 400u);
  LSE_EXPECT_EQ(s.predrain_ns, 10u);
  LSE_EXPECT_EQ(s.max_submit_ns, 30u);
  LSE_EXPECT_EQ(s.max_completion_ns, 300u);
}

LSE_TEST(dispatch_profile_has_a_bounded_overflow_bucket) {
  DispatchProfile p;
  for (std::size_t i = 0; i < DispatchProfile::kMaxShapes + 9; ++i) {
    p.record(std::to_string(i), 1, 0, 0, true);
  }
  p.record(std::string(DispatchProfile::kMaxKeyBytes + 1, 'x'), 2, 0, 0, true);
  const auto rows = p.rows(false);
  LSE_EXPECT_EQ(rows.size(), DispatchProfile::kMaxShapes + 1);
  LSE_EXPECT(rows[0].first == "<other-shapes>");
  LSE_EXPECT_EQ(rows[0].second.count, 10u);
  LSE_EXPECT_EQ(rows[0].second.submit_ns, 11u);
}
LSE_TEST_MAIN()
