#include "harness.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/server/http_server.hpp"

#include <limits>

using namespace lse;

LSE_TEST(mtp_depth_defaults_and_bounds) {
  runtime::GenerationLimits limits;
  server::ServerOptions options;
  LSE_EXPECT_EQ(limits.mtp_depth, 2u);
  LSE_EXPECT_EQ(options.mtp_depth, limits.mtp_depth);
  for (std::uint32_t depth = 1; depth <= 7; ++depth)
    LSE_EXPECT(runtime::valid_mtp_depth(depth));
  LSE_EXPECT(!runtime::valid_mtp_depth(0));
  LSE_EXPECT(!runtime::valid_mtp_depth(8));
  LSE_EXPECT(!runtime::valid_mtp_depth(std::numeric_limits<std::uint32_t>::max()));
}

LSE_TEST(mtp_acceptance_counts_compared_proposals) {
  runtime::GenerationStats stats;
  LSE_EXPECT_EQ(stats.mtp_depth, 0u);
  LSE_EXPECT_EQ(stats.acceptance_rate(), 0.0);
  stats.spec_steps = 3;
  stats.spec_tested = 6;
  stats.spec_accepted = 5;
  LSE_EXPECT_EQ(stats.acceptance_rate(), 5.0 / 6.0);
  stats.spec_accepted = stats.spec_tested;
  LSE_EXPECT_EQ(stats.acceptance_rate(), 1.0);
}
LSE_TEST(mtp_verifier_width_respects_tokens_and_kv_capacity) {
  using runtime::mtp_verify_rows;
  LSE_EXPECT_EQ(mtp_verify_rows(2, 100, 100), 3u);
  LSE_EXPECT_EQ(mtp_verify_rows(7, 100, 100), 8u);
  LSE_EXPECT_EQ(mtp_verify_rows(1, 100, 100), 2u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 2, 100), 2u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 1, 100), 1u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 0, 100), 0u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 100, 2), 2u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 100, 1), 1u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 100, 0), 0u);
  LSE_EXPECT_EQ(mtp_verify_rows(2, 100, -1), 0u);
  LSE_EXPECT_EQ(mtp_verify_rows(0, 100, 100), 0u);
  LSE_EXPECT_EQ(mtp_verify_rows(8, 100, 100), 0u);
}
LSE_TEST_MAIN()
