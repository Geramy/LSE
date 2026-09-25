#include "harness.hpp"
#include "jit_timings.hpp"

using namespace lse;

LSE_TEST(http_jit_timings_preserve_cumulative_counts_and_convert_nanoseconds) {
  runtime::GenerationStats stats;
  stats.jit_memory_hits = (std::uint64_t{1} << 54) + 7;
  stats.jit_disk_hits = 19;
  stats.jit_compiles = 23;
  stats.jit_compile_ns = 123456789;
  nlohmann::json timing{{"prompt_ms", 4.5}};
  server::detail::JitTotals::from(stats).append_to(timing);
  LSE_EXPECT_EQ(timing.at("jit_memory_hits_total").get<std::uint64_t>(), stats.jit_memory_hits);
  LSE_EXPECT_EQ(timing.at("jit_disk_hits_total").get<std::uint64_t>(), 19u);
  LSE_EXPECT_EQ(timing.at("jit_compiles_total").get<std::uint64_t>(), 23u);
  LSE_EXPECT_NEAR(timing.at("jit_compile_ms_total").get<double>(), 123.456789, 1e-9);
  LSE_EXPECT_NEAR(timing.at("prompt_ms").get<double>(), 4.5, 0.0);
  LSE_EXPECT(!timing.contains("jit_compiles"));
  LSE_EXPECT(!timing.contains("jit_compile_ms"));
  // JSON round trips must retain integer precision beyond 2^53.
  auto parsed = nlohmann::json::parse(timing.dump());
  LSE_EXPECT_EQ(parsed.at("jit_memory_hits_total").get<std::uint64_t>(), stats.jit_memory_hits);
}

LSE_TEST(http_jit_timings_snapshot_is_stable_and_zero_totals_are_explicit) {
  runtime::GenerationStats stats;
  stats.jit_compiles = 5;
  auto snapshot = server::detail::JitTotals::from(stats);
  stats = {};
  nlohmann::json old, fresh;
  snapshot.append_to(old);
  server::detail::JitTotals::from(stats).append_to(fresh);
  LSE_EXPECT_EQ(old.at("jit_compiles_total").get<unsigned>(), 5u);
  LSE_EXPECT_EQ(fresh.at("jit_compiles_total").get<unsigned>(), 0u);
  LSE_EXPECT_EQ(fresh.at("jit_memory_hits_total").get<unsigned>(), 0u);
  LSE_EXPECT_EQ(fresh.at("jit_disk_hits_total").get<unsigned>(), 0u);
  LSE_EXPECT_EQ(fresh.at("jit_compile_ms_total").get<double>(), 0.0);
}

LSE_TEST_MAIN()
