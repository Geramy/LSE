#include "harness.hpp"
#include "lse/runtime/generator.hpp"

LSE_TEST(prefill_token_is_not_decode_throughput) {
  lse::runtime::GenerationStats s;
  for (int generated : {0, 1}) {
    s.generated_tokens = generated;
    s.decode_ns = 745000;
    LSE_EXPECT_EQ(s.decoded_tokens(), 0);
    LSE_EXPECT_EQ(s.decode_tokens_per_second(), 0.0);
  }
  s.generated_tokens = 17;
  s.decode_ns = 2000000000;
  LSE_EXPECT_EQ(s.decoded_tokens(), 16);
  LSE_EXPECT_EQ(s.decode_tokens_per_second(), 8.0);
  s.decode_ns = 0;
  LSE_EXPECT_EQ(s.decode_tokens_per_second(), 0.0);
}

LSE_TEST(prefill_rate_counts_only_evaluated_prompt_tokens) {
  lse::runtime::GenerationStats s;
  s.prompt_tokens = 5;
  s.prefill_ns = 2000000000;
  LSE_EXPECT_EQ(s.prompt_tokens_per_second(), 2.5);
  s.prefill_ns = 0;
  LSE_EXPECT_EQ(s.prompt_tokens_per_second(), 0.0);
  s.prefill_ns = 1;
  s.prompt_tokens = 0;
  LSE_EXPECT_EQ(s.prompt_tokens_per_second(), 0.0);
}
LSE_TEST_MAIN()
