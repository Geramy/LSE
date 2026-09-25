#include "harness.hpp"
#include "../src/cli/token_ids.hpp"

#include <stdexcept>
#include <string>

namespace {
std::string capture(std::span<const std::uint32_t> prompt,
                    std::span<const std::uint32_t> generated) {
  FILE* file = std::tmpfile();
  if (!file) throw std::runtime_error("tmpfile failed");
  const bool written = lse::cli::print_token_ids(file, prompt, generated);
  if (!written) {
    std::fclose(file);
    throw std::runtime_error("writing diagnostic failed");
  }
  std::rewind(file);
  std::string text;
  char buffer[64];
  while (std::fgets(buffer, sizeof(buffer), file)) text += buffer;
  std::fclose(file);
  return text;
}
}

LSE_TEST(token_ids_preserve_order_repeats_and_unsigned_range) {
  const std::uint32_t prompt[] = {760, 6511, 314, 9338, 369};
  const std::uint32_t generated[] = {0, UINT32_MAX, 0, 17};
  LSE_EXPECT(capture(prompt, generated) ==
      "[token-ids] {\"prompt_ids\":[760,6511,314,9338,369],"
      "\"generated_ids\":[0,4294967295,0,17],\"stop_tokens_excluded\":true}\n");
}

LSE_TEST(token_ids_empty_arrays_remain_valid_json) {
  LSE_EXPECT(capture({}, {}) ==
      "[token-ids] {\"prompt_ids\":[],\"generated_ids\":[],"
      "\"stop_tokens_excluded\":true}\n");
}
LSE_TEST_MAIN()
