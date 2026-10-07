#include "harness.hpp"
#include "../src/graph/variant_book.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <string>

using lse::graph::detail::VariantBook;

namespace {
std::string fresh_cache_dir() {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("lse-variant-book-" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  ::setenv("LSE_CACHE_DIR", dir.c_str(), 1);
  ::unsetenv("LSE_AUTOTUNE");
  return dir.string();
}
}  // namespace

LSE_TEST(variant_book_key_names_identity_menu_and_compiler) {
  const auto k = VariantBook::key(42, 3, "loomc 1");
  LSE_EXPECT(k == VariantBook::key(42, 3, "loomc 1"));
  LSE_EXPECT(k != VariantBook::key(43, 3, "loomc 1"));
  LSE_EXPECT(k != VariantBook::key(42, 2, "loomc 1"));
  LSE_EXPECT(k != VariantBook::key(42, 3, "loomc 2"));
}

LSE_TEST(variant_book_decisions_outlive_the_process_and_holds_do_not) {
  const std::string dir = fresh_cache_dir();
  {
    VariantBook book;
    LSE_EXPECT(book.enabled());
    LSE_EXPECT(!book.decided(1).has_value());
    book.record(1, 2, "0=10.0us 1=9.0us 2=8.0us");
    book.hold(2, 0);
    LSE_EXPECT(book.decided(1).value_or(9) == 2u);
    LSE_EXPECT(book.decided(2).value_or(9) == 0u);
  }
  {
    VariantBook later;
    LSE_EXPECT(later.decided(1).value_or(9) == 2u);
    LSE_EXPECT(!later.decided(2).has_value());
    // A decision measured again is appended; the last one stands.
    later.record(1, 0, "remeasured");
  }
  VariantBook third;
  LSE_EXPECT(third.decided(1).value_or(9) == 0u);
  std::filesystem::remove_all(dir);
}

LSE_TEST(variant_book_off_and_budget) {
  const std::string dir = fresh_cache_dir();
  ::setenv("LSE_AUTOTUNE", "off", 1);
  LSE_EXPECT(!VariantBook{}.enabled());
  ::unsetenv("LSE_AUTOTUNE");
  ::setenv("LSE_AUTOTUNE_BUDGET_MS", "1", 1);
  VariantBook book;
  LSE_EXPECT(book.has_budget());
  book.spend(2'000'000);
  LSE_EXPECT(!book.has_budget());
  ::unsetenv("LSE_AUTOTUNE_BUDGET_MS");
  std::filesystem::remove_all(dir);
}
LSE_TEST_MAIN()
