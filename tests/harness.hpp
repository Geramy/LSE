// Minimal test harness — no external dependency, so the suite builds anywhere.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace lse::test {

struct Registry {
  struct Case {
    std::string name;
    void (*fn)();
  };
  std::vector<Case> cases;
  int failures = 0;
  std::string skip_reason;
  std::string current;

  static Registry& get() {
    static Registry r;
    return r;
  }
};

struct Registrar {
  Registrar(const char* name, void (*fn)()) {
    Registry::get().cases.push_back({name, fn});
  }
};

inline void fail(const char* file, int line, const std::string& what) {
  Registry& r = Registry::get();
  ++r.failures;
  std::fprintf(stderr, "  FAIL %s\n    at %s:%d\n    %s\n", r.current.c_str(),
               file, line, what.c_str());
}

inline void skip(const char* reason) {
  Registry::get().skip_reason = reason;
}

// The cases of this binary listed in tests/KNOWN_FAILURES.md as
// `<binary>:<case>` entries. Those fail for a known reason and are skipped,
// so a run reports only new failures. LSE_RUN_KNOWN_FAILURES=1 runs them.
inline std::vector<std::string> known_failures() {
  std::vector<std::string> cases;
#if defined(LSE_TEST_BINARY) && defined(LSE_KNOWN_FAILURES_FILE)
  const char* run_them = std::getenv("LSE_RUN_KNOWN_FAILURES");
  if (run_them != nullptr && *run_them != '\0' && *run_them != '0')
    return cases;
  std::ifstream in(LSE_KNOWN_FAILURES_FILE);
  const std::string prefix = std::string(LSE_TEST_BINARY) + ":";
  for (std::string line; std::getline(in, line);) {
    if (line.rfind("- `", 0) != 0) continue;
    const auto end = line.find('`', 3);
    if (end == std::string::npos) continue;
    const std::string entry = line.substr(3, end - 3);
    if (entry.rfind(prefix, 0) == 0) cases.push_back(entry.substr(prefix.size()));
  }
#endif
  return cases;
}

inline int run_all() {
  Registry& r = Registry::get();
  int passed = 0;
  int skipped = 0;
  // LSE_TEST_ONLY=<name>[,<name>...] runs just those cases, so one gated case
  // can be run without the cases before it leaving process-wide knobs set.
  // A case named there runs even when it is a known failure.
  const char* only = std::getenv("LSE_TEST_ONLY");
  const std::vector<std::string> known = known_failures();
  std::size_t run = 0;
  for (const auto& c : r.cases) {
    bool named = false;
    if (only != nullptr && *only != '\0') {
      const std::string list = std::string(",") + only + ",";
      if (list.find("," + c.name + ",") == std::string::npos) continue;
      named = true;
    }
    ++run;
    bool is_known = false;
    for (const auto& k : known) is_known = is_known || k == c.name;
    if (is_known && !named) {
      ++skipped;
      std::printf("  skip %s (known failure, tests/KNOWN_FAILURES.md)\n",
                  c.name.c_str());
      continue;
    }
    if (std::getenv("LSE_TEST_TRACE") != nullptr) {
      std::fprintf(stderr, "[test] start %s\n", c.name.c_str());
    }
    r.current = c.name;
    r.skip_reason.clear();
    const int before = r.failures;
    c.fn();
    if (r.failures != before) {
      continue;
    }
    if (!r.skip_reason.empty()) {
      ++skipped;
      std::printf("  skip %s (%s)\n", c.name.c_str(), r.skip_reason.c_str());
    } else {
      ++passed;
      std::printf("  ok   %s\n", c.name.c_str());
    }
  }
  if (skipped == 0) {
    std::printf("\n%d/%zu passed\n", passed, run);
  } else {
    std::printf("\n%d passed, %d skipped / %zu\n", passed, skipped, run);
  }
  return r.failures == 0 ? 0 : 1;
}

}  // namespace lse::test

#define LSE_TEST(name)                                            \
  static void name();                                             \
  static ::lse::test::Registrar _lse_reg_##name{#name, name};      \
  static void name()

#define LSE_SKIP(reason)            \
  do {                              \
    ::lse::test::skip(reason);       \
    return;                         \
  } while (0)

// Variadic: an argument like `Shape{1, 64}.is_broadcastable_to(x)` contains a
// comma at preprocessor level, so a single-parameter macro would split it.
#define LSE_EXPECT(...)                                                      \
  do {                                                                       \
    if (!(__VA_ARGS__))                                                       \
      ::lse::test::fail(__FILE__, __LINE__, "expected: " #__VA_ARGS__);       \
  } while (0)

#define LSE_EXPECT_EQ(a, b)                                                 \
  do {                                                                      \
    auto _a = (a);                                                          \
    auto _b = (b);                                                          \
    if (!(_a == _b))                                                        \
      ::lse::test::fail(__FILE__, __LINE__,                                 \
                        std::string(#a " == " #b " (got ") +                \
                            std::to_string(_a) + " vs " +                   \
                            std::to_string(_b) + ")");                      \
  } while (0)

#define LSE_EXPECT_NEAR(a, b, tol)                                          \
  do {                                                                      \
    const double _a = static_cast<double>(a);                               \
    const double _b = static_cast<double>(b);                               \
    if (std::fabs(_a - _b) > (tol))                                         \
      ::lse::test::fail(__FILE__, __LINE__,                                 \
                        std::string(#a " ~= " #b " (got ") +                \
                            std::to_string(_a) + " vs " +                   \
                            std::to_string(_b) + ", tol " +                 \
                            std::to_string(static_cast<double>(tol)) + ")"); \
  } while (0)

#define LSE_EXPECT_OK(expr)                                                 \
  do {                                                                      \
    auto _s = (expr);                                                       \
    if (!_s.ok())                                                           \
      ::lse::test::fail(__FILE__, __LINE__, "expected OK, got: " +          \
                                                _s.to_string());            \
  } while (0)

#define LSE_TEST_MAIN() \
  int main() { return ::lse::test::run_all(); }
