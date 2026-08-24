// Minimal test harness — no external dependency, so the suite builds anywhere.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

inline int run_all() {
  Registry& r = Registry::get();
  int passed = 0;
  for (const auto& c : r.cases) {
    if (std::getenv("LSE_TEST_TRACE") != nullptr) {
      std::fprintf(stderr, "[test] start %s\n", c.name.c_str());
    }
    r.current = c.name;
    const int before = r.failures;
    c.fn();
    if (r.failures == before) {
      ++passed;
      std::printf("  ok   %s\n", c.name.c_str());
    }
  }
  std::printf("\n%d/%zu passed\n", passed, r.cases.size());
  return r.failures == 0 ? 0 : 1;
}

}  // namespace lse::test

#define LSE_TEST(name)                                            \
  static void name();                                             \
  static ::lse::test::Registrar _lse_reg_##name{#name, name};      \
  static void name()

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
