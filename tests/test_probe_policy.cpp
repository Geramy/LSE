#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/probe/pool.hpp"
#include <cstdlib>
#include <optional>
#include <string>

// Host storage with an HRX identity exercises profile selection without
// loading a GPU runtime or issuing any device operation.
struct ProfileCpu : lse::backend::CpuBackend {
  static constexpr std::string_view kName = "hrx";
};
LSE_TEST(hrx_submission_policy_changes_the_calibration_fingerprint) {
  const char* original = std::getenv("LSE_FLUSH_INTERVAL");
  const std::optional<std::string> saved =
      original ? std::optional<std::string>(original) : std::nullopt;
  struct Restore {
    const std::optional<std::string>& saved;
    ~Restore() {
      if (saved) setenv("LSE_FLUSH_INTERVAL", saved->c_str(), 1);
      else unsetenv("LSE_FLUSH_INTERVAL");
    }
  } restore{saved};
  lse::backend::BackendAdapter<ProfileCpu> backend;
  LSE_EXPECT_OK(backend.init(0));
  lse::probe::PoolMember member;
  member.id = {"hrx", 0};
  member.rank = 0;
  member.host = "profile-test";
  member.backend = &backend;
  unsetenv("LSE_FLUSH_INTERVAL");
  auto normal = lse::probe::pool_fingerprint(std::span(&member, 1), nullptr);
  setenv("LSE_FLUSH_INTERVAL", "16", 1);
  auto explicit_default = lse::probe::pool_fingerprint(std::span(&member, 1), nullptr);
  setenv("LSE_FLUSH_INTERVAL", "64", 1);
  auto larger_batch = lse::probe::pool_fingerprint(std::span(&member, 1), nullptr);
  LSE_EXPECT(normal.ok() && explicit_default.ok() && larger_batch.ok());
  if (normal.ok() && explicit_default.ok() && larger_batch.ok()) {
    LSE_EXPECT(*normal == *explicit_default);
    LSE_EXPECT(*normal != *larger_batch);
  }
  backend.shutdown();
}
#if defined(__APPLE__)
LSE_TEST(mac_hsa_wait_policy_changes_the_calibration_fingerprint) {
  const char* original = std::getenv("MAC_HSA_BLOCKED_POLL_US");
  const std::optional<std::string> saved =
      original ? std::optional<std::string>(original) : std::nullopt;
  struct Restore {
    const std::optional<std::string>& saved;
    ~Restore() {
      if (saved) setenv("MAC_HSA_BLOCKED_POLL_US", saved->c_str(), 1);
      else unsetenv("MAC_HSA_BLOCKED_POLL_US");
    }
  } restore{saved};
  lse::backend::BackendAdapter<ProfileCpu> backend;
  LSE_EXPECT_OK(backend.init(0));
  lse::probe::PoolMember member;
  member.id = {"hrx", 0};
  member.rank = 0;
  member.host = "profile-test";
  member.backend = &backend;
  auto fingerprint = [&] {
    return lse::probe::pool_fingerprint(std::span(&member, 1), nullptr);
  };
  unsetenv("MAC_HSA_BLOCKED_POLL_US");
  const auto normal = fingerprint();
  LSE_EXPECT(normal.ok());
  for (const char* setting : {"1000", "", "9", "1001", "-1", "64junk", "4294967296"}) {
    setenv("MAC_HSA_BLOCKED_POLL_US", setting, 1);
    const auto value = fingerprint();
    LSE_EXPECT(value.ok());
    if (normal.ok() && value.ok()) LSE_EXPECT(*normal == *value);
  }
  setenv("MAC_HSA_BLOCKED_POLL_US", "64", 1);
  const auto faster = fingerprint();
  LSE_EXPECT(faster.ok());
  if (normal.ok() && faster.ok()) LSE_EXPECT(*normal != *faster);
  setenv("MAC_HSA_BLOCKED_POLL_US", "0064", 1);
  const auto canonical = fingerprint();
  LSE_EXPECT(canonical.ok());
  if (faster.ok() && canonical.ok()) LSE_EXPECT(*faster == *canonical);
  setenv("MAC_HSA_BLOCKED_POLL_US", "10", 1);
  const auto lower_bound = fingerprint();
  LSE_EXPECT(lower_bound.ok());
  if (normal.ok() && lower_bound.ok()) LSE_EXPECT(*normal != *lower_bound);
  backend.shutdown();
}
#endif
LSE_TEST_MAIN()
