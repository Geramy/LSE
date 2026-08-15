// Exercises the Backend<Derived> CRTP contract through the CPU reference
// backend, and the registry that selects backends by name.
#include "../src/backends/cpu/cpu_backend.hpp"

#include <numeric>
#include <vector>

#include <cstdlib>

#include "harness.hpp"

using namespace lse;
using namespace lse::backend;

LSE_TEST(cpu_backend_initializes) {
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));
  const DeviceInfo& info = be.device_info();
  LSE_EXPECT(info.arch == "host");
  LSE_EXPECT(info.unified_memory);
  LSE_EXPECT(info.compute_units >= 1);
  be.shutdown();
}

LSE_TEST(cpu_backend_rejects_bad_ordinal) {
  CpuBackend be;
  auto s = be.init(3);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kInvalidArgument);
}

LSE_TEST(allocate_upload_download_round_trip) {
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));

  std::vector<float> src(256);
  std::iota(src.begin(), src.end(), 1.0f);

  // upload() and download() are the CRTP base's shared algorithms, written
  // once in terms of allocate/copy — this checks that layer, not just the impl.
  auto buf = be.upload(src.data(), src.size() * sizeof(float));
  LSE_EXPECT(buf.ok());

  auto out = be.download<float>(*buf, src.size());
  LSE_EXPECT(out.ok());
  for (std::size_t i = 0; i < src.size(); ++i) {
    LSE_EXPECT_NEAR((*out)[i], src[i], 0.0);
  }

  DeviceBuffer owned = buf.release();
  be.deallocate(owned);
  LSE_EXPECT(!owned.valid());
  be.shutdown();
}

LSE_TEST(copies_are_bounds_checked) {
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));
  auto buf = be.allocate(64);
  LSE_EXPECT(buf.ok());
  DeviceBuffer owned = buf.release();

  std::vector<std::byte> src(128);
  auto s = be.copy_h2d(src.data(), owned, 128, 0);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kOutOfRange);

  // Offset that pushes a legal-size copy past the end is also caught.
  auto s2 = be.copy_h2d(src.data(), owned, 32, 48);
  LSE_EXPECT(!s2.ok());

  be.deallocate(owned);
  be.shutdown();
}

LSE_TEST(zero_size_allocation_is_an_error) {
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));
  auto buf = be.allocate(0);
  LSE_EXPECT(!buf.ok());
  be.shutdown();
}

LSE_TEST(cpu_backend_declines_code_objects_explicitly) {
  // The CPU backend runs graphs through the reference interpreter; routing JIT
  // output at it should fail loudly rather than silently no-op.
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));
  std::vector<std::byte> fake_code(16);
  auto k = be.load_executable("k", fake_code);
  LSE_EXPECT(!k.ok());
  LSE_EXPECT(k.status().code() == StatusCode::kUnimplemented);
  be.shutdown();
}

LSE_TEST(registry_finds_the_cpu_backend) {
  const auto names = available_backends();
  bool found = false;
  for (const auto& n : names) found = found || (n == "cpu");
  LSE_EXPECT(found);

  auto be = create_backend("cpu");
  LSE_EXPECT(be.ok());
  LSE_EXPECT((*be)->name() == "cpu");
  LSE_EXPECT_OK((*be)->init(0));
  (*be)->shutdown();
}

LSE_TEST(registry_reports_unknown_backends) {
  auto be = create_backend("does-not-exist");
  LSE_EXPECT(!be.ok());
  LSE_EXPECT(be.status().code() == StatusCode::kNotFound);
  // The error should list what *is* available, to be actionable.
  LSE_EXPECT(be.status().message().find("cpu") != std::string::npos);
}

LSE_TEST(default_backend_prefers_the_gpu_and_falls_back_to_cpu) {
  // The preference order itself, not whatever backend this machine happens to
  // have: LSE_BACKEND pins the choice, and honouring it here would test the
  // override instead of the default.
  if (std::getenv("LSE_BACKEND") != nullptr) return;
  auto be = create_default_backend();
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;

  // hrx wins when it can actually be constructed — the build may not have it,
  // and it declines at runtime when no device answers. Either way the CPU
  // backend is the fallback, never the preference.
  auto gpu = create_backend("hrx");
  LSE_EXPECT((*be)->name() == (gpu.ok() ? "hrx" : "cpu"));
}

LSE_TEST_MAIN()
