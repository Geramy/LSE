// Exercises the Backend<Derived> CRTP contract through the CPU reference
// backend, and the registry that selects backends by name.
#include "lse/backends/cpu/cpu_backend.hpp"

#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include <cstdint>
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

LSE_TEST(cpu_backend_declines_the_device_clock) {
  // The CPU backend's device *is* the host, so the one clock it could reach is
  // a host clock — and handing that back under a device-clock name is the
  // substitution the whole seam exists to prevent. It implements neither
  // *_impl and the seam refuses for it, naming the backend that could not
  // answer. kUnimplemented, never kDeviceError: nothing is broken.
  CpuBackend be;
  LSE_EXPECT_OK(be.init(0));

  auto clock = be.device_clock();
  LSE_EXPECT(!clock.ok());
  LSE_EXPECT(clock.status().code() == StatusCode::kUnimplemented);
  LSE_EXPECT(clock.status().message().find("cpu") != std::string::npos);

  auto stamp = be.sample_device_time();
  LSE_EXPECT(!stamp.ok());
  LSE_EXPECT(stamp.status().code() == StatusCode::kUnimplemented);
  LSE_EXPECT(stamp.status().message().find("cpu") != std::string::npos);

  // And through IBackend, which is what the engine actually holds.
  auto boxed = create_backend("cpu");
  LSE_EXPECT(boxed.ok());
  LSE_EXPECT(!(*boxed)->device_clock().ok());
  LSE_EXPECT((*boxed)->sample_device_time().status().code() ==
             StatusCode::kUnimplemented);
  be.shutdown();
}

LSE_TEST(timestamps_convert_with_the_rate_they_carry) {
  // 99810000 Hz is gfx1151's measured agent rate, read from
  // HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY. It is used here because the two
  // near-misses are the whole point: a tick count that is exactly 5 ms on the
  // real rate is 4.9905 ms if you round the rate to 100 MHz, and 0.499 ms if
  // you reach for the host-scope 1 GHz that the same runtime reports.
  DeviceClock agent;
  agent.domain = ClockDomain::kDeviceAgent;
  agent.ticks_per_second = 99810000;
  agent.valid_bits = 64;
  LSE_EXPECT(agent.known());

  constexpr double kIntervalNs = 5'000'000.0;  // 5 ms
  const std::uint64_t ticks =
      static_cast<std::uint64_t>(kIntervalNs * 99810000.0 / 1e9);
  DeviceTimestamp start{1'000'000, agent};
  DeviceTimestamp end{1'000'000 + ticks, agent};

  auto elapsed = nanoseconds_between(start, end);
  LSE_EXPECT(elapsed.ok());
  // Within one tick of the known interval — a tick is ~10 ns here.
  LSE_EXPECT_NEAR(*elapsed, kIntervalNs, 11.0);

  auto raw = ticks_between(start, end);
  LSE_EXPECT(raw.ok());
  LSE_EXPECT_EQ(*raw, ticks);

  // Same ticks read as if the clock ran at the host-scope rate: off by the
  // ratio, and plausible enough to go unnoticed without the domain and the
  // rate travelling with the tick.
  DeviceClock system_rate = agent;
  system_rate.ticks_per_second = 1'000'000'000;
  auto wrong = nanoseconds_between(DeviceTimestamp{start.tick, system_rate},
                                   DeviceTimestamp{end.tick, system_rate});
  LSE_EXPECT(wrong.ok());
  LSE_EXPECT_NEAR(kIntervalNs / *wrong, 10.019, 0.001);
}

LSE_TEST(timestamps_refuse_to_subtract_across_clocks) {
  DeviceClock agent;
  agent.domain = ClockDomain::kDeviceAgent;
  agent.ticks_per_second = 99810000;
  agent.valid_bits = 64;

  const DeviceTimestamp start{100, agent};

  // A different domain on the same device: a host tick and a device tick are
  // not two readings of one clock however close the numbers look.
  DeviceClock host = agent;
  host.domain = ClockDomain::kHostSteady;
  LSE_EXPECT(!ticks_between(start, DeviceTimestamp{200, host}).ok());
  LSE_EXPECT(ticks_between(start, DeviceTimestamp{200, host}).status().code() ==
             StatusCode::kInvalidArgument);

  // Same domain, different device. Two agents count independently from
  // unrelated origins, so this is the subtraction a multi-device pool would
  // otherwise make silently.
  DeviceClock other_device = agent;
  other_device.ordinal = 1;
  LSE_EXPECT(!ticks_between(start, DeviceTimestamp{200, other_device}).ok());

  // An unknown rate makes the duration unknown, not approximate.
  DeviceClock rateless = agent;
  rateless.ticks_per_second = 0;
  LSE_EXPECT(!rateless.known());
  LSE_EXPECT(!ticks_between(DeviceTimestamp{100, rateless},
                            DeviceTimestamp{200, rateless})
                  .ok());
  LSE_EXPECT(!nanoseconds_between(start, DeviceTimestamp{200, DeviceClock{}}).ok());
}

LSE_TEST(timestamp_subtraction_survives_a_counter_wrap) {
  // A 32-bit counter that wrapped between the two readings. Ignoring
  // valid_bits here yields 2^64 minus a little, which converts to a duration of
  // several thousand years and would be the largest number in any profile.
  DeviceClock narrow;
  narrow.domain = ClockDomain::kDeviceAgent;
  narrow.ticks_per_second = 99810000;
  narrow.valid_bits = 32;

  const DeviceTimestamp start{0xFFFF'FF00ull, narrow};
  const DeviceTimestamp end{0x0000'00FFull, narrow};
  // 0xFFFFFF00 -> 0xFFFFFFFF is 255, the wrap to 0 is one more, then 255 to
  // 0xFF: 511 ticks.
  auto ticks = ticks_between(start, end);
  LSE_EXPECT(ticks.ok());
  LSE_EXPECT_EQ(*ticks, std::uint64_t{511});

  // The same two readings on a full-width clock are not a wrap at all: 2^64
  // ticks is thousands of years at this rate, so a tick that went down is a
  // clock that went backwards, and the honest answer is that the duration is
  // unknown rather than ~5900 years.
  DeviceClock wide = narrow;
  wide.valid_bits = 64;
  auto backwards =
      ticks_between(DeviceTimestamp{start.tick, wide}, DeviceTimestamp{end.tick, wide});
  LSE_EXPECT(!backwards.ok());
  LSE_EXPECT(backwards.status().code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(backwards.status().message().find("backwards") != std::string::npos);

  // Forwards on the same wide clock still subtracts normally.
  auto forwards =
      ticks_between(DeviceTimestamp{end.tick, wide}, DeviceTimestamp{start.tick, wide});
  LSE_EXPECT(forwards.ok());
  LSE_EXPECT_EQ(*forwards, std::uint64_t{0xFFFF'FF00ull - 0xFFull});
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
