// Exercises the Backend<Derived> CRTP contract through the CPU reference
// backend, and the registry that selects backends by name.
#include "lse/backends/cpu/cpu_backend.hpp"

#include <chrono>
#include <numeric>
#include <string>
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

LSE_TEST(enumeration_describes_devices_without_binding_one) {
  // Through the registry, and with no CpuBackend in scope: asking what devices
  // exist must not require one to have been initialised, which is the whole
  // reason this is a free function.
  auto found = enumerate_devices("cpu");
  LSE_EXPECT_OK(found.status());
  if (!found.ok()) return;
  LSE_EXPECT_EQ(found->size(), 1u);
  const DeviceDescriptor& d = found->front();
  LSE_EXPECT(d.backend == "cpu");
  LSE_EXPECT_EQ(d.ordinal, 0);
  LSE_EXPECT(d.id() == "cpu:0");
  LSE_EXPECT(d.product.known());
  LSE_EXPECT(d.arch.known());
  LSE_EXPECT(d.arch.value == "host");
  // The host answers this one itself.
  LSE_EXPECT(d.compute_units.known());
  LSE_EXPECT(d.compute_units.source == FactSource::kQueried);
  LSE_EXPECT(d.compute_units.value >= 1u);
  // A one-device backend still has a peer row, and its own entry is the only
  // reach that needs no query.
  LSE_EXPECT_EQ(d.peers.size(), 1u);
  LSE_EXPECT(d.peers[0] == PeerAccess::kSelf);
}

LSE_TEST(a_property_nothing_answers_stays_unknown) {
  auto found = enumerate_devices("cpu");
  LSE_EXPECT_OK(found.status());
  if (!found.ok()) return;
  const DeviceDescriptor& d = found->front();

  // This backend asks the host for no memory figure. Unknown is then the only
  // honest answer: zero would read as a full device and the machine's RAM
  // would read as a device budget nothing has agreed to.
  LSE_EXPECT(!d.total_memory.known());
  LSE_EXPECT(d.total_memory.source == FactSource::kUnknown);
  LSE_EXPECT_EQ(d.total_memory.value, 0u);
  LSE_EXPECT(!d.free_memory.known());
  LSE_EXPECT_EQ(d.free_memory.value, 0u);
  // A host has no wavefront and no LDS at all, which is a different fact from
  // not knowing — and neither of them is a number.
  LSE_EXPECT(d.wavefront_size.source == FactSource::kInapplicable);
  LSE_EXPECT(!d.wavefront_size.known());
  LSE_EXPECT(d.lds_bytes_per_workgroup.source == FactSource::kInapplicable);
  // A hole the reader can name, not one it has to infer from a zero.
  LSE_EXPECT(!d.declined.empty());

  const std::string text = d.describe();
  LSE_EXPECT(text.find("unknown") != std::string::npos);
  LSE_EXPECT(text.find("n/a") != std::string::npos);
  // The old report printed "0 MiB" for exactly this device.
  LSE_EXPECT(text.find("0 MiB") == std::string::npos);
}

LSE_TEST(a_backend_with_no_enumerator_refuses_rather_than_answering_zero) {
  auto found = enumerate_devices("does-not-exist");
  LSE_EXPECT(!found.ok());
  LSE_EXPECT(found.status().code() == StatusCode::kNotFound);
}

LSE_TEST(a_buffer_says_which_device_holds_it) {
  CpuBackend be;
  // Nothing is bound yet, so nothing can be claimed.
  LSE_EXPECT(!be.device_index().bound());
  LSE_EXPECT_OK(be.init(0));
  const DeviceIndex mine = be.device_index();
  LSE_EXPECT(mine.bound());

  auto buf = be.allocate(64, MemoryClass::kDevice);
  LSE_EXPECT(buf.ok());
  LSE_EXPECT(buf->residency == mine);

  // A view is the same bytes on the same device.
  DeviceBuffer view = *buf;
  view.offset += 16;
  view.size_bytes -= 16;
  LSE_EXPECT(view.residency == mine);

  be.deallocate(*buf);
  // Released: no device holds these bytes any more, and a stale token would
  // resolve to whoever was bound next.
  LSE_EXPECT(!buf->residency.bound());
  be.shutdown();
  LSE_EXPECT(!be.device_index().bound());
}

LSE_TEST(two_instances_are_two_devices_even_on_one_ordinal) {
  CpuBackend a;
  CpuBackend b;
  LSE_EXPECT_OK(a.init(0));
  LSE_EXPECT_OK(b.init(0));
  // Same board, two owners. The buffer a allocated is released by a and the
  // executable a loaded runs on a, so the tokens must differ.
  LSE_EXPECT(a.device_index() != b.device_index());

  auto mine = a.allocate(64, MemoryClass::kDevice);
  LSE_EXPECT(mine.ok());
  LSE_EXPECT(mine->residency == a.device_index());
  LSE_EXPECT(!(mine->residency == b.device_index()));
  a.deallocate(*mine);
  a.shutdown();
  b.shutdown();
}

LSE_TEST(a_launch_addressed_elsewhere_is_refused) {
  CpuBackend a;
  CpuBackend b;
  LSE_EXPECT_OK(a.init(0));
  LSE_EXPECT_OK(b.init(0));

  KernelHandle kernel;
  kernel.executable = 1;
  LaunchDims dims;
  DispatchArgs args;

  // Addressed at b, handed to a. A kernel handle is loaded on one device and
  // its bindings are on one device: landing on the wrong one would run and
  // return numbers nothing flags.
  DispatchTarget elsewhere;
  elsewhere.device = b.device_index();
  const Status refused = a.launch(kernel, dims, args, elsewhere);
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(refused.message().find("reached the backend") != std::string::npos);

  // Unaddressed still means "whatever device this is", which is every call
  // site that holds one device.
  const Status unaddressed = a.launch(kernel, dims, args, DispatchTarget{});
  LSE_EXPECT(unaddressed.code() != StatusCode::kInvalidArgument ||
             unaddressed.message().find("reached the backend") ==
                 std::string::npos);
  a.shutdown();
  b.shutdown();
}

LSE_TEST(a_set_of_one_refuses_another_devices_bytes) {
  auto a = create_backend("cpu");
  auto b = create_backend("cpu");
  LSE_EXPECT(a.ok() && b.ok());
  LSE_EXPECT_OK((*a)->init(0));
  LSE_EXPECT_OK((*b)->init(0));
  SingleDevice set(**a);

  LSE_EXPECT_EQ(set.size(), std::size_t{1});
  LSE_EXPECT_EQ(set.primary(), std::size_t{0});
  LSE_EXPECT_EQ(set.member_of((*a)->device_index()), std::size_t{0});
  LSE_EXPECT_EQ(set.member_of((*b)->device_index()), std::size_t{1});
  LSE_EXPECT(set.residency(0) == (*a)->device_index());

  // Unclaimed bytes are nobody's, so there is nothing to violate.
  LSE_EXPECT_OK(set.may_read(kNoDevice, 0));
  LSE_EXPECT_OK(set.may_read((*a)->device_index(), 0));
  LSE_EXPECT(!set.may_read((*b)->device_index(), 0).ok());
  (*a)->shutdown();
  (*b)->shutdown();
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
