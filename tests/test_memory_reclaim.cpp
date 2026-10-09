// A long-running server must never fail a device allocation while memory it
// could give back is still cached. These cases drive the pieces the HRX
// backend allocates through (BufferRecycler, DeviceHeadroom and the staged
// memory trimmers) against a simulated device with a hard memory cap, in the
// pattern that ran an R9700 out of VRAM: one backend instance serving
// requests of growing context length, each pass freeing buffers whose sizes
// depend on that length and are never asked for again.
#include "harness.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backend/memory_reclaim.hpp"

using namespace lse;
using namespace lse::backend;

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

// A device that hands out memory until its cap and, like an HRX stream
// whose stream-ordered allocation failed, refuses everything after the first
// failure: memory given back after that comes too late.
class SimulatedDevice {
 public:
  SimulatedDevice(std::size_t cap, bool pressure_check)
      : cap_(cap),
        pressure_check_(pressure_check),
        state_(std::make_shared<State>()),
        recycler_(std::make_shared<BufferRecycler>(cap / 8, [state = state_](std::uint64_t h) {
          state->used -= h;  // the handle is the size
        })),
        headroom_(std::make_shared<DeviceHeadroom>(4 * kMiB)) {
    trimmer_ = register_memory_trimmer([r = recycler_] { return r->release_all(); },
                                       [r = recycler_] { return r->held(); },
                                       TrimStage::kRuntime);
  }
  ~SimulatedDevice() {
    unregister_memory_trimmer(trimmer_);
    (void)recycler_->disable();
  }

  // The HRX allocate_impl order: a kept buffer of this size, else room is
  // made and the device asked.
  Result<std::shared_ptr<void>> allocate(std::size_t bytes) {
    if (state_->poisoned) return LSE_ERROR(kDeviceError, "stream failed by an earlier allocation");
    std::uint64_t handle = recycler_->take(bytes);
    if (handle == 0) {
      if (pressure_check_) {
        DeviceHeadroom::Hooks hooks;
        hooks.sample_free = [this]() -> Result<std::size_t> { return cap_ - state_->used; };
        hooks.trim = [] { return trim_device_memory_under_pressure(); };
        hooks.settle = [] { return OkStatus(); };
        hooks.driver_backed = true;
        LSE_RETURN_IF_ERROR(headroom_->make_room(bytes, hooks));
      }
      if (state_->used + bytes > cap_) {
        state_->poisoned = true;
        return Status(StatusCode::kOutOfMemory,
                      "simulated device out of memory allocating " + std::to_string(bytes));
      }
      state_->used += bytes;
      handle = bytes;
    }
    ++live_;
    return std::shared_ptr<void>(reinterpret_cast<void*>(handle),
                                 [this, bytes](void* p) {
                                   --live_;
                                   headroom_->note_release();
                                   const auto h = reinterpret_cast<std::uint64_t>(p);
                                   if (!recycler_->keep(h, bytes)) state_->used -= bytes;
                                 });
  }

  [[nodiscard]] std::size_t used() const { return state_->used; }
  [[nodiscard]] std::size_t cached() const { return recycler_->held(); }
  [[nodiscard]] std::size_t free_bytes() const { return cap_ - state_->used; }
  [[nodiscard]] DeviceHeadroom::Counters counters() const { return headroom_->counters(); }

 private:
  struct State {
    std::size_t used = 0;
    bool poisoned = false;
  };
  std::size_t cap_;
  bool pressure_check_;
  std::shared_ptr<State> state_;
  std::shared_ptr<BufferRecycler> recycler_;
  std::shared_ptr<DeviceHeadroom> headroom_;
  std::uint64_t trimmer_ = 0;
  int live_ = 0;
};

// One request of `tokens` context: K/V grows in 16 MiB arenas as chunks
// land, and every chunk allocates and frees pass buffers sized by the
// context so far, so no two chunks of any request share a size. The engine
// keeps nothing between requests (as a one-shot request ends).
Status run_request(SimulatedDevice& device, std::size_t tokens) {
  constexpr std::size_t kChunk = 1024;
  constexpr std::size_t kArena = 16 * kMiB;
  constexpr std::size_t kKvPerToken = 16 * 1024;  // bytes
  std::vector<std::shared_ptr<void>> kv;
  for (std::size_t done = 0; done < tokens; done += kChunk) {
    const std::size_t context = std::min(tokens, done + kChunk);
    while (kv.size() * kArena < context * kKvPerToken) {
      LSE_ASSIGN_OR(auto arena, device.allocate(kArena));
      kv.push_back(std::move(arena));
    }
    // Attention scratch and activations for this chunk: sizes follow the
    // context, so they are only ever reused within the chunk.
    std::vector<std::shared_ptr<void>> pass;
    for (std::size_t i = 0; i < 4; ++i) {
      LSE_ASSIGN_OR(auto scratch, device.allocate(context * 256 + (i + 1) * 4096));
      pass.push_back(std::move(scratch));
    }
    pass.clear();
  }
  return OkStatus();
}

}  // namespace

LSE_TEST(recycler_keeps_the_newest_sizes_within_its_cap) {
  std::vector<std::uint64_t> released;
  BufferRecycler recycler(100, [&](std::uint64_t h) { released.push_back(h); });
  LSE_EXPECT(recycler.keep(1, 40));
  LSE_EXPECT(recycler.keep(2, 40));
  LSE_EXPECT_EQ(recycler.held(), std::size_t{80});
  // Past the cap the least recently kept goes, not the newcomer.
  LSE_EXPECT(recycler.keep(3, 30));
  LSE_EXPECT_EQ(released.size(), std::size_t{1});
  if (!released.empty()) LSE_EXPECT_EQ(released[0], std::uint64_t{1});
  LSE_EXPECT_EQ(recycler.held(), std::size_t{70});
  LSE_EXPECT_EQ(recycler.take(40), std::uint64_t{2});
  LSE_EXPECT_EQ(recycler.take(40), std::uint64_t{0});
  LSE_EXPECT_EQ(recycler.take(30), std::uint64_t{3});
  LSE_EXPECT_EQ(recycler.held(), std::size_t{0});
  // A buffer bigger than the whole cap is never kept.
  LSE_EXPECT(!recycler.keep(4, 101));
  LSE_EXPECT(recycler.keep(5, 10));
  LSE_EXPECT(recycler.keep(6, 10));
  LSE_EXPECT_EQ(recycler.release_all(), std::size_t{20});
  LSE_EXPECT_EQ(recycler.held(), std::size_t{0});
  LSE_EXPECT(recycler.keep(7, 10));
  LSE_EXPECT_EQ(recycler.disable(), std::size_t{10});
  LSE_EXPECT(!recycler.keep(8, 10));
  LSE_EXPECT(!recycler.enabled());
}

LSE_TEST(recycler_hands_out_the_most_recent_buffer_of_a_size) {
  BufferRecycler recycler(1000, [](std::uint64_t) {});
  LSE_EXPECT(recycler.keep(11, 64));
  LSE_EXPECT(recycler.keep(12, 64));
  LSE_EXPECT(recycler.keep(13, 32));
  LSE_EXPECT_EQ(recycler.take(64), std::uint64_t{12});
  LSE_EXPECT_EQ(recycler.take(64), std::uint64_t{11});
  LSE_EXPECT_EQ(recycler.held(), std::size_t{32});
  // Eviction after takes still finds the right entries.
  BufferRecycler small(64, [](std::uint64_t) {});
  LSE_EXPECT(small.keep(21, 32));
  LSE_EXPECT(small.keep(22, 32));
  LSE_EXPECT_EQ(small.take(32), std::uint64_t{22});
  LSE_EXPECT(small.keep(23, 16));
  LSE_EXPECT(small.keep(24, 32));  // 32 + 16 + 32 > 64: 21 goes
  LSE_EXPECT_EQ(small.held(), std::size_t{48});
  LSE_EXPECT_EQ(small.take(32), std::uint64_t{24});
  LSE_EXPECT_EQ(small.take(32), std::uint64_t{0});
}

LSE_TEST(runtime_trimmers_run_after_cache_trimmers) {
  std::string order;
  // Registered first, still runs last: a cache's buffers must reach the
  // runtime's pool before the pool is emptied.
  const auto runtime = register_memory_trimmer([&] { order += "R"; return std::size_t{1}; },
                                               [] { return std::size_t{3}; },
                                               TrimStage::kRuntime);
  const auto cache = register_memory_trimmer([&] { order += "C"; return std::size_t{2}; },
                                             [] { return std::size_t{4}; });
  LSE_EXPECT_EQ(trim_device_memory(), std::size_t{3});
  LSE_EXPECT(order == "CR");
  LSE_EXPECT_EQ(cached_device_memory(), std::size_t{7});
  unregister_memory_trimmer(runtime);
  unregister_memory_trimmer(cache);
  LSE_EXPECT_EQ(cached_device_memory(), std::size_t{0});
}

LSE_TEST(only_a_pressure_trim_reports_pressure) {
  int under_pressure = 0;
  int housekeeping = 0;
  const auto id = register_memory_trimmer([&] {
    (device_memory_pressure() ? under_pressure : housekeeping) += 1;
    return std::size_t{0};
  });
  (void)trim_device_memory();
  (void)trim_device_memory_under_pressure();
  LSE_EXPECT(!device_memory_pressure());
  unregister_memory_trimmer(id);
  LSE_EXPECT_EQ(housekeeping, 1);
  LSE_EXPECT_EQ(under_pressure, 1);
}

// The failure this fixes, reproduced: without the pressure check the cache
// fills with sizes of shorter requests and the long request fails while the
// cache still holds them. With it, every request up to the longest a fresh
// device can run completes on the same device.
LSE_TEST(growing_requests_never_fail_while_memory_is_cached) {
  constexpr std::size_t kCap = 512 * kMiB;
  // The longest request a fresh device runs: K/V of 16 KiB a token in 16 MiB
  // arenas plus the last chunk's scratch has to fit.
  constexpr std::size_t kLongest = 28 * 1024;
  {
    SimulatedDevice fresh(kCap, /*pressure_check=*/true);
    LSE_EXPECT_OK(run_request(fresh, kLongest));
  }
  {
    // As before the fix: the cache is only emptied after a failure, which
    // has already failed the stream.
    SimulatedDevice before(kCap, /*pressure_check=*/false);
    Status last = OkStatus();
    for (std::size_t tokens = 1024; tokens <= kLongest && last.ok(); tokens *= 2)
      last = run_request(before, tokens);
    if (last.ok()) last = run_request(before, kLongest);
    LSE_EXPECT(!last.ok());
    LSE_EXPECT(before.cached() > 0);
  }
  {
    SimulatedDevice after(kCap, /*pressure_check=*/true);
    for (std::size_t tokens = 1024; tokens <= kLongest; tokens *= 2)
      LSE_EXPECT_OK(run_request(after, tokens));
    LSE_EXPECT_OK(run_request(after, kLongest));
    // The check cost a trim only where the device ran short.
    LSE_EXPECT(after.counters().trims > 0);
    LSE_EXPECT(after.counters().released > 0);
  }
}

LSE_TEST(an_allocation_that_cannot_fit_fails_with_the_sizes) {
  constexpr std::size_t kCap = 64 * kMiB;
  SimulatedDevice device(kCap, /*pressure_check=*/true);
  auto held = device.allocate(32 * kMiB);
  LSE_EXPECT_OK(held.status());
  {
    auto small = device.allocate(4 * kMiB);
    LSE_EXPECT_OK(small.status());
  }  // dropped: kept by the recycler
  LSE_EXPECT(device.cached() > 0);
  // Bigger than everything that is free once the cache is trimmed: refused
  // before the device is asked, so the stream survives it.
  auto big = device.allocate(40 * kMiB);
  LSE_EXPECT(!big.ok());
  LSE_EXPECT(big.status().code() == StatusCode::kOutOfMemory);
  LSE_EXPECT_EQ(device.cached(), std::size_t{0});
  auto after = device.allocate(16 * kMiB);
  LSE_EXPECT_OK(after.status());
  // The report a failed allocation produces names what was asked for, what
  // was free and what was still cached.
  const Status report = out_of_device_memory(40 * kMiB, big.status(), device.free_bytes());
  const std::string text(report.message());
  LSE_EXPECT(text.find("requested " + std::to_string(40 * kMiB) + " bytes") != std::string::npos);
  LSE_EXPECT(text.find(std::to_string(device.free_bytes()) + " bytes free") != std::string::npos);
  LSE_EXPECT(text.find("bytes cached") != std::string::npos);
}

LSE_TEST(the_check_samples_the_device_only_when_its_estimate_runs_out) {
  DeviceHeadroom headroom(1 * kMiB);
  std::size_t samples = 0;
  std::size_t trims = 0;
  DeviceHeadroom::Hooks hooks;
  hooks.sample_free = [&]() -> Result<std::size_t> { ++samples; return 100 * kMiB; };
  hooks.trim = [&] { ++trims; return std::size_t{0}; };
  for (int i = 0; i < 50; ++i) LSE_EXPECT_OK(headroom.make_room(1 * kMiB, hooks));
  // One sample covers the first ~98 MiB of allocations; frees are not
  // credited, so it errs toward sampling again.
  LSE_EXPECT_EQ(samples, std::size_t{1});
  LSE_EXPECT_EQ(trims, std::size_t{0});
  for (int i = 0; i < 60; ++i) LSE_EXPECT_OK(headroom.make_room(1 * kMiB, hooks));
  LSE_EXPECT_EQ(samples, std::size_t{2});
  LSE_EXPECT_EQ(trims, std::size_t{0});
}

LSE_TEST_MAIN()
