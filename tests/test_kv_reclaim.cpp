// Memory reclaim from inside a K/V allocation. When a session's K/V grows
// and the device cannot cover a new arena, the allocation runs the memory
// trimmers on the growing thread, and under pressure the server's trimmer
// evicts idle sessions. Their fragments go back to the K/V memory manager the
// growing thread is allocating through: with that manager's lock held across
// the allocation, the thread waited on itself and the server hung (several
// long sessions open, VRAM full). These cases drive kv::MemoryManager against
// a capped device and a trimmer that evicts the way the server's does, each
// under a deadline: a hang fails the run instead of stalling it.
#include "harness.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backend/memory_reclaim.hpp"
#include "lse/kv/memory.hpp"

using namespace lse;

namespace {

// What the capped device has handed out.
struct DeviceState {
  std::size_t cap = 0;
  std::size_t used = 0;
  std::size_t allocations = 0;
  std::uint64_t next = 0;
  // The HRX order: room is made before the allocation is issued
  // (DeviceHeadroom, trimming every cache), rather than the trimmers running
  // after the device refused it.
  std::unique_ptr<backend::DeviceHeadroom> headroom;
};

std::shared_ptr<DeviceState> capped(std::size_t cap, bool check_first = false) {
  auto state = std::make_shared<DeviceState>();
  state->cap = cap;
  if (check_first) state->headroom = std::make_unique<backend::DeviceHeadroom>(0);
  return state;
}

// A device with a hard cap. Its buffers have no memory behind them: the K/V
// manager hands out windows of an arena and never touches the bytes.
class CappedBackend final : public backend::Backend<CappedBackend> {
 public:
  static constexpr std::string_view kName = "capped-test";
  explicit CappedBackend(std::shared_ptr<DeviceState> state) : state_(std::move(state)) {}
  Status init_impl(int ordinal) {
    if (ordinal != 0) return LSE_ERROR(kInvalidArgument, "invalid test ordinal");
    info_.arch = "host";
    info_.compute_units = 1;
    info_.max_threads_per_workgroup = 1;
    return OkStatus();
  }
  void shutdown_impl() noexcept {}
  const backend::DeviceInfo& device_info_impl() const noexcept { return info_; }
  Result<backend::DeviceBuffer> allocate_impl(std::size_t bytes, backend::MemoryClass,
                                              backend::Stream) {
    if (state_->headroom) {
      backend::DeviceHeadroom::Hooks hooks;
      hooks.sample_free = [state = state_.get()]() -> Result<std::size_t> {
        return state->cap - state->used;
      };
      hooks.trim = [] { return backend::trim_device_memory_under_pressure(); };
      hooks.driver_backed = true;
      LSE_RETURN_IF_ERROR(state_->headroom->make_room(bytes, hooks));
    }
    if (state_->used + bytes > state_->cap) {
      return Status(StatusCode::kOutOfMemory,
                    "capped test device: " + std::to_string(bytes) + " bytes do not fit");
    }
    state_->used += bytes;
    ++state_->allocations;
    backend::DeviceBuffer buffer;
    buffer.handle = ++state_->next;
    buffer.storage = std::shared_ptr<void>(reinterpret_cast<void*>(buffer.handle),
                                           [state = state_, bytes](void*) {
                                             state->used -= bytes;
                                             if (state->headroom) state->headroom->note_release();
                                           });
    buffer.size_bytes = bytes;
    return buffer;
  }
  void deallocate_impl(backend::DeviceBuffer& b) noexcept { b = {}; }
  Status copy_h2d_impl(const void*, backend::DeviceBuffer&, std::size_t, std::size_t) {
    return LSE_ERROR(kUnimplemented, "host test only");
  }
  Status copy_d2h_impl(const backend::DeviceBuffer&, void*, std::size_t, std::size_t) {
    return LSE_ERROR(kUnimplemented, "host test only");
  }
  Status copy_peer_impl(const backend::DeviceBuffer&, backend::DeviceBuffer&, std::size_t,
                        std::size_t, std::size_t) {
    return LSE_ERROR(kUnimplemented, "host test only");
  }
  Result<backend::KernelHandle> load_executable_impl(std::string_view, std::span<const std::byte>) {
    return LSE_ERROR(kUnimplemented, "host test only");
  }
  Status launch_impl(const backend::KernelHandle&, const backend::LaunchDims&,
                     const backend::DispatchArgs&) {
    return LSE_ERROR(kUnimplemented, "host test only");
  }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const graph::KernelToolchain> toolchains_impl() const noexcept { return {}; }


 private:
  std::shared_ptr<DeviceState> state_;
  backend::DeviceInfo info_;
};

constexpr std::size_t kFragmentsPerArena = kv::kArenaBytes / kv::kFragmentBytes;

// Sessions holding K/V fragments, and the server's pressure trimmer over
// them: under pressure every idle session goes, least recently used first;
// the active one never does. The fragments are released outside the
// sessions' lock, as the server releases an evicted session.
struct Sessions {
  std::mutex mu;
  std::vector<std::vector<backend::DeviceBuffer>> held;
  std::vector<std::uint64_t> last_used;
  std::size_t active = 0;
  std::uint64_t clock = 0;
  std::size_t evictions = 0;
  std::uint64_t trimmer = 0;

  explicit Sessions(std::size_t n) : held(n), last_used(n, 0) {
    trimmer = backend::register_memory_trimmer([this] {
      if (!backend::device_memory_pressure()) return std::size_t{0};
      std::vector<std::vector<backend::DeviceBuffer>> evicted;
      {
        backend::ReclaimLock lock(mu, "the test sessions");
        for (std::size_t i = 0; i < held.size(); ++i) {
          if (i == active || held[i].empty()) continue;
          evicted.push_back(std::move(held[i]));
          held[i].clear();
          ++evictions;
        }
      }
      std::size_t bytes = 0;
      for (const auto& s : evicted) bytes += s.size() * kv::kFragmentBytes;
      evicted.clear();  // the fragments' leases go back to the manager here
      return bytes;
    });
  }
  ~Sessions() { backend::unregister_memory_trimmer(trimmer); }

  // Session `s` becomes the active one and takes `fragments` more.
  Status grow(kv::MemoryManager& manager, backend::IBackend& be, std::size_t s,
              std::size_t fragments) {
    {
      backend::ReclaimLock lock(mu, "the test sessions");
      active = s;
      last_used[s] = ++clock;
    }
    for (std::size_t i = 0; i < fragments; ++i) {
      LSE_ASSIGN_OR(auto fragment, manager.acquire(be, backend::kDefaultStream));
      backend::ReclaimLock lock(mu, "the test sessions");
      held[s].push_back(std::move(fragment));
    }
    return OkStatus();
  }
};

// Runs `body` on its own thread; a body that has not finished by the deadline
// is a hang, and the run stops there with a failure.
void within_deadline(const char* what, const std::function<void()>& body) {
  std::packaged_task<void()> task(body);
  auto done = task.get_future();
  std::thread worker(std::move(task));
  if (done.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
    std::fprintf(stderr, "FAIL %s: still running after 20 s, a reclaim deadlock\n", what);
    std::fflush(stderr);
    std::_Exit(1);
  }
  worker.join();
  done.get();
}

}  // namespace

namespace {
void evicts_an_idle_session(bool check_first) {
  within_deadline("kv arena eviction", [check_first] {
    auto dev = capped(2 * kv::kArenaBytes, check_first);
    backend::BackendAdapter<CappedBackend> device(dev);
    LSE_EXPECT_OK(device.init(0));
    auto manager = kv::MemoryManager::create();
    Sessions sessions(3);
    // Two sessions fill the device, an arena each, and go idle.
    LSE_EXPECT_OK(sessions.grow(*manager, device, 0, kFragmentsPerArena));
    LSE_EXPECT_OK(sessions.grow(*manager, device, 1, kFragmentsPerArena));
    LSE_EXPECT_EQ(dev->used, 2 * kv::kArenaBytes);
    // The third needs a new arena: the device cannot cover it, the trimmers
    // run on this thread inside the manager's acquire, and the idle sessions'
    // fragments come back to that manager before the arena is allocated.
    LSE_EXPECT_OK(sessions.grow(*manager, device, 2, 1));
    LSE_EXPECT_EQ(sessions.evictions, std::size_t{2});
    LSE_EXPECT_EQ(sessions.held[2].size(), std::size_t{1});
    LSE_EXPECT_EQ(dev->used, kv::kArenaBytes);
    LSE_EXPECT_EQ(manager->stats().arenas, std::size_t{1});
    sessions.held.clear();
    LSE_EXPECT_EQ(dev->used, std::size_t{0});
  });
}

void serves_from_a_held_arena(bool check_first) {
  within_deadline("eviction into a held arena", [check_first] {
    // One arena fits. An idle session and the growing one share it, full;
    // the growth's new arena is refused, and the eviction it ran returned
    // the idle session's fragments to the arena the growing one still holds.
    auto dev = capped(kv::kArenaBytes, check_first);
    backend::BackendAdapter<CappedBackend> device(dev);
    LSE_EXPECT_OK(device.init(0));
    auto manager = kv::MemoryManager::create();
    Sessions sessions(2);
    LSE_EXPECT_OK(sessions.grow(*manager, device, 0, kFragmentsPerArena - 24));
    LSE_EXPECT_OK(sessions.grow(*manager, device, 1, 24));
    LSE_EXPECT_OK(sessions.grow(*manager, device, 1, 100));
    LSE_EXPECT_EQ(sessions.evictions, std::size_t{1});
    LSE_EXPECT_EQ(sessions.held[1].size(), std::size_t{124});
    LSE_EXPECT_EQ(manager->stats().arenas, std::size_t{1});
    LSE_EXPECT_EQ(dev->used, kv::kArenaBytes);
  });
}
}  // namespace

LSE_TEST(a_kv_arena_allocation_that_evicts_an_idle_session_completes) {
  evicts_an_idle_session(false);
}

LSE_TEST(the_same_with_room_made_before_the_allocation) {
  evicts_an_idle_session(true);
}

LSE_TEST(many_sessions_growing_past_the_device_evict_and_never_hang) {
  within_deadline("many sessions", [] {
    // Room for four arenas; twelve sessions take turns growing by most of an
    // arena, so nearly every turn has to evict to grow.
    auto dev = capped(4 * kv::kArenaBytes);
    backend::BackendAdapter<CappedBackend> device(dev);
    LSE_EXPECT_OK(device.init(0));
    auto manager = kv::MemoryManager::create();
    Sessions sessions(12);
    for (int round = 0; round < 3; ++round) {
      for (std::size_t s = 0; s < 12; ++s) {
        LSE_EXPECT_OK(sessions.grow(*manager, device, s, kFragmentsPerArena * 3 / 4));
        LSE_EXPECT(dev->used <= 4 * kv::kArenaBytes);
      }
    }
    LSE_EXPECT(sessions.evictions >= 20u);
    // Every fragment held is backed, and nothing beyond the arenas they need.
    std::size_t fragments = 0;
    for (const auto& s : sessions.held) fragments += s.size();
    const std::size_t arenas = manager->stats().arenas;
    LSE_EXPECT(arenas * kFragmentsPerArena >= fragments);
    LSE_EXPECT_EQ(dev->used, arenas * kv::kArenaBytes);
  });
}

LSE_TEST(fragments_an_eviction_returns_to_a_held_arena_serve_the_growth) {
  serves_from_a_held_arena(false);
  serves_from_a_held_arena(true);
}

LSE_TEST(an_arena_emptied_while_another_was_allocated_goes_back_unlocked) {
  within_deadline("arena reuse", [] {
    auto dev = capped(2 * kv::kArenaBytes);
    backend::BackendAdapter<CappedBackend> device(dev);
    LSE_EXPECT_OK(device.init(0));
    auto manager = kv::MemoryManager::create();
    std::vector<backend::DeviceBuffer> held;
    for (std::size_t i = 0; i < kFragmentsPerArena; ++i) {
      auto f = manager->acquire(device, backend::kDefaultStream);
      LSE_EXPECT_OK(f.status());
      if (f.ok()) held.push_back(f.release());
    }
    LSE_EXPECT_EQ(manager->stats().arenas, std::size_t{1});
    // The last fragment of an arena going back frees the arena, outside the
    // manager's lock, and the next acquire takes a fresh one.
    held.clear();
    LSE_EXPECT_EQ(manager->stats().arenas, std::size_t{0});
    LSE_EXPECT_EQ(dev->used, std::size_t{0});
    auto again = manager->acquire(device, backend::kDefaultStream);
    LSE_EXPECT_OK(again.status());
    LSE_EXPECT_EQ(manager->stats().arenas, std::size_t{1});
  });
}

LSE_TEST(a_trimmer_that_allocates_is_refused_not_deadlocked) {
  within_deadline("allocating trimmer", [] {
    backend::DeviceHeadroom headroom(0);
    Status inner = OkStatus();
    backend::DeviceHeadroom::Hooks hooks;
    hooks.sample_free = []() -> Result<std::size_t> { return std::size_t{1} << 20; };
    hooks.trim = [&] {
      // A trimmer must only release; one that allocates through the same
      // check would wait on the lock its own thread holds.
      inner = headroom.make_room(std::size_t{4} << 20, hooks);
      return std::size_t{0};
    };
    LSE_EXPECT_OK(headroom.make_room(std::size_t{4} << 20, hooks));
    LSE_EXPECT(inner.code() == StatusCode::kInternal);
    LSE_EXPECT(std::string(inner.message()).find("trimmer allocated") != std::string::npos);
  });
}

LSE_TEST_MAIN()
