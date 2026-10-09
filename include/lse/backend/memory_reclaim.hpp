// Device memory a backend keeps after its owner let go, and the check that
// gives it back before an allocation that would not fit.
//
// A long-running server sees allocations whose sizes follow the prompt: a
// 64K prompt's pass buffers are not a 128K prompt's. A cache that keeps freed
// buffers for reuse fills with sizes no later pass asks for, and once it is
// full the device has that much less for the request that needs it. Giving
// it back only after an allocation has failed is too late on a runtime whose
// failed stream-ordered allocation fails the stream with it (HRX), so the
// caches are trimmed before the allocation that would not fit is issued.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::backend {

// Freed device buffers kept by exact size for the next allocation of that
// size, within a byte cap. Least recently kept goes first when the cap is
// reached, so the cache follows the sizes in use now instead of keeping the
// first sizes it saw. The release callback frees a handle for real; it runs
// outside the lock.
class BufferRecycler {
 public:
  using Release = std::function<void(std::uint64_t handle)>;

  BufferRecycler(std::size_t cap, Release release)
      : cap_(cap), release_(std::move(release)) {}
  BufferRecycler(const BufferRecycler&) = delete;
  BufferRecycler& operator=(const BufferRecycler&) = delete;
  ~BufferRecycler() { (void)release_all(); }

  // True when `handle` was kept; the caller releases it otherwise (disabled,
  // or bigger than the whole cap). Keeping it may release older entries.
  bool keep(std::uint64_t handle, std::size_t size);
  // A kept handle of exactly `size` bytes, most recently kept first, or 0.
  [[nodiscard]] std::uint64_t take(std::size_t size);
  // Releases everything kept; returns the bytes.
  std::size_t release_all();
  // Stops keeping and releases everything kept; returns the bytes.
  std::size_t disable();
  [[nodiscard]] bool enabled() const;
  [[nodiscard]] std::size_t held() const;
  [[nodiscard]] std::size_t cap() const noexcept { return cap_; }

 private:
  struct Entry {
    std::uint64_t handle;
    std::size_t size;
  };
  using Order = std::list<Entry>;
  mutable std::mutex mu_;
  Order order_;  // oldest first
  std::unordered_map<std::size_t, std::vector<Order::iterator>> bins_;
  std::size_t bytes_ = 0;
  const std::size_t cap_;
  bool enabled_ = true;
  Release release_;
};

// Free device memory as last sampled, less what was allocated since. While it
// covers the request it is the whole check, one atomic subtraction; only an
// allocation it does not cover samples the device, and only one the device
// cannot cover trims the caches first. Frees do not raise the estimate, so it
// errs toward sampling again, never toward skipping the trim.
class DeviceHeadroom {
 public:
  struct Hooks {
    // The device's free bytes now (IBackend::sample_free_memory).
    std::function<Result<std::size_t>()> sample_free;
    // trim_device_memory: every cache, then the runtime's pools.
    std::function<std::size_t()> trim;
    // Waits for released buffers to reach the driver: device work that
    // still referenced them retires first.
    std::function<Status()> settle;
    // Every buffer comes straight from the driver, so a request bigger than
    // the free bytes cannot succeed and is refused here, before the runtime
    // sees it. False where the runtime suballocates from pools it already
    // holds, which may fit a request the driver could not.
    bool driver_backed = false;
  };

  // `reserve` is kept free beyond each request: memory the device runtime
  // takes on the side, such as a new pool slab for a small buffer.
  explicit DeviceHeadroom(std::size_t reserve) : reserve_(reserve) {}

  // Called before a fresh device allocation of `bytes`. kOutOfMemory only
  // when the caches are trimmed, the device still has less than `bytes`
  // free and the runtime is driver_backed; otherwise OK, and an allocation
  // the trim could not make room for is issued and reports its own failure.
  Status make_room(std::size_t bytes, const Hooks& hooks);
  // A buffer went back to a cache or to the device: a later pressure check
  // has something new to trim.
  void note_release() noexcept { releases_.fetch_add(1, std::memory_order_relaxed); }
  // Forget the estimate: the next allocation samples the device.
  void forget() noexcept { estimate_.store(0, std::memory_order_relaxed); }

  struct Counters {
    std::uint64_t samples = 0;    // device free-memory queries
    std::uint64_t trims = 0;      // pressure trims run
    std::uint64_t released = 0;   // bytes those trims reported
  };
  [[nodiscard]] Counters counters() const noexcept;

 private:
  const std::size_t reserve_;
  std::atomic<std::size_t> estimate_{0};
  std::atomic<std::uint64_t> releases_{0};
  std::mutex pressure_mu_;
  std::uint64_t releases_at_trim_ = ~std::uint64_t{0};
  std::atomic<std::uint64_t> samples_{0};
  std::atomic<std::uint64_t> trims_{0};
  std::atomic<std::uint64_t> released_{0};
};

}  // namespace lse::backend
