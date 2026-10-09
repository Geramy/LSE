#include "lse/backend/memory_reclaim.hpp"

#include <chrono>
#include <string>
#include <thread>

namespace lse::backend {

bool BufferRecycler::keep(std::uint64_t handle, std::size_t size) {
  std::vector<std::uint64_t> evicted;
  {
    std::lock_guard lock(mu_);
    if (!enabled_ || size > cap_) return false;
    order_.push_back(Entry{handle, size});
    bins_[size].push_back(std::prev(order_.end()));
    bytes_ += size;
    // Oldest first: a size kept long ago and not asked for since is the one
    // least likely to be asked for next.
    while (bytes_ > cap_) {
      const Entry oldest = order_.front();
      std::vector<Order::iterator>& bin = bins_[oldest.size];
      // The oldest of its size is the first in its bin, since both are kept
      // in the order they arrived.
      bin.erase(bin.begin());
      if (bin.empty()) bins_.erase(oldest.size);
      order_.pop_front();
      bytes_ -= oldest.size;
      evicted.push_back(oldest.handle);
    }
  }
  for (const std::uint64_t h : evicted) release_(h);
  return true;
}

std::uint64_t BufferRecycler::take(std::size_t size) {
  std::lock_guard lock(mu_);
  auto it = bins_.find(size);
  if (it == bins_.end()) return 0;
  const Order::iterator entry = it->second.back();
  it->second.pop_back();
  if (it->second.empty()) bins_.erase(it);
  const std::uint64_t handle = entry->handle;
  order_.erase(entry);
  bytes_ -= size;
  return handle;
}

std::size_t BufferRecycler::release_all() {
  Order taken;
  std::size_t released = 0;
  {
    std::lock_guard lock(mu_);
    taken.swap(order_);
    bins_.clear();
    released = bytes_;
    bytes_ = 0;
  }
  for (const Entry& e : taken) release_(e.handle);
  return released;
}

std::size_t BufferRecycler::disable() {
  {
    std::lock_guard lock(mu_);
    enabled_ = false;
  }
  return release_all();
}

bool BufferRecycler::enabled() const {
  std::lock_guard lock(mu_);
  return enabled_;
}

std::size_t BufferRecycler::held() const {
  std::lock_guard lock(mu_);
  return bytes_;
}

Status DeviceHeadroom::make_room(std::size_t bytes, const Hooks& hooks) {
  const std::size_t need = bytes + reserve_;
  std::size_t estimate = estimate_.load(std::memory_order_relaxed);
  // The common case: the last sample still covers this allocation.
  while (estimate >= need) {
    if (estimate_.compare_exchange_weak(estimate, estimate - bytes,
                                        std::memory_order_relaxed)) {
      return OkStatus();
    }
  }
  if (!hooks.sample_free) return OkStatus();

  const std::lock_guard lock(pressure_mu_);
  const auto sample = [&]() -> Result<std::size_t> {
    samples_.fetch_add(1, std::memory_order_relaxed);
    return hooks.sample_free();
  };
  auto sampled = sample();
  // A device that cannot say what is free leaves the allocation to report
  // its own failure; nothing here can know better.
  if (!sampled.ok()) return OkStatus();
  std::size_t free_bytes = *sampled;
  const auto settle_estimate = [&] {
    estimate_.store(free_bytes > bytes ? free_bytes - bytes : 0, std::memory_order_relaxed);
  };
  if (free_bytes >= need) {
    settle_estimate();
    return OkStatus();
  }

  // Short of room. Trim, unless nothing was released since the last trim:
  // then every cache is as empty as that trim left it, and trimming again
  // would only stall the allocation behind a drain.
  const std::uint64_t releases = releases_.load(std::memory_order_relaxed);
  if (hooks.trim && releases != releases_at_trim_) {
    releases_at_trim_ = releases;
    trims_.fetch_add(1, std::memory_order_relaxed);
    std::size_t released = hooks.trim();
    if (hooks.settle) {
      // Buffers the trim let go that device work still referenced reach the
      // runtime's pools only once that work retires; a second trim then
      // hands what they freed in those pools to the driver.
      LSE_RETURN_IF_ERROR(hooks.settle());
      released += hooks.trim();
    }
    released_.fetch_add(released, std::memory_order_relaxed);
    if (auto again = sample(); again.ok()) free_bytes = *again;
    // A runtime may hand freed pool memory back to the driver on its own
    // thread. While the free figure is still rising and short, give it a
    // moment; the first sample that does not rise ends the wait.
    for (int i = 0; i < 8 && free_bytes < need; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      auto next = sample();
      if (!next.ok() || *next <= free_bytes) break;
      free_bytes = *next;
    }
  }
  settle_estimate();
  if (free_bytes < bytes && hooks.driver_backed) {
    return Status(StatusCode::kOutOfMemory,
                  "the device has " + std::to_string(free_bytes) + " bytes free after "
                  "trimming every cache, short of the " + std::to_string(bytes) +
                  " requested");
  }
  return OkStatus();
}

DeviceHeadroom::Counters DeviceHeadroom::counters() const noexcept {
  return Counters{samples_.load(std::memory_order_relaxed),
                  trims_.load(std::memory_order_relaxed),
                  released_.load(std::memory_order_relaxed)};
}

}  // namespace lse::backend
