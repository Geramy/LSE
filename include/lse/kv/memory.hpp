#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include "lse/backend/backend.hpp"

namespace lse::kv {

inline constexpr std::size_t kArenaBytes = 256u * 1024u * 1024u;
inline constexpr std::size_t kFragmentBytes = 256u * 1024u;
static_assert(kArenaBytes % kFragmentBytes == 0);

// K/V-only storage. Each arena is completely partitioned into reusable slots.
class MemoryManager : public std::enable_shared_from_this<MemoryManager> {
 public:
  struct Stats {
    std::size_t reserved_bytes = 0;
    std::size_t assigned_bytes = 0;
    std::size_t arenas = 0;
  };
  static std::shared_ptr<MemoryManager> create() {
    return std::shared_ptr<MemoryManager>(new MemoryManager);
  }
  Result<backend::DeviceBuffer> acquire(backend::IBackend& backend,
                                        backend::Stream stream);
  [[nodiscard]] Stats stats() const;

 private:
  MemoryManager() = default;
  struct Arena {
    backend::DeviceBuffer buffer;
    backend::IBackend* backend = nullptr;
    backend::Stream stream;
    std::vector<std::uint32_t> available;
    std::size_t assigned = 0;
  };
  struct Lease;
  void release(std::size_t arena, std::uint32_t slot);
  mutable std::mutex mutex_;
  std::vector<Arena> arenas_;
};

// Logical K/V bytes backed by fixed fragments. Growing changes only the
// address table; existing fragments and their contents remain in place.
class FragmentStorage : public std::enable_shared_from_this<FragmentStorage> {
 public:
  FragmentStorage(std::shared_ptr<MemoryManager> manager,
                  backend::IBackend& backend, backend::Stream stream)
      : manager_(std::move(manager)), backend_(backend), stream_(stream) {}
  Status grow(std::size_t bytes);
  Status reserve(std::size_t bytes);
  Result<backend::DeviceBuffer> binding();
  Status read(void* dst, std::size_t bytes, std::size_t offset = 0) const;
  Status write(const void* src, std::size_t bytes, std::size_t offset = 0);
  [[nodiscard]] std::size_t size_bytes() const { return size_bytes_; }
  [[nodiscard]] std::size_t fragment_count() const { return fragments_.size(); }

 private:
  std::shared_ptr<MemoryManager> manager_;
  backend::IBackend& backend_;
  backend::Stream stream_;
  std::vector<backend::DeviceBuffer> fragments_;
  backend::DeviceBuffer table_;
  std::size_t size_bytes_ = 0;
};

}  // namespace lse::kv
