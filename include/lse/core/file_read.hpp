// Positioned reads of a file into caller memory: what weight loading uses in
// place of touching a checkpoint mapping. A mapping is read by page faults,
// one 16 KiB page at a time on the faulting thread, which measured 1.4 GB/s
// from a cold cache on an M5 Max against 11.6 GB/s for 16 MiB preads of the
// same file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "lse/core/status.hpp"

namespace lse {

// Reads exactly `bytes` at `offset` of `fd` into `dst`. A short read (the file
// ends early) or any read error is an error that names `what`.
Status read_file_range(int fd, std::uint64_t offset, void* dst,
                       std::size_t bytes, const std::string& what);

// Opens `path` read-only for bulk reads that should not fill the page cache:
// on Apple platforms F_NOCACHE, so a 17 GB checkpoint streamed once to a GPU
// does not evict everything else. Data already cached is still read from the
// cache. Returns the descriptor or an error naming the path.
Result<int> open_for_streaming(const std::string& path);

}  // namespace lse

namespace lse {

// Host memory for one transient load buffer (a tensor read whole, a gather's
// result), mapped from the VM system and unmapped when it goes out of scope.
// Not std::vector: the allocator keeps freed large blocks resident for reuse,
// and a load that freed a few hundred of them held 730 MB of dirty, empty
// MALLOC_LARGE regions afterwards.
class HostScratch {
 public:
  HostScratch() = default;
  ~HostScratch();
  HostScratch(HostScratch&& other) noexcept;
  HostScratch& operator=(HostScratch&& other) noexcept;
  HostScratch(const HostScratch&) = delete;
  HostScratch& operator=(const HostScratch&) = delete;

  // `bytes` of zero-filled memory, or an error naming `what`.
  static Result<HostScratch> allocate(std::size_t bytes, const std::string& what);

  [[nodiscard]] std::byte* data() noexcept { return data_; }
  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t mapped_ = 0;
};

}  // namespace lse
