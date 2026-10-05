#include "lse/core/file_read.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace lse {

Status read_file_range(int fd, std::uint64_t offset, void* dst,
                       std::size_t bytes, const std::string& what) {
  if (fd < 0) return LSE_ERROR(kInvalidArgument, "no file to read ", what, " from");
  // One pread moves at most INT_MAX bytes on Darwin; larger ranges loop.
  constexpr std::size_t kMaxRead = std::size_t{1} << 30;
  auto* out = static_cast<std::byte*>(dst);
  std::size_t done = 0;
  while (done < bytes) {
    const std::size_t want = std::min(kMaxRead, bytes - done);
    const ssize_t got = ::pread(fd, out + done, want,
                                static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) continue;
      return LSE_ERROR(kIoError, "reading ", what, " (", std::to_string(bytes),
                       " bytes at offset ", std::to_string(offset),
                       "): ", std::strerror(errno));
    }
    if (got == 0) {
      return LSE_ERROR(kIoError, "reading ", what, ": the file ends ",
                       std::to_string(bytes - done), " bytes short of offset ",
                       std::to_string(offset + bytes));
    }
    done += static_cast<std::size_t>(got);
  }
  return OkStatus();
}

Result<int> open_for_streaming(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return LSE_ERROR(kIoError, "cannot open '", path, "': ", std::strerror(errno));
  }
#if defined(__APPLE__)
  if (::fcntl(fd, F_NOCACHE, 1) != 0) {
    const int error = errno;
    ::close(fd);
    return LSE_ERROR(kIoError, "F_NOCACHE on '", path, "': ", std::strerror(error));
  }
#endif
  return fd;
}

HostScratch::~HostScratch() {
  if (data_ != nullptr) ::munmap(data_, mapped_);
}

HostScratch::HostScratch(HostScratch&& other) noexcept
    : data_(other.data_), size_(other.size_), mapped_(other.mapped_) {
  other.data_ = nullptr;
  other.size_ = other.mapped_ = 0;
}

HostScratch& HostScratch::operator=(HostScratch&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) ::munmap(data_, mapped_);
    data_ = other.data_;
    size_ = other.size_;
    mapped_ = other.mapped_;
    other.data_ = nullptr;
    other.size_ = other.mapped_ = 0;
  }
  return *this;
}

Result<HostScratch> HostScratch::allocate(std::size_t bytes, const std::string& what) {
  HostScratch out;
  if (bytes == 0) return out;
  const auto page = static_cast<std::size_t>(::getpagesize());
  const std::size_t mapped = (bytes + page - 1) / page * page;
  void* p = ::mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED) {
    return LSE_ERROR(kOutOfMemory, "cannot map ", std::to_string(bytes),
                     " bytes of host memory for ", what, ": ", std::strerror(errno));
  }
  out.data_ = static_cast<std::byte*>(p);
  out.size_ = bytes;
  out.mapped_ = mapped;
  return out;
}

}  // namespace lse
