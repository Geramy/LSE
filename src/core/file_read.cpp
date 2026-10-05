#include "lse/core/file_read.hpp"

#include <fcntl.h>
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

}  // namespace lse
