// Cache identity for the compiler image selected by the dynamic loader.
#pragma once

#include "lse/core/hash.hpp"
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/loader.h>
#endif

namespace lse::backend::detail {
inline std::string hex_identity(std::uint64_t value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string text(16, '0');
  for (std::size_t i = 0; i < 16; ++i) {
    text[15 - i] = digits[value & 15];
    value >>= 4;
  }
  return text;
}

inline auto file_mtime(const struct stat &st) {
#if defined(__APPLE__)
  return st.st_mtimespec;
#else
  return st.st_mtim;
#endif
}
inline auto file_ctime(const struct stat &st) {
#if defined(__APPLE__)
  return st.st_ctimespec;
#else
  return st.st_ctim;
#endif
}
inline bool same_compiler_file(const struct stat &a, const struct stat &b) {
  const auto am = file_mtime(a), bm = file_mtime(b), ac = file_ctime(a),
             bc = file_ctime(b);
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
         a.st_size == b.st_size && am.tv_sec == bm.tv_sec &&
         am.tv_nsec == bm.tv_nsec && ac.tv_sec == bc.tv_sec &&
         ac.tv_nsec == bc.tv_nsec;
}

// Content covers an equal-size rebuild with its original timestamp restored.
// The descriptor and before/after stat checks reject concurrent replacement or
// mutation during the read. Callers cache this result, never hash per kernel.
inline std::optional<std::string> compiler_file_identity(const char *path) {
  if (!path || !*path)
    return std::nullopt;
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return std::nullopt;
  struct Close {
    int fd;
    ~Close() { ::close(fd); }
  } close{fd};
  struct stat before{}, after{};
  if (::fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0)
    return std::nullopt;
  std::array<char, 64 * 1024> bytes{};
  std::uint64_t digest = kHashSeed, total = 0;
  while (total < static_cast<std::uint64_t>(before.st_size)) {
    const auto remaining = static_cast<std::uint64_t>(before.st_size) - total;
    const auto length = remaining < bytes.size()
                            ? static_cast<std::size_t>(remaining)
                            : bytes.size();
    const auto n = ::read(fd, bytes.data(), length);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return std::nullopt;
    }
    if (n == 0)
      break;
    digest = hash_bytes(
        std::string_view(bytes.data(), static_cast<std::size_t>(n)), digest);
    total += static_cast<std::uint64_t>(n);
  }
  if (::fstat(fd, &after) != 0 || !same_compiler_file(before, after) ||
      total != static_cast<std::uint64_t>(before.st_size))
    return std::nullopt;
  std::error_code ec;
  const auto resolved = std::filesystem::canonical(path, ec);
  if (ec)
    return std::nullopt;
  struct stat named{};
  if (::stat(resolved.c_str(), &named) != 0 ||
      !same_compiler_file(after, named))
    return std::nullopt;
  const auto modified = file_mtime(after);
  return "file=" + resolved.string() + " bytes=" + std::to_string(total) +
         " content=" + hex_identity(digest) +
         " mtime=" + std::to_string(modified.tv_sec) + ":" +
         std::to_string(modified.tv_nsec);
}

// Also identify the resident Mach-O image, so replacing the file after dyld
// mapped it cannot label old code solely with the replacement's file digest.
inline std::string resident_compiler_identity(const void *base) {
#if defined(__APPLE__)
  if (!base)
    return {};
  const auto *header = static_cast<const mach_header_64 *>(base);
  if (header->magic != MH_MAGIC_64 || header->sizeofcmds > 16 * 1024 * 1024 ||
      header->ncmds > header->sizeofcmds / sizeof(load_command))
    return {};
  const auto *current = reinterpret_cast<const std::byte *>(header + 1);
  const auto *end = current + header->sizeofcmds;
  for (std::uint32_t i = 0; i < header->ncmds; ++i) {
    if (static_cast<std::size_t>(end - current) < sizeof(load_command))
      return {};
    const auto *cmd = reinterpret_cast<const load_command *>(current);
    if (cmd->cmdsize < sizeof(load_command) ||
        cmd->cmdsize > static_cast<std::size_t>(end - current))
      return {};
    if (cmd->cmd == LC_UUID) {
      if (cmd->cmdsize < sizeof(uuid_command))
        return {};
      const auto *uuid = reinterpret_cast<const uuid_command *>(cmd);
      constexpr char digits[] = "0123456789abcdef";
      std::string text = "resident_uuid=";
      for (auto byte : uuid->uuid) {
        text += digits[byte >> 4];
        text += digits[byte & 15];
      }
      return text;
    }
    current += cmd->cmdsize;
  }
#else
  (void)base;
#endif
  return {};
}

inline const std::string &compiler_process_nonce() {
  static const std::string value =
      "process=" + std::to_string(::getpid()) + ":" +
      std::to_string(std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count()) +
      ":" +
      hex_identity(reinterpret_cast<std::uintptr_t>(&compiler_process_nonce));
  return value;
}

// Use a function pointer from the actual call site, not a configure-time path
// or a fresh dlopen. Unknown identity prevents cross-process disk-cache reuse.
inline std::string compiler_image_identity(const void *symbol) {
  Dl_info image{};
  if (!symbol || ::dladdr(symbol, &image) == 0 || !image.dli_fname ||
      !image.dli_fbase)
    return "image=unresolved " + compiler_process_nonce();
  const auto file = compiler_file_identity(image.dli_fname);
  const auto resident = resident_compiler_identity(image.dli_fbase);
  if (!file)
    return "image=unreadable " + resident + " " + compiler_process_nonce();
#if defined(__APPLE__)
  // A nonstandard/no-UUID image has no independent resident-file correlation.
  if (resident.empty())
    return *file + " resident=unknown " + compiler_process_nonce();
#endif
  return *file + (resident.empty() ? "" : " " + resident);
}
} // namespace lse::backend::detail
