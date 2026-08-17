#include "lse/graph/jit.hpp"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace lse::graph {

namespace fs = std::filesystem;

namespace {

std::vector<std::byte> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto size = static_cast<std::streamsize>(in.tellg());
  if (size <= 0) return {};
  std::vector<std::byte> out(static_cast<std::size_t>(size));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(out.data()), size);
  if (in.gcount() != size) return {};
  return out;
}

std::uint64_t fnv(std::string_view s) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (char ch : s) {
    h ^= static_cast<unsigned char>(ch);
    h *= 1099511628211ull;
  }
  return h;
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v;
  h *= 1099511628211ull;
  return h;
}

struct DiskMeta {
  std::string arch;
  std::uint64_t source_hash = 0;
  std::string entry;
};

bool read_meta(const fs::path& path, DiskMeta* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string hash;
  if (!std::getline(in, out->arch)) return false;
  if (!std::getline(in, hash)) return false;
  if (!std::getline(in, out->entry)) return false;
  if (out->arch.empty() || hash.empty()) return false;
  try {
    out->source_hash = std::stoull(hash, nullptr, 16);
  } catch (...) {
    return false;
  }
  return true;
}

void write_meta(const fs::path& path, const DiskMeta& meta) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp);
  if (!out) return;
  out << meta.arch << '\n';
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(meta.source_hash));
  out << buf << '\n' << meta.entry << '\n';
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

void write_code(const fs::path& path, const std::vector<std::byte>& code) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp, std::ios::binary);
  if (!out) return;
  out.write(reinterpret_cast<const char*>(code.data()),
            static_cast<std::streamsize>(code.size()));
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

void write_text(const fs::path& path, std::string_view text) {
  const fs::path tmp =
      path.string() + ".tmp" + std::to_string(::getpid());
  std::ofstream out(tmp);
  if (!out) return;
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.close();
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) fs::remove(tmp, ec);
}

}  // namespace

void dump_hip_source(const EmittedKernel& emitted, std::uint64_t key) {
  if (emitted.source.empty()) return;
  const fs::path dir = hip_dump_directory();
  if (dir.empty()) return;

  std::string name = emitted.entry_name.empty()
                         ? ("kernel_" + std::to_string(key))
                         : emitted.entry_name;
  for (char& c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
          c == '-' || c == '.')) {
      c = '_';
    }
  }

  static std::mutex mu;
  static std::unordered_set<std::string> written;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!written.insert(name).second) return;
  }

  std::error_code ec;
  fs::create_directories(dir, ec);
  write_text(dir / (name + ".hip"), emitted.source);
}

std::string hip_dump_directory() {
  if (const char* env = std::getenv("LSE_HIP_DUMP"); env != nullptr && env[0] != '\0') {
    return env;
  }
#ifdef LSE_BUILD_DIR
  return std::string(LSE_BUILD_DIR) + "/hip";
#else
  return {};
#endif
}

void purge_kernel_artifacts() {
  static std::once_flag once;
  std::call_once(once, [] {
    const auto wipe = [](const fs::path& dir) {
      if (dir.empty()) return;
      const fs::path n = dir.lexically_normal();
      if (n.empty() || n == n.root_path()) return;
      const std::string s = n.generic_string();
      if (s.find("lse") == std::string::npos &&
          s.find("hip") == std::string::npos &&
          s.find("kernel") == std::string::npos) {
        return;
      }
      std::error_code ec;
      fs::remove_all(n, ec);
    };
    // The dump dir is debug output; clearing it keeps build/hip in step with
    // this process's emission. The .co cache is NOT wiped: entries carry an
    // arch + source-hash check on read, so stale ones self-invalidate, and
    // wiping here forced every process to recompile every kernel.
    wipe(hip_dump_directory());
    if (const char* env = std::getenv("LSE_JIT_PURGE");
        env != nullptr && env[0] == '1') {
      wipe(default_cache_dir());
    }
  });
}

std::string default_cache_dir() {
  if (const char* env = std::getenv("LSE_CACHE_DIR")) return env;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    return std::string(xdg) + "/lse/kernels";
  }
  if (const char* home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/lse/kernels";
  }
  return ".lse-cache";
}

struct JitCache::Impl {
  struct Slot {
    backend::KernelHandle handle;
    std::uint64_t source_hash = 0;
    std::string arch;
  };
  // One map per member. A KernelHandle is an executable loaded on ONE device;
  // handing member B the handle member A loaded is a wrong-device dispatch that
  // no runtime here reports, and two same-arch devices would otherwise share
  // the entry because the disk key is deliberately the same for them.
  std::vector<std::unordered_map<std::uint64_t, Slot>> memory;
};

JitCache::JitCache(backend::IDeviceSet& devices, std::string cache_dir)
    : devices_(devices),
      cache_dir_(std::move(cache_dir)),
      impl_(std::make_unique<Impl>()) {
  impl_->memory.resize(devices_.size());
  compiler_id_.resize(devices_.size(), 0);
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    const IKernelCompiler* cc = devices_.device(i).compiler();
    compiler_id_[i] = cc != nullptr ? fnv(cc->identity()) : 0;
  }
  purge_kernel_artifacts();
}

JitCache::JitCache(backend::IBackend& backend, const IKernelCompiler& compiler,
                   std::string cache_dir)
    : own_set_(std::make_unique<backend::SingleDevice>(backend)),
      devices_(*own_set_),
      named_compiler_(&compiler),
      cache_dir_(std::move(cache_dir)),
      impl_(std::make_unique<Impl>()) {
  compiler_id_.push_back(fnv(compiler.identity()));
  impl_->memory.resize(1);
  purge_kernel_artifacts();
}

JitCache::~JitCache() = default;

const IKernelCompiler* JitCache::compiler_for(
    std::size_t member) const noexcept {
  if (named_compiler_ != nullptr) return named_compiler_;
  return devices_.device(member).compiler();
}

std::uint64_t JitCache::slot_key(std::size_t member,
                                 std::uint64_t signature) const noexcept {
  // A cached object is only valid for the toolchain that built it, and
  // source_hash cannot tell two toolchains apart. The compiler reports its own
  // identity (version + option lists) rather than a human bumping a revision
  // constant here, which was one forgotten increment away from serving an
  // object built by a different pipeline.
  // Arch in the key so a device change cannot reuse another target's object.
  const backend::DeviceInfo& info = devices_.device(member).device_info();
  std::uint64_t h = mix(mix(signature, compiler_id_[member]), fnv(info.arch));
  // Arch is NOT enough between two devices of the same ISA. The emitter chooses
  // workgroup dimensions, an LDS budget and a persistent-grid decision from the
  // CU count, the LDS pool and the workgroup ceiling, so two gfx1151 parts with
  // different geometry are handed different source under one arch string. Left
  // out, they collide on one key: the source-hash guard then forces a recompile
  // per alternation AND each device deletes the other's object as dead — a
  // ~350 ms stall per kernel per switch, on a key that looked like a hit.
  //
  // Deliberately NOT the device's identity. Two members with the same geometry
  // emit byte-identical source and must share the object; keying on which
  // device asked would compile it once per device for nothing.
  h = mix(h, static_cast<std::uint64_t>(info.compute_units));
  h = mix(h, static_cast<std::uint64_t>(info.lds_bytes_per_workgroup));
  h = mix(h, static_cast<std::uint64_t>(info.max_threads_per_workgroup));
  h = mix(h, static_cast<std::uint64_t>(info.wavefront_size));
  h = mix(h, static_cast<std::uint64_t>(info.cus_per_lds_pool));
  return h;
}

const backend::KernelHandle* JitCache::try_get(std::size_t member,
                                               std::uint64_t signature) noexcept {
  if (member >= impl_->memory.size()) return nullptr;
  auto& slots = impl_->memory[member];
  const auto it = slots.find(slot_key(member, signature));
  if (it == slots.end()) return nullptr;
  if (it->second.arch != devices_.device(member).device_info().arch) {
    return nullptr;
  }
  ++stats_.memory_hits;
  return &it->second.handle;
}

Result<backend::KernelHandle> JitCache::get_or_compile(
    std::size_t member, std::uint64_t signature, const EmittedKernel& emitted) {
  if (member >= impl_->memory.size()) {
    return LSE_ERROR(kOutOfRange, "device set has ",
                     std::to_string(impl_->memory.size()),
                     " members; there is no member ", std::to_string(member));
  }
  backend::IBackend& be = devices_.device(member);
  const IKernelCompiler* compiler = compiler_for(member);
  if (compiler == nullptr) {
    return LSE_ERROR(kUnimplemented, "backend '", std::string(be.name()),
                     "' has no kernel compiler");
  }
  auto& slots = impl_->memory[member];
  const std::string arch(be.device_info().arch);
  const std::uint64_t key = slot_key(member, signature);
  const std::uint64_t src_hash =
      emitted.source.empty() ? 0 : fnv(emitted.source);

  dump_hip_source(emitted, key);

  if (const auto it = slots.find(key); it != slots.end()) {
    const Impl::Slot& slot = it->second;
    if (slot.arch == arch &&
        (src_hash == 0 || src_hash == slot.source_hash)) {
      ++stats_.memory_hits;
      return slot.handle;
    }
  }

  const fs::path stem = fs::path(cache_dir_) / std::to_string(key);
  const fs::path meta_path = stem.string() + ".meta";

  DiskMeta meta;
  const bool meta_ok = read_meta(meta_path, &meta);
  const bool source_matches =
      src_hash == 0 || (meta_ok && src_hash == meta.source_hash);
  const bool device_matches = meta_ok && meta.arch == arch;

  // The object is named by the source hash it was built from, so a load can
  // never pair one source's meta with another source's code: the two renames
  // below are individually atomic but not atomic as a pair, and a crash
  // between them must not be able to poison the key forever.
  const std::uint64_t co_hash =
      src_hash != 0 ? src_hash : (meta_ok ? meta.source_hash : 0);
  const fs::path co_path =
      stem.string() + "." + std::to_string(co_hash) + ".co";

  std::vector<std::byte> code;
  if (device_matches && source_matches) {
    code = read_file(co_path);
    if (!code.empty()) ++stats_.disk_hits;
  }

  if (code.empty()) {
    if (emitted.source.empty()) {
      return LSE_ERROR(kCompileError,
                       "no cached kernel for this device and no source to compile");
    }
    const auto begin = std::chrono::steady_clock::now();
    auto compiled = compiler->compile(emitted.source, arch);
    if (!compiled.ok()) return compiled.status();
    code = compiled.release();
    stats_.compile_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count());
    ++stats_.compiles;

    std::error_code ec;
    fs::create_directories(cache_dir_, ec);
    // Best-effort reclaim of objects this key no longer references (older
    // source revisions); the hash-suffixed name makes them dead, not wrong.
    for (fs::directory_iterator it(cache_dir_, ec), end; !ec && it != end;
         ++it) {
      const std::string name = it->path().filename().string();
      const std::string prefix = std::to_string(key) + ".";
      if (name.rfind(prefix, 0) == 0 && name.size() > 3 &&
          name.compare(name.size() - 3, 3, ".co") == 0 &&
          it->path() != co_path) {
        std::error_code rm;
        fs::remove(it->path(), rm);
      }
    }
    write_code(co_path, code);
    write_meta(meta_path, DiskMeta{arch, src_hash, emitted.entry_name});
  }

  auto handle = be.load_executable(
      emitted.entry_name.empty() && meta_ok ? meta.entry : emitted.entry_name,
      code);
  if (!handle.ok()) return handle.status();
  backend::KernelHandle kernel = handle.release();
  const std::uint64_t stored_hash =
      src_hash != 0 ? src_hash : (meta_ok ? meta.source_hash : 0);
  slots[key] = Impl::Slot{kernel, stored_hash, arch};
  return kernel;
}

}  // namespace lse::graph
