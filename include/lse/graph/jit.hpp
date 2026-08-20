// Kernel cache: EmittedKernel -> code object -> backend executable.
//
// Device-agnostic. Compilation itself is delegated to the IKernelCompiler the
// backend supplies, so nothing here knows about a particular toolchain.
//
// A kernel is compiled only when the device (arch) changed, the cache has
// no entry for this (arch, signature), or the generated source changed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/codegen.hpp"

namespace lse::graph {

// $LSE_CACHE_DIR, else $XDG_CACHE_HOME/lse/kernels, else ~/.cache/lse/kernels.
std::string default_cache_dir();

// $LSE_HIP_DUMP, else ${CMAKE_BINARY_DIR}/hip.
std::string hip_dump_directory();

// Writes `emitted.source` under hip_dump_directory(). One file per
// entry name per process; later launches of the same kernel skip.
void dump_hip_source(const EmittedKernel& emitted, std::uint64_t key = 0);

// Drops on-disk code objects and HIP dumps from the previous process.
// Called once the first time this process constructs a JitCache.
void purge_kernel_artifacts();

class JitCache {
 public:
  // One cache for the whole set. Not one per device: two members of the same
  // arch and geometry emit the same source and compile to the same object, so a
  // cache per device would pay ~350 ms per kernel per device for bytes it
  // already had. What IS per device is the loaded handle — an executable is
  // loaded on one device and running it on another is a wrong-device dispatch
  // that no runtime here reports — so the object is shared and the handle is not.
  explicit JitCache(backend::IDeviceSet& devices,
                    std::string cache_dir = default_cache_dir());
  // One device, for a caller that holds a backend rather than a set. The
  // compiler is taken from the backend either way; the parameter is here
  // because a caller may hold a compiler the backend does not publish.
  JitCache(backend::IBackend& backend, const IKernelCompiler& compiler,
           std::string cache_dir = default_cache_dir());
  ~JitCache();

  // `signature` is the emitter's cache identity (group + specialization);
  // `member` is which device of the set will run it. The device's arch and the
  // geometry the source was emitted against are mixed in here, and so is
  // `emitted.dialect` — the text names its own language, and the compiler this
  // object is built with is the one declared beside the emitter that wrote it.
  Result<backend::KernelHandle> get_or_compile(std::size_t member,
                                               std::uint64_t signature,
                                               const EmittedKernel& emitted);

  // Live handle if this process already loaded the kernel ON THIS MEMBER in
  // THIS DIALECT. Does not compile, emit, or read disk. Counts as a memory
  // hit. `dialect` is not optional and has no default: this is the one lookup
  // that returns a handle without seeing the source it was built from, so a
  // caller that forgot to say which language it asked for would be handed the
  // other one's kernel.
  const backend::KernelHandle* try_get(std::size_t member,
                                       std::uint64_t signature,
                                       Dialect dialect) noexcept;

  // What the toolchain reported about the object behind this entry, or
  // nullptr when nothing was reported for it. Available on a warm start too:
  // the numbers are persisted beside the cached object, so they are a property
  // of the kernel and not of how long this process has been running. An empty
  // `entry` matches when the object defines exactly one kernel.
  [[nodiscard]] const backend::KernelResources* resources(
      std::size_t member, std::uint64_t signature, Dialect dialect,
      std::string_view entry = {}) const noexcept;

  // What the object's instructions counted up to, on the same terms: persisted
  // beside it, so a cached kernel is as well described as a freshly compiled
  // one, and counted from the cached bytes when an older note has no counts.
  [[nodiscard]] const backend::KernelCensus* census(
      std::size_t member, std::uint64_t signature, Dialect dialect,
      std::string_view entry = {}) const noexcept;

  struct Stats {
    std::uint64_t memory_hits = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t compiles = 0;
    std::uint64_t compile_ns = 0;
  };
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

 private:
  // Everything that decides what SOURCE this device would be given, which is
  // what an object on disk is only valid for.
  [[nodiscard]] std::uint64_t slot_key(std::size_t member, Dialect dialect,
                                       std::uint64_t signature) const noexcept;
  // The compiler declared beside the emitter that writes `dialect`, or nullptr
  // when this member declares no such dialect.
  [[nodiscard]] const IKernelCompiler* compiler_for(
      std::size_t member, Dialect dialect) const noexcept;
  [[nodiscard]] std::size_t toolchain_slot(std::size_t member,
                                           Dialect dialect) const noexcept {
    return member * kDialectCount + static_cast<std::size_t>(dialect);
  }

  // Only when the single-device constructor was used: the set that backend is,
  // and the compiler the caller named for it. Declared before devices_ so the
  // reference is bound to something already built.
  std::unique_ptr<backend::SingleDevice> own_set_;
  backend::IDeviceSet& devices_;
  const IKernelCompiler* named_compiler_ = nullptr;
  // Hash of each (member, dialect)'s compiler identity, taken once: slot_key
  // runs per group per token and identity() builds strings. Indexed by
  // toolchain_slot, NOT by member: a member declaring two dialects declares two
  // compilers, and one entry per member would have hashed the front one's
  // identity into the other one's cache slots.
  std::vector<std::uint64_t> compiler_id_;
  std::string cache_dir_;
  Stats stats_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::graph
