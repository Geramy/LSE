// Kernel cache: EmittedKernel -> code object -> backend executable.
//
// Device-agnostic. Compilation itself is delegated to the IKernelCompiler the
// backend supplies, so nothing here knows about a particular toolchain.
//
// A kernel is compiled only when the device (arch) changed, the cache has
// no entry for this (arch, signature), or the generated source changed.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

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
  JitCache(backend::IBackend& backend, const IKernelCompiler& compiler,
           std::string cache_dir = default_cache_dir());
  ~JitCache();

  // `signature` is the emitter's cache identity (group + specialization).
  // Arch is taken from the live device and mixed in here.
  Result<backend::KernelHandle> get_or_compile(std::uint64_t signature,
                                               const EmittedKernel& emitted);

  // Live handle if this process already loaded the kernel for the current
  // device. Does not compile, emit, or read disk. Counts as a memory hit.
  const backend::KernelHandle* try_get(std::uint64_t signature) noexcept;

  struct Stats {
    std::uint64_t memory_hits = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t compiles = 0;
    std::uint64_t compile_ns = 0;
  };
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

 private:
  [[nodiscard]] std::uint64_t slot_key(std::uint64_t signature) const noexcept;

  backend::IBackend& backend_;
  const IKernelCompiler& compiler_;
  // Hash of compiler_.identity(), taken once: slot_key runs per group per
  // token and identity() builds strings.
  std::uint64_t compiler_id_;
  std::string cache_dir_;
  Stats stats_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::graph
