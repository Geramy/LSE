// Kernel cache: EmittedKernel -> code object -> backend executable.
//
// Device-agnostic. Compilation itself is delegated to the IKernelCompiler the
// backend supplies, so nothing here knows about a particular toolchain.
//
// Cache ownership is scoped to the engine release. Source, compiler, and
// device checks still distinguish kernels within that release.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/codegen.hpp"

namespace lse::graph {

// The release compiled into the cache implementation.
std::string_view kernel_cache_version() noexcept;

// Digest of every engine source file and the compiler configuration this
// binary was built from (generated at build time; see
// cmake/LSEBuildIdentity.cmake). Kernel templates live in that source, so two
// builds of one release that could write different text for the same group
// carry different identities. The launch index is keyed on it.
std::string_view engine_build_identity() noexcept;

// $LSE_CACHE_DIR when set, otherwise ~/.lse/cache.
std::string default_cache_dir();

// Called at startup; an explicit path takes precedence over the environment.
Status prepare_cache_dir(std::string_view requested = {});

// $LSE_HIP_DUMP, else ${CMAKE_BINARY_DIR}/hip.
std::string hip_dump_directory();

// Writes `emitted.source` under hip_dump_directory(). One file per
// entry name per process; later launches of the same kernel skip.
void dump_hip_source(const EmittedKernel& emitted, std::uint64_t key = 0);

// Removes identifiable older LSE cache families in the selected directory.
// Same/newer releases, incomplete entries, symlinks and unrelated files remain.
// Generated-source dumps are cleared separately, once per process.
void purge_kernel_artifacts(std::string_view cache_dir = {});

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

  // Makes every kernel in `kernels` resident on `member`, loading those that
  // are not as ONE executable: one code object, compiled (or read back from
  // disk) as one, for the whole set. An executable holds device allocations
  // of its own and a device grants a process a few thousand, so a server that
  // makes every pass shape resident at load cannot spend one per kernel. Each
  // kernel's source must be present; all share one dialect.
  struct Preload {
    std::uint64_t signature = 0;
    const EmittedKernel* emitted = nullptr;
  };
  //
  // With `index`, the set is also written to the LAUNCH INDEX: under a key
  // made only of things known before any source exists (the kernels' cache
  // keys as slot_key extends them, the dialect, and engine_build_identity()),
  // the launch description and source fingerprints of each kernel and the
  // bundle object they were loaded from. restore() reads it back.
  Status preload(std::size_t member, std::span<const Preload> kernels,
                 bool index = false);

  // Makes the kernels `signatures` name resident on `member` from the launch
  // index, without their source: the bundle object an earlier preload of the
  // same set loaded is loaded again, and each kernel's launch description is
  // handed to `emitter` (IKernelEmitter::adopt_launch), so launching one
  // writes no text either. `signatures` is the list preload was given, in
  // its order.
  //
  // True when every kernel is resident. False is a MISS and changes nothing:
  // the caller emits the sources and preloads them, exactly as with no
  // index, and that preload rewrites the entry. A miss is any entry not on
  // disk, or one whose set no longer matches what this process has resident,
  // or whose bundle object is no longer on disk.
  //
  // CORRUPTION POLICY: rejected and rebuilt, never served and never fatal.
  // Every entry carries a checksum and echoes its own key, build identity,
  // release, device and kernel list; the bundle object's bytes are checked
  // against the checksum recorded when the entry was written. An entry or
  // object that fails any check is reported on stderr with its path and the
  // check it failed, removed, and counted in Stats::index_rejects, and the
  // call returns false so the kernels are written and compiled from source.
  // Nothing in the index can make a kernel run that its source would not
  // have produced: a damaged entry costs one cold preparation of its set.
  Result<bool> restore(std::size_t member, Dialect dialect,
                       std::span<const std::uint64_t> signatures,
                       const IKernelEmitter& emitter);

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
    // Launch index: sets made resident without source, sets that had to be
    // emitted, and entries discarded as corrupt (also counted as misses).
    std::uint64_t index_hits = 0;
    std::uint64_t index_misses = 0;
    std::uint64_t index_rejects = 0;
  };
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

  // Where this binary's launch index lives, inside the cache directory.
  [[nodiscard]] const std::string& launch_index_dir() const noexcept {
    return index_dir_;
  }

 private:
  // Everything that decides what SOURCE this device would be given, which is
  // what an object on disk is only valid for.
  [[nodiscard]] std::uint64_t slot_key(std::size_t member, Dialect dialect,
                                       std::uint64_t signature) const noexcept;
  // Finds or makes this build's index directory and prunes other builds'.
  void open_launch_index();
  // The launch index entry for `signatures` on `member`: their slot keys,
  // the dialect and the build identity, all known before any source exists.
  [[nodiscard]] std::uint64_t index_key(
      std::size_t member, Dialect dialect,
      std::span<const std::uint64_t> signatures) const noexcept;
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
  // <cache_dir>/launch-<release>-<build identity prefix>: one directory per
  // build, so two builds sharing a cache never read each other's entries.
  std::string index_dir_;
  Stats stats_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::graph
