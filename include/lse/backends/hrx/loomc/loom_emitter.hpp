// Fusion group -> Loom source, for AMDGPU targets.
//
// The second generator inside the HRX runtime, beside hipc/. It answers the
// same IKernelEmitter contract HipEmitter does and produces the same
// EmittedKernel — the same binding order, the same dispatch constants, the
// same launch dims — so the runtime below it does not know which one wrote the
// text. What differs is the language, and everything that follows from Loom
// being SSA rather than C.
//
// It covers strictly less than the HIP emitter, and says so: a group whose
// body Loom cannot express is a kUnimplemented naming the reason, and the
// scheduler falls back to the HIP toolchain the same way it falls back from a
// declined primitive. Nothing here approximates.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>

#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/graph/codegen.hpp"

namespace lse::graph {
class Node;
}

namespace lse::backend {

class LoomEmitter final : public graph::IKernelEmitter {
 public:
  struct CacheStats {
    std::size_t hits, misses, entries, source_bytes;
  };
  [[nodiscard]] CacheStats cache_stats() const;

  Result<graph::EmittedKernel> emit(const graph::FusionGroup& group,
                                    const DeviceInfo& device) const override;
  Result<graph::EmittedKernel> emit_launch(const graph::FusionGroup& group,
                                           const DeviceInfo& device) const override;

  [[nodiscard]] std::uint64_t cache_key(
      const graph::FusionGroup& group,
      const DeviceInfo& device) const override;
  [[nodiscard]] std::uint32_t variants(const graph::FusionGroup& group,
                                       const DeviceInfo& device) const override;

  // Every description emit() writes is kept under cache_key(), so one read
  // back from the JIT's launch index stands in for writing the source.
  [[nodiscard]] bool keeps_launches() const noexcept override { return true; }
  bool adopt_launch(std::uint64_t key,
                    const graph::EmittedKernel& launch) const override;

  [[nodiscard]] graph::Dialect dialect() const noexcept override {
    return graph::Dialect::kLoom;
  }

  // A primitive whose chosen specialization owns its indexing (a wave-
  // cooperative body) can only be a group's sole output, so a run holding
  // one is not joined.
  [[nodiscard]] bool joins_run(std::span<const graph::NodePtr> run,
                               const DeviceInfo& device) const override;

  // Loom has no header: a `.loom` file is a bare sequence of top-level ops,
  // with no module wrapper and nothing to include. A primitive that owns a
  // whole translation unit has nothing to be prefixed with, which is also why
  // no such primitive can be written in this dialect.
  [[nodiscard]] std::string_view prelude() const noexcept override {
    return {};
  }

  [[nodiscard]] graph::DialectSourceTable sources() const noexcept override {
    return loom_sources();
  }

  // No staged phase form. A phase body is a resident grid in waiting, and Loom
  // refuses grid-wide synchronization by design — see the note in emit().
  [[nodiscard]] const graph::IPhaseStaging* staging() const noexcept override {
    return nullptr;
  }

 private:
  mutable std::mutex cache_mutex_;
  Result<graph::EmittedKernel> emit_kernel(const graph::FusionGroup& group,
                                           const DeviceInfo& device,
                                           bool with_source) const;
  // One entry per emission identity: the launch description (no source, no
  // bindings) and the source text beside it while the text budget lasts. A
  // server prepares every pass shape at load, so the descriptions of all of
  // them stay; only text, which a resident kernel never needs, is budgeted.
  struct CachedEmission {
    graph::EmittedKernel launch;
    std::shared_ptr<const std::string> source;
  };
  // Keyed by cache_key(): the identity is hashed as it is walked, never
  // spelled out as text, so a key costs no allocation.
  mutable std::unordered_map<std::uint64_t, CachedEmission> emit_cache_;
  mutable std::size_t cache_hits_ = 0, cache_misses_ = 0, cache_bytes_ = 0;
};

}  // namespace lse::backend
