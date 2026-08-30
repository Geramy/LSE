// Fusion group -> HIP source, for AMDGPU targets.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/graph/codegen.hpp"

namespace lse::graph {
class Node;
}

namespace lse::backend {

class HipEmitter final : public graph::IKernelEmitter,
                         public graph::IPhaseStaging {
 public:
  Result<graph::EmittedKernel> emit(const graph::FusionGroup& group,
                                    const DeviceInfo& device) const override;

  static Result<graph::EmittedKernel> emit_phase(
      const graph::FusionGroup& group, const DeviceInfo& device);

  [[nodiscard]] bool can_stage(const graph::Node& n) const noexcept override;
  [[nodiscard]] std::uint32_t stage_threads(
      const graph::Node& n, const DeviceInfo& device) const override;
  [[nodiscard]] bool lane_stage(const graph::Node& n) const noexcept override;
  [[nodiscard]] bool lane_aligned(
      const graph::Node& producer,
      const graph::Node& consumer) const noexcept override;
  [[nodiscard]] bool lane_writes(const graph::Node& n) const noexcept override;

  static void bind_phase(const graph::FusionGroup& group,
                         graph::EmittedKernel& out);

  [[nodiscard]] std::uint64_t cache_key(
      const graph::FusionGroup& group,
      const DeviceInfo& device) const override;

  [[nodiscard]] graph::Dialect dialect() const noexcept override {
    return graph::Dialect::kHip;
  }

  // The HIP runtime headers and the dispatch-constants struct; a kernel
  // primitive owning a whole translation unit is prefixed with this.
  [[nodiscard]] std::string_view prelude() const noexcept override;

  [[nodiscard]] graph::DialectSourceTable sources() const noexcept override;

  [[nodiscard]] const graph::IPhaseStaging* staging() const noexcept override {
    return this;
  }

  // What this emitter's two arrangements of `run` would declare in workgroup
  // scratch. See IKernelEmitter::run_scratch: the bytes are this file's, the
  // residency they buy is the engine's.
  [[nodiscard]] RunScratch run_scratch(
      std::span<const graph::NodePtr> run,
      const DeviceInfo& device) const override;

  // ConstantsLayout is offsets and sizes; the C spelling of a 4- or 8-byte
  // scalar is HIP's business, so the struct text is rendered here.
  [[nodiscard]] static std::string constants_decl(
      const graph::ConstantsLayout& layout);

  // Workgroup scratch a generated kernel declares, in bytes, read off its
  // source.
  //
  // A phase kernel is assembled as text — braced stage bodies, plus `__device__`
  // helpers in the preamble that may hold scratch of their own — so the text is
  // where its total lives; there is no single ir::Body to ask. Every declaration
  // is charged once and 16-byte aligned, disjoint block scopes included:
  // separate declarations get separate offsets, and two
  // `__shared__ float[2176]` in sibling braces report sharedSizeBytes 17408
  // against 8704 for one (measured, gfx1151, hipFuncGetAttributes).
  //
  // Fails on a declaration whose element type is not in the dialect's spelling
  // table rather than guessing its width, since a guess is how an accounting
  // comes to under-report the one case it was built to catch.
  [[nodiscard]] static Result<std::uint32_t> shared_bytes(
      std::string_view source);

 private:
  // Workgroup size and per-thread element count, chosen against the device's
  // occupancy limits. Returns the candidate with the highest occupancy that
  // does not exceed lds_bytes_per_workgroup.
  static LaunchDims choose_dims(const graph::FusionGroup& group,
                                const DeviceInfo& device,
                                std::uint32_t lds_bytes);

  struct CachedEmit {
    std::string source;
    std::string entry_name;
    LaunchDims dims{};
    // Measured off the cached text, not re-predicted on a cache hit.
    std::uint32_t lds_bytes = 0;
    std::size_t scratch_bytes = 0;
    bool persist_grid = false;
    // Whether the cached text's signature takes a pointer table. Restored on
    // a hit — a table-mode kernel served with direct bindings launches
    // against the wrong signature and reads garbage addresses.
    bool pointer_table = false;
  };
  mutable std::unordered_map<std::uint64_t, CachedEmit> emit_cache_;
  // Priced runs, keyed on the arrangement's own signature. run_scratch is a
  // pure function of the run and the device, and the scheduler asks it once
  // per candidate per join attempt — quadratic in the run length, each ask
  // specializing and planning every member. Without this the 27B pays 0.23 s
  // of extra partition time per process for answers it already has.
  mutable std::unordered_map<std::uint64_t, RunScratch> run_scratch_cache_;
  // Sibling runs whose assembled body overran the workgroup budget, and by how
  // much. The verdict is a function of the group and the device, so it is
  // reached once; without this a prefill pass rebuilds the whole IR body of
  // every such run on every pass only to refuse it again.
  mutable std::unordered_map<std::uint64_t, std::uint32_t> lds_refused_;
};

}  // namespace lse::backend
