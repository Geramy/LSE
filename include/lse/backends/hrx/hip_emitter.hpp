// Fusion group -> HIP source, for AMDGPU targets.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "lse/backends/hrx/hip_sources.hpp"
#include "lse/graph/codegen.hpp"

namespace lse::graph {
class Node;
}

namespace lse::backend {

class HipEmitter final : public graph::IKernelEmitter {
 public:
  Result<graph::EmittedKernel> emit(const graph::FusionGroup& group,
                                    const DeviceInfo& device) const override;

  static Result<graph::EmittedKernel> emit_phase(
      const graph::FusionGroup& group, const DeviceInfo& device);
  static bool phase_can_stage(const graph::Node& n) noexcept;

  // Independent work items the node could spend if it owned the launch. The
  // phase splitter breaks a chain here: a stage wanting thousands of them
  // cannot share the one-workgroup fallback its dependent neighbours need.
  static std::uint32_t phase_stage_threads(const graph::Node& n,
                                           const DeviceInfo& device);

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

  // ConstantsLayout is offsets and sizes; the C spelling of a 4- or 8-byte
  // scalar is HIP's business, so the struct text is rendered here.
  [[nodiscard]] static std::string constants_decl(
      const graph::ConstantsLayout& layout);

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
    std::size_t scratch_bytes = 0;
    bool persist_grid = false;
  };
  mutable std::unordered_map<std::uint64_t, CachedEmit> emit_cache_;
};

}  // namespace lse::backend
