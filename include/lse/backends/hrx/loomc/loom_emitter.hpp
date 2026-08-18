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
  Result<graph::EmittedKernel> emit(const graph::FusionGroup& group,
                                    const DeviceInfo& device) const override;

  [[nodiscard]] std::uint64_t cache_key(
      const graph::FusionGroup& group,
      const DeviceInfo& device) const override;

  [[nodiscard]] graph::Dialect dialect() const noexcept override {
    return graph::Dialect::kLoom;
  }

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
};

}  // namespace lse::backend
