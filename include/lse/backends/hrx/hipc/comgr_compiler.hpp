// HIP source -> AMDGPU code object via amd_comgr.
//
// comgr rather than hipRTC: it produces a code object without linking the HIP
// runtime, which is what lets dispatch stay on the native HRX ABI.
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/graph/codegen.hpp"

namespace lse::backend {

class ComgrCompiler final : public graph::IKernelCompiler {
 public:
  // The object, plus the resources its own metadata note states — read at the
  // point the linked object is already in hand, which costs one metadata walk
  // and no second compile.
  Result<graph::CompiledKernel> compile(std::string_view source,
                                        std::string_view arch) const override;

  // The census of an object comgr produced, disassembled through comgr's own
  // disassembler. Serves the warm-cache path, where the bytes exist and no
  // compile ran.
  [[nodiscard]] std::vector<KernelCensus> census(
      std::span<const std::byte> object) const override;

  [[nodiscard]] bool available() const override;

  // comgr's own version plus the exact option lists both actions run with.
  [[nodiscard]] std::string identity() const override;
};

}  // namespace lse::backend
