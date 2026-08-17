// HIP source -> AMDGPU code object via amd_comgr.
//
// comgr rather than hipRTC: it produces a code object without linking the HIP
// runtime, which is what lets dispatch stay on the native HRX ABI.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "lse/graph/codegen.hpp"

namespace lse::backend {

class ComgrCompiler final : public graph::IKernelCompiler {
 public:
  Result<std::vector<std::byte>> compile(std::string_view source,
                                         std::string_view arch) const override;

  [[nodiscard]] bool available() const override;

  // comgr's own version plus the exact option lists both actions run with.
  [[nodiscard]] std::string identity() const override;
};

}  // namespace lse::backend
