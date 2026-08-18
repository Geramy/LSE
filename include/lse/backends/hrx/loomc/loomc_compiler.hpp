// Loom source -> AMDGPU code object via loomc.
//
// The second generator inside the HRX runtime, beside hipc/. loomc parses
// `.loom` text and emits an HSACO itself: no LLVM in the graph and no HIP
// runtime to link, so the object loads through the same native
// hrx_executable_load_data path a comgr-built one does.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/graph/codegen.hpp"

namespace lse::backend {

class LoomcCompiler final : public graph::IKernelCompiler {
 public:
  LoomcCompiler();
  ~LoomcCompiler() override;

  LoomcCompiler(const LoomcCompiler&) = delete;
  LoomcCompiler& operator=(const LoomcCompiler&) = delete;

  // Deserialize, compile and emit in one call. The entry symbols are read back
  // out of the module rather than taken from the caller: `kernel.def` already
  // names them, and a second statement of the same name is a second place for
  // it to be wrong.
  Result<std::vector<std::byte>> compile(std::string_view source,
                                         std::string_view arch) const override;

  [[nodiscard]] bool available() const override;

  // The loomc install this was linked against plus the exact option set every
  // invocation runs with.
  [[nodiscard]] std::string identity() const override;

 private:
  struct State;
  // The loomc handles are built on the first compile, not in the constructor:
  // every creation entry point returns a status and a constructor has nowhere
  // to put one.
  std::unique_ptr<State> state_;
};

}  // namespace lse::backend
