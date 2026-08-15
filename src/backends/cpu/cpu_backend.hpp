// CPU reference backend.
//
// Purpose is correctness, not speed: it is the oracle the HRX backend is
// checked against, and it lets the whole engine build and run on a machine with
// no GPU. It satisfies the Backend<Derived> contract with plain host memory.
//
// It deliberately does NOT implement load_executable: there is no code object
// to load on the host. CPU fusion groups are executed by an interpreter over
// the graph instead (see src/backends/cpu/cpu_interpreter.cpp), which is also
// what makes it a useful differential-testing oracle for the JIT.
#pragma once

#include "lse/backend/backend.hpp"

namespace lse::backend {

class CpuBackend : public Backend<CpuBackend> {
 public:
  static constexpr std::string_view kName = "cpu";

  Status init_impl(int device_ordinal);
  void shutdown_impl() noexcept;
  const DeviceInfo& device_info_impl() const noexcept { return info_; }

  Result<DeviceBuffer> allocate_impl(std::size_t bytes, MemoryClass cls);
  void deallocate_impl(DeviceBuffer& buf) noexcept;

  Status copy_h2d_impl(const void* src, DeviceBuffer& dst, std::size_t bytes,
                       std::size_t dst_offset);
  Status copy_d2h_impl(const DeviceBuffer& src, void* dst, std::size_t bytes,
                       std::size_t src_offset);

  Result<KernelHandle> load_executable_impl(std::string_view name,
                                            std::span<const std::byte> code_object);
  Status launch_impl(const KernelHandle& kernel, const LaunchDims& dims,
                     const DispatchArgs& args);
  Status synchronize_impl();

  // No codegen: CPU fusion groups run through the interpreter.
  const graph::IKernelEmitter* emitter_impl() const noexcept { return nullptr; }
  const graph::IKernelCompiler* compiler_impl() const noexcept { return nullptr; }

 private:
  DeviceInfo info_;
  bool initialized_ = false;
};

}  // namespace lse::backend
