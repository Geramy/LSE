// HRX backend — native low-latency path.
//
// Binds the native HRX C ABI directly: hrx_gpu_* for lifecycle,
// hrx_buffer_allocate on the dispatch stream for device memory (stream-ordered,
// same contract as hipMallocAsync/hipFreeAsync), hrx_stream_dispatch for
// launch, and hrx_executable_load_data for code objects. Staging maps still
// go through the device allocator. The HIP compatibility layer that HRX also
// ships (libamdhip64.so) is intentionally not used — it exists to run
// unmodified HIP binaries, and routing through it would discard the
// low-latency dispatch path that is the whole reason to be on HRX.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/comgr_compiler.hpp"
#include "lse/backends/hrx/hip_emitter.hpp"

namespace lse::backend {

class HrxBackend : public Backend<HrxBackend> {
 public:
  static constexpr std::string_view kName = "hrx";

  ~HrxBackend();

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
  Result<void*> device_pointer_impl(const DeviceBuffer& buf) const;

  const graph::IKernelEmitter* emitter_impl() const noexcept { return &emitter_; }
  const graph::IKernelCompiler* compiler_impl() const noexcept {
    return &compiler_;
  }

  // True when this build actually linked libhrx. When false the backend
  // compiles and type-checks against the real header but refuses to init, so a
  // half-built HRX cannot masquerade as a working device.
  static bool available() noexcept;

 private:
  DeviceInfo info_;
  // Owned so DeviceInfo::extension stays valid for the backend's lifetime.
  AmdDeviceInfo amd_{};
  HipEmitter emitter_;
  ComgrCompiler compiler_;
  void* device_ = nullptr;    // hrx_device_t
  void* stream_ = nullptr;    // hrx_stream_t
  void* allocator_ = nullptr; // hrx_allocator_t (borrowed)
  bool initialized_ = false;
  // Dispatches accumulate in the stream's open command buffer and are
  // submitted every flush_interval_ launches (LSE_FLUSH_INTERVAL; 1 = submit
  // per launch, 0 = only at sync/transfer boundaries). Counts launches
  // recorded since the last submit.
  std::uint32_t flush_interval_ = 0;
  std::uint32_t unflushed_launches_ = 0;

  Status flush_pending();
  void adopt(DeviceBuffer& buf, std::uint64_t handle, std::size_t bytes);
  void release_buffer(std::uint64_t handle) noexcept;
};

}  // namespace lse::backend
