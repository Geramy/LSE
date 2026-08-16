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
                     const DispatchArgs& args, const DispatchTarget& target);
  Status synchronize_impl();
  Result<void*> device_pointer_impl(const DeviceBuffer& buf) const;

  // --- execution streams ---
  // Each Stream is a real hrx_stream_t: its own timeline semaphore and its own
  // open command buffer, so two streams are two submissions the device may run
  // at once. Ordering between them is a timeline wait, never a global sync.
  const StreamCapabilities& stream_capabilities_impl() const noexcept {
    return stream_caps_;
  }
  Result<StreamEvent> record_event_impl(Stream stream);
  Status wait_event_impl(Stream stream, const StreamEvent& event);
  Status synchronize_stream_impl(Stream stream);

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
  void* allocator_ = nullptr; // hrx_allocator_t (borrowed)
  bool initialized_ = false;
  // hrx_stream_t per Stream index, sized to stream_caps_.stream_count and
  // filled on first use — an unused stream costs nothing, so the count can be
  // the device's ceiling rather than a guess at what the scheduler will want.
  std::vector<void*> streams_;
  // Queue affinity bit per stream. hrx_stream_dispatch submits its command
  // buffer with IREE_HAL_QUEUE_AFFINITY_ANY, which this HAL resolves by
  // first-set-bit to logical queue 0, so a batched stream cannot name a queue;
  // hrx_queue_dispatch can. One bit per stream is what puts two streams on two
  // AQL rings.
  std::vector<std::uint64_t> stream_affinity_;
  // Logical queues this device's submission path can actually address, probed
  // at init rather than read off a header (see probe_queue_count).
  std::uint32_t queue_count_ = 1;
  StreamCapabilities stream_caps_;
  // Dispatches accumulate in each stream's open command buffer and are
  // submitted every flush_interval_ launches (LSE_FLUSH_INTERVAL; 1 = submit
  // per launch, 0 = only at sync/transfer boundaries). Counted per stream:
  // batching is a property of one command buffer, not of the device.
  std::uint32_t flush_interval_ = 0;
  std::vector<std::uint32_t> unflushed_launches_;

  Result<void*> stream_at(std::uint32_t index);
  Status flush_stream(std::uint32_t index);
  void adopt(DeviceBuffer& buf, std::uint64_t handle, std::size_t bytes);
  void release_buffer(std::uint64_t handle) noexcept;
};

}  // namespace lse::backend
