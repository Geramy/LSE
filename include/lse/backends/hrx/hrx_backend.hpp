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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hipc/comgr_compiler.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"

namespace lse::backend {

class HrxBackend : public Backend<HrxBackend> {
 public:
  static constexpr std::string_view kName = "hrx";

  ~HrxBackend();

  Status init_impl(int device_ordinal);
  void shutdown_impl() noexcept;
  const DeviceInfo& device_info_impl() const noexcept { return info_; }

  // Every GPU this runtime can drive, described without binding one. Static
  // because the answer cannot depend on a device already being bound, and
  // cheap for the same reason: it reads identities and asks the accelerator
  // for nothing else — no queue probe, no stream, no allocation. Bringing the
  // accelerator up is unavoidable (hrx_gpu_device_count is UNAVAILABLE before
  // it) and is idempotent, so enumerating first does not stop a later bind.
  static Result<std::vector<DeviceDescriptor>> enumerate_devices();

  // What a PCI path or UUID names, as an ordinal, WITHOUT bringing the
  // accelerator up. A pool named by a stable reference must register as a
  // spanning group before the first init() fixes the topology, and
  // enumerate_devices above is exactly what fixes it (it brings the
  // accelerator up with whatever group is registered at that moment — empty,
  // if this has not run yet, which is every GPU on the box). So a reference is
  // resolved the cheap way: the HSA runtime publishes each GPU agent's PCI and
  // UUID to a plain enumeration, no accelerator needed. The ordinal is the
  // GPU agent's index in hsa_iterate_agents order, the same join enumerate
  // devices makes and then checks by node. nullopt when the runtime is not
  // reachable or no agent answers for the reference.
  static std::optional<int> resolve_stable_ref(std::string_view pci,
                                               std::string_view uuid);

  Result<DeviceBuffer> allocate_impl(std::size_t bytes, MemoryClass cls,
                                     Stream stream);
  void deallocate_impl(DeviceBuffer& buf) noexcept;
  Result<std::size_t> sample_free_memory_impl() const;
  // The agent counter this device's dispatches run against — named and rated
  // from the device, though nothing hrx exposes can currently read a tick off
  // it. Both answers are stated here rather than left to the seam's generic
  // refusal so the missing piece is the one named at the call site.
  Result<DeviceClock> device_clock_impl() const;
  Result<DeviceTimestamp> sample_device_time_impl() const;

  Status copy_h2d_impl(const void* src, DeviceBuffer& dst, std::size_t bytes,
                       std::size_t dst_offset);
  // Peer to this device, no host bounce. Declines when the runtime refuses the
  // copy, which is what it does when the source's memory was never granted to
  // this agent.
  // The token every member of a spanning device stamps, or kNoDevice when this
  // instance holds a GPU of its own.
  Status copy_peer_impl(const DeviceBuffer& src, DeviceBuffer& dst,
                        std::size_t bytes, std::size_t src_offset,
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

  std::span<const graph::KernelToolchain> toolchains_impl() const noexcept {
    return toolchains_;
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
  LoomEmitter loom_emitter_;
  LoomcCompiler loom_compiler_;
  // Declared, not derived: a second dialect is one more entry, and the first
  // entry is what a caller with no dialect opinion gets. Points into this
  // object, which is why the backend is constructed in place and never moved —
  // the same reason DeviceInfo::extension can point at amd_.
  //
  // kHip stays first, so nothing that asks the device for "its" emitter sees a
  // change. Loom is declared beside it rather than chosen by a flag: a caller
  // that wants it looks the dialect up and either finds it or does not, which
  // is the whole of the negotiation.
  std::array<graph::KernelToolchain, 2> toolchains_{
      graph::KernelToolchain{graph::Dialect::kHip, &emitter_, &compiler_},
      graph::KernelToolchain{graph::Dialect::kLoom, &loom_emitter_,
                             &loom_compiler_}};
  void* device_ = nullptr;    // hrx_device_t
  void* allocator_ = nullptr; // hrx_allocator_t (borrowed)
  bool initialized_ = false;
  // hrx_stream_t per Stream index, sized to stream_caps_.stream_count and
  // filled on first use — an unused stream costs nothing, so the count can be
  // the device's ceiling rather than a guess at what the scheduler will want.
  std::vector<void*> streams_;
  // A staging buffer kept for the life of the backend, grown as needed.
  //
  // IREE's transfer helper allocates one of these per call for anything over
  // 64 KB and frees it again, which costs more than the transfer: measured at
  // 4 MB, that path runs about 1.3 GB/s where the copy engine does 42. Holding
  // one and driving the two halves -- a host memcpy and a queued copy -- keeps
  // the DMA and drops the allocation.
  void* staging_buffer_ = nullptr;  // hrx_buffer_t
  void* staging_host_ = nullptr;    // mapped host address of the above
  std::size_t staging_bytes_ = 0;
  Status ensure_staging(std::size_t bytes);
  Status dma_host_transfer(void* host, const DeviceBuffer& device,
                           std::size_t bytes, std::size_t device_offset,
                           bool to_device);
  // GPU index within hsa_iterate_agents order, which is what names this device
  // to the copy engine. Stamped at init.
  int gpu_ordinal_ = 0;
  Status copy_h2d_imported(const void* src, DeviceBuffer& dst,
                           std::size_t bytes, std::size_t dst_offset);
  // Queue affinity bit per stream. hrx_stream_dispatch submits its command
  // buffer with IREE_HAL_QUEUE_AFFINITY_ANY, which this HAL resolves by
  // first-set-bit to logical queue 0, so a batched stream cannot name a queue;
  // hrx_queue_dispatch can. One bit per stream is what puts two streams on two
  // AQL rings.
  std::vector<std::uint64_t> stream_affinity_;
  // Where an event goes when the last StreamEvent naming it is dropped. A
  // queued wait names the event object, so a drop alone cannot free it — the
  // wait may not have executed yet. It is freed at the next full-device
  // synchronize, the first point every queued wait has provably retired, or at
  // shutdown. Shared with the StreamEvent deleters so an event dropped after
  // this backend is gone releases itself instead of writing into a dead list.
  struct EventGraveyard {
    std::mutex mu;
    std::vector<void*> retired;
    bool backend_alive = true;
  };
  std::shared_ptr<EventGraveyard> graveyard_ =
      std::make_shared<EventGraveyard>();
  // Monotonic stamp for StreamEvent::timeline; identification only.
  std::uint64_t event_serial_ = 0;
  // Executables loaded on this device, released at shutdown. The seam hands
  // out KernelHandle as a raw ordinal pair with no unload call, so the load
  // path is where ownership has to be kept or the objects — each of which
  // retains the device — outlive every model that used them.
  std::vector<void*> loaded_executables_;
  // How many physical GPUs this device spans. 1 for an ordinary device; more
  // when a pool asked for one device over several, in which case stream i
  // drives GPU i.
  std::uint32_t physical_count_ = 1;
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
