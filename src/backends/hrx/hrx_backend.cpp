#include "lse/backends/hrx/hrx_backend.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "lse/backends/hrx/arch_database.hpp"

extern "C" {
#include "hrx_runtime.h"
}

namespace lse::backend {

namespace {

// Translates an hrx_status_t into our Status, taking ownership of the message.
[[maybe_unused]] Status from_hrx(hrx_status_t status, const char* context) {
  if (hrx_status_is_ok(status)) return OkStatus();

  std::string detail;
  char* message = nullptr;
  std::size_t length = 0;
  hrx_status_t fmt = hrx_status_to_string(status, &message, &length);
  if (hrx_status_is_ok(fmt) && message != nullptr) {
    detail.assign(message, length);
  }
  if (message != nullptr) hrx_status_free_message(message);
  hrx_status_ignore(fmt);

  const hrx_status_code_t code = hrx_status_code(status);
  StatusCode mapped = StatusCode::kDeviceError;
  switch (code) {
    case HRX_STATUS_INVALID_ARGUMENT: mapped = StatusCode::kInvalidArgument; break;
    case HRX_STATUS_OUT_OF_MEMORY:    mapped = StatusCode::kOutOfMemory; break;
    case HRX_STATUS_NOT_FOUND:        mapped = StatusCode::kNotFound; break;
    case HRX_STATUS_UNIMPLEMENTED:    mapped = StatusCode::kUnimplemented; break;
    default:                          mapped = StatusCode::kDeviceError; break;
  }
  return Status(mapped, std::string(context) + ": " +
                            (detail.empty() ? "hrx error" : detail));
}

template <typename T>
[[maybe_unused]] Status query_property(hrx_device_t device, hrx_device_property_t prop, T* out) {
  return from_hrx(hrx_device_get_property(device, prop, out, sizeof(T)),
                  "hrx_device_get_property");
}

[[maybe_unused]] std::string query_string_property(hrx_device_t device,
                                  hrx_device_property_t prop) {
  char buffer[256] = {};
  hrx_status_t s = hrx_device_get_property(device, prop, buffer, sizeof(buffer));
  if (!hrx_status_is_ok(s)) {
    hrx_status_ignore(s);
    return {};
  }
  buffer[sizeof(buffer) - 1] = '\0';
  return std::string(buffer);
}

// One attribute of the `ordinal`-th GPU agent, read from the copy of
// libhsa-runtime64 that hrx has *already* loaded — dlopen by soname returns
// that same handle, never a second runtime. The three entry points below have
// been ABI-stable since HSA 1.0 and are declared here rather than by including
// <hsa/hsa.h>, which is not a build input of this backend. hsa_agent_get_info
// writes exactly the attribute's own width, so `out` must be that wide and
// there is no size to pass.
//
// False when the runtime is not reachable or the agent will not answer. Each
// caller has its own answer for not knowing; neither invents one.
bool agent_attribute(std::uint8_t ordinal, int attribute, void* out) noexcept {
  struct HsaAgent {
    std::uint64_t handle;
  };
  using HsaStatus = int;
  constexpr HsaStatus kSuccess = 0;
  constexpr int kInfoDevice = 17;
  constexpr int kDeviceTypeGpu = 1;

  void* lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (lib == nullptr) return false;

  using IterateFn = HsaStatus (*)(HsaStatus (*)(HsaAgent, void*), void*);
  using GetInfoFn = HsaStatus (*)(HsaAgent, int, void*);
  auto iterate = reinterpret_cast<IterateFn>(dlsym(lib, "hsa_iterate_agents"));
  auto get_info = reinterpret_cast<GetInfoFn>(dlsym(lib, "hsa_agent_get_info"));
  if (iterate == nullptr || get_info == nullptr) {
    dlclose(lib);
    return false;
  }

  struct Visit {
    GetInfoFn get_info;
    std::uint8_t want;
    std::uint8_t seen;
    int attribute;
    void* out;
    bool filled;
  } visit{get_info, ordinal, 0, attribute, out, false};

  iterate(
      [](HsaAgent agent, void* data) -> HsaStatus {
        auto* v = static_cast<Visit*>(data);
        int type = 0;
        if (v->get_info(agent, kInfoDevice, &type) != kSuccess) return kSuccess;
        if (type != kDeviceTypeGpu) return kSuccess;
        if (v->seen++ != v->want) return kSuccess;
        v->filled = v->get_info(agent, v->attribute, v->out) == kSuccess;
        return kSuccess;
      },
      &visit);

  dlclose(lib);
  return visit.filled;
}

// The agent's own answer to "how many queues may exist on you at once"
// (HSA_AGENT_INFO_QUEUES_MAX; 128 on gfx1151). hrx has no property for it.
// Returns 0 when the agent cannot be asked, which is not an error: the caller
// then bounds the stream count by the compute units alone.
std::uint32_t agent_queue_maximum(std::uint8_t ordinal) noexcept {
  constexpr int kInfoQueuesMax = 12;
  std::uint32_t queues = 0;
  if (!agent_attribute(ordinal, kInfoQueuesMax, &queues)) return 0;
  return queues;
}

#if LSE_HRX_LINKED
// Bytes free across every global pool this agent owns
// (HSA_AMD_AGENT_INFO_MEMORY_AVAIL, hsa_ext_amd.h). A live figure: it falls as
// any process on the box allocates and rises as they free. This is the same
// query hipMemGetInfo answers with, one layer further out than hrx.
bool agent_memory_available(std::uint8_t ordinal, std::uint64_t* out) noexcept {
  constexpr int kInfoMemoryAvail = 0xA015;
  return agent_attribute(ordinal, kInfoMemoryAvail, out);
}
#endif

// How many logical queues this device's submission path can address, asked of
// the device rather than read off a header.
//
// The AMDGPU HAL resolves a queue affinity by ANDing the request with the
// queues the logical device actually created and taking the first set bit
// (queue_affinity.c); a bit past the end normalizes to empty and the call
// fails. So walking the bits with a barrier that signals a scratch timeline
// asks exactly the question that matters — "does a submission carrying this
// bit reach a queue" — and the answer is the count of bits that do. Each such
// queue is its own AQL ring (one hsa_queue_create per host queue).
//
// Cheap: N empty barriers once, at init. Returns 1 if the probe cannot run,
// which is the answer that costs nothing to be wrong about.
#if LSE_HRX_LINKED
std::uint32_t probe_queue_count(hrx_device_t device) noexcept {
  hrx_semaphore_t semaphore = nullptr;
  if (!hrx_status_is_ok(hrx_semaphore_create(device, 0, &semaphore))) return 1;

  std::uint64_t value = 0;
  std::uint32_t count = 0;
  // 64 is the width of the affinity mask; the loop stops at the first bit the
  // device refuses, so this is bounded by the queue count in practice.
  for (std::uint32_t bit = 0; bit < 64; ++bit) {
    std::uint64_t next = value + 1;
    hrx_semaphore_list_t signals = {};
    signals.semaphores = &semaphore;
    signals.values = &next;
    signals.count = 1;
    const hrx_status_t status = hrx_queue_barrier(
        device, static_cast<hrx_queue_affinity_t>(1) << bit,
        /*wait_semaphores=*/nullptr, &signals);
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      break;
    }
    value = next;
    ++count;
  }
  if (value > 0) {
    hrx_status_ignore(hrx_semaphore_wait(semaphore, value, UINT64_MAX));
  }
  hrx_semaphore_release(semaphore);
  return count != 0 ? count : 1u;
}
#endif

// How many execution streams this device is worth handing out.
//
// One per addressable hardware queue, and no more. A stream past that shares a
// ring with another, so it cannot overlap with it — it would only cost the
// events the scheduler spends to spread onto it. `queues` is what the probe
// found; the agent's queue maximum and the compute units still bound it,
// because a queue with no CU to run on is a submission that returns nothing.
StreamCapabilities derive_stream_capabilities(const DeviceInfo& info,
                                              std::uint32_t queues) noexcept {
  StreamCapabilities caps;
  const std::uint32_t agent_max = agent_queue_maximum(info.ordinal);
  std::uint32_t n = queues != 0 ? queues : 1u;
  if (agent_max != 0) n = std::min(n, agent_max);
  if (info.compute_units != 0) {
    n = std::min(n, static_cast<std::uint32_t>(info.compute_units));
  }
  caps.stream_count = std::max(n, 1u);
  // Every stream this backend hands out has its own AQL ring, so every one of
  // them can run at the same time as the others. That is the probe's answer,
  // not a promise from the header, and rocprofv3 agrees: spreading a decode
  // step across two rings produced 1.73 ms of genuinely concurrent kernel time
  // per token on two distinct Queue_Ids, where the single-ring path produced
  // 0.000 ms.
  caps.concurrent_streams = caps.stream_count;
  caps.needs_explicit_events = true;
  // ...and it still does not pay, which is a different question and belongs to
  // the cost model rather than to the capability.
  //
  // Only one stream can use the batched path: hrx_stream_dispatch accumulates
  // into a command buffer whose packets the hardware walks with the barrier bit
  // already set, but hrx submits it with IREE_HAL_QUEUE_AFFINITY_ANY and cannot
  // name a queue. Every other stream must go through hrx_queue_dispatch, one
  // submission and one completion-signal round trip per kernel. Measured on
  // gfx1151, -n 32, median of 5: 93.7 tok/s batched vs 79.8 tok/s when the same
  // single-stream step goes through the queue path — 3.06 us per dispatch
  // against a 7.3 us median kernel.
  //
  // Overlap cannot win that back here, because the memory system was already
  // the limit: the largest decode kernel reads 508 MB in 2.1 ms, which is
  // 242 GB/s on a part whose roof is about that. Kernels run beside each other
  // therefore take longer by about what the concurrency saves — per token,
  // 598 kernels summing 8.51 ms ordered against 10.33 ms spread, for 1.735 ms
  // of concurrency won and 8.51 vs 8.62 ms of device busy time. End to end, a
  // balanced 297/295 split cost 93.7 -> 80.9 tok/s.
  //
  // So the seam is real and the planner is right to decline. This flips the day
  // a stream can carry a queue affinity into the batched path — a libhrx
  // change, not an LSE one — and nothing above this line moves when it does.
  caps.uniform_launch_cost = false;
  // A dispatch is a grid, and a grid cannot be cut. Flips when a kernel
  // declares work items instead; nothing else about the seam changes.
  caps.splittable_work = false;
  return caps;
}

}  // namespace

bool HrxBackend::available() noexcept {
#if LSE_HRX_LINKED
  return true;
#else
  return false;
#endif
}

HrxBackend::~HrxBackend() { shutdown_impl(); }

Status HrxBackend::init_impl(int device_ordinal) {
#if !LSE_HRX_LINKED
  (void)device_ordinal;
  return LSE_ERROR(kUnimplemented,
                   "this build was configured with HRX headers but libhrx was "
                   "not linked; build hrx-system and reconfigure with "
                   "-DLSE_HRX_ROOT=<install prefix>");
#else
  LSE_RETURN_IF_ERROR(from_hrx(hrx_gpu_initialize(0), "hrx_gpu_initialize"));

  int count = 0;
  LSE_RETURN_IF_ERROR(from_hrx(hrx_gpu_device_count(&count), "hrx_gpu_device_count"));
  if (device_ordinal < 0 || device_ordinal >= count) {
    return LSE_ERROR(kInvalidArgument, "device ordinal ",
                     std::to_string(device_ordinal), " out of range; ",
                     std::to_string(count), " HRX device(s) present");
  }

  hrx_device_t device = nullptr;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_gpu_device_get(device_ordinal, &device), "hrx_gpu_device_get"));
  device_ = device;

  // Borrowed reference, valid for the device's lifetime — do not release.
  allocator_ = hrx_device_allocator(device);

  // --- device info ---
  info_ = DeviceInfo{};
  info_.ordinal = static_cast<std::uint8_t>(device_ordinal);
  info_.name = query_string_property(device, HRX_DEVICE_PROPERTY_NAME);
  info_.arch = query_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);

  std::uint64_t total_memory = 0;
  if (query_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &total_memory).ok()) {
    info_.total_memory = static_cast<std::size_t>(total_memory);
  }
  std::uint32_t u32 = 0;
  if (query_property(device, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &u32).ok()) {
    info_.compute_units = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE, &u32).ok()) {
    info_.max_threads_per_workgroup = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_WARP_SIZE, &u32).ok()) {
    info_.wavefront_size = static_cast<std::uint16_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_MAX_SHARED_MEMORY, &u32).ok()) {
    info_.lds_bytes_per_workgroup = u32;
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_CLOCK_RATE, &u32).ok()) {
    amd_.clock_khz = u32;
  }
  info_.ordinal = static_cast<std::uint8_t>(device_ordinal);

  // HRX values already on info_/amd_ stay. Tables fill only zeros.
  apply_arch_defaults(info_, amd_);
  info_.extension_id = AmdDeviceInfo::kExtensionId;
  info_.extension = &amd_;

  flush_interval_ = 16;
  if (const char* env = std::getenv("LSE_FLUSH_INTERVAL");
      env != nullptr && env[0] != '\0') {
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && v >= 0) {
      flush_interval_ = static_cast<std::uint32_t>(v);
    }
  }

  queue_count_ = probe_queue_count(device);
  stream_caps_ = derive_stream_capabilities(info_, queue_count_);
  streams_.assign(stream_caps_.stream_count, nullptr);
  unflushed_launches_.assign(stream_caps_.stream_count, 0);
  stream_affinity_.resize(stream_caps_.stream_count);
  for (std::uint32_t i = 0; i < stream_caps_.stream_count; ++i) {
    stream_affinity_[i] = static_cast<std::uint64_t>(1) << (i % queue_count_);
  }
  // Stream 0 exists from the start: every path that does not name a stream
  // uses it, including the stream-ordered allocator.
  LSE_RETURN_IF_ERROR(stream_at(0).status());

  initialized_ = true;
  return OkStatus();
#endif
}

Result<void*> HrxBackend::stream_at(std::uint32_t index) {
#if !LSE_HRX_LINKED
  (void)index;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  if (streams_[index] != nullptr) return streams_[index];
  hrx_stream_t stream = nullptr;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_stream_create(static_cast<hrx_device_t>(device_), 0, &stream),
               "hrx_stream_create"));
  streams_[index] = stream;
  return streams_[index];
#endif
}

#if LSE_HRX_LINKED
Status HrxBackend::flush_stream(std::uint32_t index) {
  if (index >= streams_.size() || streams_[index] == nullptr) return OkStatus();
  unflushed_launches_[index] = 0;
  return from_hrx(hrx_stream_flush(static_cast<hrx_stream_t>(streams_[index])),
                  "hrx_stream_flush");
}

#else
Status HrxBackend::flush_stream(std::uint32_t) {
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
}
#endif

void HrxBackend::adopt(DeviceBuffer& buf, std::uint64_t handle,
                       std::size_t bytes) {
  buf.handle = handle;
  buf.size_bytes = bytes;
  buf.storage = std::shared_ptr<void>(
      reinterpret_cast<void*>(handle), [this](void* p) {
        release_buffer(reinterpret_cast<std::uint64_t>(p));
      });
}

void HrxBackend::release_buffer(std::uint64_t handle) noexcept {
#if LSE_HRX_LINKED
  if (handle == 0) return;
  // Safe against in-flight AND still-unflushed dispatches: hrx command
  // buffers are ONE_SHOT (not UNRETAINED), so recording a dispatch inserts
  // its buffers into the CB's resource set, which keeps the hal buffer (and
  // its backing pool) alive until the CB itself retires. This release only
  // drops our reference.
  hrx_buffer_release(reinterpret_cast<hrx_buffer_t>(handle));
#else
  (void)handle;
#endif
}

void HrxBackend::shutdown_impl() noexcept {
#if LSE_HRX_LINKED
  for (void*& s : streams_) {
    if (s == nullptr) continue;
    hrx_stream_release(static_cast<hrx_stream_t>(s));
    s = nullptr;
  }
  streams_.clear();
  unflushed_launches_.clear();
  if (device_ != nullptr) {
    hrx_device_release(static_cast<hrx_device_t>(device_));
    device_ = nullptr;
  }
  allocator_ = nullptr;
  if (initialized_) {
    hrx_status_ignore(hrx_gpu_shutdown());
    initialized_ = false;
  }
#endif
}

Result<DeviceBuffer> HrxBackend::allocate_impl(std::size_t bytes,
                                               MemoryClass cls) {
#if !LSE_HRX_LINKED
  (void)bytes;
  (void)cls;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  if (bytes == 0) return LSE_ERROR(kInvalidArgument, "zero-size allocation");

  const bool staging = cls == MemoryClass::kStaging;
  const hrx_buffer_usage_t usage =
      HRX_BUFFER_USAGE_DISPATCH_STORAGE | HRX_BUFFER_USAGE_TRANSFER |
      (staging ? (HRX_BUFFER_USAGE_MAPPING_SCOPED |
                  HRX_BUFFER_USAGE_MAPPING_PERSISTENT)
               : 0);

  hrx_buffer_t buffer = nullptr;
  if (!staging) {
    // hipMallocAsync: ordered against work already on this stream.
    // hrx_buffer_allocate flushes the stream internally (libhrx buffer.c), so
    // an allocation mid-batch submits the open command buffer for us.
    //
    // The other streams are deliberately not flushed. A one-shot command
    // buffer inserts every buffer it dispatches against into its own resource
    // set at *record* time, so memory another stream still references cannot
    // be handed back to the pool here, flushed or not.
    auto stream = stream_at(0);
    if (!stream.ok()) return stream.status();
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_buffer_allocate(static_cast<hrx_stream_t>(*stream), bytes,
                            HRX_MEMORY_TYPE_DEVICE_LOCAL, usage, &buffer),
        "hrx_buffer_allocate"));
    unflushed_launches_[0] = 0;
  } else {
    // Staging has to stay mapped; the stream-ordered path is device-local.
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_COHERENT |
                  HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = HRX_MEMORY_ACCESS_ALL;
    params.usage = usage;
    params.queue_affinity = 0;
    LSE_RETURN_IF_ERROR(
        from_hrx(hrx_allocator_allocate_buffer(
                     static_cast<hrx_allocator_t>(allocator_), params, bytes,
                     &buffer),
                 "hrx_allocator_allocate_buffer"));
  }

  DeviceBuffer out;
  const auto handle = reinterpret_cast<std::uint64_t>(buffer);
  out.size_bytes = bytes;

  // A staging buffer stays mapped for its whole lifetime; a device-local one
  // has no host address at all, and reaching it means copy_h2d/copy_d2h.
  if (staging) {
    void* mapped = nullptr;
    const hrx_status_t mapped_status = hrx_buffer_map(
        buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, bytes, &mapped);
    if (!hrx_status_is_ok(mapped_status)) {
      hrx_buffer_release(buffer);
      return from_hrx(mapped_status, "hrx_buffer_map");
    }
    out.ptr = mapped;
    out.handle = handle;
    // Staging is not pooled: the mapping has to die with the buffer.
    out.storage = std::shared_ptr<void>(
        reinterpret_cast<void*>(handle), [](void* p) {
#if LSE_HRX_LINKED
          auto* b = reinterpret_cast<hrx_buffer_t>(p);
          hrx_status_ignore(hrx_buffer_unmap(b));
          hrx_buffer_release(b);
#else
          (void)p;
#endif
        });
    return out;
  }

  adopt(out, handle, bytes);
  return out;
#endif
}

void HrxBackend::deallocate_impl(DeviceBuffer& buf) noexcept {
  buf.storage.reset();
  buf.handle = 0;
  buf.ptr = nullptr;
  buf.size_bytes = 0;
}

Result<std::size_t> HrxBackend::sample_free_memory_impl() const {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // hrx_device_memory_info rejects a null total_bytes, so the declared total
  // is read and dropped — info_.total_memory already carries it, and it is a
  // different quantity from the one being sampled here.
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const hrx_status_t status = hrx_device_memory_info(
      static_cast<hrx_device_t>(device_), &free_bytes, &total_bytes);
  if (hrx_status_is_ok(status)) return free_bytes;
  // UNAVAILABLE is this HAL declining to publish an availability observation,
  // which is an absent query and not a broken device — from_hrx would fold it
  // into kDeviceError and a caller that stops on device errors would turn an
  // unanswerable question into a failed run. It is also the answer on every
  // device LSE runs on today: the AMDGPU HAL populates its memory observation
  // from the device spec and therefore sets the total only
  // (iree/hal/drivers/amdgpu/logical_device.c). The agent underneath still
  // knows, so ask it before giving up; hrx stays first so that the day the HAL
  // publishes the figure it is the one that is used.
  if (hrx_status_code(status) != HRX_STATUS_UNAVAILABLE) {
    return from_hrx(status, "hrx_device_memory_info");
  }
  hrx_status_ignore(status);
  std::uint64_t available = 0;
  if (!agent_memory_available(info_.ordinal, &available)) {
    return LSE_ERROR(kUnimplemented,
                     "neither this device's HAL nor its agent publishes an "
                     "available-memory figure");
  }
  return static_cast<std::size_t>(available);
#endif
}

Status HrxBackend::copy_h2d_impl(const void* src, DeviceBuffer& dst,
                                 std::size_t bytes, std::size_t dst_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)dst_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (src == nullptr || dst.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_h2d");
  }
  if (dst_offset + bytes > dst.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_h2d writes past the end of the buffer");
  }
  // hrx_synchronous_* is a separate device submission carrying no semaphore
  // dependency on any stream, and a HAL queue does not order entries by
  // submission — only by semaphore edges. Flushing is therefore not enough:
  // the streams have to be *waited* for, or the transfer races whatever they
  // still hold on this buffer. With one stream that race was invisible; with
  // two it is not, and it is the same race either way.
  LSE_RETURN_IF_ERROR(synchronize_impl());
  return from_hrx(hrx_synchronous_h2d(static_cast<hrx_device_t>(device_), src,
                                      reinterpret_cast<hrx_buffer_t>(dst.handle),
                                      dst.offset + dst_offset, bytes),
                  "hrx_synchronous_h2d");
#endif
}

Status HrxBackend::copy_d2h_impl(const DeviceBuffer& src, void* dst,
                                 std::size_t bytes, std::size_t src_offset) {
#if !LSE_HRX_LINKED
  (void)src; (void)dst; (void)bytes; (void)src_offset;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (dst == nullptr || src.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_d2h");
  }
  if (src_offset + bytes > src.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_d2h reads past the end of the buffer");
  }
  // Same hazard as copy_h2d: the transfer is not ordered against any stream
  // by anything but this wait.
  LSE_RETURN_IF_ERROR(synchronize_impl());
  return from_hrx(hrx_synchronous_d2h(static_cast<hrx_device_t>(device_),
                                      reinterpret_cast<hrx_buffer_t>(src.handle),
                                      src.offset + src_offset, dst, bytes),
                  "hrx_synchronous_d2h");
#endif
}

Result<KernelHandle> HrxBackend::load_executable_impl(
    std::string_view name, std::span<const std::byte> code_object) {
#if !LSE_HRX_LINKED
  (void)name; (void)code_object;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (code_object.empty()) {
    return LSE_ERROR(kInvalidArgument, "empty code object");
  }

  hrx_executable_t executable = nullptr;
  // target_family/target_key tell HRX which advertised device target this code
  // object was built for; the arch string is exactly that key.
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_executable_load_data(static_cast<hrx_device_t>(device_),
                               code_object.data(), code_object.size(), "amdgpu",
                               info_.arch.c_str(), &executable),
      "hrx_executable_load_data"));

  std::size_t export_count = 0;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_executable_export_count(executable, &export_count),
      "hrx_executable_export_count"));
  if (export_count == 0) {
    hrx_executable_release(executable);
    return LSE_ERROR(kCompileError, "code object for '", std::string(name),
                     "' advertises no exports");
  }

  // A code object may hold many kernels (one source file can define several
  // primitives plus shared helpers), so resolve by name rather than assuming
  // ordinal 0. Fall back to the sole export when there is exactly one.
  const std::string entry(name);
  std::uint32_t ordinal = 0;
  hrx_status_t looked_up =
      hrx_executable_lookup_export_by_name(executable, entry.c_str(), &ordinal);
  if (!hrx_status_is_ok(looked_up)) {
    hrx_status_ignore(looked_up);
    if (export_count != 1) {
      std::string available;
      for (std::uint32_t i = 0; i < export_count; ++i) {
        hrx_executable_export_info_t info = {};
        if (hrx_status_is_ok(hrx_executable_export_info(executable, i, &info)) &&
            info.name != nullptr) {
          if (!available.empty()) available += ", ";
          available += info.name;
        }
      }
      hrx_executable_release(executable);
      return LSE_ERROR(kNotFound, "no export '", entry, "' in code object; has: ",
                       available);
    }
  }

  KernelHandle handle;
  handle.executable = reinterpret_cast<std::uint64_t>(executable);
  handle.export_ordinal = ordinal;
  handle.name = entry;
  return handle;
#endif
}

Status HrxBackend::launch_impl(const KernelHandle& kernel, const LaunchDims& dims,
                               const DispatchArgs& args,
                               const DispatchTarget& target) {
#if !LSE_HRX_LINKED
  (void)kernel; (void)dims; (void)args; (void)target;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!kernel.valid()) return LSE_ERROR(kInvalidArgument, "invalid kernel handle");
  if (!target.work.whole()) {
    // A dispatch is still a grid here, and a grid has no subrange. When a
    // kernel declares work items the range lands in the dispatch config and
    // this refusal goes away; nothing else on this path changes.
    return LSE_ERROR(kUnimplemented,
                     "hrx dispatch covers a whole grid; work ranges arrive "
                     "with work-item kernels");
  }
  auto stream = stream_at(target.stream.index);
  if (!stream.ok()) return stream.status();

  hrx_dispatch_config_t config = {};
  for (int i = 0; i < 3; ++i) {
    config.workgroup_count[i] = dims.workgroup_count[i];
    config.workgroup_size[i] = dims.workgroup_size[i];
  }
  config.subgroup_size = dims.subgroup_size;

  std::vector<hrx_buffer_ref_t> bindings;
  bindings.reserve(args.bindings.size());
  for (const BufferRef& ref : args.bindings) {
    if (ref.buffer == nullptr || ref.buffer->handle == 0) {
      return LSE_ERROR(kInvalidArgument, "null buffer binding in dispatch");
    }
    hrx_buffer_ref_t out = {};
    out.buffer = reinterpret_cast<hrx_buffer_t>(ref.buffer->handle);
    out.offset = ref.buffer->offset + ref.offset;
    out.length = ref.length != 0 ? ref.length : ref.buffer->size_bytes;
    bindings.push_back(out);
  }

  const std::uint32_t index = target.stream.index;
  auto* s = static_cast<hrx_stream_t>(*stream);

  // Two submission shapes, and which one a stream gets is a property of the
  // runtime rather than a policy:
  //
  //   - a command buffer batches many dispatches behind one submission and one
  //     semaphore hop, and the hardware walks its packets with the barrier bit
  //     already set — no host round trip between kernels. But hrx submits it
  //     with IREE_HAL_QUEUE_AFFINITY_ANY, which this HAL resolves by
  //     first-set-bit to ring 0, so it cannot name a queue. Exactly one stream
  //     can use it, and that stream is 0.
  //   - hrx_queue_dispatch names a queue, which is what puts a second stream on
  //     a second AQL ring, but it submits one dispatch at a time and pays a
  //     completion-signal round trip per kernel (measured ~3 us on gfx1151).
  //
  // So the chain, which is most of the work, keeps the cheap path, and only
  // the groups the planner deliberately moves off it pay for the ring change.
  if (index == 0) {
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_stream_dispatch(s,
                            reinterpret_cast<hrx_executable_t>(kernel.executable),
                            kernel.export_ordinal, &config,
                            args.constants.empty() ? nullptr : args.constants.data(),
                            args.constants.size(),
                            bindings.empty() ? nullptr : bindings.data(),
                            bindings.size(), args.flags),
        "hrx_stream_dispatch"));
    // The periodic flush keeps the GPU fed while the host records the rest of
    // the token's launches. Do not reach for hrx_stream_begin_capture or
    // hrx_graph_exec_update to go further: both are UNIMPLEMENTED stubs in
    // libhrx (graph.c) as of this writing.
    if (flush_interval_ != 0 &&
        ++unflushed_launches_[index] >= flush_interval_) {
      return flush_stream(index);
    }
    return OkStatus();
  }

  // Order inside the stream is the timeline: wait on the position this stream
  // is at, signal the next. That is the same guarantee the command buffer's
  // dispatch->dispatch barrier gives, spelled with a semaphore instead, and it
  // is required rather than implied — this HAL states outright that queue
  // entries are not ordered by submission (host_queue_waits.c).
  hrx_timeline_point_t self = {};
  LSE_RETURN_IF_ERROR(from_hrx(hrx_stream_get_timeline_position(s, &self),
                               "hrx_stream_get_timeline_position"));
  std::uint64_t next = 0;
  LSE_RETURN_IF_ERROR(from_hrx(hrx_stream_advance_timeline(s, &next),
                               "hrx_stream_advance_timeline"));

  hrx_semaphore_list_t waits = {};
  waits.semaphores = &self.semaphore;
  waits.values = &self.value;
  waits.count = self.value > 0 ? 1 : 0;
  hrx_semaphore_list_t signals = {};
  signals.semaphores = &self.semaphore;
  signals.values = &next;
  signals.count = 1;

  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_queue_dispatch(static_cast<hrx_device_t>(device_),
                         stream_affinity_[index], &waits, &signals,
                         reinterpret_cast<hrx_executable_t>(kernel.executable),
                         kernel.export_ordinal, &config,
                         args.constants.empty() ? nullptr : args.constants.data(),
                         args.constants.size(),
                         bindings.empty() ? nullptr : bindings.data(),
                         bindings.size(), args.flags),
      "hrx_queue_dispatch"));
  unflushed_launches_[index] = 0;
  return OkStatus();
#endif
}

Result<StreamEvent> HrxBackend::record_event_impl(Stream stream) {
#if !LSE_HRX_LINKED
  (void)stream;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (stream.index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(stream.index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  // A stream nobody has used has nothing to wait for. The invalid event says
  // exactly that, and wait_event treats it as satisfied.
  if (streams_[stream.index] == nullptr) return StreamEvent{stream, 0};

  // The timeline only advances on submit, so the batch has to go now — this
  // is the one place batching yields, and it yields because a consumer is
  // about to depend on the result.
  LSE_RETURN_IF_ERROR(flush_stream(stream.index));

  hrx_timeline_point_t point = {};
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_stream_get_timeline_position(
          static_cast<hrx_stream_t>(streams_[stream.index]), &point),
      "hrx_stream_get_timeline_position"));
  return StreamEvent{stream, point.value};
#endif
}

Status HrxBackend::wait_event_impl(Stream stream, const StreamEvent& event) {
#if !LSE_HRX_LINKED
  (void)stream; (void)event;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!event.valid()) return OkStatus();
  // Work on one stream is already ordered against itself.
  if (event.stream == stream) return OkStatus();
  if (event.stream.index >= streams_.size() ||
      streams_[event.stream.index] == nullptr) {
    return LSE_ERROR(kInvalidArgument, "event names a stream that has no work");
  }
  auto waiter = stream_at(stream.index);
  if (!waiter.ok()) return waiter.status();
  auto* waiting = static_cast<hrx_stream_t>(*waiter);

  hrx_semaphore_t produced = nullptr;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_stream_get_semaphore(
          static_cast<hrx_stream_t>(streams_[event.stream.index]), &produced),
      "hrx_stream_get_semaphore"));

  // Everything already recorded here has to be submitted before the barrier
  // that follows it, or the barrier would be ordered ahead of it.
  LSE_RETURN_IF_ERROR(flush_stream(stream.index));

  hrx_timeline_point_t self = {};
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_stream_get_timeline_position(waiting, &self),
               "hrx_stream_get_timeline_position"));

  std::uint64_t next = 0;
  LSE_RETURN_IF_ERROR(from_hrx(hrx_stream_advance_timeline(waiting, &next),
                               "hrx_stream_advance_timeline"));

  // The barrier waits on the producer AND on this stream's own current
  // position before signalling the position everything after it will wait on.
  //
  // hrx_stream_wait_on() would be the obvious call and is wrong here: it omits
  // the second wait, so on a stream with work still in flight the barrier can
  // signal position+1 while position itself is unreached. A timeline that
  // jumps ahead of its own work releases every later wait early — including
  // the one inside hrx_stream_synchronize — and the stream silently stops
  // being ordered against itself.
  hrx_semaphore_t wait_semaphores[2] = {produced, self.semaphore};
  std::uint64_t wait_values[2] = {event.timeline, self.value};
  hrx_semaphore_list_t waits = {};
  waits.semaphores = wait_semaphores;
  waits.values = wait_values;
  waits.count = self.value > 0 ? 2 : 1;

  hrx_semaphore_t signal_semaphore = self.semaphore;
  std::uint64_t signal_value = next;
  hrx_semaphore_list_t signals = {};
  signals.semaphores = &signal_semaphore;
  signals.values = &signal_value;
  signals.count = 1;

  // On the waiting stream's own queue: a barrier submitted anywhere else would
  // order that queue instead of this one.
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_queue_barrier(static_cast<hrx_device_t>(device_),
                                 stream_affinity_[stream.index], &waits,
                                 &signals),
               "hrx_queue_barrier"));
  unflushed_launches_[stream.index] = 0;
  return OkStatus();
#endif
}

Status HrxBackend::synchronize_stream_impl(Stream stream) {
#if !LSE_HRX_LINKED
  (void)stream;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  if (stream.index >= streams_.size()) {
    return LSE_ERROR(kOutOfRange, "stream ", std::to_string(stream.index),
                     " past the ", std::to_string(streams_.size()),
                     " this device offers");
  }
  if (streams_[stream.index] == nullptr) return OkStatus();
  unflushed_launches_[stream.index] = 0;
  return from_hrx(
      hrx_stream_synchronize(static_cast<hrx_stream_t>(streams_[stream.index])),
      "hrx_stream_synchronize");
#endif
}

Result<void*> HrxBackend::device_pointer_impl(const DeviceBuffer& buf) const {
#if !LSE_HRX_LINKED
  (void)buf;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (buf.handle == 0) {
    return LSE_ERROR(kInvalidArgument, "null buffer in device_pointer");
  }
  void* p = nullptr;
  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_buffer_get_device_ptr(reinterpret_cast<hrx_buffer_t>(buf.handle), &p),
      "hrx_buffer_get_device_ptr"));
  return static_cast<void*>(static_cast<std::byte*>(p) + buf.offset);
#endif
}

Status HrxBackend::synchronize_impl() {
#if !LSE_HRX_LINKED
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!initialized_) return LSE_ERROR(kInternal, "hrx backend not initialized");
  // Every stream, because "the device is idle" is the only thing a caller can
  // mean by synchronize() with no stream named. hrx_stream_synchronize flushes
  // the open command buffer itself before waiting (libhrx stream.c), and a
  // stream that was never used is skipped inside synchronize_impl(Stream).
  for (std::uint32_t i = 0; i < streams_.size(); ++i) {
    LSE_RETURN_IF_ERROR(synchronize_stream_impl(Stream{i}));
  }
  return OkStatus();
#endif
}

}  // namespace lse::backend

LSE_REGISTER_BACKEND("hrx", ::lse::backend::HrxBackend)
