#include "lse/backends/hrx/hrx_backend.hpp"

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

  hrx_stream_t stream = nullptr;
  LSE_RETURN_IF_ERROR(
      from_hrx(hrx_stream_create(device, 0, &stream), "hrx_stream_create"));
  stream_ = stream;

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
    amd_.wavefront_size = static_cast<std::uint8_t>(u32);
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_MAX_SHARED_MEMORY, &u32).ok()) {
    amd_.lds_bytes_per_workgroup = u32;
  }
  if (query_property(device, HRX_DEVICE_PROPERTY_CLOCK_RATE, &u32).ok()) {
    amd_.clock_khz = u32;
  }
  info_.ordinal = static_cast<std::uint8_t>(device_ordinal);

  // HRX values already on info_/amd_ stay. Tables fill only zeros.
  apply_arch_defaults(info_, amd_);
  info_.extension_id = AmdDeviceInfo::kExtensionId;
  info_.extension = &amd_;

  initialized_ = true;
  return OkStatus();
#endif
}

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
  // Stream-ordered: the device free waits for work already recorded on
  // stream_, matching hipFreeAsync.
  hrx_buffer_release(reinterpret_cast<hrx_buffer_t>(handle));
#else
  (void)handle;
#endif
}

void HrxBackend::shutdown_impl() noexcept {
#if LSE_HRX_LINKED
  if (stream_ != nullptr) {
    hrx_stream_release(static_cast<hrx_stream_t>(stream_));
    stream_ = nullptr;
  }
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
    LSE_RETURN_IF_ERROR(from_hrx(
        hrx_buffer_allocate(static_cast<hrx_stream_t>(stream_), bytes,
                            HRX_MEMORY_TYPE_DEVICE_LOCAL, usage, &buffer),
        "hrx_buffer_allocate"));
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
                               const DispatchArgs& args) {
#if !LSE_HRX_LINKED
  (void)kernel; (void)dims; (void)args;
  return LSE_ERROR(kUnimplemented, "libhrx not linked");
#else
  if (!kernel.valid()) return LSE_ERROR(kInvalidArgument, "invalid kernel handle");

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

  LSE_RETURN_IF_ERROR(from_hrx(
      hrx_stream_dispatch(static_cast<hrx_stream_t>(stream_),
                          reinterpret_cast<hrx_executable_t>(kernel.executable),
                          kernel.export_ordinal, &config,
                          args.constants.empty() ? nullptr : args.constants.data(),
                          args.constants.size(),
                          bindings.empty() ? nullptr : bindings.data(),
                          bindings.size(), args.flags),
      "hrx_stream_dispatch"));
  // Submit now so the GPU runs this kernel while the host records the next
  // one. Leaving everything in pending_cb until synchronize() keeps the
  // device idle for the whole emit/alloc walk.
  return from_hrx(hrx_stream_flush(static_cast<hrx_stream_t>(stream_)),
                  "hrx_stream_flush");
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
  return from_hrx(hrx_stream_synchronize(static_cast<hrx_stream_t>(stream_)),
                  "hrx_stream_synchronize");
#endif
}

}  // namespace lse::backend

LSE_REGISTER_BACKEND("hrx", ::lse::backend::HrxBackend)
