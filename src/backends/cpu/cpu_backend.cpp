#include "cpu_backend.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

namespace lse::backend {

namespace {
// Match the widest vector load the host is likely to use, so reference kernels
// can assume alignment the same way device kernels do.
constexpr std::size_t kAlignment = 64;
}  // namespace

Status CpuBackend::init_impl(int device_ordinal) {
  if (device_ordinal != 0) {
    return LSE_ERROR(kInvalidArgument,
                     "cpu backend has exactly one device (ordinal 0)");
  }

  info_ = DeviceInfo{};
  info_.name = "host cpu";
  info_.arch = "host";
  info_.ordinal = 0;
  info_.unified_memory = true;
  // No vendor extension: nothing here has a wavefront or an LDS budget.
  info_.extension_id = {};
  info_.extension = nullptr;  // emulated in software by the reference kernels

  const unsigned hw = std::thread::hardware_concurrency();
  info_.compute_units = static_cast<std::uint16_t>(hw ? hw : 1u);
  info_.max_threads_per_workgroup = 1;


  initialized_ = true;
  return OkStatus();
}

void CpuBackend::shutdown_impl() noexcept { initialized_ = false; }

Result<DeviceBuffer> CpuBackend::allocate_impl(std::size_t bytes,
                                              MemoryClass cls) {
  // Host and device are the same memory here, so the class changes nothing.
  (void)cls;
  if (bytes == 0) return LSE_ERROR(kInvalidArgument, "zero-size allocation");
  // Round up so the tail of the buffer is safe to touch with aligned vector
  // stores, mirroring how device allocators over-allocate.
  const std::size_t padded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
  void* ptr = std::aligned_alloc(kAlignment, padded);
  if (ptr == nullptr) {
    return LSE_ERROR(kOutOfMemory, "failed to allocate ", std::to_string(bytes),
                     " bytes on the host");
  }
  DeviceBuffer buf;
  buf.ptr = ptr;
  buf.size_bytes = bytes;
  buf.handle = reinterpret_cast<std::uint64_t>(ptr);
  return buf;
}

void CpuBackend::deallocate_impl(DeviceBuffer& buf) noexcept {
  std::free(buf.ptr);
  buf.ptr = nullptr;
  buf.handle = 0;
  buf.size_bytes = 0;
}

Status CpuBackend::copy_h2d_impl(const void* src, DeviceBuffer& dst,
                                 std::size_t bytes, std::size_t dst_offset) {
  if (src == nullptr || !dst.valid()) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_h2d");
  }
  if (dst_offset + bytes > dst.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_h2d writes past the end of the buffer");
  }
  std::memcpy(static_cast<std::byte*>(dst.ptr) + dst.offset + dst_offset, src,
              bytes);
  return OkStatus();
}

Status CpuBackend::copy_d2h_impl(const DeviceBuffer& src, void* dst,
                                 std::size_t bytes, std::size_t src_offset) {
  if (dst == nullptr || !src.valid()) {
    return LSE_ERROR(kInvalidArgument, "null buffer in copy_d2h");
  }
  if (src_offset + bytes > src.size_bytes) {
    return LSE_ERROR(kOutOfRange, "copy_d2h reads past the end of the buffer");
  }
  std::memcpy(dst, static_cast<const std::byte*>(src.ptr) + src.offset + src_offset,
              bytes);
  return OkStatus();
}

Result<KernelHandle> CpuBackend::load_executable_impl(
    std::string_view name, std::span<const std::byte> code_object) {
  (void)name;
  (void)code_object;
  // By design. The CPU backend executes fusion groups through the reference
  // interpreter, so there is no code object to load. Callers that see this are
  // routing JIT output to the wrong backend.
  return LSE_ERROR(kUnimplemented,
                   "cpu backend executes graphs via the reference interpreter; "
                   "it does not load code objects");
}

Status CpuBackend::launch_impl(const KernelHandle& kernel,
                               const LaunchDims& dims, const DispatchArgs& args) {
  (void)kernel;
  (void)dims;
  (void)args;
  return LSE_ERROR(kUnimplemented,
                   "cpu backend does not dispatch code objects; use the "
                   "reference interpreter path");
}

Status CpuBackend::synchronize_impl() {
  // Host work is synchronous; nothing is ever in flight.
  return initialized_ ? OkStatus()
                      : LSE_ERROR(kInternal, "cpu backend is not initialized");
}

}  // namespace lse::backend

LSE_REGISTER_BACKEND("cpu", ::lse::backend::CpuBackend)
