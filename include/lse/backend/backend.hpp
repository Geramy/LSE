// Backend contract.
//
// A backend author writes the *_impl surface and nothing else. Backend<Derived>
// turns that into the public API and adds the algorithms that are the same for
// every device — upload, download — written once against allocate/copy rather
// than repeated per backend. BackendAdapter<Derived> wraps the result in
// IBackend, and that is what the engine actually holds: dispatch is virtual,
// because the backend is chosen from a config string at startup. The CRTP layer
// buys code reuse and compile-time checking of the *_impl set, not devirtualized
// dispatch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::graph {
class IKernelEmitter;
class IKernelCompiler;
}  // namespace lse::graph

namespace lse::backend {

// Where an allocation lives. Device memory is not host-addressable on a real
// GPU — mapping it is refused outright — so anything the host must read or
// write goes through an explicit transfer, never a pointer.
enum class MemoryClass : std::uint8_t {
  // Only kernels touch it. This is the default because a compiled kernel chain
  // should never involve host memory at all.
  kDevice,
  // Host-visible, for uploads, readbacks, and the host fallback path.
  kStaging,
};

struct DeviceBuffer {
  // Host address of the allocation base, or null when the allocation is
  // device-local. A view keeps the same pointer and names its window with
  // `offset` + `size_bytes`; never free a view, only the original.
  void* ptr = nullptr;
  std::size_t size_bytes = 0;
  std::uint64_t handle = 0;     // opaque, backend-private
  std::size_t offset = 0;       // bytes from the start of the allocation
  // When set, the allocation is released when the last copy dies. Views
  // share it so a reshape cannot free a buffer the residual still holds.
  std::shared_ptr<void> storage;

  [[nodiscard]] bool valid() const noexcept { return ptr != nullptr || handle != 0; }
};

// A loaded code object plus which of its exports to dispatch. Selection is by
// ordinal, not by symbol lookup, so no runtime carries a module table here.
struct KernelHandle {
  std::uint64_t executable = 0;      // opaque, backend-private
  std::uint32_t export_ordinal = 0;
  std::string name;

  [[nodiscard]] bool valid() const noexcept { return executable != 0; }
};

struct LaunchDims {
  std::uint32_t workgroup_count[3] = {1, 1, 1};
  std::uint32_t workgroup_size[3] = {1, 1, 1};
  std::uint32_t subgroup_size = 0;   // 0 = let the runtime choose
};

struct BufferRef {
  const DeviceBuffer* buffer = nullptr;
  std::size_t offset = 0;
  std::size_t length = 0;
};

// Kernel arguments split into bound buffers and a flat constants block, rather
// than one flattened argv. Backends that want an argv build it from these.
struct DispatchArgs {
  std::span<const BufferRef> bindings;
  std::span<const std::byte> constants;
  std::uint32_t flags = 0;
};

// What any device has. Nothing here is specific to a vendor or an ISA: every
// field is either an identity a cache key needs or a limit every backend can
// answer. Vendor detail lives in `extension`.
//
// Two things that look missing are missing on purpose:
//   - register budgets (VGPR/SGPR): a tile is legal only if its *actual*
//     register usage fits, and that is not known until the kernel is compiled.
//     It comes from the code object's metadata, not from a device query.
//   - bandwidth / clocks: only useful once there is a roofline cost model to
//     consume them. Add them with the model, not before.
struct DeviceInfo {
  std::string name;   // "Radeon 8060S Graphics"
  std::string arch;   // "gfx1151" — part of the JIT cache key

  std::size_t total_memory = 0;              // working-set / KV-cache budgeting
  std::uint16_t compute_units = 0;
  std::uint16_t max_threads_per_workgroup = 0;
  std::uint8_t ordinal = 0;
  // Host and device memory are one physical pool, so uploads are pointer
  // handoffs and staging buffers are pure waste. A real behavioural branch.
  bool unified_memory = false;

  // Vendor-specific block the backend owns and outlives this struct: wavefront
  // width, LDS budget, matrix-core generation and so on, which only code that
  // already targets that vendor may read. Reach it through device_extension().
  std::string_view extension_id;
  const void* extension = nullptr;

  [[nodiscard]] std::string describe() const;
};

// Scalars: 8 (size_t) + 2 + 2 + 2x1 (padded to 8) + 16 (string_view) + 8 (ptr).
// Pinned so adding a field is a deliberate act, not a drift.
static_assert(sizeof(DeviceInfo) == 2 * sizeof(std::string) + 40,
              "DeviceInfo scalar budget changed — justify the field, then "
              "update this assert");

// Null unless this device published exactly T. T supplies
// `static constexpr std::string_view kExtensionId`.
template <typename T>
[[nodiscard]] const T* device_extension(const DeviceInfo& info) noexcept {
  if (info.extension == nullptr || info.extension_id != T::kExtensionId) {
    return nullptr;
  }
  return static_cast<const T*>(info.extension);
}

// Derived implements each *_impl below. Kernel generation and compilation are
// device-specific, so a backend that has them exposes them through
// graph::IKernelEmitter / graph::IKernelCompiler; one that does not returns
// nullptr and only ever loads an already-compiled code object.
template <typename Derived>
class Backend {
 public:
  Status init(int device_ordinal = 0) {
    return derived().init_impl(device_ordinal);
  }

  void shutdown() noexcept { derived().shutdown_impl(); }

  [[nodiscard]] const DeviceInfo& device_info() const noexcept {
    return derived().device_info_impl();
  }

  [[nodiscard]] std::string_view arch() const noexcept {
    return device_info().arch;
  }

  Result<DeviceBuffer> allocate(std::size_t bytes,
                                MemoryClass cls = MemoryClass::kDevice) {
    return derived().allocate_impl(bytes, cls);
  }

  void deallocate(DeviceBuffer& buf) noexcept { derived().deallocate_impl(buf); }

  Status copy_h2d(const void* src, DeviceBuffer& dst, std::size_t bytes,
                  std::size_t dst_offset = 0) {
    return derived().copy_h2d_impl(src, dst, bytes, dst_offset);
  }

  Status copy_d2h(const DeviceBuffer& src, void* dst, std::size_t bytes,
                  std::size_t src_offset = 0) {
    return derived().copy_d2h_impl(src, dst, bytes, src_offset);
  }

  Result<KernelHandle> load_executable(std::string_view name,
                                       std::span<const std::byte> code_object) {
    return derived().load_executable_impl(name, code_object);
  }

  Status launch(const KernelHandle& kernel, const LaunchDims& dims,
                const DispatchArgs& args) {
    return derived().launch_impl(kernel, dims, args);
  }

  Status synchronize() { return derived().synchronize_impl(); }

  Result<void*> device_pointer(const DeviceBuffer& buf) const {
    if constexpr (requires { derived().device_pointer_impl(buf); }) {
      return derived().device_pointer_impl(buf);
    }
    if (buf.ptr == nullptr) {
      return LSE_ERROR(kUnimplemented, "backend has no device pointer");
    }
    return static_cast<void*>(static_cast<std::byte*>(buf.ptr) + buf.offset);
  }

  [[nodiscard]] const graph::IKernelEmitter* emitter() const noexcept {
    return derived().emitter_impl();
  }

  [[nodiscard]] const graph::IKernelCompiler* compiler() const noexcept {
    return derived().compiler_impl();
  }

  // Written once against the primitives above — the reason this is a base
  // class rather than a bare concept.
  Result<DeviceBuffer> upload(const void* src, std::size_t bytes) {
    auto buf = allocate(bytes);
    if (!buf.ok()) return buf.status();
    DeviceBuffer owned = buf.release();
    Status s = copy_h2d(src, owned, bytes);
    if (!s.ok()) {
      deallocate(owned);
      return s;
    }
    return owned;
  }

  template <typename T>
  Result<std::vector<T>> download(const DeviceBuffer& src, std::size_t count) {
    std::vector<T> out(count);
    Status s = copy_d2h(src, out.data(), count * sizeof(T));
    if (!s.ok()) return s;
    return out;
  }

 protected:
  Backend() = default;

 private:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept {
    return static_cast<const Derived&>(*this);
  }
};

class IBackend {
 public:
  virtual ~IBackend() = default;
  virtual Status init(int device_ordinal) = 0;
  virtual void shutdown() noexcept = 0;
  virtual const DeviceInfo& device_info() const noexcept = 0;
  virtual Result<DeviceBuffer> allocate(std::size_t bytes,
                                        MemoryClass cls) = 0;
  virtual void deallocate(DeviceBuffer& buf) noexcept = 0;
  virtual Status copy_h2d(const void* src, DeviceBuffer& dst, std::size_t bytes,
                          std::size_t dst_offset) = 0;
  virtual Status copy_d2h(const DeviceBuffer& src, void* dst, std::size_t bytes,
                          std::size_t src_offset) = 0;
  virtual Result<KernelHandle> load_executable(
      std::string_view name, std::span<const std::byte> code_object) = 0;
  virtual Status launch(const KernelHandle& kernel, const LaunchDims& dims,
                        const DispatchArgs& args) = 0;
  virtual Status synchronize() = 0;
  virtual std::string_view name() const noexcept = 0;
  // nullptr when the backend has no codegen of its own.
  virtual const graph::IKernelEmitter* emitter() const noexcept = 0;
  virtual const graph::IKernelCompiler* compiler() const noexcept = 0;

  // Device address of `buf`, including its offset. Used to build a pointer
  // table so a phase kernel is not limited by the kernarg slot count.
  virtual Result<void*> device_pointer(const DeviceBuffer& buf) const {
    if (buf.ptr == nullptr) {
      return LSE_ERROR(kUnimplemented, "backend has no device pointer");
    }
    return static_cast<void*>(static_cast<std::byte*>(buf.ptr) + buf.offset);
  }
};

template <typename Derived>
class BackendAdapter final : public IBackend {
 public:
  template <typename... Args>
  explicit BackendAdapter(Args&&... args) : impl_(std::forward<Args>(args)...) {}

  Status init(int d) override { return impl_.init(d); }
  void shutdown() noexcept override { impl_.shutdown(); }
  const DeviceInfo& device_info() const noexcept override {
    return impl_.device_info();
  }
  Result<DeviceBuffer> allocate(std::size_t b, MemoryClass c) override {
    return impl_.allocate(b, c);
  }
  void deallocate(DeviceBuffer& b) noexcept override { impl_.deallocate(b); }
  Status copy_h2d(const void* s, DeviceBuffer& d, std::size_t n,
                  std::size_t off) override {
    return impl_.copy_h2d(s, d, n, off);
  }
  Status copy_d2h(const DeviceBuffer& s, void* d, std::size_t n,
                  std::size_t off) override {
    return impl_.copy_d2h(s, d, n, off);
  }
  Result<KernelHandle> load_executable(
      std::string_view n, std::span<const std::byte> code) override {
    return impl_.load_executable(n, code);
  }
  Status launch(const KernelHandle& k, const LaunchDims& d,
                const DispatchArgs& a) override {
    return impl_.launch(k, d, a);
  }
  Status synchronize() override { return impl_.synchronize(); }
  std::string_view name() const noexcept override { return Derived::kName; }
  const graph::IKernelEmitter* emitter() const noexcept override {
    return impl_.emitter();
  }
  const graph::IKernelCompiler* compiler() const noexcept override {
    return impl_.compiler();
  }
  Result<void*> device_pointer(const DeviceBuffer& buf) const override {
    return impl_.device_pointer(buf);
  }

  Derived& impl() noexcept { return impl_; }

 private:
  Derived impl_;
};

// Backends self-register from their own TU: linking one in is all that makes
// it selectable by name.
using BackendFactory = std::unique_ptr<IBackend> (*)();

void register_backend(std::string_view name, BackendFactory factory);
Result<std::unique_ptr<IBackend>> create_backend(std::string_view name);
std::vector<std::string> available_backends();

Result<std::unique_ptr<IBackend>> create_default_backend();

// Backends to try, best first. A backend can construct and still fail to come
// up, so a caller that initializes should walk this rather than take only the
// first name. LSE_BACKEND, when set, is the sole candidate.
std::vector<std::string> default_backend_order();

struct BackendRegistrar {
  BackendRegistrar(std::string_view name, BackendFactory factory) {
    register_backend(name, factory);
  }
};

// No token pasting: Type may be a qualified name. One registration per TU.
#define LSE_REGISTER_BACKEND(name, Type)                                  \
  namespace {                                                             \
  const ::lse::backend::BackendRegistrar _lse_backend_registrar{          \
      name, []() -> std::unique_ptr<::lse::backend::IBackend> {           \
        return std::make_unique<::lse::backend::BackendAdapter<Type>>();  \
      }};                                                                 \
  }  // namespace

}  // namespace lse::backend
