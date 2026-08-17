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

// An ordered timeline of dispatches. Work issued on one stream executes in
// issue order. Work on two different streams may execute concurrently, and any
// ordering between them has to be *stated* — record_event on the producer,
// wait_event on the consumer. There is no implicit cross-stream ordering, and
// the seam deliberately offers no "sync everything" shortcut for expressing a
// dependency: a global barrier is not a dependency, it is the absence of one.
struct Stream {
  std::uint32_t index = 0;

  friend bool operator==(Stream, Stream) noexcept = default;
};

// Every backend has at least this one, and a backend that has only this one
// satisfies the whole seam without writing a line (see Backend<Derived>).
inline constexpr Stream kDefaultStream{0};

// A point on a stream's timeline: everything that stream had been given at the
// moment it was recorded. Opaque past these two fields — the backend decides
// what `timeline` counts.
struct StreamEvent {
  Stream stream{};
  std::uint64_t timeline = 0;

  [[nodiscard]] bool valid() const noexcept { return timeline != 0; }
};

// Which part of a dispatch's work this launch covers, as a half-open range of
// work items. `count == 0` means the whole dispatch, which is every launch
// today: a kernel still declares a grid, and a grid cannot be cut.
//
// It is here now so that when a kernel declares N work items at a granularity
// instead, splitting one op across two streams — and later across two devices
// — is a different WorkRange on the same call, not a new signature.
struct WorkRange {
  std::uint64_t begin = 0;
  std::uint64_t count = 0;

  [[nodiscard]] bool whole() const noexcept { return begin == 0 && count == 0; }
};

// Where a launch goes. One struct rather than trailing parameters so that
// adding the work range above did not change launch()'s arity, and adding
// placement (which device) later will not either.
struct DispatchTarget {
  Stream stream{};
  WorkRange work{};
};

// What this device's execution streams can actually do — declared by the
// backend, read by the scheduler, never inferred from the backend's name.
//
// The scheduler asks two questions of it: may I put these two groups on
// different streams at all, and is it worth an event to do so. Everything a
// caller needs to answer them is here; nothing that only one vendor could
// interpret is.
struct StreamCapabilities {
  // Streams the backend will hand out. Indices [0, stream_count) are valid.
  std::uint32_t stream_count = 1;
  // How many of those the device can genuinely run at the same time. 1 means
  // extra streams are legal but buy nothing, and a scheduler that reads this
  // will keep everything ordered rather than pay for events it cannot cash.
  std::uint32_t concurrent_streams = 1;
  // Ordering between two streams needs record_event/wait_event. False when the
  // backend already keeps every stream mutually ordered, which makes the
  // events no-ops rather than a special case at the call site.
  bool needs_explicit_events = false;
  // A launch may cover a strict subrange of its work items. False until a
  // kernel declares work items rather than a grid; while it is false a launch
  // whose WorkRange is not whole() is refused instead of silently widened.
  bool splittable_work = false;
  // A launch costs the same whatever stream it goes to. False when the
  // backend's cheap submission path can only address one queue, so every other
  // stream pays extra per dispatch to reach a queue of its own. While it is
  // false, moving a group is a loss even when it genuinely overlaps — the
  // overlap has to beat a per-kernel tax the group did not owe before — so the
  // cost model refuses to spread and the seam is exercised without being paid
  // for. It flips when the backend can batch onto more than one queue, and
  // nothing else about the seam changes when it does.
  bool uniform_launch_cost = true;

  [[nodiscard]] bool concurrent() const noexcept {
    return concurrent_streams > 1 && stream_count > 1;
  }

  // Whether any group could end up on a stream other than the one it is
  // already ordered on. False makes the whole placement pass dead work — the
  // answer is the order the caller already had — so the planner may skip it
  // rather than compute a plan it is forbidden to act on.
  [[nodiscard]] bool may_spread() const noexcept {
    return concurrent() && uniform_launch_cost;
  }

  // The cost model, in the only currency the caller has at placement time.
  // Moving independent work to another stream costs one event and buys the
  // chance to run beside something, so it is worth it only for work that does
  // not already fill the machine: `device_width` is the work-item count that
  // saturates it. A group at or above that leaves nothing to overlap with and
  // stays on the stream it is already on, however independent it is.
  //
  // Deliberately not priced in nanoseconds. A group's device time is not known
  // until there is a roofline model to predict it, and a made-up estimate
  // would make this look like a decision it is not.
  [[nodiscard]] bool worth_moving(std::uint64_t work_items,
                                  std::uint64_t device_width) const noexcept {
    return may_spread() && work_items > 0 && work_items < device_width;
  }
};

// One stream, everything ordered. What a device with no concurrency has, and
// what a backend that implements none of the stream *_impl surface reports.
[[nodiscard]] inline const StreamCapabilities&
ordered_single_stream() noexcept {
  static const StreamCapabilities caps{};
  return caps;
}

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
  // Occupancy limits. Every GPU has all three under some name — wave/warp/
  // subgroup, LDS/shared memory, resident waves per CU/SM — so a scheduler
  // deciding whether a tile can co-reside reads them without knowing the
  // vendor. 0 is unknown, never a guess.
  std::uint32_t lds_bytes_per_workgroup = 0;
  std::uint16_t compute_units = 0;
  std::uint16_t max_threads_per_workgroup = 0;
  std::uint16_t wavefront_size = 0;
  std::uint16_t max_waves_per_cu = 0;
  std::uint8_t ordinal = 0;
  // Host and device memory are one physical pool, so uploads are pointer
  // handoffs and staging buffers are pure waste. A real behavioural branch.
  bool unified_memory = false;

  // Vendor-specific block the backend owns and outlives this struct: cache
  // sizes, matrix-core generation, per-type arithmetic support and so on,
  // which only code that already targets that vendor may read. Reach it
  // through device_extension().
  std::string_view extension_id;
  const void* extension = nullptr;

  [[nodiscard]] std::string describe() const;
};

// Scalars: 8 (size_t) + 4 + 4x2 + 2x1 (padded to 24) + 16 (string_view)
// + 8 (ptr). Pinned so adding a field is a deliberate act, not a drift.
static_assert(sizeof(DeviceInfo) == 2 * sizeof(std::string) + 48,
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

  // Bytes free on this device at the instant of the call.
  //
  // A sample, not an identity. DeviceInfo::total_memory is the part number and
  // never moves; this moves with every allocation on the device, this process's
  // or another process's, so two calls a second apart may legitimately
  // disagree and no caller may memoize it.
  //
  // A backend whose runtime cannot answer refuses here, and in particular never
  // substitutes total_memory: a capacity inferred from the total is exactly the
  // number that hands a 40 GB shard to a device with 2 GB left.
  Result<std::size_t> sample_free_memory() const {
    if constexpr (requires { derived().sample_free_memory_impl(); }) {
      return derived().sample_free_memory_impl();
    } else {
      return LSE_ERROR(kUnimplemented, "backend '", std::string(Derived::kName),
                       "' has no device memory query");
    }
  }

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
                const DispatchArgs& args, const DispatchTarget& target = {}) {
    if constexpr (requires {
                    derived().launch_impl(kernel, dims, args, target);
                  }) {
      return derived().launch_impl(kernel, dims, args, target);
    } else {
      // A backend with one ordered stream: the target names the only stream
      // there is, and a partial work range has no way to be honoured.
      if (!target.work.whole()) {
        return LSE_ERROR(kUnimplemented, "backend '", std::string(Derived::kName),
                         "' cannot split a dispatch's work range");
      }
      return derived().launch_impl(kernel, dims, args);
    }
  }

  Status synchronize() { return derived().synchronize_impl(); }

  // --- execution streams -----------------------------------------------
  // A backend that implements none of the *_impl below still satisfies the
  // whole seam: it gets one stream, everything ordered, and events that are
  // already true the moment they are recorded. Nothing here assumes a GPU.

  [[nodiscard]] const StreamCapabilities& stream_capabilities() const noexcept {
    if constexpr (requires { derived().stream_capabilities_impl(); }) {
      return derived().stream_capabilities_impl();
    } else {
      return ordered_single_stream();
    }
  }

  // Names everything issued on `stream` so far, so another stream can be made
  // to wait for exactly that and no more.
  Result<StreamEvent> record_event(Stream stream) {
    if constexpr (requires { derived().record_event_impl(stream); }) {
      return derived().record_event_impl(stream);
    } else {
      // Nothing can be issued out of order, so every point is already past.
      return StreamEvent{stream, 1};
    }
  }

  // Orders `stream` after `event`. Does not block the host, and never becomes
  // a device-wide barrier: only the waiting stream is held.
  Status wait_event(Stream stream, const StreamEvent& event) {
    if constexpr (requires { derived().wait_event_impl(stream, event); }) {
      return derived().wait_event_impl(stream, event);
    } else {
      (void)stream;
      (void)event;
      return OkStatus();
    }
  }

  // Distinct name rather than an overload of synchronize(): a backend that
  // wants only the device-wide one must not have to restate this to keep it
  // visible.
  Status synchronize_stream(Stream stream) {
    if constexpr (requires { derived().synchronize_stream_impl(stream); }) {
      return derived().synchronize_stream_impl(stream);
    } else {
      (void)stream;
      return derived().synchronize_impl();
    }
  }

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
  // Bytes free on this device right now — a sample that moves under the
  // caller, never the declared total in DeviceInfo. A backend whose runtime
  // has no such query refuses, and a refusal means unknown: the reader is a
  // capacity constraint, and an unknown there declines a placement while an
  // invented figure would approve one.
  virtual Result<std::size_t> sample_free_memory() const {
    return LSE_ERROR(kUnimplemented, "backend has no device memory query");
  }
  virtual Status copy_h2d(const void* src, DeviceBuffer& dst, std::size_t bytes,
                          std::size_t dst_offset) = 0;
  virtual Status copy_d2h(const DeviceBuffer& src, void* dst, std::size_t bytes,
                          std::size_t src_offset) = 0;
  virtual Result<KernelHandle> load_executable(
      std::string_view name, std::span<const std::byte> code_object) = 0;
  virtual Status launch(const KernelHandle& kernel, const LaunchDims& dims,
                        const DispatchArgs& args,
                        const DispatchTarget& target) = 0;
  // Whole dispatch, default stream — what a caller that has no opinion wants.
  Status launch(const KernelHandle& kernel, const LaunchDims& dims,
                const DispatchArgs& args) {
    return launch(kernel, dims, args, DispatchTarget{});
  }
  virtual Status synchronize() = 0;

  // What this device's streams can do, and how to order two of them. A caller
  // reads the capabilities and decides; it never asks which backend this is.
  //
  // The defaults are a backend with one stream on which everything is already
  // ordered — every point on its timeline is behind anything issued later, so
  // an event is satisfied the moment it is recorded. A backend with real
  // streams overrides all four; one without overrides none.
  virtual const StreamCapabilities& stream_capabilities() const noexcept {
    return ordered_single_stream();
  }
  virtual Result<StreamEvent> record_event(Stream stream) {
    return StreamEvent{stream, 1};
  }
  virtual Status wait_event(Stream, const StreamEvent&) { return OkStatus(); }
  virtual Status synchronize_stream(Stream) { return synchronize(); }

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
  Result<std::size_t> sample_free_memory() const override {
    return impl_.sample_free_memory();
  }
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
                const DispatchArgs& a, const DispatchTarget& t) override {
    return impl_.launch(k, d, a, t);
  }
  using IBackend::launch;
  Status synchronize() override { return impl_.synchronize(); }
  const StreamCapabilities& stream_capabilities() const noexcept override {
    return impl_.stream_capabilities();
  }
  Result<StreamEvent> record_event(Stream s) override {
    return impl_.record_event(s);
  }
  Status wait_event(Stream s, const StreamEvent& e) override {
    return impl_.wait_event(s, e);
  }
  Status synchronize_stream(Stream s) override {
    return impl_.synchronize_stream(s);
  }
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
