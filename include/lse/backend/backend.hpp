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

#include "lse/backend/resources.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/toolchain.hpp"

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

// Which device holds bytes, and which device a launch is for.
//
// Not an ordinal and not a backend name: it identifies the bound backend
// INSTANCE. Two instances driving the same board are two values here, because
// a buffer one of them allocated is released by that one and an executable one
// of them loaded runs on that one — the identity a dispatch turns on is the
// instance, not the part number. Resolving one back to an "hrx:0" is the
// device set's job one layer up; nothing at this level can name a device,
// which is the point.
//
// Zero is "no device": what a default-constructed buffer carries, what a
// released one goes back to, and what an IBackend that stamps nothing reports.
struct DeviceIndex {
  std::uint16_t value = 0;

  [[nodiscard]] bool bound() const noexcept { return value != 0; }
  friend bool operator==(DeviceIndex, DeviceIndex) noexcept = default;
  friend auto operator<=>(DeviceIndex, DeviceIndex) noexcept = default;
};

inline constexpr DeviceIndex kNoDevice{};

// The next index no device in this process has held. Called by Backend<Derived>
// when init() binds a device; nothing picks its own.
//
// Indices are never recycled: a stale one must resolve to nothing rather than
// to whoever was bound next. Past 65535 binds in one process it stops handing
// them out and returns kNoDevice, so the failure is unstamped buffers — which
// read as "no device holds these" and are refused a cross-device check — and
// never two live devices sharing a token.
[[nodiscard]] DeviceIndex next_device_index() noexcept;

struct DeviceBuffer {
  // Host address of the allocation base, or null when the allocation is
  // device-local. A view keeps the same pointer and names its window with
  // `offset` + `size_bytes`; never free a view, only the original.
  void* ptr = nullptr;
  std::size_t size_bytes = 0;
  std::uint64_t handle = 0;     // opaque, backend-private
  std::size_t offset = 0;       // bytes from the start of the allocation
  // Which device holds these bytes. Stamped by the backend that allocated
  // them, carried unchanged by a view, cleared when the allocation is
  // released. Without it a launch on the wrong device is accepted by every
  // runtime here, runs, and returns plausible numbers — the whole point is
  // that a kernel's right to read a buffer becomes a question that can be
  // asked.
  DeviceIndex residency{};
  // When set, the allocation is released when the last copy dies. Views
  // share it so a reshape cannot free a buffer the residual still holds.
  std::shared_ptr<void> storage;

  [[nodiscard]] bool valid() const noexcept { return ptr != nullptr || handle != 0; }
};

// Four words of addressing, then residency in the padding its own alignment
// leaves, then the shared_ptr's two. Pinned because this struct is a member of
// every graph Node and every workgroup slot: a field here is paid for per
// tensor, so adding one has to be a deliberate act.
static_assert(sizeof(DeviceBuffer) == 40 + sizeof(std::shared_ptr<void>),
              "DeviceBuffer grew — justify the field, then update this assert");

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

// Which clock a tick was read from.
//
// The distinction is not bookkeeping. A host clock read at the moment the host
// hands work to a device measures submission, queueing and interrupt wake, and
// on a batched submission path that is most of what it measures — so a host
// tick labelled as a device one turns dispatch jitter into a kernel duration
// and a cost model acts on it. Naming the domain is what stops that, and it is
// the reason kHostSystem exists as its own value rather than being folded into
// the device case just because a vendor runtime published it.
enum class ClockDomain : std::uint8_t {
  kUnknown = 0,
  // The device's own counter — the only domain in which a dispatch's duration
  // is the dispatch's duration. Counted per *physical device*: two devices on
  // one box count independently from unrelated origins, so a matching domain is
  // necessary for a subtraction and not sufficient (see DeviceClock::ordinal).
  kDeviceAgent,
  // A host clock the vendor runtime publishes under a device-scoped name. A
  // backend that has only this says so here instead of passing it off as
  // kDeviceAgent, which is the substitution this whole seam exists to prevent.
  kHostSystem,
  // std::chrono::steady_clock — the domain every host-side span is already in.
  kHostSteady,
};

[[nodiscard]] constexpr std::string_view clock_domain_name(
    ClockDomain domain) noexcept {
  switch (domain) {
    case ClockDomain::kDeviceAgent: return "device-agent";
    case ClockDomain::kHostSystem:  return "host-system";
    case ClockDomain::kHostSteady:  return "host-steady";
    case ClockDomain::kUnknown:     break;
  }
  return "unknown";
}

// The clock a device's timestamps are counted on, as the facts a duration can
// actually be computed from.
//
// `ticks_per_second` is queried from the device or measured on it, never
// assumed, because the plausible guesses are all wrong: on gfx1151 the agent
// counter runs at 99,810,000 Hz, so rounding to 100 MHz is 0.19% off and taking
// the host-scope rate the same runtime reports for its system clock (1 GHz) is
// off by 10.019x. A rate nobody could answer stays 0, and 0 makes every
// duration derived from it unknown rather than approximate.
struct DeviceClock {
  ClockDomain domain = ClockDomain::kUnknown;
  std::uint64_t ticks_per_second = 0;
  // Bits of a tick that count before the counter wraps; above them the value is
  // unspecified. A subtraction that ignores this reads a wrap as a few thousand
  // years, so it is carried rather than assumed to be 64. 0 is unknown.
  std::uint8_t valid_bits = 0;
  // Which device's counter this is, within the backend that published it. Two
  // ticks from different devices are unrelated however alike their clocks look,
  // and nothing in the tick itself says so. Across machines even the ordinal is
  // not enough — that comparison belongs to whoever holds the pool's device
  // identities, and it starts by refusing here.
  std::uint8_t ordinal = 0;

  // Everything a duration needs is present. False means a tick from this clock
  // converts to no number at all.
  [[nodiscard]] bool known() const noexcept {
    return domain != ClockDomain::kUnknown && ticks_per_second > 0 &&
           valid_bits > 0;
  }

  // Whether two ticks may be subtracted: same clock, same device, same rate.
  [[nodiscard]] bool same_source_as(const DeviceClock& other) const noexcept {
    return known() && other.known() && domain == other.domain &&
           ticks_per_second == other.ticks_per_second &&
           valid_bits == other.valid_bits && ordinal == other.ordinal;
  }
};

// One reading. The clock travels with the tick so that a record which outlives
// the backend that produced it still says what it means — a bare uint64 in a
// trace is a number nobody can convert and nobody can refuse.
struct DeviceTimestamp {
  std::uint64_t tick = 0;
  DeviceClock clock{};

  [[nodiscard]] bool valid() const noexcept { return clock.known(); }
};

// Ticks elapsed from `start` to `end`, wrap-corrected against the clock's
// valid_bits. Refuses across domains, across devices and on an unknown rate:
// an unknown duration is the honest answer and a plausible one is not.
//
// A narrower-than-64-bit counter really does wrap, so a decreasing tick is
// masked back into a duration; that is correct while the true interval is under
// one wrap, which is the standing assumption for such counters, and nothing
// here can detect a double wrap. At 64 bits it is the opposite: 2^64 ticks is
// ~5900 years at the ~100 MHz these counters run, so a decreasing tick is not a
// wrap but the clock moving backwards — which happens for real, because a
// device clock resynced against the host can roll back (IREE says so of these
// very timestamps in hal/drivers/amdgpu/abi/signal.h). Masking that would enter
// a several-thousand-year span into a profile as the largest number in it, so
// it refuses instead.
[[nodiscard]] inline Result<std::uint64_t> ticks_between(
    const DeviceTimestamp& start, const DeviceTimestamp& end) {
  if (!start.clock.known() || !end.clock.known()) {
    return LSE_ERROR(kInvalidArgument,
                     "cannot subtract timestamps from an unknown clock");
  }
  if (!start.clock.same_source_as(end.clock)) {
    return LSE_ERROR(kInvalidArgument, "cannot subtract a ",
                     std::string(clock_domain_name(end.clock.domain)),
                     " tick on device ", std::to_string(end.clock.ordinal),
                     " from a ",
                     std::string(clock_domain_name(start.clock.domain)),
                     " tick on device ", std::to_string(start.clock.ordinal),
                     ": unrelated counters");
  }
  const std::uint8_t bits = start.clock.valid_bits;
  if (bits >= 64) {
    if (end.tick < start.tick) {
      return LSE_ERROR(kInvalidArgument, "device clock went backwards: ",
                       std::to_string(start.tick), " to ",
                       std::to_string(end.tick));
    }
    return end.tick - start.tick;
  }
  const std::uint64_t mask = (std::uint64_t{1} << bits) - 1;
  return ((end.tick - start.tick) & mask);
}

// The same subtraction in nanoseconds, which is the only form a cost model has
// any use for. Integer ticks over an integer rate, in double, because the two
// differ by ~10x between the domains on one device and truncating first loses a
// microsecond-scale span entirely.
[[nodiscard]] inline Result<double> nanoseconds_between(
    const DeviceTimestamp& start, const DeviceTimestamp& end) {
  auto ticks = ticks_between(start, end);
  if (!ticks.ok()) return ticks.status();
  return static_cast<double>(*ticks) * 1e9 /
         static_cast<double>(start.clock.ticks_per_second);
}

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
// placement (which device) later did not either.
struct DispatchTarget {
  Stream stream{};
  // Which device this launch is for. kNoDevice means "whichever device the
  // backend being called drives" — every call site that has one device in
  // view. When it is set and names a different device than the callee drives,
  // the launch is refused: the kernel handle was loaded on one device and the
  // bindings' bytes sit on one device, so landing on the wrong one computes
  // garbage without erroring anywhere.
  //
  // Sits between `stream` and `work` rather than after them because the
  // padding after a 4-byte Stream was being spent on nothing.
  DeviceIndex device{};
  WorkRange work{};
};

static_assert(sizeof(DispatchTarget) == 24,
              "DispatchTarget grew past the padding it was fitted into");

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
// Register USAGE is deliberately not here: a tile is legal only if what the
// allocator actually gave the kernel fits, and that is not known until the
// kernel is compiled — it arrives as KernelResources, from the code object's
// metadata. Register CAPACITY is a different question with a different answer:
// it is a constant of the target that the compiler's own ISA table states
// before any compile, and it lives in `arch` below.
//
// Bandwidth and clocks are still missing on purpose: only useful once there is
// a roofline cost model to consume them. Add them with the model, not before.
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
  // How many CUs draw workgroup scratch from one physical LDS pool. 1 unless
  // the architecture pairs CUs behind a shared block, as RDNA's WGP does — and
  // that pairing is why `lds_bytes_per_workgroup / request` counts resident
  // workgroups across a *pool*, not per CU. Getting this wrong overstates
  // occupancy by exactly this factor, which is how a 16 KiB request came to be
  // described as "four workgroups per CU" when the hardware seats two.
  // Measured on gfx1151 with hipOccupancyMaxActiveBlocksPerMultiprocessor at
  // 256 threads: 8 blocks/MP at 8192 B, 4 at 16384, 3 at 20480, 2 at 32768,
  // 1 at 65536, and HIP's multiprocessor is the WGP — multiProcessorCount 20
  // against the 40 CUs rocminfo reports.
  std::uint8_t cus_per_lds_pool = 1;
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

  // Per-target capacity, queried from the toolchain where it can answer and
  // declared from a table where it cannot. Every field is individually
  // unknown-able, so a target nothing answers for yields no numbers rather
  // than zeros a model would spend.
  ArchFacts arch_facts;

  // What share of its streaming rate the memory system gives a launch at a
  // given residency. The other half of the pair an arrangement is priced on:
  // arch_facts says how many workgroups fit, this says what fitting fewer
  // costs. Unknown where nothing measured it, and a decision then prices the
  // traffic alone rather than scaling it by an invented number.
  ResidencyBandwidth residency_bandwidth;

  [[nodiscard]] std::string describe() const;
};

// Scalars: 8 (size_t) + 4 + 4x2 + 2x1 (padded to 24) + 16 (string_view)
// + 8 (ptr) + sizeof(ArchFacts). `cus_per_lds_pool` fits in the existing tail
// padding beside `ordinal`. Pinned so adding a field is a deliberate act, not
// a drift — ArchFacts was added deliberately, as the capacity half of the pair
// the occupancy model needs.
static_assert(sizeof(ArchFacts) == 13 * sizeof(DeviceFact<std::uint32_t>),
              "ArchFacts gained or lost a fact — say which and why");
// ResidencyBandwidth is a byte array, so it lands in no existing padding and
// its own tail is rounded to the struct's alignment; the round is spelled out
// rather than folded into a literal so the next field to be added has to say
// which of the two it is doing.
static_assert(sizeof(DeviceInfo) ==
                  2 * sizeof(std::string) + 48 + sizeof(ArchFacts) +
                      ((sizeof(ResidencyBandwidth) + alignof(DeviceInfo) - 1) /
                       alignof(DeviceInfo)) *
                          alignof(DeviceInfo),
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
    LSE_RETURN_IF_ERROR(derived().init_impl(device_ordinal));
    // Claimed only after the bind succeeded, so a refused ordinal leaves this
    // instance stamping nothing rather than owning a token for a device it
    // never got.
    device_ = next_device_index();
    return OkStatus();
  }

  void shutdown() noexcept {
    derived().shutdown_impl();
    device_ = kNoDevice;
  }

  // The token this instance stamps on the buffers it allocates. kNoDevice
  // before init() and after shutdown().
  [[nodiscard]] DeviceIndex device_index() const noexcept { return device_; }

  [[nodiscard]] const DeviceInfo& device_info() const noexcept {
    return derived().device_info_impl();
  }

  [[nodiscard]] std::string_view arch() const noexcept {
    return device_info().arch;
  }

  Result<DeviceBuffer> allocate(std::size_t bytes,
                                MemoryClass cls = MemoryClass::kDevice) {
    // The one funnel every allocation goes through, which is why residency is
    // stamped here rather than in each backend's allocate_impl: a backend
    // author cannot forget to do it.
    auto buf = derived().allocate_impl(bytes, cls);
    if (buf.ok()) buf->residency = device_;
    return buf;
  }

  void deallocate(DeviceBuffer& buf) noexcept {
    derived().deallocate_impl(buf);
    buf.residency = kNoDevice;
  }

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

  // The clock this device counts its timestamps on: domain, rate, width.
  //
  // A measurement seam, so it refuses rather than describing some other clock.
  // In particular a backend never answers with the host's clock because the
  // host's is the one it can reach — a duration measured on the host around a
  // device is submission latency, and the whole point of naming a domain is
  // that nothing downstream has to guess which one it got.
  //
  // Answering here says what a tick would mean, not that one can be had:
  // sample_device_time() is a separate question and a backend may well be able
  // to describe a counter it cannot read.
  Result<DeviceClock> device_clock() const {
    if constexpr (requires { derived().device_clock_impl(); }) {
      return derived().device_clock_impl();
    } else {
      return LSE_ERROR(kUnimplemented, "backend '", std::string(Derived::kName),
                       "' publishes no device clock");
    }
  }

  // One tick of that clock, read now.
  //
  // A sample, like sample_free_memory: it moves under the caller and nothing
  // may memoize it. What it is *not* is a dispatch's duration — it is the
  // primitive a duration is built from, and a backend that can only obtain a
  // tick by making the device stop and tell it should refuse here rather than
  // sell a round trip as a clock read.
  Result<DeviceTimestamp> sample_device_time() const {
    if constexpr (requires { derived().sample_device_time_impl(); }) {
      return derived().sample_device_time_impl();
    } else {
      return LSE_ERROR(kUnimplemented, "backend '", std::string(Derived::kName),
                       "' has no device timestamp source");
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

  // One device's memory to another's, without the bytes touching host memory.
  //
  // Issued on this backend, which is the destination's; `src` belongs to the
  // peer. Whether the peer's memory is reachable is settled when it is
  // allocated -- a device pool is DISALLOWED_BY_DEFAULT until somebody grants
  // access -- so a backend that cannot reach it says so here rather than
  // bouncing through the host quietly, which is a thousandfold difference
  // nobody would see in a result.
  Status copy_peer(const DeviceBuffer& src, DeviceBuffer& dst,
                   std::size_t bytes, std::size_t src_offset = 0,
                   std::size_t dst_offset = 0) {
    if constexpr (requires(Derived& d) {
                    d.copy_peer_impl(src, dst, bytes, src_offset, dst_offset);
                  }) {
      return derived().copy_peer_impl(src, dst, bytes, src_offset, dst_offset);
    } else {
      return LSE_ERROR(kUnimplemented, "backend '", std::string(Derived::kName),
                       "' has no device-to-device copy");
    }
  }

  Result<KernelHandle> load_executable(std::string_view name,
                                       std::span<const std::byte> code_object) {
    return derived().load_executable_impl(name, code_object);
  }

  Status launch(const KernelHandle& kernel, const LaunchDims& dims,
                const DispatchArgs& args, const DispatchTarget& target = {}) {
    // The one invariant this layer can settle alone: a launch addressed to
    // another device must not run here. Whether the bindings' bytes are
    // reachable is a question about two devices and belongs to whoever holds
    // the set — a backend can only see its own.
    if (target.device.bound() && target.device != device_) {
      return LSE_ERROR(kInvalidArgument, "launch is for device ",
                       std::to_string(target.device.value),
                       " but reached the backend driving device ",
                       std::to_string(device_.value));
    }
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

  [[nodiscard]] std::span<const graph::KernelToolchain> toolchains()
      const noexcept {
    return derived().toolchains_impl();
  }

  [[nodiscard]] const graph::IKernelEmitter* emitter() const noexcept {
    const auto tcs = toolchains();
    return tcs.empty() ? nullptr : tcs.front().emitter;
  }

  [[nodiscard]] const graph::IKernelCompiler* compiler() const noexcept {
    const auto tcs = toolchains();
    return tcs.empty() ? nullptr : tcs.front().compiler;
  }

  // nullptr when this device does not declare `dialect`. The same two
  // questions IBackend answers, restated here because the CRTP surface is
  // reached without going through the vtable.
  [[nodiscard]] const graph::KernelToolchain* toolchain_for(
      graph::Dialect dialect) const noexcept {
    return graph::find_toolchain(toolchains(), dialect);
  }
  [[nodiscard]] const graph::KernelToolchain* toolchain(
      graph::DialectPreference want) const noexcept {
    return graph::resolve_toolchain(toolchains(), want);
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

  DeviceIndex device_{};
};

class IBackend {
 public:
  virtual ~IBackend() = default;
  virtual Status init(int device_ordinal) = 0;
  virtual void shutdown() noexcept = 0;
  virtual const DeviceInfo& device_info() const noexcept = 0;
  // The token this instance stamps on the buffers it allocates, or kNoDevice
  // for an implementation that stamps none. An unstamped buffer says "no device
  // claims these bytes", which is exactly what a residency check should read
  // from an implementation that does not track one — so the default is honest
  // rather than merely permissive.
  virtual DeviceIndex device_index() const noexcept { return kNoDevice; }
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
  // What a tick from this device would mean, and one such tick. Both refuse by
  // default, and a refusal means unknown: the reader is a cost model, and an
  // unknown duration leaves it on whatever it already measured while a
  // substituted host clock would feed it submission jitter as device time.
  //
  // Two questions rather than one because they have different answers on the
  // same device — a backend can often name and rate the counter its dispatches
  // run against while having no way to read it.
  virtual Result<DeviceClock> device_clock() const {
    return LSE_ERROR(kUnimplemented, "backend publishes no device clock");
  }
  virtual Result<DeviceTimestamp> sample_device_time() const {
    return LSE_ERROR(kUnimplemented, "backend has no device timestamp source");
  }
  virtual Status copy_h2d(const void* src, DeviceBuffer& dst, std::size_t bytes,
                          std::size_t dst_offset) = 0;
  virtual Status copy_d2h(const DeviceBuffer& src, void* dst, std::size_t bytes,
                          std::size_t src_offset) = 0;
  // Not pure: a backend with no peer of its own is the common case, and the
  // caller's answer to "can these two talk directly" is this declining.
  virtual Status copy_peer(const DeviceBuffer& src, DeviceBuffer& dst,
                           std::size_t bytes, std::size_t src_offset,
                           std::size_t dst_offset) {
    (void)src; (void)dst; (void)bytes; (void)src_offset; (void)dst_offset;
    return LSE_ERROR(kUnimplemented, "this backend has no peer copy");
  }
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

  // Every dialect this device can generate and build, in the backend's own
  // preference order. Empty when the backend has no codegen of its own.
  //
  // This is the only virtual: emitter() and compiler() below are the same
  // question asked without a dialect, and a device answers it once.
  virtual std::span<const graph::KernelToolchain> toolchains() const noexcept = 0;

  // The default dialect's halves — nullptr when the backend has no codegen.
  [[nodiscard]] const graph::IKernelEmitter* emitter() const noexcept {
    const auto tcs = toolchains();
    return tcs.empty() ? nullptr : tcs.front().emitter;
  }
  [[nodiscard]] const graph::IKernelCompiler* compiler() const noexcept {
    const auto tcs = toolchains();
    return tcs.empty() ? nullptr : tcs.front().compiler;
  }
  // nullptr when this device does not declare `dialect`. A caller that wants a
  // specific dialect asks here and handles absence; it never assumes.
  [[nodiscard]] const graph::KernelToolchain* toolchain_for(
      graph::Dialect dialect) const noexcept {
    return graph::find_toolchain(toolchains(), dialect);
  }

  // A run's dialect preference resolved against this device: the named dialect
  // when this device declares it, and otherwise the device's first choice. The
  // degrade is the point — a preference is a runtime value naming a resource,
  // and a member that lacks it still has to run.
  [[nodiscard]] const graph::KernelToolchain* toolchain(
      graph::DialectPreference want) const noexcept {
    return graph::resolve_toolchain(toolchains(), want);
  }

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
  DeviceIndex device_index() const noexcept override {
    return impl_.device_index();
  }
  Result<DeviceBuffer> allocate(std::size_t b, MemoryClass c) override {
    return impl_.allocate(b, c);
  }
  void deallocate(DeviceBuffer& b) noexcept override { impl_.deallocate(b); }
  Result<std::size_t> sample_free_memory() const override {
    return impl_.sample_free_memory();
  }
  Result<DeviceClock> device_clock() const override {
    return impl_.device_clock();
  }
  Result<DeviceTimestamp> sample_device_time() const override {
    return impl_.sample_device_time();
  }
  Status copy_h2d(const void* s, DeviceBuffer& d, std::size_t n,
                  std::size_t off) override {
    return impl_.copy_h2d(s, d, n, off);
  }
  Status copy_d2h(const DeviceBuffer& s, void* d, std::size_t n,
                  std::size_t off) override {
    return impl_.copy_d2h(s, d, n, off);
  }
  Status copy_peer(const DeviceBuffer& s, DeviceBuffer& d, std::size_t n,
                   std::size_t soff, std::size_t doff) override {
    return impl_.copy_peer(s, d, n, soff, doff);
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
  std::span<const graph::KernelToolchain> toolchains() const noexcept override {
    return impl_.toolchains();
  }
  Result<void*> device_pointer(const DeviceBuffer& buf) const override {
    return impl_.device_pointer(buf);
  }

  Derived& impl() noexcept { return impl_; }

 private:
  Derived impl_;
};

// --- the device set --------------------------------------------------------
// What one process holds, as everything above a backend sees it. A scheduler
// binds one of these instead of a single backend, so the difference between one
// device and eight is how many members the set has and not which code path ran.
//
// A seam rather than a class because the layer that owns device lifetimes sits
// ABOVE the layer that dispatches on them: lse::place builds the set, lse::graph
// holds it, and graph must not depend on place to do so.
//
// Every method here is an index into the set. Nothing above the seam may name a
// device, so nothing here hands out a name.
class IDeviceSet {
 public:
  virtual ~IDeviceSet() = default;

  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  // The backend driving member `i`, which callers only ever ask for i < size().
  [[nodiscard]] virtual IBackend& device(std::size_t i) const = 0;
  // Where work goes when nothing has placed it. Always < size().
  [[nodiscard]] virtual std::size_t primary() const noexcept = 0;
  // Which member stamps this residency, or size() when no member does — the
  // answer for an unbound residency and for a device this set does not hold,
  // told apart by DeviceIndex::bound().
  [[nodiscard]] virtual std::size_t member_of(DeviceIndex d) const noexcept = 0;
  // Whether a kernel running on member `target` may read bytes with this
  // residency. Only the set can answer: it holds the members' identities and
  // whatever was measured or queried about the paths between them, and a
  // backend cannot see past its own device.
  //
  // An unbound residency is readable — nothing claims those bytes, so there is
  // nothing to violate. A residency this set cannot resolve is NOT: an
  // unanswerable question about two devices refuses rather than passes, because
  // the alternative is a kernel reading another device's memory and returning
  // numbers that look right.
  [[nodiscard]] virtual Status may_read(DeviceIndex held,
                                        std::size_t target) const = 0;

  // The residency member `i`'s buffers carry. Hoisted out of a binding loop by
  // callers, which is why it is worth having beside device().
  [[nodiscard]] DeviceIndex residency(std::size_t i) const {
    return device(i).device_index();
  }
};

// The set one backend is: itself, alone. What everything holding a single
// device uses to speak the set vocabulary, so the one-device case is the same
// code with size() == 1 rather than a second path beside it.
class SingleDevice final : public IDeviceSet {
 public:
  explicit SingleDevice(IBackend& backend) noexcept : backend_(&backend) {}

  std::size_t size() const noexcept override { return 1; }
  IBackend& device(std::size_t) const override { return *backend_; }
  std::size_t primary() const noexcept override { return 0; }
  std::size_t member_of(DeviceIndex d) const noexcept override {
    return d.bound() && d == backend_->device_index() ? std::size_t{0}
                                                      : std::size_t{1};
  }
  Status may_read(DeviceIndex held, std::size_t) const override {
    if (!held.bound() || held == backend_->device_index()) return OkStatus();
    return LSE_ERROR(kInvalidArgument, "bytes resident on device ",
                     std::to_string(held.value),
                     " cannot be read by work on device ",
                     std::to_string(backend_->device_index().value),
                     ": this set holds one device and it is not that one");
  }

 private:
  IBackend* backend_;
};

// --- device enumeration ----------------------------------------------------
// Asking what devices exist is a different question from binding one, and the
// answer must be available before any device is bound: an instance method
// would mean initialising a device to find out how many exist. So enumeration
// is a free function over the registry, and what it returns is a description,
// not a device — nothing here can allocate, launch or be dispatched to.

// FactSource and DeviceFact<T> — the "unknown is not zero" vocabulary every
// reported fact below is spelled in — live in lse/backend/resources.hpp,
// beside the kernel and ISA facts that use the same discipline.

// Whether one device can reach another's memory. Four answers rather than a
// bool: "not yet enabled" is a different fact from "never", and "nothing here
// can tell" is a third — a peer path that is assumed rather than queried is
// how a cost model comes to believe in a link that does not exist.
#define LSE_PEER_ACCESS_LIST(X)                                              \
  X(kUnknown, "unknown")         /* no reachable query answers this pair */  \
  X(kSelf, "self")               /* the same device */                       \
  X(kNo, "no")                   /* the runtime says never */                \
  X(kOnRequest, "on request")    /* reachable once access is granted */      \
  X(kYes, "yes")                 /* reachable as it stands */

LSE_DECLARE_ENUM(PeerAccess, std::uint8_t, LSE_PEER_ACCESS_LIST)

// One device a backend can drive, described without binding it.
//
// `backend` + `ordinal` is the device's whole identity to the engine and the
// only part of this a placement may consume: the rest is for a human reading a
// report. A product name is not an address — two boards of the same part share
// it, and an ordinal is not stable across a topology change either, which is
// what `uuid` and `pci_path` are here to say when the runtime can.
struct DeviceDescriptor {
  std::string backend;               // registry name, "hrx"
  int ordinal = 0;                   // index within that backend

  DeviceFact<std::string> product;   // "Radeon 8060S Graphics"
  DeviceFact<std::string> arch;      // "gfx1151"
  DeviceFact<std::string> uuid;      // stable across reboots when present
  DeviceFact<std::string> pci_path;  // "0000:bd:00.0"

  DeviceFact<std::size_t> total_memory;  // bytes
  // Bytes free, sampled while enumerating. A sample and not an identity: it
  // moves with every allocation on the device, this process's or another
  // process's, so a reader may act on it but must not memoize it.
  DeviceFact<std::size_t> free_memory;
  DeviceFact<std::uint32_t> compute_units;
  DeviceFact<std::uint32_t> max_threads_per_workgroup;
  DeviceFact<std::uint32_t> wavefront_size;
  DeviceFact<std::uint32_t> lds_bytes_per_workgroup;
  DeviceFact<std::uint32_t> queue_count;  // hardware queues, not streams
  DeviceFact<bool> unified_memory;

  // Reach to every device of this same backend, indexed by their ordinal, so
  // peers[ordinal] is this device's answer about itself. Empty when the
  // backend has no peer query at all, which is not the same as a row of
  // kUnknown: that row means the query exists and did not answer.
  std::vector<PeerAccess> peers;

  // What was asked and not answered, in the runtime's own terms, joined by
  // "; ". Empty when everything above is known. A hole a reader can name is
  // worth more than one it has to infer from a zero.
  std::string declined;

  // "hrx:0" — the form probe::parse_device_id round-trips.
  [[nodiscard]] std::string id() const;
  [[nodiscard]] std::string describe() const;
};

// Every device this backend can drive, in ordinal order. Empty is a valid
// answer and not an error: a backend can be present with no device attached.
// A backend with no enumerator of its own refuses.
using DeviceEnumerator = Result<std::vector<DeviceDescriptor>> (*)();

// Backends self-register from their own TU: linking one in is all that makes
// it selectable by name.
using BackendFactory = std::unique_ptr<IBackend> (*)();

void register_backend(std::string_view name, BackendFactory factory,
                      DeviceEnumerator enumerator = nullptr);
Result<std::unique_ptr<IBackend>> create_backend(std::string_view name);
Result<std::vector<DeviceDescriptor>> enumerate_devices(std::string_view name);
std::vector<std::string> available_backends();

Result<std::unique_ptr<IBackend>> create_default_backend();

// Backends to try, best first. A backend can construct and still fail to come
// up, so a caller that initializes should walk this rather than take only the
// first name. LSE_BACKEND, when set, is the sole candidate.
std::vector<std::string> default_backend_order();

struct BackendRegistrar {
  BackendRegistrar(std::string_view name, BackendFactory factory,
                   DeviceEnumerator enumerator) {
    register_backend(name, factory, enumerator);
  }
};

// The enumerator a backend supplies, or null. Detected from the type rather
// than named in the macro, so a backend gains enumeration by writing one
// static function and nothing else.
template <typename Derived>
[[nodiscard]] constexpr DeviceEnumerator device_enumerator_for() noexcept {
  if constexpr (requires { DeviceEnumerator{&Derived::enumerate_devices}; }) {
    return &Derived::enumerate_devices;
  } else {
    return nullptr;
  }
}

// No token pasting: Type may be a qualified name. One registration per TU.
#define LSE_REGISTER_BACKEND(name, Type)                                  \
  namespace {                                                             \
  const ::lse::backend::BackendRegistrar _lse_backend_registrar{          \
      name, []() -> std::unique_ptr<::lse::backend::IBackend> {           \
        return std::make_unique<::lse::backend::BackendAdapter<Type>>();  \
      },                                                                  \
      ::lse::backend::device_enumerator_for<Type>()};                     \
  }  // namespace

}  // namespace lse::backend
