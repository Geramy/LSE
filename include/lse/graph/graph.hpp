// Lazy op DAG, fusion partitioning, and scheduling.
//
// Ops record nodes and compute nothing. Work happens only when a host-visible
// read demands a value: item(), to_host(), operator<<, a data-dependent branch,
// or eval(). That is why a print splits a kernel in two — it forces a flush, so
// ops after it cannot fuse backwards across it.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/dtype.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/fallback.hpp"
#include "lse/graph/primitive.hpp"
#include "lse/graph/workgroup.hpp"
#include "lse/trace/record.hpp"

namespace lse::graph {

#define LSE_OPKIND_LIST(X) \
  X(kBuffer,        "buffer") \
  X(kConstant,      "constant") \
  X(kAdd,           "add") \
  X(kSub,           "sub") \
  X(kMul,           "mul") \
  X(kDiv,           "div") \
  X(kNeg,           "neg") \
  X(kExp,           "exp") \
  X(kLog,           "log") \
  X(kSqrt,          "sqrt") \
  X(kRsqrt,         "rsqrt") \
  X(kSiLU,          "silu") \
  X(kGELU,          "gelu") \
  X(kSigmoid,       "sigmoid") \
  X(kTanh,          "tanh") \
  X(kReLU,          "relu") \
  X(kCast,          "cast") \
  X(kClamp,         "clamp") \
  X(kWhere,         "where") \
  X(kSoftplus,      "softplus") \
  X(kL2Norm,        "l2_normalize") \
  X(kSum,           "sum") \
  X(kMax,           "max") \
  X(kMean,          "mean") \
  X(kRMS,           "rms_norm") \
  X(kSoftmax,       "softmax") \
  X(kLogSumExp,     "logsumexp") \
  X(kReshape,       "reshape") \
  X(kTranspose,     "transpose") \
  X(kBroadcast,     "broadcast") \
  X(kSlice,         "slice") \
  X(kConcat,        "concat") \
  X(kGather,        "gather") \
  X(kScatter,       "scatter") \
  X(kOverwriteSlice,"overwrite_slice") \
  X(kKvPageWrite,   "kv_page_write") \
  X(kRepeat,        "repeat") \
  X(kEmbedding,     "embedding") \
  X(kQuantEmbedding,"quant_embedding") \
  X(kMatMul,        "matmul") \
  X(kLinear,        "linear") \
  X(kQuantMatMul,   "quant_matmul") \
  X(kRoPE,          "rope") \
  X(kAttention,     "attention") \
  X(kGDNChunkScan,  "gdn_chunk_scan") \
  X(kCausalConv1d,  "causal_conv1d") \
  X(kConvTailShift, "conv_tail") \
  X(kMoEDispatch,   "moe_dispatch") \
  X(kMoECombine,    "moe_combine") \
  X(kTopK,          "topk") \
  X(kArgMax,        "argmax") \
  X(kAllReduce,     "all_reduce") \
  X(kAllGather,     "all_gather") \
  X(kReduceScatter, "reduce_scatter") \
  X(kAllToAll,      "all_to_all") \
  X(kBroadcastRank, "broadcast_rank") \
  X(kCustom,        "custom")

LSE_DECLARE_ENUM(OpKind, std::uint16_t, LSE_OPKIND_LIST)

[[nodiscard]] bool is_elementwise(OpKind k) noexcept;
[[nodiscard]] bool is_reduction(OpKind k) noexcept;
[[nodiscard]] bool is_structural(OpKind k) noexcept;
[[nodiscard]] bool is_collective(OpKind k) noexcept;
[[nodiscard]] bool is_barrier(OpKind k) noexcept;
[[nodiscard]] FusionClass fusion_class_of(OpKind k) noexcept;

class Node;
using NodePtr = std::shared_ptr<Node>;

// The member the enclosing ScopedMember named, or Node::kAnyMember when none
// is active. Distinct from preferred_member(), which resolves "nothing placed
// this" to the primary: a node has to keep the difference, because "belongs to
// member 0" and "belongs wherever its operands lead" are not the same claim
// and only the first one pins a tensor-split branch.
[[nodiscard]] std::uint16_t stamped_member() noexcept;

// A group-affine weight is three tensors, but every op that consumes a weight
// takes one Array. The packed plane carries the other two and its geometry
// here, so linear() and embedding() can dispatch on the storage format without
// every weight struct in lse::ops growing two more fields.
//
// Trace-time metadata only. The op builders read it and pass all three planes
// as ordinary inputs, so the partitioner, the emitter and the scheduler never
// see it and the packed plane stays a leaf buffer to all of them.
struct QuantPlanes {
  NodePtr scales;
  NodePtr biases;
  std::int32_t bits = 0;
  std::int32_t group_size = 0;
  // Weights per row. The packed plane's own last axis counts lanes, so this is
  // the only place the logical width survives.
  std::int64_t in_features = 0;
};

class Node {
 public:
  OpKind kind = OpKind::kBuffer;
  // Set once at construction from `kind`, or from the primitive for kCustom.
  // The partitioner reads this, never the enum, so a registered primitive fuses
  // exactly like a built-in.
  FusionClass fclass = FusionClass::kLeaf;
  const Primitive* prim = nullptr;   // non-null only for kCustom
  Shape shape;
  DType dtype = DType::kF32;
  std::vector<NodePtr> inputs;

  std::array<float, 4> attrs{};
  std::array<std::int32_t, 4> iattrs{};

  backend::DeviceBuffer buffer;
  bool materialized = false;

  // Non-null only on the packed plane of a group-affine weight.
  std::shared_ptr<const QuantPlanes> quant;

  // Host copy of `buffer`, for nodes the host has to read or write. Device
  // memory has no host address, so this is the only way the interpreter can
  // touch a value a kernel produced. It is allocated on first host access and
  // stays empty for anything that only ever moves between kernels — which is
  // every node once codegen covers the whole graph.
  std::vector<std::byte> host_mirror;
  // Exactly one side may be authoritative. Writing on the host clears
  // device_dirty and writing on the device clears host_dirty, because a sync in
  // the wrong direction would otherwise copy stale bytes over fresh ones. Both
  // true at once is the bug this pair is arranged to prevent.
  bool host_dirty = false;    // mirror holds writes the device has not seen
  bool device_dirty = false;  // device holds writes the mirror has not seen

  std::uint32_t consumer_count = 0;

  // Which pool member this node's work belongs to, or kAnyMember when nothing
  // placed it. Stamped at construction from the enclosing ScopedMember, so the
  // decision is made where the model is built -- where the layer, the shard and
  // the branch are all still known -- rather than reconstructed at dispatch
  // from which operand happens to be biggest.
  //
  // A tensor-split layer is several branches whose nodes differ ONLY in this
  // field and in which shard of the weights they read, so it has to travel on
  // the node: two branches of one layer are otherwise indistinguishable.
  static constexpr std::uint16_t kAnyMember = 0xFFFF;
  std::uint16_t member = kAnyMember;

  // Keeps kind and fclass in sync; setting kind alone silently makes a node
  // look like a leaf to the partitioner.
  void set_kind(OpKind k) noexcept {
    kind = k;
    fclass = fusion_class_of(k);
  }

  // Drives materialize-vs-recompute at a fan-out point.
  [[nodiscard]] std::uint64_t recompute_cost() const noexcept;
  [[nodiscard]] std::size_t element_count() const noexcept { return shape.elem_count(); }
};

class Array {
 public:
  Array() = default;
  explicit Array(NodePtr node) : node_(std::move(node)) {}

  static Array from_buffer(backend::DeviceBuffer buf, Shape shape, DType dtype);
  static Array zeros(Shape shape, DType dtype);
  static Array full(Shape shape, DType dtype, float value);

  [[nodiscard]] const Shape& shape() const noexcept { return node_->shape; }
  [[nodiscard]] DType dtype() const noexcept { return node_->dtype; }
  [[nodiscard]] const NodePtr& node() const noexcept { return node_; }
  [[nodiscard]] bool valid() const noexcept { return node_ != nullptr; }

  Status eval();
  // Same as eval() but leaves the value on the device. Use this at an internal
  // barrier (one block's output feeding the next) so a host-visible read does
  // not bounce the tensor through the mirror.
  Status materialize();
  Result<float> item();
  Status to_host(void* dst, std::size_t bytes);

  template <typename T>
  Result<std::vector<T>> to_vector();

 private:
  NodePtr node_;
};

// Evaluates: streaming is a host read.
std::ostream& operator<<(std::ostream& os, Array& a);

// Nodes emitted as one kernel. `outputs` escape the group; everything else
// stays in registers.
struct FusionGroup {
  std::vector<NodePtr> nodes;
  std::vector<NodePtr> inputs;
  std::vector<NodePtr> outputs;
  // What shaped the group. anchor is a display/signature tag; anchor_class is
  // what the promotion rule reads, because kCustom carries no class of its own.
  OpKind anchor = OpKind::kCustom;
  FusionClass anchor_class = FusionClass::kLeaf;
  // Dispatches the Workgroup that formed this group will issue. 1 when
  // several KernelPrimitives share a body; otherwise one per primitive.
  std::uint32_t launches = 1;
  // Whole decode or prefill phase: one staged kernel, not one launch per op.
  bool is_phase = false;

  // Identity of the computation's shape, independent of buffer addresses.
  // JIT cache key together with the arch string.
  [[nodiscard]] std::uint64_t signature() const noexcept;
};

class Partitioner {
 public:
  static std::vector<FusionGroup> partition(std::span<const NodePtr> roots);
  static std::vector<FusionGroup> partition(std::span<const NodePtr> roots,
                                            const backend::DeviceInfo* device);

  // One Workgroup per phase of the DAG (decode and/or prefill). The
  // model is ideally a single entry; emit still cuts each phase into
  // the FusionGroups `partition` returns.
  static std::vector<Workgroup> phases(std::span<const NodePtr> roots);
  static std::vector<Workgroup> phases(std::span<const NodePtr> roots,
                                       const backend::DeviceInfo* device);
  static FusionGroup phase_group(const Workgroup& wg,
                                 std::span<const NodePtr> roots);
  static std::vector<FusionGroup> phase_chunks(const FusionGroup& phase,
                                               std::size_t max_bindings);
  static std::vector<NodePtr> unmaterialized(std::span<const NodePtr> roots);

  // Expression / register fusion. Launch sharing is Workgroup::try_add —
  // linear, rms, conv, gdn, and slice are not launch barriers.
  //
  //   elementwise -> elementwise   fuse
  //   elementwise -> reduce        fuse, as the reduce's prologue
  //   reduce      -> elementwise   fuse only if the reduce output is a
  //                                broadcastable scalar/row (RMSNorm)
  //   matmul                       isolated; not workgroup-shareable
  //   matmul      -> elementwise   fuse as a matmul epilogue
  //   collective                   isolated, but schedulable async
  //   multi-consumer               materialize unless recompute_cost is low
  //
  // Exposed so tests assert fusion decisions directly instead of inferring
  // them from emitted source.
  static bool can_fuse(const Node& producer, const Node& consumer) noexcept;
};

class Program;

class Scheduler {
 public:
  // The devices this scheduler may run on. One member or eight is a difference
  // in size(), not in code path: everything below places a group on a member
  // and dispatches to it, and with one member the placement is the member.
  explicit Scheduler(backend::IDeviceSet& devices);
  // One device, for a caller that holds a backend rather than a set — a probe
  // measuring its own hardware, a tracer, a test. It is the same thing: the set
  // that backend is.
  explicit Scheduler(backend::IBackend& backend);
  ~Scheduler();

  Status eval(std::span<const NodePtr> roots, bool pull_host = true);
  // When `plan` is set, replay and retain write that Program instead of the
  // scheduler's leftover one. Same-root reuse is how decode avoids rebuild.
  Status eval(std::span<const NodePtr> roots, bool pull_host, Program* plan);

  // Device-first: every group that can be compiled is compiled and dispatched
  // on the selected backend. The host path runs only when a group cannot be
  // expressed as a kernel (missing op, or a primitive with no device impl).
  // Off forces the reference interpreter, which is what the JIT is diffed
  // against.
  enum class Mode : std::uint8_t { kDeviceFirst, kHostOnly };
  void set_mode(Mode m) noexcept { mode_ = m; }
  [[nodiscard]] Mode mode() const noexcept { return mode_; }

  // Which dialect this run would rather its kernels were written in.
  //
  // A runtime value naming a resource, the way the pool names devices: unset
  // is the ordinary case and every dialect it can name is compiled into this
  // binary either way. A member that does not declare the named dialect is
  // given its own first choice instead of failing, so naming one cannot turn a
  // run that would have worked into a run that does not.
  void set_dialect(Dialect d) noexcept { dialect_ = d; }
  void clear_dialect() noexcept { dialect_.reset(); }
  [[nodiscard]] DialectPreference dialect() const noexcept { return dialect_; }
  // What `member` will actually be handed, preference resolved against what
  // that device declares. nullptr when the member has no codegen at all.
  [[nodiscard]] const KernelToolchain* toolchain(
      std::size_t member) const noexcept {
    return devices_.device(member).toolchain(dialect_);
  }

  // Why a node cannot run on `backend`, or empty if it can.
  static std::string device_gap(const Node& node, const backend::IBackend& backend);

  // Defaults to default_fallback_chain(); override per scheduler to route
  // declined nodes somewhere else entirely.
  void set_fallback_chain(FallbackChain* chain) noexcept { fallbacks_ = chain; }
  [[nodiscard]] FallbackChain& fallback_chain() const noexcept;

 private:
  // The step itself. eval() is the wrapper that owns the step total, the
  // unattributed remainder and the accumulation, so those close on exactly one
  // path and a step that returned an error still reports where its time went.
  Status eval_step(std::span<const NodePtr> roots, bool pull_host,
                   Program* plan);
  Status try_dispatch_group(const FusionGroup& group, backend::Stream stream,
                            std::size_t member);
  // Which member of the set runs this group: the one already holding its
  // operands. Not a cost decision — a group whose inputs are resident on one
  // device has nowhere else to run until something moves them, and moving them
  // is what a Planner decides. With one member this is that member.
  [[nodiscard]] std::size_t member_for(const FusionGroup& group) const noexcept;
  // Bring one operand onto `member`, copying it there if it is held elsewhere.
  Status make_local(Node& n, std::size_t member);
  // Refuses when a binding's bytes are held by a device the target cannot read.
  Status check_residency(std::span<const backend::BufferRef> bindings,
                         std::size_t member) const;
  // Releases through the member that allocated it, which the buffer names.
  void release(backend::DeviceBuffer& buf) const noexcept;

 public:

  struct Trace {
    std::vector<std::string> group_descriptions;
    std::uint32_t kernels_launched = 0;
    std::uint32_t nodes_evaluated = 0;
    std::uint32_t collectives_issued = 0;
    // Nodes the device backend could not run, executed on the host instead.
    // Counted and named because a silent fallback is a silent perf cliff.
    std::uint32_t host_fallbacks = 0;
    std::vector<std::string> fallback_reasons;
    // Handler that took each fallen-back node, in order.
    std::vector<std::string> fallback_handlers;
    // Nodes claimed by an intercepting handler that the backend could have run.
    std::uint32_t intercepted = 0;
    // Groups compiled and dispatched as kernels vs run on the host.
    std::uint32_t device_groups = 0;
    std::uint32_t host_groups = 0;
    // Phase Workgroups the partitioner planned (1 = whole model, 2 =
    // prefill + decode). ideal_launches is 1 per connected device phase.
    std::uint32_t phase_groups = 0;
    std::uint32_t phase_ideal_launches = 0;
    // Reshape/contiguous-slice groups that reused an existing buffer.
    std::uint32_t views_aliased = 0;
    // Activation slots the Workgroup recycled vs allocated.
    std::uint32_t slots_reused = 0;
    std::uint32_t slots_allocated = 0;
    std::vector<std::string> host_group_reasons;
    // Execution streams the step's groups were spread across, how many
    // cross-stream waits that cost, and the longest chain of dependent groups
    // in it. The chain is the floor: no number of streams can make a step
    // shorter than its own dependencies, so `chain` against `device_groups` is
    // the spread that was actually available, not the one that was hoped for.
    std::uint32_t streams_used = 1;
    std::uint32_t stream_waits = 0;
    // Operands fetched from another device, and what they cost. A layer split
    // should show one migration per block boundary per step; more than that
    // means groups are landing away from their weights.
    std::uint32_t peer_migrations = 0;
    std::uint64_t peer_bytes = 0;
    std::uint32_t stream_chain = 0;
    // Compute time that overlapped an in-flight collective.
    std::uint64_t overlap_ns = 0;
    // Where a step's host time went, as spans that do not contain one another.
    //
    // Every one of these is std::chrono::steady_clock around host code, and
    // `clock` on each says so. NONE of them is device time and none ever will
    // be: the only device interval this engine could report is the execution
    // slice buried inside host_wait, and that has to arrive as `device_exec`
    // below rather than as a relabelling of a host counter.
    //
    // DISJOINT BY CONSTRUCTION, which the previous four counters were not:
    // partition_ns bracketed the whole dispatch loop and so contained emit and
    // launch, and emit_ns bracketed the JIT and so contained compile. Summing
    // them double-counted, and a cold prefill's `emit` was 98.4% compile.
    struct Spans {
      // Building the groups: Partitioner::phases, the staging walk, slot
      // planning and binding, and Partitioner::partition on the host-fallback
      // path. WHAT to run. A replay skips all of it, so this is ~0 on a
      // steady-state decode token - which is the property the old partition_ns
      // appeared to have only because it was written under `if (!replayed)`.
      trace::HostDuration partition;
      // Setting the step up: releasing last step's pointer tables, the
      // unmaterialized walk, plan_streams, and the entry-barrier snapshot.
      // WHERE and in what order. Runs on every step, replay included, and the
      // old counters charged it to nothing.
      trace::HostDuration schedule;
      // IKernelEmitter::emit and nothing else.
      trace::HostDuration emit;
      // The cache deciding whether this kernel already exists: the source dump,
      // try_get, the meta/object read and the load. The compiler call nested
      // inside get_or_compile is subtracted out into jit_compile.
      trace::HostDuration jit_lookup;
      // IKernelCompiler::compile, as measured inside JitCache and nowhere else.
      trace::HostDuration jit_compile;
      // Between emit and launch: in-place aliasing, output buffers, constant
      // materialization, the H2D of host-dirty inputs, the constants block, the
      // pointer table's allocate/copy_h2d, and check_residency. If a copy in
      // here is ever timed on the device it becomes its own span, not part of
      // this one.
      trace::HostDuration bind;
      // IBackend::launch. Recording an AQL packet into an open command buffer,
      // plus a submit one time in flush_interval_ - so it moves with the
      // submission policy and is NOT kernel start latency. Sweeping
      // LSE_FLUSH_INTERVAL over {0,64,16,1} moves it 0.212 -> 0.382 ms/token
      // for identical device work.
      trace::HostDuration submit;
      // Host wall around a blocking IBackend::synchronize. Four things in one
      // number - submission of whatever is still unflushed, the drain of what
      // was already submitted, the execution still running, and the interrupt
      // wake - and the host cannot separate them. Named host_wait because that
      // is what was measured; the execution part is device_exec's job.
      trace::HostDuration host_wait;
      // interpreter::sync_from_device of the roots, which is the D2H a
      // host-visible read forces.
      trace::HostDuration readback;
      // The interpreter and the fallback handlers running nodes on the host.
      // Zero on the device path, and the bulk of a host-only step.
      trace::HostDuration host_exec;
      // The whole of eval(), so the spans above can be checked against it.
      trace::HostDuration step;
      // step minus the spans above: the dispatch loop's own bookkeeping,
      // cross-stream event record/wait, view aliasing, group descriptions, and
      // this instrumentation's own clock reads. Reported rather than left
      // missing - a reader has to be able to tell what was not measured.
      // Saturates at zero, so `unattributed == 0` with a positive step is how
      // an overlap would show up.
      trace::HostDuration unattributed;
    };
    Spans spans;

    // The device-side execution interval that host_wait contains.
    //
    // It stays UNKNOWN and duration_ns() refuses. `clock` is filled only by a
    // reader that took both ticks off the device, and no backend here can:
    // HRX publishes its agent counter (kDeviceAgent, 99,810,000 Hz, 64 bits)
    // and declines to read a tick. Copying that published clock in would make
    // known() true and duration_ns() return 0, and filling it from steady_clock
    // is the substitution the whole clock-domain seam exists to prevent.
    trace::DeviceSpan device_exec;

    // The old names, kept only so src/runtime/generator.cpp keeps compiling
    // while it belongs to another change in flight. Read `spans` instead:
    // these are the same numbers under names that mislead - `launch_ns` is
    // submission, `sync_ns` is a host wall around a blocking wait. Delete all
    // four once generator.hpp carries Spans.
    std::uint64_t partition_ns = 0;
    std::uint64_t emit_ns = 0;
    std::uint64_t launch_ns = 0;
    std::uint64_t sync_ns = 0;
    bool replayed = false;
  };

  struct JitStats {
    std::uint64_t memory_hits = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t compiles = 0;
    std::uint64_t compile_ns = 0;
  };
  [[nodiscard]] JitStats jit_stats() const noexcept;
  [[nodiscard]] const Trace& last_trace() const noexcept { return trace_; }
  [[nodiscard]] const Trace& accumulated_trace() const noexcept { return acc_; }
  void reset_accumulated_trace() noexcept { acc_ = {}; }

  // Where data goes when the caller has no opinion about which device holds it
  // — bulk-loading weights, rather than computing them. The set's primary,
  // which is the whole answer while nothing above states a residency.
  [[nodiscard]] backend::IBackend& backend() const noexcept {
    return devices_.device(devices_.primary());
  }
  [[nodiscard]] backend::IDeviceSet& devices() const noexcept {
    return devices_;
  }

 private:
  // Declared before devices_ so it is built before the reference to it is
  // bound: the single-backend constructor makes the set it hands itself.
  std::unique_ptr<backend::SingleDevice> own_set_;
  backend::IDeviceSet& devices_;
  Mode mode_ = Mode::kDeviceFirst;
  DialectPreference dialect_;
  FallbackChain* fallbacks_ = nullptr;
  Trace trace_;
  Trace acc_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The device set default_scheduler() binds.
//
// A function pointer rather than a call into the layer that owns device
// lifetimes: that layer (lse::place) sits ABOVE this one and registers itself
// here at static-initialization time, so a binary that links it gets the whole
// set and one that does not gets the single-device fallback below. Registering
// twice keeps the first.
using DeviceSetFactory = backend::IDeviceSet* (*)();
void register_device_set_factory(DeviceSetFactory factory);

// null when no backend could be created or initialized.
Scheduler* default_scheduler();

// Which member of that scheduler's device set new allocations should land on.
//
// A model spanning a pool is a placement decision made where the layers are
// known, not where the bytes are allocated: weight loading walks the blocks in
// order and names the member each belongs to, and the tensors built underneath
// land there without every constructor in between learning about devices.
// Unset -- everything except that walk -- means the primary, so a single-device
// run behaves exactly as it did.
//
// Scoped rather than set-and-forget: an early return part-way through a layer
// must not leave the next one loading onto the wrong GPU.
[[nodiscard]] std::size_t preferred_member() noexcept;

// How a model spans a pool of more than one device.
//
// kLayer gives each member a contiguous run of layers. The members take turns
// -- layer n+1 needs layer n's output -- so it buys capacity and not latency,
// and it is what a pool falls back to when a layer's weights will not divide.
//
// kTensor gives every member a slice of EVERY layer: the output features of
// the projections that start a block, the input features of the ones that end
// it. Both members then run the same layer at the same time on their own half
// of the weights, and the partial sums are added. That is the one that makes
// two devices' memory bandwidth add rather than alternate.
enum class SplitScheme : std::uint8_t { kNone, kLayer, kTensor };

[[nodiscard]] std::string_view to_string(SplitScheme s) noexcept;

// What the pool settled on, and the setter the model calls once it knows
// whether the shapes divide. Nothing reads a flag: kTensor is chosen when it
// can be, and the fallback reports itself.
[[nodiscard]] SplitScheme split_scheme() noexcept;
void set_split_scheme(SplitScheme s) noexcept;

// Holds a different scheme for a nested submodel. Not a switch for turning a
// split off: it states that a submodel is not part of the split at all, which
// is what a speculative draft head is. Its single block would otherwise be
// sharded while it carries one mixer state, and the shards would index past it.
class ScopedSplitScheme {
 public:
  explicit ScopedSplitScheme(SplitScheme s) noexcept;
  ~ScopedSplitScheme();
  ScopedSplitScheme(const ScopedSplitScheme&) = delete;
  ScopedSplitScheme& operator=(const ScopedSplitScheme&) = delete;

 private:
  SplitScheme previous_;
};

class ScopedMember {
 public:
  explicit ScopedMember(std::size_t member) noexcept;
  ~ScopedMember();
  ScopedMember(const ScopedMember&) = delete;
  ScopedMember& operator=(const ScopedMember&) = delete;

 private:
  std::size_t previous_;
};

}  // namespace lse::graph
