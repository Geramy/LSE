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
#include "lse/graph/sharding.hpp"
#include "lse/graph/workgroup.hpp"

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
  X(kRepeat,        "repeat") \
  X(kEmbedding,     "embedding") \
  X(kMatMul,        "matmul") \
  X(kLinear,        "linear") \
  X(kQuantMatMul,   "quant_matmul") \
  X(kRoPE,          "rope") \
  X(kAttention,     "attention") \
  X(kGDNChunkScan,  "gdn_chunk_scan") \
  X(kCausalConv1d,  "causal_conv1d") \
  X(kMoEDispatch,   "moe_dispatch") \
  X(kMoECombine,    "moe_combine") \
  X(kTopK,          "topk") \
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
  Sharding sharding;
  std::vector<NodePtr> inputs;

  std::array<float, 4> attrs{};
  std::array<std::int32_t, 4> iattrs{};

  backend::DeviceBuffer buffer;
  bool materialized = false;

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

  // Why a node cannot run on `backend`, or empty if it can.
  static std::string device_gap(const Node& node, const backend::IBackend& backend);

  // Defaults to default_fallback_chain(); override per scheduler to route
  // declined nodes somewhere else entirely.
  void set_fallback_chain(FallbackChain* chain) noexcept { fallbacks_ = chain; }
  [[nodiscard]] FallbackChain& fallback_chain() const noexcept;

 private:
  Status try_dispatch_group(const FusionGroup& group);

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
    // Compute time that overlapped an in-flight collective.
    std::uint64_t overlap_ns = 0;
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

  // The backend this scheduler runs on, for callers that need to place data
  // themselves — bulk-loading weights, rather than computing them.
  [[nodiscard]] backend::IBackend& backend() const noexcept { return backend_; }

 private:
  backend::IBackend& backend_;
  Mode mode_ = Mode::kDeviceFirst;
  FallbackChain* fallbacks_ = nullptr;
  Trace trace_;
  Trace acc_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// null when no backend could be created or initialized.
Scheduler* default_scheduler();

}  // namespace lse::graph
