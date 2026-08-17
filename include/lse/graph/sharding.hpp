// Where a tensor's bytes ended up, and what changing your mind about that
// costs. This is solver *output*, never solver input.
//
// Nothing above the compiler picks a scheme. Expert, tensor, pipeline and data
// parallelism are windows over different dimensions of one iteration space,
// and which one you get is what the partitioner chose from measured cost and
// declared constraints — see PLAN.md, "Distribution and code generation are
// one decision". This header is the vocabulary that decision is *reported* in.
// A `strategy::column_parallel()` free function would be the opposite shape —
// a caller naming the answer — so there is not one.
//
// Knows nothing about how bytes move: a Reshard names a collective and the
// volume it carries, and dist/ decides the algorithm, the wire format and the
// wall time. The one dependency in the other direction is vocabulary only
// (dist::Rank, dist::ReduceOp), so that a rank means the same thing on both
// sides of the boundary.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/shape.hpp"
#include "lse/dist/transport.hpp"

namespace lse::graph {

enum class OpKind : std::uint16_t;

// Named axes over the ranks of one pool. Ranks are numbered row-major, so the
// axis listed last is contiguous: put the axis whose collectives are hottest
// there and its groups land on the fastest links.
class DeviceMesh {
 public:
  static constexpr std::size_t kMaxAxes = 4;

  struct Axis {
    std::string name;
    std::int32_t size = 1;
  };

  DeviceMesh() = default;
  explicit DeviceMesh(std::vector<Axis> axes) : axes_(std::move(axes)) {}

  [[nodiscard]] std::size_t rank_count() const noexcept;
  [[nodiscard]] std::size_t axis_count() const noexcept { return axes_.size(); }
  [[nodiscard]] const Axis& axis(std::size_t i) const noexcept { return axes_[i]; }

  // 1 for an axis that does not exist, which is "no peers" — the answer that
  // makes every collective along it free rather than the answer that makes it
  // infinite.
  [[nodiscard]] std::int32_t axis_size(std::size_t i) const noexcept {
    return i < axes_.size() ? axes_[i].size : 1;
  }

  // -1 when there is no such axis.
  [[nodiscard]] std::int32_t axis_index(std::string_view name) const noexcept;
  // -1 when the axis or the rank is outside the mesh.
  [[nodiscard]] std::int32_t coordinate(dist::Rank rank,
                                        std::size_t axis) const noexcept;

  // Ranks sharing every coordinate but `axis` — the group a collective along
  // that axis reduces over, in coordinate order. Empty when `rank` or `axis`
  // is outside the mesh.
  [[nodiscard]] std::vector<dist::Rank> group_along(dist::Rank rank,
                                                    std::size_t axis) const;

 private:
  std::vector<Axis> axes_;
};

enum class Placement : std::uint8_t {
  kReplicated,
  kSharded,
  // Holds an unreduced partial contribution: the state a contraction over a
  // sharded dimension leaves you in before its all-reduce.
  kPartial,
};

struct AxisPlacement {
  Placement placement = Placement::kReplicated;
  std::int8_t tensor_dim = -1;
  dist::ReduceOp reduce_op = dist::ReduceOp::kSum;

  static AxisPlacement replicated() noexcept { return {}; }
  static AxisPlacement shard(std::int8_t dim) noexcept {
    return {Placement::kSharded, dim, dist::ReduceOp::kSum};
  }
  static AxisPlacement partial(dist::ReduceOp op = dist::ReduceOp::kSum) noexcept {
    return {Placement::kPartial, -1, op};
  }

  [[nodiscard]] bool is_replicated() const noexcept {
    return placement == Placement::kReplicated;
  }

  [[nodiscard]] bool operator==(const AxisPlacement&) const noexcept = default;
};

// One placement per mesh axis. Two axes may shard the same tensor dimension,
// which is how a 2-D mesh splits one weight matrix both ways.
struct Sharding {
  std::array<AxisPlacement, DeviceMesh::kMaxAxes> axes{};
  std::uint8_t axis_count = 0;

  [[nodiscard]] bool is_fully_replicated() const noexcept;
  [[nodiscard]] bool has_partial() const noexcept;

  // The extent this rank actually holds. Uneven splits give the low
  // coordinates the extra element, so the local shapes tile the global one
  // exactly; a dimension shorter than its axis leaves the high coordinates
  // holding nothing, which is empty, not an error.
  //
  // A placement the mesh cannot honour — an axis past the mesh's, a tensor_dim
  // past the shape's — is ignored, so an ill-formed Sharding degrades to
  // replicated. That direction over-allocates; the other one silently returns
  // an extent nobody holds.
  [[nodiscard]] Shape local_shape(const Shape& global, const DeviceMesh& mesh,
                                  dist::Rank rank) const;

  // Per mesh axis: "R", "S<dim>", or "P(<op>)" — e.g. "[S1,R]".
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] bool operator==(const Sharding&) const noexcept = default;
};

// One collective along one mesh axis. What it does not carry is deliberate:
// no algorithm, no wire dtype, no latency. `dist::select_all_reduce` owns
// those, and it prices the buffer it is handed — so `traffic_bytes` must not
// be fed to it as an element count or the ring factor is applied twice.
struct Reshard {
  enum class Kind : std::uint8_t {
    kNone,
    kAllReduce,      // kPartial -> kReplicated
    kReduceScatter,  // kPartial -> kSharded
    kAllGather,      // kSharded -> kReplicated
    kAllToAll,       // kSharded(a) -> kSharded(b)
    kBroadcast,
  };

  Kind kind = Kind::kNone;
  std::uint8_t mesh_axis = 0;
  std::int8_t from_dim = -1;
  std::int8_t to_dim = -1;
  dist::ReduceOp reduce_op = dist::ReduceOp::kSum;

  // Payload bytes crossing one rank's link, for a tensor whose *global* shape
  // is `global`. The kinds differ by more than a constant and that is the
  // whole point of comparing them: an all-to-all moves 1/P as much as an
  // all-gather of the same tensor, which is why a free parallel axis beats
  // splitting a contraction whenever the graph has one.
  [[nodiscard]] std::size_t traffic_bytes(const Shape& global, DType dtype,
                                          const DeviceMesh& mesh) const noexcept;
};

namespace detail {

// Compiler internals. The partitioner runs this; nothing above graph/ does,
// and nothing above graph/ may state its result either.
//
// Propagation, not search: given the shardings the operands already have, it
// answers what the result is sharded like and which collectives that costs.
// Choosing the operand shardings in the first place is the solver's job
// (graph/partition.*), and it will call this to price its candidates.
//
// The consequential rule: matmul(replicated, shard[n]) stays sharded with no
// collective, while matmul(shard[k], shard[k]) yields kPartial and costs an
// all-reduce. Pairing one of each is why attention and the FFN are split as a
// column/row sandwich — one all-reduce per pair instead of per layer.
class ShardingPropagator {
 public:
  struct Result {
    Sharding output;
    // Several entries may share an index: a 2-D mesh can need a different
    // collective on each of its axes for the same operand.
    std::vector<std::pair<std::size_t, Reshard>> input_reshards;
    // Never populated by `propagate`. A kPartial result is a legal state to
    // leave the value in — reducing it is the consumer's decision, and
    // deferring it past the next linear op is the optimization. It exists so a
    // caller that has decided can record what it chose.
    Reshard output_reshard;
  };

  // `reduce_dim` is the tensor dimension a reduction collapses (Node::iattrs[0]
  // for the ops that have one); -1 means the last, matching ops.hpp's default.
  // `keepdims` decides whether the surviving shard dimensions shift down.
  // Both are ignored by every other op kind.
  static Result propagate(OpKind op, std::span<const Sharding> inputs,
                          std::span<const Shape> input_shapes,
                          const DeviceMesh& mesh, std::int8_t reduce_dim = -1,
                          bool keepdims = false);

  // The collectives that carry `from` to `to`, in the order they must run.
  //
  // No shape or dtype, because there is nothing here for a cost model to
  // choose between. Per axis the route is forced — a reduce-scatter always
  // beats an all-reduce then a slice, an all-to-all always beats an all-gather
  // then a slice — and across axes the order is a rule rather than a search:
  // shrink or hold before you grow, and among the gathers it does not matter,
  // since each byte a rank ends up holding arrives exactly once whichever axis
  // gathers first. Pricing happens between whole plans, via
  // `Reshard::traffic_bytes`, not inside one.
  static std::vector<Reshard> plan(const Sharding& from, const Sharding& to,
                                   const DeviceMesh& mesh);
};

}  // namespace detail

}  // namespace lse::graph
