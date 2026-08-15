// Tensor partitioning. Emits collective nodes into the graph; knows nothing
// about how bytes move (see dist/transport.hpp).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"
#include "lse/dist/transport.hpp"

namespace lse::graph {

enum class OpKind : std::uint16_t;

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
  [[nodiscard]] std::int32_t axis_index(std::string_view name) const noexcept;
  [[nodiscard]] std::int32_t coordinate(dist::Rank rank, std::size_t axis) const noexcept;

  // Ranks sharing every coordinate but `axis` — the group a collective along
  // that axis reduces over.
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

  [[nodiscard]] bool operator==(const AxisPlacement&) const noexcept = default;
};

struct Sharding {
  std::array<AxisPlacement, DeviceMesh::kMaxAxes> axes{};
  std::uint8_t axis_count = 0;

  [[nodiscard]] bool is_fully_replicated() const noexcept;
  [[nodiscard]] bool has_partial() const noexcept;
  [[nodiscard]] Shape local_shape(const Shape& global, const DeviceMesh& mesh,
                                  dist::Rank rank) const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] bool operator==(const Sharding&) const noexcept = default;
};

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

  [[nodiscard]] std::size_t traffic_bytes(const Shape& global, DType dtype,
                                          const DeviceMesh& mesh) const noexcept;
};

// Infers output sharding from input shardings and reports the reshards needed
// to make an op legal.
//
// The consequential rule: matmul(replicated, shard[n]) stays sharded with no
// collective, while matmul(shard[k], shard[k]) yields kPartial and costs an
// all-reduce. Pairing one of each is why attention and the FFN are split as a
// column/row sandwich — one all-reduce per pair instead of per layer.
class ShardingPropagator {
 public:
  struct Result {
    Sharding output;
    std::vector<std::pair<std::size_t, Reshard>> input_reshards;
    Reshard output_reshard;
  };

  static Result propagate(OpKind op, std::span<const Sharding> inputs,
                          std::span<const Shape> input_shapes,
                          const DeviceMesh& mesh);

  static std::vector<Reshard> plan(const Sharding& from, const Sharding& to,
                                   const Shape& global, DType dtype,
                                   const DeviceMesh& mesh);
};

namespace strategy {

Sharding expert_parallel(std::uint8_t mesh_axis);
Sharding vocab_parallel(std::uint8_t mesh_axis);
Sharding column_parallel(std::uint8_t mesh_axis);
Sharding row_parallel(std::uint8_t mesh_axis);
Sharding data_parallel(std::uint8_t mesh_axis);

// Replicated when `heads` does not divide `axis_size`, so callers fall back
// rather than silently misshard. b1.5 has 2 KV heads, capping this at 2-way.
Result<Sharding> head_parallel(std::uint8_t mesh_axis, std::int32_t heads,
                               std::int32_t axis_size);

}  // namespace strategy

}  // namespace lse::graph
