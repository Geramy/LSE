#include "lse/graph/sharding.hpp"

#include <algorithm>
#include <cstddef>

#include "lse/graph/graph.hpp"

namespace lse::graph {

namespace {

std::size_t clamped_size(std::int32_t n) noexcept {
  return n > 1 ? static_cast<std::size_t>(n) : 1;
}

// Ranks are numbered row-major over the axes, so the last axis is contiguous.
std::size_t axis_stride(const DeviceMesh& mesh, std::size_t axis) noexcept {
  std::size_t s = 1;
  for (std::size_t i = mesh.axis_count(); i-- > axis + 1;) {
    s *= clamped_size(mesh.axis(i).size);
  }
  return s;
}

std::string_view reduce_op_name(dist::ReduceOp op) noexcept {
  switch (op) {
    case dist::ReduceOp::kSum: return "sum";
    case dist::ReduceOp::kProd: return "prod";
    case dist::ReduceOp::kMin: return "min";
    case dist::ReduceOp::kMax: return "max";
    case dist::ReduceOp::kAvg: return "avg";
  }
  return "?";
}

// A partial block still occupies a whole block on the wire, so round up rather
// than take dtype_storage_bytes' zero — a reshard that reports zero bytes reads
// as free, which is the one answer a cost model must never be handed.
std::size_t wire_bytes(DType dtype, std::size_t elems) noexcept {
  const DTypeInfo& info = dtype_info(dtype);
  if (!info.is_quantized) return elems * info.size_bytes;
  const std::size_t per = info.block_elems != 0 ? info.block_elems : 1;
  return ((elems + per - 1) / per) * info.block_bytes;
}

}  // namespace

std::size_t DeviceMesh::rank_count() const noexcept {
  std::size_t n = 1;
  for (const Axis& a : axes_) n *= clamped_size(a.size);
  return n;
}

std::int32_t DeviceMesh::axis_index(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < axes_.size(); ++i) {
    if (axes_[i].name == name) return static_cast<std::int32_t>(i);
  }
  return -1;
}

std::int32_t DeviceMesh::coordinate(dist::Rank rank,
                                    std::size_t axis) const noexcept {
  if (axis >= axes_.size() || rank < 0) return -1;
  if (static_cast<std::size_t>(rank) >= rank_count()) return -1;
  const std::size_t stride = axis_stride(*this, axis);
  const std::size_t size = clamped_size(axes_[axis].size);
  return static_cast<std::int32_t>((static_cast<std::size_t>(rank) / stride) % size);
}

std::vector<dist::Rank> DeviceMesh::group_along(dist::Rank rank,
                                                std::size_t axis) const {
  const std::int32_t coord = coordinate(rank, axis);
  if (coord < 0) return {};
  const auto stride = static_cast<dist::Rank>(axis_stride(*this, axis));
  const auto size = static_cast<std::int32_t>(clamped_size(axes_[axis].size));
  const dist::Rank base = rank - coord * stride;
  std::vector<dist::Rank> group;
  group.reserve(static_cast<std::size_t>(size));
  for (std::int32_t i = 0; i < size; ++i) group.push_back(base + i * stride);
  return group;
}

namespace {

std::size_t live_axes(std::uint8_t declared) noexcept {
  return std::min<std::size_t>(declared, DeviceMesh::kMaxAxes);
}

AxisPlacement placement_on(const Sharding& s, std::size_t axis) noexcept {
  return axis < live_axes(s.axis_count) ? s.axes[axis] : AxisPlacement{};
}

}  // namespace

bool Sharding::is_fully_replicated() const noexcept {
  for (std::size_t i = 0; i < live_axes(axis_count); ++i) {
    if (!axes[i].is_replicated()) return false;
  }
  return true;
}

bool Sharding::has_partial() const noexcept {
  for (std::size_t i = 0; i < live_axes(axis_count); ++i) {
    if (axes[i].placement == Placement::kPartial) return true;
  }
  return false;
}

Shape Sharding::local_shape(const Shape& global, const DeviceMesh& mesh,
                            dist::Rank rank) const {
  std::array<std::int64_t, kMaxRank> dims{};
  for (std::size_t i = 0; i < global.rank(); ++i) dims[i] = global.dim(i);

  for (std::size_t a = 0; a < live_axes(axis_count); ++a) {
    const AxisPlacement& p = axes[a];
    if (p.placement != Placement::kSharded) continue;
    if (p.tensor_dim < 0 ||
        static_cast<std::size_t>(p.tensor_dim) >= global.rank()) {
      continue;
    }
    const auto parts = static_cast<std::int64_t>(clamped_size(mesh.axis_size(a)));
    if (parts <= 1) continue;
    const std::int32_t coord = mesh.coordinate(rank, a);
    if (coord < 0) continue;
    // Each axis splits the extent the axes before it left, so nested uneven
    // splits still tile: 10 over a 2x3 mesh is 5|5 then 2|2|1 and 2|2|1.
    std::int64_t& d = dims[static_cast<std::size_t>(p.tensor_dim)];
    d = d / parts + (coord < d % parts ? 1 : 0);
  }

  Shape local;
  for (std::size_t i = 0; i < global.rank(); ++i) local.push_back(dims[i]);
  return local;
}

std::string Sharding::to_string() const {
  std::string out = "[";
  for (std::size_t i = 0; i < live_axes(axis_count); ++i) {
    if (i != 0) out += ',';
    switch (axes[i].placement) {
      case Placement::kReplicated:
        out += 'R';
        break;
      case Placement::kSharded:
        out += 'S';
        out += std::to_string(static_cast<int>(axes[i].tensor_dim));
        break;
      case Placement::kPartial:
        out += "P(";
        out += reduce_op_name(axes[i].reduce_op);
        out += ')';
        break;
    }
  }
  out += ']';
  return out;
}

std::size_t Reshard::traffic_bytes(const Shape& global, DType dtype,
                                   const DeviceMesh& mesh) const noexcept {
  const std::size_t p = clamped_size(mesh.axis_size(mesh_axis));
  if (kind == Kind::kNone || p <= 1) return 0;
  const std::size_t bytes = wire_bytes(dtype, global.elem_count());
  if (bytes == 0) return 0;
  // Rounds up for the same reason wire_bytes does: a tensor smaller than the
  // mesh makes bytes*(P-1) smaller than the divisor, and truncating that to
  // zero prices a collective that actually runs as free.
  const auto share = [](std::size_t num, std::size_t den) noexcept {
    return (num + den - 1) / den;
  };
  switch (kind) {
    // A reduce-scatter then an all-gather: each rank sends and receives its
    // (P-1)/P share twice.
    case Kind::kAllReduce:
      return share(2 * bytes * (p - 1), p);
    case Kind::kReduceScatter:
    case Kind::kAllGather:
      return share(bytes * (p - 1), p);
    // The one that is cheap by a factor of P: a rank only holds bytes/P to
    // begin with and keeps 1/P of that. This is why an expert axis beats
    // splitting a contraction whenever the graph has one.
    case Kind::kAllToAll:
      return share(bytes * (p - 1), p * p);
    // Every non-root rank receives the whole tensor; nothing was distributed
    // beforehand to make it smaller.
    case Kind::kBroadcast:
      return bytes;
    case Kind::kNone:
      return 0;
  }
  return 0;
}

namespace detail {

namespace {

using ReshardList = std::vector<std::pair<std::size_t, Reshard>>;

// A dimension index normalised to the trailing end. Two operands of different
// rank can only agree about "the same axis" this way: under NumPy trailing-dim
// broadcast, dim 0 of a [K] is not dim 0 of an [M,K].
constexpr std::int32_t kNoDim = 1 << 20;

std::int32_t trailing(std::int8_t dim, std::size_t rank) noexcept {
  return static_cast<std::int32_t>(dim) - static_cast<std::int32_t>(rank);
}

void gather_to_replicated(std::size_t input, std::size_t axis,
                          const AxisPlacement& p, ReshardList& out) {
  if (p.is_replicated()) return;
  Reshard r;
  r.mesh_axis = static_cast<std::uint8_t>(axis);
  if (p.placement == Placement::kSharded) {
    r.kind = Reshard::Kind::kAllGather;
    r.from_dim = p.tensor_dim;
  } else {
    r.kind = Reshard::Kind::kAllReduce;
    r.reduce_op = p.reduce_op;
  }
  out.emplace_back(input, r);
}

void gather_all(std::size_t axis, std::span<const Sharding> inputs,
                ReshardList& out) {
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    gather_to_replicated(i, axis, placement_on(inputs[i], axis), out);
  }
}

AxisPlacement elementwise_axis(std::size_t axis, OpKind op,
                               std::span<const Sharding> inputs,
                               std::span<const Shape> shapes,
                               std::size_t out_rank, ReshardList& out) {
  // A pending reduction survives only an op that commutes with it: -x is the
  // sum of the -x_r, a cast is a cast of each. Nothing else here is linear in
  // the shard, and a mul of two partials is not even close.
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const AxisPlacement p = placement_on(inputs[i], axis);
    if (p.placement != Placement::kPartial) continue;
    const bool commutes =
        inputs.size() == 1 &&
        (op == OpKind::kCast ||
         (op == OpKind::kNeg && (p.reduce_op == dist::ReduceOp::kSum ||
                                 p.reduce_op == dist::ReduceOp::kAvg)));
    if (commutes) return p;
    gather_to_replicated(i, axis, p, out);
  }

  // Operands this axis has already moved. The conflict path below must not
  // emit a second collective for one it has.
  std::uint64_t moved = 0;
  const auto mark = [&moved](std::size_t i) {
    if (i < 64) moved |= std::uint64_t{1} << i;
  };
  const auto was_moved = [&moved](std::size_t i) {
    return i < 64 && (moved & (std::uint64_t{1} << i)) != 0;
  };

  // Every sharded operand must name the same logical axis. A replicated
  // operand alongside them costs nothing — it already holds every slice and
  // reads only its own — which is what makes `column_parallel_matmul + bias`
  // collective-free.
  std::int32_t agreed = kNoDim;
  bool conflict = false;
  for (std::size_t i = 0; i < inputs.size() && !conflict; ++i) {
    const AxisPlacement p = placement_on(inputs[i], axis);
    if (p.placement != Placement::kSharded) continue;
    if (p.tensor_dim < 0 ||
        static_cast<std::size_t>(p.tensor_dim) >= shapes[i].rank()) {
      conflict = true;
      break;
    }
    // A shard of a broadcast extent is not a shard the broadcast can use, and
    // it is not replicated either: local_shape hands that one element to
    // coordinate 0 and leaves every other coordinate holding nothing, so the
    // ranks that need it to broadcast do not have it. Gathering one element is
    // the cheapest all-gather there is; assuming everyone already has it is
    // wrong on every rank but the first.
    if (shapes[i].dim(static_cast<std::size_t>(p.tensor_dim)) == 1) {
      gather_to_replicated(i, axis, p, out);
      mark(i);
      continue;
    }
    const std::int32_t t = trailing(p.tensor_dim, shapes[i].rank());
    if (agreed == kNoDim) {
      agreed = t;
    } else if (agreed != t) {
      conflict = true;
    }
  }

  const std::int32_t dim = static_cast<std::int32_t>(out_rank) + agreed;
  if (conflict || agreed == kNoDim || dim < 0) {
    // Only the shards: any partial was already dealt with above, and gathering
    // it twice would emit two all-reduces for one operand.
    if (conflict || dim < 0) {
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const AxisPlacement p = placement_on(inputs[i], axis);
        if (p.placement == Placement::kSharded && !was_moved(i)) {
          gather_to_replicated(i, axis, p, out);
        }
      }
    }
    return {};
  }
  return AxisPlacement::shard(static_cast<std::int8_t>(dim));
}

// Which dimension of each operand plays which role. `linear` stores weights
// [N, K] so it contracts B's last dim, while `matmul` takes B as [K, N] and
// contracts its first — the difference the propagation rules turn on.
struct Contraction {
  std::int8_t a_k = -1;
  std::int8_t b_k = -1;
  std::int8_t b_n = -1;
  std::int8_t out_n = -1;
  bool valid = false;
};

Contraction contraction_of(OpKind op, const Shape& a, const Shape& b) noexcept {
  Contraction c;
  if (a.rank() < 1 || b.rank() < 2) return c;
  const auto ra = static_cast<std::int8_t>(a.rank());
  const auto rb = static_cast<std::int8_t>(b.rank());
  c.a_k = static_cast<std::int8_t>(ra - 1);
  if (op == OpKind::kMatMul) {
    c.b_k = static_cast<std::int8_t>(rb - 2);
    c.b_n = static_cast<std::int8_t>(rb - 1);
  } else {
    c.b_k = static_cast<std::int8_t>(rb - 1);
    c.b_n = static_cast<std::int8_t>(rb - 2);
  }
  // The output keeps A's batch and row dims and replaces K with N, so an
  // index below a_k means the same dimension on both sides.
  c.out_n = c.a_k;
  c.valid = true;
  return c;
}

AxisPlacement contraction_axis(std::size_t axis, const Contraction& c,
                               std::span<const Sharding> inputs,
                               std::span<const Shape> shapes,
                               ReshardList& out) {
  for (std::size_t i = 2; i < inputs.size(); ++i) {
    gather_to_replicated(i, axis, placement_on(inputs[i], axis), out);
  }

  AxisPlacement pa = placement_on(inputs[0], axis);
  AxisPlacement pb = placement_on(inputs[1], axis);

  // A partial weight is always reduced first: a quantized matmul is not linear
  // in its weight shard, and no plan produces a partial weight anyway.
  if (pb.placement == Placement::kPartial) {
    gather_to_replicated(1, axis, pb, out);
    pb = {};
  }
  if (pa.placement == Placement::kPartial) {
    // (sum_r A_r) @ B == sum_r (A_r @ B). Carrying the pending reduction past
    // the matmul instead of paying for it here is the entire reason kPartial
    // is a state rather than an immediate all-reduce.
    if (pb.is_replicated() && (pa.reduce_op == dist::ReduceOp::kSum ||
                               pa.reduce_op == dist::ReduceOp::kAvg)) {
      return pa;
    }
    gather_to_replicated(0, axis, pa, out);
    pa = {};
  }

  enum class A : std::uint8_t { kNone, kK, kM };
  enum class B : std::uint8_t { kNone, kK, kN, kOther };

  A as = A::kNone;
  if (pa.placement == Placement::kSharded) {
    if (pa.tensor_dim == c.a_k) {
      as = A::kK;
    } else if (pa.tensor_dim >= 0 && pa.tensor_dim < c.a_k) {
      as = A::kM;
    } else {
      gather_to_replicated(0, axis, pa, out);
    }
  }
  B bs = B::kNone;
  if (pb.placement == Placement::kSharded) {
    if (pb.tensor_dim == c.b_k) bs = B::kK;
    else if (pb.tensor_dim == c.b_n) bs = B::kN;
    else bs = B::kOther;
  }
  if (bs == B::kOther) {
    gather_to_replicated(1, axis, pb, out);
    bs = B::kNone;
  }

  const AxisPlacement partial = AxisPlacement::partial(dist::ReduceOp::kSum);
  const AxisPlacement column = AxisPlacement::shard(c.out_n);

  switch (as) {
    case A::kNone:
      // A replicated operand needs no collective to meet a sharded one on the
      // contraction dim: it holds every K slice already and reads only its own.
      if (bs == B::kK) return partial;
      if (bs == B::kN) return column;  // column-parallel: no collective at all
      return {};

    case A::kK:
      // Row-parallel when B is sharded on K too; still partial when B is
      // replicated, for the same reason A::kNone/B::kK is.
      if (bs == B::kK || bs == B::kNone) return partial;
      // B holds its N slice for every K, A holds one K slice for every N.
      // Neither placement can be kept on one axis. Gathering A costs
      // activations and leaves column-parallel, which needs no output
      // all-reduce; gathering B costs weights and leaves one that does.
      gather_to_replicated(0, axis, pa, out);
      return column;

    case A::kM:
      // Data parallel: A's batch and row dims keep their index in the output.
      if (bs == B::kNone) return AxisPlacement::shard(pa.tensor_dim);
      if (bs == B::kK) {
        // A row window and a K split cannot share one mesh axis. Gather the
        // activations, not the weights.
        gather_to_replicated(0, axis, pa, out);
        return partial;
      }
      // Both are free parallel splits and both want this axis. Keep the one on
      // the larger operand and move the smaller.
      if (shapes[1].elem_count() >= shapes[0].elem_count()) {
        gather_to_replicated(0, axis, pa, out);
        return column;
      }
      gather_to_replicated(1, axis, pb, out);
      return AxisPlacement::shard(pa.tensor_dim);
  }
  return {};
}

AxisPlacement reduction_axis(std::size_t axis, OpKind op,
                             std::span<const Sharding> inputs,
                             const Shape& shape, std::size_t reduced,
                             bool drops, ReshardList& out) {
  // A norm's scale operand is replicated; gathering an already-replicated
  // input emits nothing.
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    gather_to_replicated(i, axis, placement_on(inputs[i], axis), out);
  }

  const AxisPlacement p = placement_on(inputs[0], axis);
  if (p.placement == Placement::kPartial) {
    // A sum of partial sums is a partial sum; a max of partial maxes is a
    // partial max. Nothing else here commutes with a pending reduction.
    const bool commutes =
        (op == OpKind::kSum && p.reduce_op == dist::ReduceOp::kSum) ||
        (op == OpKind::kMax && p.reduce_op == dist::ReduceOp::kMax);
    if (commutes) return p;
    gather_to_replicated(0, axis, p, out);
    return {};
  }
  if (p.placement != Placement::kSharded) return {};
  if (p.tensor_dim < 0 ||
      static_cast<std::size_t>(p.tensor_dim) >= shape.rank()) {
    gather_to_replicated(0, axis, p, out);
    return {};
  }

  if (static_cast<std::size_t>(p.tensor_dim) == reduced) {
    if (op == OpKind::kSum) return AxisPlacement::partial(dist::ReduceOp::kSum);
    if (op == OpKind::kMax) return AxisPlacement::partial(dist::ReduceOp::kMax);
    // A softmax, an RMS norm or a logsumexp over a split dimension is a
    // different program — an online pass exchanging its own running max and
    // denominator — not this one with a collective appended. Gathering is the
    // honest answer until that program exists.
    gather_to_replicated(0, axis, p, out);
    return {};
  }

  std::int8_t kept = p.tensor_dim;
  if (drops && static_cast<std::size_t>(p.tensor_dim) > reduced) --kept;
  return AxisPlacement::shard(kept);
}

bool reduction_drops_dim(OpKind op) noexcept {
  switch (op) {
    case OpKind::kSum:
    case OpKind::kMax:
    case OpKind::kMean:
    case OpKind::kLogSumExp:
      return true;
    default:
      return false;  // kRMS, kSoftmax and kL2Norm are shape-preserving
  }
}

}  // namespace

ShardingPropagator::Result ShardingPropagator::propagate(
    OpKind op, std::span<const Sharding> inputs,
    std::span<const Shape> input_shapes, const DeviceMesh& mesh,
    std::int8_t reduce_dim, bool keepdims) {
  Result result;
  const std::size_t axes =
      std::min<std::size_t>(mesh.axis_count(), DeviceMesh::kMaxAxes);
  result.output.axis_count = static_cast<std::uint8_t>(axes);
  if (axes == 0 || inputs.empty() || inputs.size() != input_shapes.size()) {
    return result;
  }

  const bool contraction = (op == OpKind::kMatMul || op == OpKind::kLinear ||
                            op == OpKind::kQuantMatMul) &&
                           inputs.size() >= 2;
  const Contraction c =
      contraction ? contraction_of(op, input_shapes[0], input_shapes[1])
                  : Contraction{};

  Shape broadcast = input_shapes[0];
  if (is_elementwise(op)) {
    for (std::size_t i = 1; i < input_shapes.size(); ++i) {
      broadcast = Shape::broadcast(broadcast, input_shapes[i]);
    }
  }

  const std::size_t in_rank = input_shapes[0].rank();
  const std::size_t reduced =
      reduce_dim < 0 ? (in_rank == 0 ? 0 : in_rank - 1)
                     : static_cast<std::size_t>(reduce_dim);
  const bool reduction = is_reduction(op) && reduced < in_rank;

  for (std::size_t a = 0; a < axes; ++a) {
    if (mesh.axis_size(a) <= 1) continue;  // no peers on this axis
    if (contraction && c.valid) {
      result.output.axes[a] =
          contraction_axis(a, c, inputs, input_shapes, result.input_reshards);
    } else if (is_elementwise(op) && broadcast.rank() != 0) {
      result.output.axes[a] =
          elementwise_axis(a, op, inputs, input_shapes, broadcast.rank(),
                           result.input_reshards);
    } else if (reduction) {
      result.output.axes[a] =
          reduction_axis(a, op, inputs, input_shapes[0], reduced,
                         !keepdims && reduction_drops_dim(op),
                         result.input_reshards);
    } else {
      // Everything else — reshape, transpose, slice, concat, attention, MoE,
      // embedding, top-k, the collectives themselves, registered primitives —
      // gathers to replicated. Deliberately suboptimal and deliberately not
      // guessed: a transpose could relabel its shard dim for free, but the
      // permutation is a node attribute this signature does not carry, and a
      // sharding inferred from nothing is a wrong answer rather than a slow one.
      gather_all(a, inputs, result.input_reshards);
    }
  }
  return result;
}

std::vector<Reshard> ShardingPropagator::plan(const Sharding& from,
                                              const Sharding& to,
                                              const DeviceMesh& mesh) {
  std::vector<Reshard> reduce_first;
  std::vector<Reshard> shuffle;
  std::vector<Reshard> grow_last;

  const std::size_t axes =
      std::min<std::size_t>(mesh.axis_count(), DeviceMesh::kMaxAxes);
  for (std::size_t a = 0; a < axes; ++a) {
    if (mesh.axis_size(a) <= 1) continue;
    const AxisPlacement f = placement_on(from, a);
    const AxisPlacement t = placement_on(to, a);
    if (f == t) continue;

    Reshard r;
    r.mesh_axis = static_cast<std::uint8_t>(a);
    switch (f.placement) {
      case Placement::kReplicated:
        // Replicated -> sharded moves nothing: every rank already holds the
        // bytes and keeps its own slice. It is absent from the plan rather
        // than present at zero cost, because a zero-cost entry would still be
        // a collective the emitter has to place and synchronize.
        // Replicated -> partial is not producible by movement at all.
        continue;
      case Placement::kSharded:
        if (t.placement == Placement::kSharded) {
          if (f.tensor_dim == t.tensor_dim) continue;
          r.kind = Reshard::Kind::kAllToAll;
          r.from_dim = f.tensor_dim;
          r.to_dim = t.tensor_dim;
        } else {
          // -> replicated, and -> partial: un-reducing a value is not a
          // collective, so the only defined move is to gather and let the
          // consumer contract again.
          r.kind = Reshard::Kind::kAllGather;
          r.from_dim = f.tensor_dim;
        }
        break;
      case Placement::kPartial:
        if (t.placement == Placement::kSharded) {
          r.kind = Reshard::Kind::kReduceScatter;
          r.to_dim = t.tensor_dim;
        } else {
          // -> replicated, and -> partial under a different op: finishing the
          // reduction `from` is holding is the only thing that is defined.
          r.kind = Reshard::Kind::kAllReduce;
        }
        r.reduce_op = f.reduce_op;
        break;
    }

    // Order is not a search, it is a rule: shrink or hold before you grow.
    // Reducing while the tensor is still split across the other axes moves
    // fewer bytes than gathering first and reducing the whole thing, and among
    // the gathers the order is immaterial — each byte a rank ends up holding
    // arrives exactly once whichever axis gathers first.
    switch (r.kind) {
      case Reshard::Kind::kAllReduce:
      case Reshard::Kind::kReduceScatter:
        reduce_first.push_back(r);
        break;
      case Reshard::Kind::kAllToAll:
        shuffle.push_back(r);
        break;
      default:
        grow_last.push_back(r);
        break;
    }
  }

  reduce_first.insert(reduce_first.end(), shuffle.begin(), shuffle.end());
  reduce_first.insert(reduce_first.end(), grow_last.begin(), grow_last.end());
  return reduce_first;
}

}  // namespace detail

}  // namespace lse::graph
