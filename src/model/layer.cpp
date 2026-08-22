// Model-independent layer machinery: weight binding and the generic block.
#include "lse/model/layer.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// host_bytes / sync_to_device: a checkpoint tensor is host data and Array has
// no host-write path, so the buffer is filled behind the graph.
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/norm.hpp"

namespace lse::model {

Result<Array> WeightBinder::require(std::string_view name) {
  auto got = optional(name);
  if (got.ok()) return got;
  // Only a missing tensor becomes a "no such tensor" report; anything else
  // (no backend, allocation failure) would otherwise be misattributed.
  if (got.status().code() != StatusCode::kNotFound) return got.status();
  return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                   "'");
}

namespace {

// The dtype a checkpoint tensor is held in on the device. Float formats keep
// the format they were stored in — widening them in memory is a pure bandwidth
// tax on every token, and converting in register costs nothing. So does the
// packed plane of a group-affine weight: quant_linear unpacks it in register,
// and widening it here would not just cost bandwidth, it would destroy the
// codes. Block-quantized storage still widens; no kernel reads it directly.
DType device_storage(DType checkpoint) noexcept {
  switch (checkpoint) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kU32:
      return checkpoint;
    default:
      return DType::kF32;
  }
}

// Uploads one checkpoint tensor and reports it as `shape`. `order`, when
// non-null, selects and reorders rows — a row being one span of the tensor's
// last axis.
//
// A weight is data, not a computation. Going through eval() would dispatch a
// fill kernel across every element and then overwrite the result one element
// at a time — for the tied head that is a quarter of a billion pointless
// writes on each side, and it dominated model load. Allocate the buffer and
// read the tensor straight into it instead.
// `window`, when non-empty, keeps only that span of every row's last axis.
// Staged whole and copied span by span for the same reason the row path is:
// one sequential pass over the mapping beats a scattered read per row.
Result<Array> upload(const TensorView& v, Shape shape,
                     const std::vector<std::int64_t>* order,
                     std::int64_t win_first = 0, std::int64_t win_count = 0) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to load '", v.name,
                     "' into");
  }
  // Where this tensor lives. Weight loading names a member per layer so a
  // model spans the pool; everything else leaves it unset and gets the
  // primary, which is what a single-device run has always done.
  backend::IDeviceSet& set = sched->devices();
  const std::size_t member = graph::preferred_member();
  backend::IBackend& be =
      member < set.size() ? set.device(member) : sched->backend();

  const DType dt = device_storage(v.dtype);
  Array a = Array::zeros(shape, dt);
  graph::Node& n = *a.node();
  LSE_RETURN_IF_ERROR(graph::interpreter::ensure_output_buffer(n, be));
  const bool native = dt == v.dtype;

  if (order == nullptr && win_count <= 0) {
    if (native) {
      LSE_RETURN_IF_ERROR(
          v.read_native(graph::interpreter::host_bytes(n),
                        dtype_storage_bytes(dt, n.element_count())));
    } else {
      LSE_RETURN_IF_ERROR(v.read_f32(
          static_cast<float*>(graph::interpreter::host_bytes(n)),
          n.element_count()));
    }
  } else if (order == nullptr) {
    const std::size_t rank = v.shape.rank();
    const auto width =
        static_cast<std::size_t>(rank >= 2 ? v.shape.dim(rank - 1) : 1);
    const std::size_t elem = dtype_storage_bytes(dt, 1);
    const std::size_t rows = width > 0 ? v.element_count() / width : 0;
    std::vector<std::byte> staged(dtype_storage_bytes(dt, v.element_count()));
    if (native) {
      LSE_RETURN_IF_ERROR(v.read_native(staged.data(), staged.size()));
    } else {
      LSE_RETURN_IF_ERROR(v.read_f32(reinterpret_cast<float*>(staged.data()),
                                     v.element_count()));
    }
    auto* dst = static_cast<std::byte*>(graph::interpreter::host_bytes(n));
    const auto first = static_cast<std::size_t>(win_first);
    const auto count = static_cast<std::size_t>(win_count);
    for (std::size_t r = 0; r < rows; ++r) {
      std::memcpy(dst + r * count * elem,
                  staged.data() + (r * width + first) * elem, count * elem);
    }
  } else {
    // Staged whole rather than read row by row: TensorView reads the mapping,
    // and one sequential pass over it beats `order->size()` scattered ones.
    const std::size_t rank = v.shape.rank();
    const auto width =
        static_cast<std::size_t>(rank >= 2 ? v.shape.dim(rank - 1) : 1);
    const std::size_t elem = dtype_storage_bytes(dt, 1);
    std::vector<std::byte> staged(dtype_storage_bytes(dt, v.element_count()));
    if (native) {
      LSE_RETURN_IF_ERROR(v.read_native(staged.data(), staged.size()));
    } else {
      LSE_RETURN_IF_ERROR(v.read_f32(reinterpret_cast<float*>(staged.data()),
                                     v.element_count()));
    }
    auto* dst = static_cast<std::byte*>(graph::interpreter::host_bytes(n));
    const std::size_t row_bytes = width * elem;
    for (std::size_t i = 0; i < order->size(); ++i) {
      std::memcpy(
          dst + i * row_bytes,
          staged.data() + static_cast<std::size_t>((*order)[i]) * row_bytes,
          row_bytes);
    }
  }

  // The mirror now holds the only copy; it is the authority until it is pushed.
  n.host_dirty = true;
  n.device_dirty = false;
  n.materialized = true;
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(n, be));
  return a;
}

std::int64_t last_dim(const Shape& s) {
  return s.rank() == 0 ? 0 : s.dim(s.rank() - 1);
}

Status check_rows(std::string_view name, const Shape& shape,
                  const std::vector<std::int64_t>& order) {
  const auto width = static_cast<std::size_t>(
      shape.rank() >= 2 ? shape.dim(shape.rank() - 1) : 1);
  const std::size_t rows = width > 0 ? shape.elem_count() / width : 0;
  if (width == 0) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' has no rows to take");
  }
  for (std::int64_t r : order) {
    if (r < 0 || static_cast<std::size_t>(r) >= rows) {
      return LSE_ERROR(kOutOfRange, "row ", std::to_string(r), " of '",
                       std::string(name), "' is outside its ",
                       std::to_string(rows), " rows");
    }
  }
  return OkStatus();
}

}  // namespace

Result<std::array<const TensorView*, 2>> WeightBinder::quant_planes(
    std::string_view name) const {
  constexpr std::string_view kWeight = ".weight";
  std::array<const TensorView*, 2> planes{nullptr, nullptr};
  if (name.size() <= kWeight.size() ||
      name.substr(name.size() - kWeight.size()) != kWeight) {
    return planes;
  }
  const std::string base(name.substr(0, name.size() - kWeight.size()));
  planes[0] = weights_->find(base + ".scales");
  planes[1] = weights_->find(base + ".biases");
  if ((planes[0] == nullptr) != (planes[1] == nullptr)) {
    const bool have_scales = planes[0] != nullptr;
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has a '",
                     have_scales ? ".scales" : ".biases", "' plane but no '",
                     have_scales ? ".biases" : ".scales",
                     "'; a group-affine weight needs both");
  }
  return planes;
}

Result<Array> WeightBinder::optional(std::string_view name) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) return LSE_ERROR(kNotFound, std::string(name));

  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] == nullptr) {
    LSE_ASSIGN_OR(Array a, upload(*v, v->shape, nullptr));
    claimed_.emplace_back(name);
    return a;
  }
  return bind_quantized(name, *v, *planes[0], *planes[1], nullptr, Shape{});
}

Result<Array> WeightBinder::require_rows(std::string_view name,
                                         const std::vector<std::int64_t>& order,
                                         Shape shape) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] != nullptr) {
    return bind_quantized(name, *v, *planes[0], *planes[1], &order, shape);
  }

  const std::size_t rank = v->shape.rank();
  const auto width =
      static_cast<std::size_t>(rank >= 2 ? v->shape.dim(rank - 1) : 1);
  const std::size_t rows = width > 0 ? v->element_count() / width : 0;
  if (width == 0 || shape.elem_count() != order.size() * width) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has ",
                     std::to_string(rows), " rows of ", std::to_string(width),
                     "; taking ", std::to_string(order.size()),
                     " of them cannot fill the requested ",
                     std::to_string(shape.elem_count()), " elements");
  }
  LSE_RETURN_IF_ERROR(check_rows(name, v->shape, order));
  LSE_ASSIGN_OR(Array a, upload(*v, shape, &order));
  claimed_.emplace_back(name);
  return a;
}

Result<Array> WeightBinder::require_as(std::string_view name, Shape shape) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  if (v->element_count() != shape.elem_count()) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                     v->shape.to_string(), ", which is not ",
                     shape.to_string(), " read differently");
  }
  LSE_ASSIGN_OR(Array a, upload(*v, shape, nullptr));
  claimed_.emplace_back(name);
  return a;
}

Result<Array> WeightBinder::bind_quantized(
    std::string_view name, const TensorView& packed, const TensorView& scales,
    const TensorView& biases, const std::vector<std::int64_t>* order,
    Shape logical, TensorWindow window) {
  if (quantization_ == nullptr) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' is stored with .scales/.biases planes but the config "
                     "declared no quantization block, so its group size is "
                     "unknown");
  }
  if (packed.dtype != DType::kU32) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' has .scales/.biases planes but is stored as ",
                     to_string(packed.dtype),
                     "; a group-affine plane is packed into u32 lanes");
  }
  if (scales.dtype != biases.dtype) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' stores its scales as ", to_string(scales.dtype),
                     " and its biases as ", to_string(biases.dtype));
  }
  // Rank 2 is one [out, in] matrix; rank 3 is MLX's SwitchGLU stack, read by
  // quant_linear_indexed one expert at a time. Nothing is unstacked here — the
  // expert axis stays the leading axis of the plane all the way to the kernel.
  const std::size_t rank = packed.shape.rank();
  if (rank != 2 && rank != 3) {
    return LSE_ERROR(kUnimplemented, "'", std::string(name), "' is ",
                     packed.shape.to_string(),
                     "; a group-affine weight is read as an [out, in] matrix "
                     "or an [expert, out, in] stack of them");
  }
  bool agree = scales.shape == biases.shape && scales.shape.rank() == rank;
  for (std::size_t i = 0; agree && i + 1 < rank; ++i) {
    if (scales.shape.dim(i) != packed.shape.dim(i)) agree = false;
  }
  if (!agree) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                     packed.shape.to_string(), " with scales ",
                     scales.shape.to_string(), " and biases ",
                     biases.shape.to_string(),
                     "; the three planes must agree on every axis but the "
                     "last, which counts lanes on one and groups on the other");
  }
  if (order != nullptr && rank != 2) {
    return LSE_ERROR(kUnimplemented, "'", std::string(name),
                     "' is a stack; taking rows out of one would have to name "
                     "an expert as well as a row");
  }

  LSE_ASSIGN_OR(const quant::GroupAffine spec,
                quantization_->resolve_checked(name, last_dim(packed.shape),
                                               last_dim(scales.shape)));
  const std::int64_t in_features =
      last_dim(packed.shape) * 32 / spec.bits;

  Shape packed_shape = packed.shape;
  Shape group_shape = scales.shape;
  if (order != nullptr) {
    LSE_RETURN_IF_ERROR(check_rows(name, packed.shape, *order));
    const auto rows = static_cast<std::int64_t>(order->size());
    if (logical.elem_count() != static_cast<std::size_t>(rows * in_features)) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has ",
                       std::to_string(packed.shape.dim(0)), " rows of ",
                       std::to_string(in_features), " weights; taking ",
                       std::to_string(order->size()),
                       " of them cannot fill the requested ",
                       std::to_string(logical.elem_count()), " elements");
    }
    packed_shape = Shape{rows, last_dim(packed.shape)};
    group_shape = Shape{rows, last_dim(scales.shape)};
  }

  // An input-feature window becomes two different windows: one over packed
  // lanes and one over groups. Both are exact only on a boundary, which is
  // checked in require_columns before we get here.
  std::int64_t lane_first = 0, lane_count = 0;
  std::int64_t grp_first = 0, grp_count = 0;
  std::int64_t sliced_in = in_features;
  if (!window.empty()) {
    lane_first = window.first * spec.bits / 32;
    lane_count = window.count * spec.bits / 32;
    grp_first = window.first / spec.group_size;
    grp_count = window.count / spec.group_size;
    sliced_in = window.count;
    packed_shape = Shape{packed_shape.dim(0), lane_count};
    group_shape = Shape{group_shape.dim(0), grp_count};
  }

  LSE_ASSIGN_OR(Array a,
                upload(packed, packed_shape, order, lane_first, lane_count));
  LSE_ASSIGN_OR(Array s,
                upload(scales, group_shape, order, grp_first, grp_count));
  LSE_ASSIGN_OR(Array b,
                upload(biases, group_shape, order, grp_first, grp_count));

  auto planes = std::make_shared<graph::QuantPlanes>();
  planes->scales = s.node();
  planes->biases = b.node();
  planes->bits = spec.bits;
  planes->group_size = spec.group_size;
  planes->in_features = sliced_in;
  a.node()->quant = std::move(planes);

  claimed_.emplace_back(name);
  claimed_.emplace_back(scales.name);
  claimed_.emplace_back(biases.name);
  return a;
}

Result<Array> WeightBinder::require_columns(std::string_view name,
                                            std::int64_t first,
                                            std::int64_t count, Shape shape) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  if (first < 0 || count <= 0) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' asked for an empty column window");
  }
  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] != nullptr) {
    if (quantization_ == nullptr) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                       "' is group-affine but no quantization was declared");
    }
    LSE_ASSIGN_OR(const quant::GroupAffine spec,
                  quantization_->resolve_checked(name, last_dim(v->shape),
                                                 last_dim(planes[0]->shape)));
    const std::int64_t per_lane = 32 / spec.bits;
    if (first % spec.group_size != 0 || count % spec.group_size != 0 ||
        first % per_lane != 0 || count % per_lane != 0) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                       "' is quantized in groups of ",
                       std::to_string(spec.group_size), " with ",
                       std::to_string(per_lane),
                       " weights to a lane; input features [",
                       std::to_string(first), ", ",
                       std::to_string(first + count),
                       ") would split a group or a lane, and slicing one would "
                       "mean re-quantizing rather than reading");
    }
    return bind_quantized(name, *v, *planes[0], *planes[1], nullptr, shape,
                          TensorWindow{first, count});
  }

  const std::size_t rank = v->shape.rank();
  const auto width =
      static_cast<std::size_t>(rank >= 2 ? v->shape.dim(rank - 1) : 1);
  if (static_cast<std::size_t>(first + count) > width) {
    return LSE_ERROR(kOutOfRange, "'", std::string(name), "' rows are ",
                     std::to_string(width), " wide; [", std::to_string(first),
                     ", ", std::to_string(first + count), ") runs past that");
  }
  LSE_ASSIGN_OR(Array a, upload(*v, shape, nullptr, first, count));
  claimed_.emplace_back(name);
  return a;
}

std::vector<std::string> WeightBinder::unclaimed() const {
  std::vector<std::string> out;
  for (const auto& [name, _] : weights_->tensors()) {
    if (std::find(claimed_.begin(), claimed_.end(), name) == claimed_.end()) {
      out.push_back(name);
    }
  }
  return out;
}

Status HybridBlock::load(WeightBinder& binder, std::string_view prefix,
                         const LayerContext& ctx) {
  const std::string p(prefix);
  LSE_ASSIGN_OR(norm1_weight_, binder.require(p + spec_.norm1_name));
  LSE_ASSIGN_OR(norm2_weight_, binder.require(p + spec_.norm2_name));
  // From the context, the same place the mixers take it. Reading the scheme
  // and the device set here instead made the block's idea of how many ways it
  // was split disagree with its own mixer's, which is only invisible while
  // both happen to be derived from the same run -- a block asked to split two
  // ways with one copy of its norms indexes past what it has.
  const auto shards =
      static_cast<std::size_t>(ctx.shards > 0 ? ctx.shards : 1);
  norm1_shards_.resize(shards);
  norm2_shards_.resize(shards);
  for (std::size_t m = 0; m < shards; ++m) {
    if (m == 0) {
      norm1_shards_[0] = norm1_weight_;
      norm2_shards_[0] = norm2_weight_;
      continue;
    }
    // A norm is elementwise over the hidden width and is not cut; each member
    // keeps its own copy so the work either side of a reduce stays local.
    const graph::ScopedMember on(m);
    LSE_ASSIGN_OR(norm1_shards_[m], binder.require(p + spec_.norm1_name));
    LSE_ASSIGN_OR(norm2_shards_[m], binder.require(p + spec_.norm2_name));
  }
  LSE_RETURN_IF_ERROR(mixer_->load(binder, prefix, ctx));
  LSE_RETURN_IF_ERROR(ffn_->load(binder, prefix, ctx));
  if (mod_) LSE_RETURN_IF_ERROR(mod_->load(binder, prefix));
  return OkStatus();
}

Result<Array> HybridBlock::forward(const Array& x, MixerState* state,
                                   Array* aux_loss, const LayerContext& ctx) {
  if (ctx.config == nullptr) {
    return LSE_ERROR(kInvalidArgument, "HybridBlock::forward needs a config");
  }
  const float eps = ctx.config->rms_eps;
  auto norm = [&](const Array& v, const Array& w) {
    return zero_centered_norm_ ? ops::rms_norm_zero_centered(v, w, eps)
                               : ops::rms_norm(v, w, eps);
  };

  LSE_ASSIGN_OR(Array mixed,
                mixer_->forward(norm(x, norm1_weight_), state, ctx));
  Array h = x + mixed;

  Array h2 = norm(h, norm2_weight_);
  LSE_ASSIGN_OR(Array ff, ffn_->forward(h2, aux_loss, ctx));
  if (mod_) {
    LSE_ASSIGN_OR(ff, mod_->gate_all(h2, ff));
  }
  LSE_ASSIGN_OR(Array dense, ffn_->ungated(h2));
  if (dense.valid()) ff = ff + dense;
  return h + ff;
}

Result<std::vector<Array>> HybridBlock::forward_shards(
    const std::vector<Array>& xs, MixerState* state, Array* aux_loss,
    const LayerContext& ctx) {
  if (ctx.config == nullptr) {
    return LSE_ERROR(kInvalidArgument, "HybridBlock::forward needs a config");
  }
  const std::size_t n = xs.size();
  const float eps = ctx.config->rms_eps;
  auto norm = [&](const Array& v, const Array& w) {
    return zero_centered_norm_ ? ops::rms_norm_zero_centered(v, w, eps)
                               : ops::rms_norm(v, w, eps);
  };
  // Every member adds every partial, so each ends up with the same total and
  // the work after it is local. What crosses is one partial per member per
  // reduce, in opposite directions, instead of the activation on the way in
  // AND the sum on the way out.
  auto reduce = [&](const std::vector<Array>& parts) {
    // An op that did not shard hands back ONE value, and that value is already
    // the whole answer: every member takes it as it is. Summing it per member
    // would add it to itself, and indexing parts[m] past what it returned is
    // how a block that shards its feed-forward but not its mixer -- or the
    // other way round -- reads off the end of the vector.
    std::vector<Array> out(n);
    if (parts.size() < n) {
      for (std::size_t m = 0; m < n; ++m) out[m] = parts.front();
      return out;
    }
    for (std::size_t m = 0; m < n; ++m) {
      const graph::ScopedMember on(m);
      Array acc = parts[m];
      for (std::size_t j = 0; j < parts.size(); ++j) {
        if (j != m) acc = graph::add(acc, parts[j]);
      }
      out[m] = acc;
    }
    return out;
  };

  std::vector<Array> normed(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    normed[m] = norm(xs[m], norm1_shards_[m]);
  }
  LSE_ASSIGN_OR(std::vector<Array> mixed,
                mixer_->forward_shards(normed, state, ctx));
  const std::vector<Array> mix_sum = reduce(mixed);

  std::vector<Array> h(n), h2(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    h[m] = xs[m] + mix_sum[m];
    h2[m] = norm(h[m], norm2_shards_[m]);
  }

  LSE_ASSIGN_OR(std::vector<Array> ff, ffn_->forward_shards(h2, aux_loss, ctx));
  const std::vector<Array> ff_sum = reduce(ff);

  std::vector<Array> out(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    out[m] = h[m] + ff_sum[m];
  }
  return out;
}

}  // namespace lse::model
