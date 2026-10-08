#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/graph/ops.hpp"

#include <algorithm>
#include <cstdlib>

#include <cmath>
#include "lse/graph/kernel_primitive.hpp"
#include <string_view>

namespace lse::graph {

namespace {

NodePtr make(OpKind kind, Shape shape, DType dtype, std::vector<NodePtr> inputs) {
  auto n = std::make_shared<Node>();
  n->set_kind(kind);
  n->shape = shape;
  n->dtype = dtype;
  n->member = stamped_member();
  n->proposal_only = stamped_proposal();
  // Distinct consumers, not edges: `y * y` reads y twice but is one consumer,
  // and counting it as two makes the partitioner split a chain that could
  // have stayed in one kernel.
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const bool repeat =
        std::find(inputs.begin(), inputs.begin() + static_cast<std::ptrdiff_t>(i),
                  inputs[i]) != inputs.begin() + static_cast<std::ptrdiff_t>(i);
    if (!repeat) ++inputs[i]->consumer_count;
  }
  n->inputs = std::move(inputs);
  return n;
}

// Result dtype of a binary op: the wider float wins, so bf16 x f32 -> f32.
DType promote(DType a, DType b) noexcept {
  if (a == b) return a;
  if (a == DType::kF32 || b == DType::kF32) return DType::kF32;
  if (a == DType::kBF16 || b == DType::kBF16) return DType::kBF16;
  return a;
}

// Built-in elementwise ops are ordinary registered primitives; the OpKind is
// kept only as a stable tag for tracing and fusion signatures.
Array binary(OpKind kind, std::string_view prim, const Array& a, const Array& b) {
  const Shape out = Shape::broadcast(a.shape(), b.shape());
  auto n = make(kind, out, promote(a.dtype(), b.dtype()), {a.node(), b.node()});
  n->prim = find_primitive(prim);
  return Array(n);
}

Array unary(OpKind kind, std::string_view prim, const Array& x) {
  auto n = make(kind, x.shape(), x.dtype(), {x.node()});
  n->prim = find_primitive(prim);
  return Array(n);
}

std::size_t normalize_axis(int axis, std::size_t rank) noexcept {
  if (rank == 0) return 0;
  return axis < 0 ? rank + static_cast<std::size_t>(axis) : static_cast<std::size_t>(axis);
}

Shape reduced_shape(const Shape& in, std::size_t axis, bool keepdims) {
  Shape out;
  for (std::size_t i = 0; i < in.rank(); ++i) {
    if (i == axis) {
      if (keepdims) out.push_back(1);
    } else {
      out.push_back(in.dim(i));
    }
  }
  return out;
}

Array reduce(OpKind kind, const Array& x, int axis, bool keepdims) {
  const std::size_t a = normalize_axis(axis, x.shape().rank());
  auto n = make(kind, reduced_shape(x.shape(), a, keepdims), x.dtype(), {x.node()});
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->iattrs[1] = keepdims ? 1 : 0;
  return Array(n);
}

std::string join(const std::vector<std::string>& parts) {
  std::string out;
  for (const std::string& p : parts) {
    if (!out.empty()) out += ", ";
    out += p;
  }
  return out.empty() ? "(none)" : out;
}

}  // namespace

Array add(const Array& a, const Array& b) { return binary(OpKind::kAdd, "add", a, b); }
Array sub(const Array& a, const Array& b) { return binary(OpKind::kSub, "sub", a, b); }
Array mul(const Array& a, const Array& b) { return binary(OpKind::kMul, "mul", a, b); }
Array div(const Array& a, const Array& b) { return binary(OpKind::kDiv, "div", a, b); }

Array eq(const Array& a, const Array& b) {
  const Shape out = Shape::broadcast(a.shape(), b.shape());
  auto n = make(OpKind::kCustom, out, promote(a.dtype(), b.dtype()),
                {a.node(), b.node()});
  n->prim = find_primitive("eq");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array ge(const Array& a, const Array& b) {
  const Shape out = Shape::broadcast(a.shape(), b.shape());
  auto n = make(OpKind::kCustom, out, promote(a.dtype(), b.dtype()),
                {a.node(), b.node()});
  n->prim = find_primitive("ge");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array neg(const Array& x) { return unary(OpKind::kNeg, "neg", x); }
Array exp(const Array& x) { return unary(OpKind::kExp, "exp", x); }
Array log(const Array& x) { return unary(OpKind::kLog, "log", x); }
Array sqrt(const Array& x) { return unary(OpKind::kSqrt, "sqrt", x); }
Array rsqrt(const Array& x) { return unary(OpKind::kRsqrt, "rsqrt", x); }
Array silu(const Array& x) { return unary(OpKind::kSiLU, "silu", x); }
Array gelu(const Array& x) { return unary(OpKind::kGELU, "gelu", x); }
Array sigmoid(const Array& x) { return unary(OpKind::kSigmoid, "sigmoid", x); }
Array tanh_(const Array& x) { return unary(OpKind::kTanh, "tanh", x); }
Array relu(const Array& x) { return unary(OpKind::kReLU, "relu", x); }

Array cast(const Array& x, DType to) {
  return Array(make(OpKind::kCast, x.shape(), to, {x.node()}));
}

Array clamp(const Array& x, float lo, float hi) {
  auto n = make(OpKind::kClamp, x.shape(), x.dtype(), {x.node()});
  n->prim = find_primitive("clamp");
  n->attrs[0] = lo;
  n->attrs[1] = hi;
  return Array(n);
}

Array sum(const Array& x, int axis, bool keepdims) {
  return reduce(OpKind::kSum, x, axis, keepdims);
}
Array max(const Array& x, int axis, bool keepdims) {
  return reduce(OpKind::kMax, x, axis, keepdims);
}
Array mean(const Array& x, int axis, bool keepdims) {
  return reduce(OpKind::kMean, x, axis, keepdims);
}

Array softmax(const Array& x, int axis) {
  const std::size_t a = normalize_axis(axis, x.shape().rank());
  auto n = make(OpKind::kSoftmax, x.shape(), x.dtype(), {x.node()});
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->prim = find_primitive("softmax");
  return Array(n);
}

Array reshape(const Array& x, Shape shape) {
  return Array(make(OpKind::kReshape, shape, x.dtype(), {x.node()}));
}

Array matmul(const Array& a, const Array& b) {
  const Shape& sa = a.shape();
  const Shape& sb = b.shape();
  Shape out;
  for (std::size_t i = 0; i + 1 < sa.rank(); ++i) out.push_back(sa.dim(i));
  out.push_back(sb.dim(sb.rank() - 1));
  auto n = make(OpKind::kMatMul, out, promote(a.dtype(), b.dtype()),
                {a.node(), b.node()});
  n->prim = find_primitive("matmul");
  return Array(n);
}

Array transpose(const Array& x, std::vector<int> perm) {
  const Shape& in = x.shape();
  Shape out;
  for (int p : perm) out.push_back(in.dim(static_cast<std::size_t>(p)));
  auto n = make(OpKind::kTranspose, out, x.dtype(), {x.node()});
  for (std::size_t i = 0; i < perm.size() && i < 4; ++i) {
    n->iattrs[i] = perm[i];
  }
  n->prim = find_primitive("transpose");
  return Array(n);
}

Array concat(const std::vector<Array>& parts, int axis) {
  const Shape& first = parts.front().shape();
  const std::size_t a = normalize_axis(axis, first.rank());
  std::int64_t total = 0;
  std::vector<NodePtr> inputs;
  inputs.reserve(parts.size());
  for (const Array& p : parts) {
    total += p.shape().dim(a);
    inputs.push_back(p.node());
  }
  Shape out;
  for (std::size_t i = 0; i < first.rank(); ++i) {
    out.push_back(i == a ? total : first.dim(i));
  }
  auto n = make(OpKind::kConcat, out, parts.front().dtype(), std::move(inputs));
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->prim = find_primitive("concat");
  return Array(n);
}

Array slice(const Array& x, int axis, std::int64_t begin, std::int64_t end) {
  const Shape& in = x.shape();
  const std::size_t a = normalize_axis(axis, in.rank());
  Shape out;
  for (std::size_t i = 0; i < in.rank(); ++i) {
    out.push_back(i == a ? end - begin : in.dim(i));
  }
  auto n = make(OpKind::kSlice, out, x.dtype(), {x.node()});
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->iattrs[1] = static_cast<std::int32_t>(begin);
  n->iattrs[2] = static_cast<std::int32_t>(end);
  // The scheduler aliases contiguous windows; strided windows keep indexed copies.
  n->prim = find_primitive("slice");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array repeat(const Array& x, int count, int axis) {
  const Shape& in = x.shape();
  const std::size_t a = normalize_axis(axis, in.rank());
  Shape out;
  for (std::size_t i = 0; i < in.rank(); ++i) {
    out.push_back(i == a ? in.dim(i) * count : in.dim(i));
  }
  auto n = make(OpKind::kRepeat, out, x.dtype(), {x.node()});
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->iattrs[1] = count;
  return Array(n);
}

Shape weight_shape(const Array& w) {
  if (!w.valid()) return Shape{};
  const NodePtr& n = w.node();
  if (!n->quant) return n->shape;
  // A stacked weight keeps its expert axis: only the last one counts lanes.
  if (n->shape.rank() == 3) {
    return Shape{n->shape.dim(0), n->shape.dim(1), n->quant->in_features};
  }
  return Shape{n->shape.dim(0), n->quant->in_features};
}

Array linear(const Array& x, const Array& w) {
  if (w.valid() && w.node()->quant) {
    const QuantPlanes& q = *w.node()->quant;
    return quant_linear(x, w, Array(q.scales), Array(q.biases), q.bits,
                        q.group_size);
  }
  const Shape& sx = x.shape();
  Shape out;
  for (std::size_t i = 0; i + 1 < sx.rank(); ++i) out.push_back(sx.dim(i));
  out.push_back(w.shape().dim(0));
  auto n = make(OpKind::kLinear, out, promote(x.dtype(), w.dtype()),
                {x.node(), w.node()});
  // Carries the backend's kernel so the partitioner can fuse an epilogue into
  // it; without a primitive the node is a barrier with nowhere to run but the
  // host.
  n->prim = find_primitive("linear");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array quant_linear(const Array& x, const Array& packed, const Array& scales,
                   const Array& biases, int bits, int group_size) {
  const Shape& sx = x.shape();
  Shape out;
  for (std::size_t i = 0; i + 1 < sx.rank(); ++i) out.push_back(sx.dim(i));
  out.push_back(packed.shape().dim(0));
  // f32 regardless of how the scales are stored: the codes carry the weight,
  // the scale only places it, and the product is an activation.
  auto n = make(OpKind::kQuantMatMul, out, DType::kF32,
                {x.node(), packed.node(), scales.node(), biases.node()});
  n->iattrs[0] = bits;
  n->iattrs[1] = group_size;
  const std::array<Shape, 4> shapes{sx, packed.shape(), scales.shape(), biases.shape()};
  const std::array<DType, 4> dtypes{x.dtype(), packed.dtype(), scales.dtype(), biases.dtype()};
  KernelShapes geometry;
  geometry.inputs = shapes;
  geometry.input_dtypes = dtypes;
  geometry.output = out;
  geometry.iattrs = n->iattrs;
  const auto matrix_storage = packed.node()->quant
      ? packed.node()->quant->matrix_storage : nullptr;
  const std::array<NodePtr, 3> sources{packed.node(), scales.node(), biases.node()};
  const backend::DeviceInfo* device = nullptr;
  if (Scheduler* scheduler = default_scheduler())
    device = &scheduler->backend().device_info();
  if (dispatch::q4_gemm_shape(geometry) &&
      (device == nullptr || dispatch::q4_gemm_device(*device))) {
    // The tiled GEMM reads an f16 copy of the activation, made once and
    // shared by every contraction over the same activation.
    const Shape panel_shape{
        static_cast<std::int64_t>(sx.elem_count() /
                                  static_cast<std::uint64_t>(sx.dim(sx.rank() - 1))),
        sx.dim(sx.rank() - 1)};
    auto panel = x.node()->f16_activation_panel.lock();
    if (!panel || panel->inputs.size() != 1 || panel->inputs[0] != x.node() ||
        panel->shape != panel_shape || panel->dtype != DType::kF16) {
      panel = make(OpKind::kCustom, panel_shape, DType::kF16, {x.node()});
      panel->prim = find_primitive("quant_activation.f16_panel.v1");
      if (panel->prim) panel->fclass = panel->prim->fusion_class();
      x.node()->f16_activation_panel = panel;
    }
    n->inputs.push_back(panel);
    ++panel->consumer_count;
    n->prim = find_primitive("quant_linear.q4_gemm_f16.v1");
    const std::uint32_t compute_units = device ? device->compute_units : 0u;
    const auto k = static_cast<std::uint64_t>(sx.dim(sx.rank() - 1));
    const auto m = sx.elem_count() / k;
    const std::uint32_t slices = dispatch::q4_gemm_slices(
        m, static_cast<std::uint64_t>(packed.shape().dim(0)), k, compute_units,
        dispatch::arch::tuning(device ? std::string_view(device->arch)
                                      : std::string_view{}));
    if (slices > 1) {
      // K cut into slices: one launch writes a partial product per slice,
      // a second sums them, and trailing elementwise work fuses into the sum.
      n->prim = find_primitive("quant_linear.q4_gemm_f16.slices.v1");
      Shape partial_shape{static_cast<std::int64_t>(slices)};
      for (std::size_t i = 0; i < out.rank(); ++i) partial_shape.push_back(out.dim(i));
      n->shape = partial_shape;
      n->set_kind(OpKind::kCustom);
      n->iattrs[2] = static_cast<std::int32_t>(slices);
      n->fclass = n->prim ? n->prim->fusion_class() : n->fclass;
      auto sum = make(OpKind::kCustom, out, DType::kF32, {n});
      sum->prim = find_primitive("quant_linear.q4_gemm_f16.slice_sum.v1");
      if (sum->prim) sum->fclass = sum->prim->fusion_class();
      return Array(sum);
    }
  } else if (dispatch::q8_packed_matrix_shape(geometry) && matrix_storage &&
      matrix_storage->matches(sources)) {
    for (const auto& leaf : matrix_storage->packed) {
      n->inputs.push_back(leaf);
      ++leaf->consumer_count;
    }
    n->prim = find_primitive("quant_linear.q8.wmma16.packed.v1");
  } else if (dispatch::q4_matrix_panel_shape(
                 geometry, device ? std::string_view(device->arch)
                                  : std::string_view{}) ||
             dispatch::q8_matrix_panel_shape(geometry)) {
    // 8-bit weights read the panel in byte order (attribute 0 = 8), 4-bit
    // weights in nibble-plane order; siblings share it only in one order.
    const bool bytes = dispatch::q8_matrix_panel_shape(geometry);
    const std::int32_t order = bytes ? 8 : 0;
    const Shape panel_shape = dispatch::q4_matrix_panel_storage_shape(sx);
    auto panel = x.node()->quant_activation_panel.lock();
    if (!panel || panel->inputs.size() != 1 || panel->inputs[0] != x.node() ||
        panel->shape != panel_shape || panel->dtype != DType::kU32 ||
        !panel->prim || panel->prim->name() != "quant_activation.q4_matrix_panel.v1" ||
        panel->iattrs[0] != order) {
      panel = make(OpKind::kCustom, panel_shape, DType::kU32, {x.node()});
      panel->prim = find_primitive("quant_activation.q4_matrix_panel.v1");
      panel->iattrs[0] = order;
      if (panel->prim) panel->fclass = panel->prim->fusion_class();
      x.node()->quant_activation_panel = panel;
    }
    n->inputs.push_back(panel);
    ++panel->consumer_count;
    n->prim = find_primitive(bytes ? "quant_linear.q8_matrix_panel.v1"
                                   : "quant_linear.q4_matrix_panel.v1");
  } else if (dispatch::q4_shared_panel_shape(
                 geometry, device ? std::string_view(device->arch)
                                  : std::string_view{})) {
    const auto k = sx.dim(sx.rank() - 1);
    const auto m = static_cast<std::int64_t>(sx.elem_count() / static_cast<std::uint64_t>(k));
    const Shape panel_shape{m, (k / 64) * 25};
    auto panel = x.node()->quant_activation_panel.lock();
    if (!panel || panel->inputs.size() != 1 || panel->inputs[0] != x.node() ||
        panel->shape != panel_shape || panel->dtype != DType::kU32) {
      panel = make(OpKind::kCustom, panel_shape, DType::kU32, {x.node()});
      panel->prim = find_primitive("quant_activation.q4_shared_panel.v1");
      if (panel->prim) panel->fclass = panel->prim->fusion_class();
      x.node()->quant_activation_panel = panel;
    }
    n->inputs.push_back(panel);
    ++panel->consumer_count;
    n->prim = find_primitive("quant_linear.q4_global_panel.v1");
  } else {
    n->prim = find_primitive("quant_linear");
  }
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array quant_linear_indexed(const Array& x, const Array& packed,
                           const Array& scales, const Array& biases,
                           const Array& idx, int slot, int bits,
                           int group_size) {
  const Shape& sx = x.shape();
  Shape out;
  for (std::size_t i = 0; i + 1 < sx.rank(); ++i) out.push_back(sx.dim(i));
  out.push_back(packed.shape().dim(1));
  // kMoEDispatch like the dense form, f32 like quant_linear: the packed plane
  // is kU32 and promote() would take that for the result's width.
  auto n = make(OpKind::kMoEDispatch, out, DType::kF32,
                {x.node(), packed.node(), scales.node(), biases.node(),
                 idx.node()});
  // iattrs[0] stays the expert slot, as it is for linear_indexed, so the
  // emitter's device_fn_name keeps slot-1 off the slot-0 body. The geometry
  // follows it, and mixes into the same name — which is what gives a 6-bit
  // expert and an 8-bit router distinct device functions.
  n->iattrs[0] = slot;
  n->iattrs[1] = bits;
  n->iattrs[2] = group_size;
  n->prim = find_primitive("quant_linear_indexed");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array quant_embedding(const Array& packed, const Array& scales,
                      const Array& biases, const Array& ids, int bits,
                      int group_size) {
  Shape out;
  for (std::size_t i = 0; i < ids.shape().rank(); ++i) {
    out.push_back(ids.shape().dim(i));
  }
  out.push_back(packed.shape().dim(1) * 32 / bits);
  auto n = make(OpKind::kQuantEmbedding, out, DType::kF32,
                {packed.node(), scales.node(), biases.node(), ids.node()});
  n->iattrs[0] = bits;
  n->iattrs[1] = group_size;
  n->prim = find_primitive("quant_embedding");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array embedding(const Array& table, const Array& ids) {
  if (table.valid() && table.node()->quant) {
    const QuantPlanes& q = *table.node()->quant;
    return quant_embedding(table, Array(q.scales), Array(q.biases), ids, q.bits,
                           q.group_size);
  }
  Shape out;
  for (std::size_t i = 0; i < ids.shape().rank(); ++i) out.push_back(ids.shape().dim(i));
  out.push_back(table.shape().dim(1));
  // Not the table's dtype: this is where the residual stream begins, and a
  // narrow embedding table would make the whole stream narrow for the rest of
  // the network. Storage format is the table's business; the value it yields
  // is an activation.
  auto n = make(OpKind::kEmbedding, out, promote(table.dtype(), DType::kF32),
                {table.node(), ids.node()});
  n->prim = find_primitive("embedding");
  return Array(n);
}

Array gather_rows(const Array& x, const Array& rows) {
  const std::int64_t width = x.shape().dim(x.shape().rank() - 1);
  Shape out;
  out.push_back(rows.shape().elem_count() > 0
                    ? static_cast<std::int64_t>(rows.shape().elem_count())
                    : 0);
  out.push_back(width);
  // Same rule as embedding: gathering rows out of a stored table produces
  // activations, so the result does not inherit the table's storage width.
  auto n = make(OpKind::kGather, out, promote(x.dtype(), DType::kF32),
                {x.node(), rows.node()});
  n->prim = find_primitive("gather");
  return Array(n);
}

Array scatter_add_rows(const Array& base, const Array& rows,
                       const Array& values) {
  auto n = make(OpKind::kScatter, base.shape(), base.dtype(),
                {base.node(), rows.node(), values.node()});
  n->prim = find_primitive("scatter");
  return Array(n);
}

bool topk_pairs_fits(std::int64_t width, int k) noexcept {
  return width >= 4096 && width < 16777215 && k >= 1 && k <= 32;
}

Array topk_pairs(const Array& x, int k) {
  if (!x.valid() || !x.shape().rank() || x.dtype() != DType::kF32) return {};
  const auto width = x.shape().dim(x.shape().rank() - 1);
  if (!topk_pairs_fits(width, k)) return {};
  // Scan stages of 256 threads x `per` registers each: a wide first stage
  // leaves a few chunks' k candidates, which the next stage (or two) reduce
  // to one chunk.
  constexpr std::int64_t kThreads = 256, kWidePer = 32;
  Array candidates = x;
  auto count = width;
  bool pairs = false;
  while (true) {
    const std::int64_t need = (count + kThreads - 1) / kThreads;
    const auto per = std::min<std::int64_t>(need, kWidePer);
    const auto chunks = (count + kThreads * per - 1) / (kThreads * per);
    Shape stage_shape{static_cast<std::int64_t>(x.shape().elem_count()) / width, chunks, k, 2};
    auto stage = make(OpKind::kCustom, stage_shape, DType::kF32, {candidates.node()});
    stage->attrs = {static_cast<float>(k), static_cast<float>(count), pairs ? 1.0f : 0.0f,
                    static_cast<float>(per)};
    stage->prim = find_primitive("topk.scan.v1");
    candidates = Array(stage);
    if (chunks == 1) break;
    count = chunks * k;
    pairs = true;
  }
  return candidates;
}

Array topk(const Array& x, int k, int axis, Array* indices, float score_band) {
  const std::size_t a = normalize_axis(axis, x.shape().rank());
  Shape out;
  for (std::size_t i = 0; i < x.shape().rank(); ++i) {
    out.push_back(i == a ? static_cast<std::int64_t>(k) : x.shape().dim(i));
  }
  const auto width = x.shape().dim(a);
  if (a + 1 == x.shape().rank() && x.dtype() == DType::kF32 &&
      width >= 4096 && width < 16777215 && k >= 1 && k <= 16) {
    Array candidates = topk_pairs(x, k);
    auto extract = [&](bool write_index) {
      auto node = make(OpKind::kCustom, out, DType::kF32, {candidates.node()});
      node->attrs = {static_cast<float>(k), write_index ? 1.0f : 0.0f, score_band, 0.0f};
      node->prim = find_primitive("topk.extract");
      return Array(node);
    };
    Array values = extract(false);
    if (indices != nullptr) *indices = extract(true);
    return values;
  }
  auto make_topk = [&](std::int32_t write_idx) {
    auto n = make(OpKind::kTopK, out, x.dtype(), {x.node()});
    n->iattrs[0] = static_cast<std::int32_t>(a);
    n->iattrs[1] = k;
    n->iattrs[2] = write_idx;
    n->attrs[0] = score_band;
    n->prim = find_primitive("topk");
    return Array(n);
  };
  Array values = make_topk(0);
  if (indices != nullptr) *indices = make_topk(1);
  return values;
}

Array argmax(const Array& x) {
  const Shape& in = x.shape();
  if (in.rank() == 0) return {};
  const std::int64_t n = in.dim(in.rank() - 1);
  if (n <= 0) return {};
  // One workgroup reduces one chunk; must agree with the device kernel's
  // per-workgroup span and the host interpreter (both read iattrs[1]).
  constexpr std::int64_t kChunk = 4096;
  const std::int64_t nchunks = (n + kChunk - 1) / kChunk;

  Shape partial_shape;
  for (std::size_t i = 0; i + 1 < in.rank(); ++i) partial_shape.push_back(in.dim(i));
  partial_shape.push_back(nchunks);
  partial_shape.push_back(2);
  auto p = make(OpKind::kArgMax, partial_shape, DType::kF32, {x.node()});
  p->iattrs[0] = 0;
  p->iattrs[1] = static_cast<std::int32_t>(kChunk);
  p->prim = find_primitive("argmax.partial");

  Shape final_shape;
  for (std::size_t i = 0; i + 1 < in.rank(); ++i) final_shape.push_back(in.dim(i));
  if (final_shape.rank() == 0) final_shape.push_back(1);
  auto f = make(OpKind::kArgMax, final_shape, DType::kF32, {NodePtr(p)});
  f->iattrs[0] = 1;
  f->prim = find_primitive("argmax.final");
  return Array(f);
}

Array linear_indexed(const Array& x, const Array& w, const Array& idx,
                     int slot) {
  if (w.valid() && w.node()->quant) {
    const QuantPlanes& q = *w.node()->quant;
    return quant_linear_indexed(x, w, Array(q.scales), Array(q.biases), idx,
                                slot, q.bits, q.group_size);
  }
  Shape out;
  for (std::size_t i = 0; i + 1 < x.shape().rank(); ++i) {
    out.push_back(x.shape().dim(i));
  }
  out.push_back(w.shape().dim(1));
  auto n = make(OpKind::kMoEDispatch, out, promote(x.dtype(), w.dtype()),
                {x.node(), w.node(), idx.node()});
  n->iattrs[0] = slot;
  n->prim = find_primitive("linear_indexed");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array overwrite_slice(const Array& dst, const Array& src, int axis,
                      const Array& begin) {
  const std::size_t a = normalize_axis(axis, dst.shape().rank());
  auto n = make(OpKind::kOverwriteSlice, dst.shape(), dst.dtype(),
                {dst.node(), src.node(), begin.node()});
  n->iattrs[0] = static_cast<std::int32_t>(a);
  n->prim = find_primitive("overwrite_slice");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array rope(const Array& x, const Array& cos, const Array& sin, int offset,
           int rotary) {
  auto n = make(OpKind::kRoPE, x.shape(), x.dtype(),
                {x.node(), cos.node(), sin.node()});
  n->iattrs[0] = offset;
  n->iattrs[1] = rotary;
  n->prim = find_primitive("rope");
  return Array(n);
}

Array rope(const Array& x, const Array& cos, const Array& sin,
           const Array& offset, int rotary) {
  auto n = make(OpKind::kRoPE, x.shape(), x.dtype(),
                {x.node(), cos.node(), sin.node(), offset.node()});
  n->iattrs[0] = 0;
  n->iattrs[1] = rotary;
  n->prim = find_primitive("rope");
  return Array(n);
}

Array rope_rows(const Array& x, const Array& cos, const Array& sin,
                const Array& positions, int rotary) {
  auto n = make(OpKind::kRoPE, x.shape(), x.dtype(),
                {x.node(), cos.node(), sin.node(), positions.node()});
  n->iattrs[0] = 0;
  n->iattrs[1] = rotary;
  n->iattrs[2] = 1;
  n->prim = find_primitive("rope");
  return Array(n);
}

Array sdpa(const Array& q, const Array& k, const Array& v, float scale,
           MaskKind mask, int window, int offset) {
  const Shape& sq = q.shape();
  Shape out{sq.dim(0), sq.dim(1), sq.dim(2), v.shape().dim(3)};
  auto n = make(OpKind::kAttention, out, q.dtype(), {q.node(), k.node(), v.node()});
  n->attrs[0] = scale;
  n->iattrs[0] = static_cast<std::int32_t>(mask);
  n->iattrs[1] = window;
  n->iattrs[2] = offset;
  n->prim = find_primitive("attention");
  return Array(n);
}

Array sdpa(const Array& q, const Array& k, const Array& v, float scale,
           MaskKind mask, int window, const Array& offset) {
  const Shape& sq = q.shape();
  Shape out{sq.dim(0), sq.dim(1), sq.dim(2), v.shape().dim(3)};
  auto n = make(OpKind::kAttention, out, q.dtype(),
                {q.node(), k.node(), v.node(), offset.node()});
  n->attrs[0] = scale;
  n->iattrs[0] = static_cast<std::int32_t>(mask);
  n->iattrs[1] = window;
  n->iattrs[2] = 0;
  n->prim = find_primitive("attention");
  return Array(n);
}

namespace {
Array split_paged_attention(const NodePtr& baseline, bool blasst=false) {
  const auto* partial_prim = find_primitive(blasst ? "attention.blasst.partial1024.v2" : "attention.split_partial128.wg128c2.v1");
  const auto* merge_prim = find_primitive("attention.split_merge128.wg128c2.v1");
  if (!partial_prim || !merge_prim) return Array(baseline);
  const auto& q = baseline->inputs[0]->shape;
  const bool short_query = q.dim(2) > 1;
  const auto capacity = baseline->inputs[1]->shape.dim(2) * baseline->inputs[4]->shape.dim(1);
  const auto partition_keys=blasst?1024:128;
  const auto parts = (capacity + partition_keys - 1) / partition_keys;
  const Shape shape = short_query ? Shape{q.dim(0), q.dim(1), q.dim(2), parts, 258}
                                  : Shape{q.dim(0), q.dim(1), parts, 258};
  auto partial = make(OpKind::kCustom, shape, DType::kF32, baseline->inputs);
  partial->prim = partial_prim;
  partial->fclass = partial_prim->fusion_class();
  partial->attrs = baseline->attrs;
  partial->iattrs = baseline->iattrs;
  auto merged = make(OpKind::kCustom, baseline->shape, DType::kF32, {partial});
  merged->prim = merge_prim;
  merged->fclass = merge_prim->fusion_class();
  const auto& inputs = baseline->inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (std::find(inputs.begin(), inputs.begin() + static_cast<std::ptrdiff_t>(i), inputs[i]) ==
        inputs.begin() + static_cast<std::ptrdiff_t>(i)) --inputs[i]->consumer_count;
  return Array(merged);
}
}  // namespace

Array sdpa_paged(const Array& q, const Array& k, const Array& v, float scale,
                 MaskKind mask, int window, const Array& meta,
                 const Array& table, int block_size,
                 const backend::DeviceInfo* device, kv::CacheDType storage,
                 ops::SparseAttentionPhase sparse) {
  const Shape& sq = q.shape();
  Shape out{sq.dim(0), sq.dim(1), sq.dim(2), kv::logical_width(storage, v.shape().dim(3))};
  auto n = make(OpKind::kAttention, out, q.dtype(),
                {q.node(), k.node(), v.node(), meta.node(), table.node()});
  n->attrs[0] = scale;
  n->attrs[1] = static_cast<float>(storage);
  n->attrs[2] = sparse.scale;
  n->attrs[3] = sparse.flashprefill ? 2.0f : sparse.blasst ? 1.0f : 0.0f;
  n->iattrs[0] = static_cast<std::int32_t>(mask);
  n->iattrs[1] = window;
  n->iattrs[2] = 0;
  n->iattrs[3] = block_size;
  n->prim = find_primitive(sparse.blasst
      ? (sq.dim(2) == 1 ? "attention.blasst.partial1024.v2" : "attention.flash.wmma16.v3")
      : "attention");
  if(sparse.blasst && sq.dim(2)==1) return split_paged_attention(n,true);
  if (sparse.flashprefill) {
    const auto blocks = (table.shape().dim(1) * block_size + 255) / 256;
    const auto tiles = (sq.dim(2) + 15) / 16;
    auto pooled = make(OpKind::kCustom, Shape{sq.dim(0), k.shape().dim(1), blocks, 513},
                       DType::kF32, n->inputs);
    pooled->prim = find_primitive("attention.flashprefill.pool.v1");
    pooled->fclass = pooled->prim->fusion_class();
    pooled->attrs = n->attrs; pooled->iattrs = n->iattrs;
    auto selected = make(OpKind::kCustom, Shape{sq.dim(0), sq.dim(1), tiles, blocks},
                         DType::kF32, {q.node(), pooled, meta.node()});
    selected->prim = find_primitive("attention.flashprefill.select.v2");
    selected->fclass = selected->prim->fusion_class();
    selected->attrs = n->attrs; selected->iattrs = n->iattrs;
    auto inputs = n->inputs; inputs.push_back(pooled); inputs.push_back(selected);
    auto result = make(OpKind::kCustom, out, DType::kF32, inputs);
    result->prim = find_primitive("attention.flashprefill.wmma.v1");
    result->fclass = result->prim->fusion_class();
    result->attrs = n->attrs; result->iattrs = n->iattrs;
    for (std::size_t i = 0; i < n->inputs.size(); ++i)
      if (std::find(n->inputs.begin(), n->inputs.begin() + static_cast<std::ptrdiff_t>(i), n->inputs[i]) ==
          n->inputs.begin() + static_cast<std::ptrdiff_t>(i)) --n->inputs[i]->consumer_count;
    return Array(result);
  }
  if (device != nullptr) {
    const Shape shapes[] = {q.shape(), k.shape(), v.shape(), meta.shape(), table.shape()};
    const DType dtypes[] = {q.dtype(), k.dtype(), v.dtype(), meta.dtype(), table.dtype()};
    KernelShapes request;
    request.inputs = shapes;
    request.input_dtypes = dtypes;
    request.output = out;
    request.output_dtype = q.dtype();
    request.attrs = n->attrs;
    request.iattrs = n->iattrs;
    request.device = device;
    if (sparse.blasst) {
      n->prim = find_primitive(sq.dim(2) == 1
          ? "attention.blasst.partial1024.v2" : "attention.flash.wmma16.v3");
      return Array(n);
    }
    if (dispatch::split_short_default_supported(request) ||
        dispatch::split_decode_supported(request))
      return split_paged_attention(n);
  }
  return Array(n);
}

Array kv_page_write(const Array& dst, const Array& src, const Array& meta,
                    const Array& table, int block_size, kv::CacheDType storage) {
  auto n = make(OpKind::kKvPageWrite, dst.shape(), dst.dtype(),
                {dst.node(), src.node(), meta.node(), table.node()});
  n->iattrs[0] = block_size;
  n->iattrs[1] = static_cast<std::int32_t>(storage);
  n->prim = find_primitive("kv_page_write");
  n->kv_fragments = dst.node()->kv_fragments;
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array kv_page_write_rows(const Array& dst, const Array& src, const Array& table,
                         const Array& path, int block_size, kv::CacheDType storage) {
  auto n = make(OpKind::kCustom, dst.shape(), dst.dtype(),
                {dst.node(), src.node(), table.node(), path.node()});
  n->iattrs[0] = block_size;
  n->iattrs[1] = static_cast<std::int32_t>(storage);
  n->prim = find_primitive("kv_page_write.rows.v1");
  n->kv_fragments = dst.node()->kv_fragments;
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

namespace {
Array custom_node(std::string_view name, const Shape& shape, const std::vector<Array>& inputs) {
  std::vector<NodePtr> in;
  for (const Array& a : inputs) in.push_back(a.node());
  auto n = make(OpKind::kCustom, shape, DType::kF32, std::move(in));
  n->prim = find_primitive(name);
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}
}  // namespace

Array gated_delta_tree(const Array& q, const Array& k, const Array& v, const Array& alpha,
                       const Array& beta, const Array& state_in, const Array& depth) {
  return custom_node("gdn.tree_scan.v1", v.shape(), {q, k, v, alpha, beta, state_in, depth});
}

Array gated_delta_path(const Array& k, const Array& v, const Array& alpha, const Array& beta,
                       const Array& state_in, const Array& path) {
  return custom_node("gdn.path_state.v1", state_in.shape(), {k, v, alpha, beta, state_in, path});
}

Array causal_conv1d_tree(const Array& x, const Array& weight, const Array& bias,
                         const Array& tail, const Array& ancestors) {
  std::vector<NodePtr> in{x.node(), weight.node(), bias.node(), tail.node(), ancestors.node()};
  auto n = make(OpKind::kCustom, x.shape(), x.dtype(), std::move(in));
  n->prim = find_primitive("causal_conv1d.tree.v1");
  if (n->prim != nullptr) n->fclass = n->prim->fusion_class();
  return Array(n);
}

Array conv_tail_rows(const Array& tail, const Array& x, const Array& path) {
  return custom_node("conv_tail.rows.v1", tail.shape(), {tail, x, path});
}

Result<Array> custom(std::string_view primitive, const std::vector<Array>& inputs,
                     std::array<float, 4> attrs) {
  const Primitive* p = find_primitive(primitive);
  if (p == nullptr) {
    return LSE_ERROR(kNotFound, "no primitive named '", std::string(primitive),
                     "'; registered: ", join(registered_primitives()));
  }
  if (inputs.size() != p->arity()) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(primitive), "' takes ",
                     std::to_string(p->arity()), " inputs, got ",
                     std::to_string(inputs.size()));
  }

  std::vector<Shape> shapes;
  std::vector<DType> dtypes;
  std::vector<NodePtr> nodes;
  shapes.reserve(inputs.size());
  dtypes.reserve(inputs.size());
  nodes.reserve(inputs.size());
  for (const Array& a : inputs) {
    if (!a.valid()) return LSE_ERROR(kInvalidArgument, "invalid input array");
    shapes.push_back(a.shape());
    dtypes.push_back(a.dtype());
    nodes.push_back(a.node());
  }

  auto shape = p->infer_shape(shapes);
  if (!shape.ok()) return shape.status();

  auto n = make(OpKind::kCustom, shape.release(), p->infer_dtype(dtypes),
                std::move(nodes));
  n->prim = p;
  n->fclass = p->fusion_class();
  n->attrs = attrs;
  return Array(n);
}

Array softplus(const Array& x) { return unary(OpKind::kSoftplus, "softplus", x); }

Array l2_normalize(const Array& x, float eps) {
  auto n = make(OpKind::kL2Norm, x.shape(), x.dtype(), {x.node()});
  n->attrs[0] = eps;
  n->prim = find_primitive("l2_normalize");
  return Array(n);
}

Array causal_conv1d(const Array& x, const Array& weight, const Array& bias) {
  auto n = make(OpKind::kCausalConv1d, x.shape(), x.dtype(),
                {x.node(), weight.node(), bias.node()});
  n->prim = find_primitive("causal_conv1d");
  return Array(n);
}

Array causal_conv1d(const Array& x, const Array& weight, const Array& bias,
                    const Array& tail) {
  auto n = make(OpKind::kCausalConv1d, x.shape(), x.dtype(),
                {x.node(), weight.node(), bias.node(), tail.node()});
  n->prim = find_primitive("causal_conv1d");
  return Array(n);
}

Array conv_tail(const Array& tail, const Array& x) {
  auto n = make(OpKind::kConvTailShift, tail.shape(), x.dtype(),
                {tail.node(), x.node()});
  n->prim = find_primitive("conv_tail");
  return Array(n);
}

Array gated_delta_step(const Array& q, const Array& k, const Array& v,
                       const Array& alpha, const Array& beta,
                       const Array& state_in, Array* state_out) {
  auto make_gdn = [&](Shape out, std::int32_t write_state) {
    auto n = make(OpKind::kGDNChunkScan, std::move(out), q.dtype(),
                  {q.node(), k.node(), v.node(), alpha.node(), beta.node(),
                   state_in.node()});
    n->iattrs[0] = write_state;
    n->prim = find_primitive("gdn_chunk_scan");
    return Array(n);
  };
  // The output has the value heads; q and k may share theirs across them.
  Array o = make_gdn(v.shape(), 0);
  if (state_out != nullptr) *state_out = make_gdn(state_in.shape(), 1);
  return o;
}

Array rms_norm(const Array& x, const Array& weight, float eps,
               bool zero_centered) {
  auto n = make(OpKind::kRMS, x.shape(), x.dtype(), {x.node(), weight.node()});
  n->attrs[0] = eps;
  n->iattrs[0] = zero_centered ? 1 : 0;
  n->prim = find_primitive("rms_norm");
  return Array(n);
}

}  // namespace lse::graph
