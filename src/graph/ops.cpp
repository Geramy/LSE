#include "lse/graph/ops.hpp"

#include <algorithm>

#include <cmath>
#include <string_view>

namespace lse::graph {

namespace {

NodePtr make(OpKind kind, Shape shape, DType dtype, std::vector<NodePtr> inputs) {
  auto n = std::make_shared<Node>();
  n->set_kind(kind);
  n->shape = shape;
  n->dtype = dtype;
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
  // A device copy, not a host bounce: the window stays where the producer put
  // it. The host only sees a tensor when something actually reads it.
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

Array linear(const Array& x, const Array& w) {
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

Array embedding(const Array& table, const Array& ids) {
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

Array topk(const Array& x, int k, int axis, Array* indices, float score_band) {
  const std::size_t a = normalize_axis(axis, x.shape().rank());
  Shape out;
  for (std::size_t i = 0; i < x.shape().rank(); ++i) {
    out.push_back(i == a ? static_cast<std::int64_t>(k) : x.shape().dim(i));
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

Array rope(const Array& x, const Array& cos, const Array& sin, int offset) {
  auto n = make(OpKind::kRoPE, x.shape(), x.dtype(),
                {x.node(), cos.node(), sin.node()});
  n->iattrs[0] = offset;
  n->prim = find_primitive("rope");
  return Array(n);
}

Array rope(const Array& x, const Array& cos, const Array& sin,
           const Array& offset) {
  auto n = make(OpKind::kRoPE, x.shape(), x.dtype(),
                {x.node(), cos.node(), sin.node(), offset.node()});
  n->iattrs[0] = 0;
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
  Array o = make_gdn(q.shape(), 0);
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
