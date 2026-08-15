#include "lse/ops/rope.hpp"

#include <cmath>

// store_element: Array has no host-write path, and the tables are built on the
// host from cos/sin rather than by any graph op.
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"

namespace lse::ops {

Result<RopeTables> build_rope(std::int32_t rope_dim, std::int32_t max_seq,
                              float theta) {
  if (rope_dim <= 0 || rope_dim % 2 != 0) {
    return LSE_ERROR(kInvalidArgument, "rope_dim must be positive and even");
  }
  RopeTables t;
  t.dim = rope_dim;
  t.max_seq = max_seq;
  t.cos = Array::zeros(Shape{max_seq, rope_dim}, DType::kF32);
  t.sin = Array::zeros(Shape{max_seq, rope_dim}, DType::kF32);
  LSE_RETURN_IF_ERROR(t.cos.eval());
  LSE_RETURN_IF_ERROR(t.sin.eval());

  const std::int32_t half = rope_dim / 2;
  for (std::int32_t p = 0; p < max_seq; ++p) {
    for (std::int32_t i = 0; i < half; ++i) {
      const double freq =
          1.0 / std::pow(static_cast<double>(theta),
                         static_cast<double>(i) / static_cast<double>(half));
      const double angle = static_cast<double>(p) * freq;
      const auto c = static_cast<float>(std::cos(angle));
      const auto s = static_cast<float>(std::sin(angle));
      // Interleaved so channel pairs (2i, 2i+1) share an angle.
      const auto base = static_cast<std::size_t>(p) *
                            static_cast<std::size_t>(rope_dim) +
                        static_cast<std::size_t>(2 * i);
      graph::interpreter::store_element(*t.cos.node(), base, c);
      graph::interpreter::store_element(*t.cos.node(), base + 1, c);
      graph::interpreter::store_element(*t.sin.node(), base, s);
      graph::interpreter::store_element(*t.sin.node(), base + 1, s);
    }
  }
  return t;
}

Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         std::int32_t offset) {
  const Shape& s = x.shape();
  const std::int64_t head_dim = s.dim(s.rank() - 1);
  if (tables.dim > head_dim) {
    return LSE_ERROR(kInvalidArgument, "rope dim exceeds head_dim");
  }
  if (tables.dim == head_dim) {
    return graph::rope(x, tables.cos, tables.sin, offset);
  }
  // Partial rotation: rotate the leading channels, pass the rest through.
  Array rotated = graph::rope(graph::slice(x, -1, 0, tables.dim), tables.cos,
                              tables.sin, offset);
  Array passthrough = graph::slice(x, -1, tables.dim, head_dim);
  return graph::concat({rotated, passthrough}, -1);
}

Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         const Array& offset) {
  const Shape& s = x.shape();
  const std::int64_t head_dim = s.dim(s.rank() - 1);
  if (tables.dim > head_dim) {
    return LSE_ERROR(kInvalidArgument, "rope dim exceeds head_dim");
  }
  if (tables.dim == head_dim) {
    return graph::rope(x, tables.cos, tables.sin, offset);
  }
  Array rotated = graph::rope(graph::slice(x, -1, 0, tables.dim), tables.cos,
                              tables.sin, offset);
  Array passthrough = graph::slice(x, -1, tables.dim, head_dim);
  return graph::concat({rotated, passthrough}, -1);
}

}  // namespace lse::ops
