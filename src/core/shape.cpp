#include "lse/core/shape.hpp"

#include <algorithm>

namespace lse {

std::string Shape::to_string() const {
  std::string out = "(";
  for (std::size_t i = 0; i < rank_; ++i) {
    if (i) out += ", ";
    out += std::to_string(dims_[i]);
  }
  out += ")";
  return out;
}

Shape Shape::broadcast(const Shape& a, const Shape& b) noexcept {
  const std::size_t rank = std::max(a.rank_, b.rank_);
  Shape out;
  out.rank_ = rank;
  // Align trailing dimensions, NumPy-style.
  for (std::size_t i = 0; i < rank; ++i) {
    const std::size_t ai = a.rank_ + i - rank;  // wraps when i < rank - a.rank_
    const std::size_t bi = b.rank_ + i - rank;
    const std::int64_t da = (i + a.rank_ >= rank) ? a.dims_[ai] : 1;
    const std::int64_t db = (i + b.rank_ >= rank) ? b.dims_[bi] : 1;
    if (da != db && da != 1 && db != 1) return Shape{};  // incompatible
    out.dims_[i] = std::max(da, db);
  }
  out.recompute();
  return out;
}

bool Shape::is_broadcastable_to(const Shape& target) const noexcept {
  if (rank_ > target.rank_) return false;
  for (std::size_t i = 0; i < rank_; ++i) {
    const std::int64_t d = dims_[rank_ - 1 - i];
    const std::int64_t t = target.dims_[target.rank_ - 1 - i];
    if (d != t && d != 1) return false;
  }
  return true;
}

BroadcastMap BroadcastMap::build(const Shape& src, const Shape& out) noexcept {
  BroadcastMap m;
  m.rank = out.rank();
  // Equal element counts covers both src==out and a pure reshape. Checking it
  // first also guards the `gap` subtraction below, which underflows when src
  // has HIGHER rank than out.
  if (src == out || src.elem_count() == out.elem_count() ||
      src.rank() > out.rank()) {
    m.identity = true;
    return m;
  }
  if (src.elem_count() == 1) {
    m.scalar = true;
    return m;
  }

  m.gap = out.rank() - src.rank();
  m.out_stride = out.strides();
  m.out_dim = {};
  for (std::size_t i = 0; i < out.rank(); ++i) m.out_dim[i] = out.dim(i);

  const auto s = src.strides();
  for (std::size_t i = 0; i < src.rank(); ++i) {
    // A size-1 source dim contributes nothing: stride 0 means "broadcast".
    m.src_stride[i] = src.dim(i) == 1 ? 0 : s[i];
  }
  return m;
}

}  // namespace lse
