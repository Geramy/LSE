#pragma once
#include "lse/graph/graph.hpp"
#include <limits>
#include <utility>

namespace lse::runtime {
// B1 feature rows form a contiguous window in an owned, materialized buffer.
[[nodiscard]] inline Result<graph::Array> materialized_feature_prefix(
    const graph::Array& source, std::size_t rows) {
  if (!source.valid() || source.dtype() != DType::kF32 ||
      source.shape().rank() != 3 || source.shape().dim(0) != 1 ||
      source.shape().dim(1) <= 0 || source.shape().dim(2) <= 0 || rows == 0 ||
      rows > static_cast<std::size_t>(source.shape().dim(1)))
    return LSE_ERROR(kInvalidArgument, "invalid materialized feature prefix shape");
  const graph::Node& owner = *source.node();
  if (!owner.materialized || !owner.buffer.valid() || !owner.buffer.storage ||
      (owner.buffer.ptr == nullptr && owner.host_dirty))
    return LSE_ERROR(kInvalidArgument, "feature prefix needs owned device-current storage");
  const auto width = static_cast<std::size_t>(source.shape().dim(2));
  const auto count = static_cast<std::size_t>(source.shape().dim(1));
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  if (width > limit / sizeof(float) || count > limit / (width * sizeof(float)))
    return LSE_ERROR(kOutOfRange, "feature prefix byte extent overflows");
  const auto row_bytes = width * sizeof(float);
  const auto bytes = count * row_bytes;
  if (bytes > owner.buffer.size_bytes || owner.buffer.offset > limit - bytes)
    return LSE_ERROR(kOutOfRange, "feature prefix exceeds its source buffer window");
  auto buffer = owner.buffer;
  buffer.size_bytes = rows * row_bytes;
  auto view = graph::Array::from_buffer(std::move(buffer),
      Shape{1, static_cast<std::int64_t>(rows), source.shape().dim(2)}, DType::kF32);
  view.node()->member = owner.member;
  return view;
}
}  // namespace lse::runtime
