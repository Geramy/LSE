#pragma once

#include <algorithm>
#include <limits>
#include <optional>

#include "lse/graph/graph.hpp"

namespace lse::graph {

struct BufferViewWindow {
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

// A flat view can only name contiguous, byte-addressable storage.
inline Result<std::optional<BufferViewWindow>> buffer_view_window(const Node& n) {
  if (n.kind != OpKind::kReshape && n.kind != OpKind::kSlice)
    return std::optional<BufferViewWindow>{};
  if (n.inputs.size() != 1 || !n.inputs[0])
    return LSE_ERROR(kInvalidArgument, "view requires one input");
  const Node& source = *n.inputs[0];
  if (n.dtype != source.dtype)
    return LSE_ERROR(kInvalidArgument, "view changes dtype");
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  auto count = [&](const Shape& shape) -> Result<std::size_t> {
    std::size_t value = 1;
    if (shape.rank() == 0)
      return LSE_ERROR(kInvalidArgument, "view requires a nonempty shape");
    for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
      const auto dimension = shape.dim(axis);
      if (dimension <= 0 || static_cast<std::uint64_t>(dimension) > limit / value)
        return LSE_ERROR(kOutOfRange, "view shape is empty or overflows");
      value *= static_cast<std::size_t>(dimension);
    }
    return value;
  };
  LSE_ASSIGN_OR(const auto source_count, count(source.shape));
  LSE_ASSIGN_OR(const auto output_count, count(n.shape));
  const auto element_bytes = dtype_info(n.dtype).size_bytes;
  if (element_bytes != 0 && output_count > limit / element_bytes)
    return LSE_ERROR(kOutOfRange, "view byte extent overflows");
  const auto& dtype = dtype_info(n.dtype);
  if (element_bytes == 0 && (dtype.block_elems == 0 || dtype.block_bytes == 0 ||
      output_count / dtype.block_elems > limit / dtype.block_bytes))
    return LSE_ERROR(kOutOfRange, "view block storage extent overflows");
  const auto bytes = dtype_storage_bytes(n.dtype, output_count);
  if (bytes == 0) return LSE_ERROR(kOutOfRange, "view storage extent is invalid");
  if (n.kind == OpKind::kReshape) {
    if (source_count != output_count)
      return LSE_ERROR(kInvalidArgument, "reshape changes element count");
    return std::optional<BufferViewWindow>{{0, bytes}};
  }
  const auto axis = n.iattrs[0], begin = n.iattrs[1], end = n.iattrs[2];
  if (axis < 0 || static_cast<std::size_t>(axis) >= source.shape.rank() ||
      begin < 0 || end <= begin || end > source.shape.dim(static_cast<std::size_t>(axis)) ||
      n.shape.rank() != source.shape.rank())
    return LSE_ERROR(kOutOfRange, "slice window is outside its source");
  std::size_t outer = 1, inner = 1;
  for (std::size_t a = 0; a < source.shape.rank(); ++a) {
    const auto expected = a == static_cast<std::size_t>(axis)
        ? static_cast<std::int64_t>(end) - begin : source.shape.dim(a);
    if (n.shape.dim(a) != expected)
      return LSE_ERROR(kInvalidArgument, "slice output shape does not match its window");
    if (a < static_cast<std::size_t>(axis)) outer *= static_cast<std::size_t>(source.shape.dim(a));
    if (a > static_cast<std::size_t>(axis)) inner *= static_cast<std::size_t>(source.shape.dim(a));
  }
  if (n.requires_owned_storage || element_bytes == 0 ||
      (outer != 1 && (begin != 0 || end != source.shape.dim(static_cast<std::size_t>(axis)))))
    return std::optional<BufferViewWindow>{};
  if (inner > limit / element_bytes || static_cast<std::size_t>(begin) > limit / (inner * element_bytes) ||
      output_count > limit / element_bytes)
    return LSE_ERROR(kOutOfRange, "slice byte window overflows");
  return std::optional<BufferViewWindow>{{static_cast<std::size_t>(begin) * inner * element_bytes, bytes}};
}

inline bool is_buffer_view(const Node& n) {
  auto window = buffer_view_window(n);
  return window.ok() && window->has_value();
}

inline Status bind_buffer_view(Node& n, bool require_materialized = true) {
  LSE_ASSIGN_OR(auto window, buffer_view_window(n));
  if (!window) return LSE_ERROR(kUnimplemented, "node requires materialized storage");
  const Node& source = *n.inputs[0];
  if (!source.buffer.valid() || (require_materialized && !source.materialized))
    return LSE_ERROR(kUnimplemented, "view ", std::string(to_string(n.kind)),
        n.shape.to_string(), " needs a materialized input; source ",
        std::string(to_string(source.kind)), source.shape.to_string(),
        " materialized=", source.materialized ? "true" : "false",
        " buffer_valid=", source.buffer.valid() ? "true" : "false");
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  if (window->offset > source.buffer.size_bytes ||
      window->bytes > source.buffer.size_bytes - window->offset ||
      source.buffer.offset > limit - window->offset ||
      source.buffer.offset + window->offset > limit - window->bytes)
    return LSE_ERROR(kOutOfRange, "view exceeds its source buffer window");
  if (source.host_dirty && source.buffer.ptr == nullptr &&
      (window->offset > source.host_mirror.size() ||
       window->bytes > source.host_mirror.size() - window->offset))
    return LSE_ERROR(kInvalidArgument, "view has no complete authoritative host window");
  n.buffer = source.buffer;
  n.buffer.offset += window->offset;
  n.buffer.size_bytes = window->bytes;
  n.materialized = source.materialized;
  n.device_dirty = source.device_dirty && !source.host_dirty;
  n.host_dirty = source.host_dirty;
  n.host_mirror.clear();
  const bool complete_mirror = window->offset <= source.host_mirror.size() &&
      window->bytes <= source.host_mirror.size() - window->offset;
  if (complete_mirror && (source.host_dirty || !source.device_dirty)) {
    const auto first = source.host_mirror.begin() + static_cast<std::ptrdiff_t>(window->offset);
    n.host_mirror.assign(first, first + static_cast<std::ptrdiff_t>(window->bytes));
  } else if (source.buffer.ptr == nullptr && !source.host_dirty) {
    n.device_dirty = true;
  }
  return OkStatus();
}

// Ownership is structural even before buffers have been allocated.
inline const Node* buffer_alias_source(const Node* node) {
  if (!node) return nullptr;
  if (is_buffer_view(*node)) return node->inputs[0].get();
  const int input = node->prim ? node->prim->inplace_input() : -1;
  return input >= 0 && static_cast<std::size_t>(input) < node->inputs.size()
      ? node->inputs[static_cast<std::size_t>(input)].get() : nullptr;
}

inline const Node* buffer_allocation_owner(const Node* node) {
  const Node* slow = node;
  const Node* fast = node;
  while (const Node* next = buffer_alias_source(node)) {
    node = next;
    slow = buffer_alias_source(slow);
    fast = buffer_alias_source(buffer_alias_source(fast));
    if (slow && slow == fast) return nullptr;
  }
  return node;
}

inline bool buffer_bindings_may_alias(const Node& a, const Node& b) {
  const auto* a_owner = buffer_allocation_owner(&a);
  const auto* b_owner = buffer_allocation_owner(&b);
  if (!a_owner || !b_owner) return true;
  const auto& ab = a.buffer;
  const auto& bb = b.buffer;
  if (ab.valid() && bb.valid() && ab.residency == bb.residency && ab.member == bb.member &&
      ((ab.handle != 0 && ab.handle == bb.handle) || (ab.ptr != nullptr && ab.ptr == bb.ptr))) {
    constexpr auto limit = std::numeric_limits<std::size_t>::max();
    if (ab.offset > limit - ab.size_bytes || bb.offset > limit - bb.size_bytes) return true;
    return ab.offset < bb.offset + bb.size_bytes && bb.offset < ab.offset + ab.size_bytes;
  }
  return a_owner == b_owner;
}

inline bool bindings_may_alias(std::span<const NodePtr> bindings) {
  for (std::size_t i = 0; i < bindings.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (bindings[i] && bindings[j] &&
          buffer_bindings_may_alias(*bindings[i], *bindings[j])) return true;
  return false;
}

inline bool group_bindings_may_alias(const FusionGroup& group) {
  const auto count = group.inputs.size() + group.outputs.size();
  auto binding = [&](std::size_t i) -> const NodePtr& {
    return i < group.inputs.size() ? group.inputs[i]
                                  : group.outputs[i - group.inputs.size()];
  };
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const auto& a = binding(i);
      const auto& b = binding(j);
      if (a && b && a != b && buffer_bindings_may_alias(*a, *b)) return true;
    }
  return false;
}

inline Status refresh_buffer_aliases(std::span<const NodePtr> order) {
  for (const auto& node : order) {
    if (!node) continue;
    const int input = node->prim ? node->prim->inplace_input() : -1;
    if (input >= 0 && static_cast<std::size_t>(input) < node->inputs.size() &&
        node->inputs[static_cast<std::size_t>(input)] &&
        node->inputs[static_cast<std::size_t>(input)]->buffer.valid())
      node->buffer = node->inputs[static_cast<std::size_t>(input)]->buffer;
    if (is_buffer_view(*node) && node->inputs[0]->buffer.valid())
      LSE_RETURN_IF_ERROR(bind_buffer_view(*node, false));
  }
  return OkStatus();
}

}  // namespace lse::graph
