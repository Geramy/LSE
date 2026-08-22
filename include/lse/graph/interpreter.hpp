// Node-by-node host evaluation. Deliberately simple and unfused: this is the
// reference the JIT's generated kernels are diffed against.
#pragma once

#include <cstddef>

#include "lse/backend/backend.hpp"
#include "lse/graph/graph.hpp"

namespace lse::graph::interpreter {

Status evaluate(const NodePtr& node, backend::IBackend& backend);

// Allocates the node's output buffer if it has none. The scheduler calls this
// before handing a node to a fallback handler, so handlers can write straight
// into it.
Status ensure_output_buffer(Node& node, backend::IBackend& backend,
                            backend::Stream stream = backend::kDefaultStream);

Result<float> read_scalar(const Node& node);
Status read_raw(const Node& node, void* dst, std::size_t bytes);

float load_element(const Node& node, std::size_t index) noexcept;
void store_element(Node& node, std::size_t index, float value) noexcept;

// Host-addressable bytes for the node: the buffer itself when it is host
// memory, otherwise the mirror. Never null for a sized node.
void* host_bytes(Node& node);
const void* host_bytes(const Node& node) noexcept;

// Move whichever side is stale. Both are no-ops when the buffer is already
// host memory, so the CPU backend pays nothing for them.
Status sync_to_device(Node& node, backend::IBackend& backend);
Status sync_from_device(Node& node, backend::IBackend& backend);

}  // namespace lse::graph::interpreter
