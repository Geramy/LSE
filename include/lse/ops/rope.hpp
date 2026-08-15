// Rotary position embedding tables and application.
#pragma once

#include <cstdint>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"

namespace lse::ops {

using graph::Array;

struct RopeTables {
  Array cos;
  Array sin;
  std::int32_t dim = 0;
  std::int32_t max_seq = 0;
};

// Pair-interleaved cos/sin of shape [max_seq, dim], matching
// mx.repeat(angles, 2, axis=-1).
Result<RopeTables> build_rope(std::int32_t rope_dim, std::int32_t max_seq,
                              float theta);

// Rotates the leading `tables.dim` channels and passes the rest through, which
// is what a partial_rotary_factor < 1 needs (Qwen3.6 rotates 0.25 of head_dim).
// When tables.dim == head_dim this is a plain full rotation.
Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         std::int32_t offset);
Result<Array> apply_rope(const Array& x, const RopeTables& tables,
                         const Array& offset);

}  // namespace lse::ops
