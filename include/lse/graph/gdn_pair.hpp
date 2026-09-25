#pragma once
#include "lse/graph/graph.hpp"

namespace lse::graph {
// Only exact pure recurrence siblings qualify. Other consumers still observe
// their original output nodes; this changes scheduling, never graph identity.
struct GdnPair {
  NodePtr output;
  NodePtr state;
  explicit operator bool() const { return output && state; }
};
inline GdnPair exact_gdn_pair(const NodePtr& a, const NodePtr& b) {
  if (!a || !b || a == b || a->kind != OpKind::kGDNChunkScan ||
      b->kind != OpKind::kGDNChunkScan || a->inputs.size() != 6 || b->inputs.size() != 6 ||
      a->member != b->member || a->dtype != DType::kF32 || b->dtype != DType::kF32 ||
      !a->prim || !b->prim || a->prim->name() != "gdn_chunk_scan" ||
      b->prim->name() != "gdn_chunk_scan" || a->attrs != b->attrs) return {};
  for (std::size_t i = 1; i < a->iattrs.size(); ++i)
    if (a->iattrs[i] != b->iattrs[i]) return {};
  if (!((a->iattrs[0] == 0 && b->iattrs[0] == 1) ||
        (a->iattrs[0] == 1 && b->iattrs[0] == 0))) return {};
  for (std::size_t i = 0; i < 6; ++i)
    if (!a->inputs[i] || a->inputs[i] != b->inputs[i] || a->inputs[i]->dtype != DType::kF32) return {};
  const auto& q = a->inputs[0]->shape;
  if (q.rank() != 4 || q.dim(0) <= 0 || q.dim(1) <= 0 || q.dim(2) <= 0 ||
      (q.dim(3) != 16 && q.dim(3) != 32 && q.dim(3) != 64 && q.dim(3) != 128) ||
      a->inputs[1]->shape != q || a->inputs[2]->shape != q) return {};
  const Shape scalar{q.dim(0), q.dim(1), q.dim(2)};
  const Shape state_shape{q.dim(0), q.dim(2), q.dim(3), q.dim(3)};
  if (a->inputs[3]->shape != scalar || a->inputs[4]->shape != scalar ||
      a->inputs[5]->shape != state_shape) return {};
  GdnPair pair{a->iattrs[0] == 0 ? a : b, a->iattrs[0] == 1 ? a : b};
  if (pair.output->shape != q || pair.state->shape != state_shape) return {};
  return pair;
}
} // namespace lse::graph
