#include "lse/graph/quant_swiglu.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include <algorithm>
#include <array>
#include <unordered_set>

namespace lse::graph {
std::size_t optimize_quant_swiglu(std::span<const NodePtr> roots,
                                  const backend::DeviceInfo &device,
                                  const IKernelEmitter &emitter,
                                  std::size_t member) {
  if (emitter.dialect() != Dialect::kLoom)
    return 0;
  const auto *pair = dynamic_cast<const KernelPrimitiveBase *>(
      find_primitive("quant_swiglu.q4_shared_panel.v2"));
  if (!pair)
    return 0;
  const auto sources = emitter.sources();
  const std::unordered_set<NodePtr> requested(roots.begin(), roots.end());
  std::size_t changed = 0;
  for (const auto &out : Partitioner::unmaterialized(roots)) {
    if (out->kind != OpKind::kMul || out->inputs.size() != 2 ||
        out->materialized || out->buffer.valid())
      continue;
    auto silu = out->inputs[0], up = out->inputs[1];
    if (silu->kind != OpKind::kSiLU)
      std::swap(silu, up);
    if (silu->kind != OpKind::kSiLU || silu->inputs.size() != 1)
      continue;
    const auto gate = silu->inputs[0];
    const auto projection = [](const NodePtr &n) {
      return n->kind == OpKind::kQuantMatMul && n->prim &&
             n->prim->name() == "quant_linear.q4_global_panel.v1" &&
             n->inputs.size() == 5;
    };
    if (!projection(gate) || !projection(up) || gate == up ||
        gate->inputs[0] != up->inputs[0] || gate->inputs[4] != up->inputs[4] ||
        gate->iattrs != up->iattrs)
      continue;
    bool exclusive = true;
    for (const auto &n : {gate, up, silu})
      exclusive &= !n->materialized && !n->buffer.valid() &&
                   n->consumer_count == 1 && !requested.contains(n) &&
                   n->shape == out->shape && n->dtype == out->dtype;
    std::vector<NodePtr> inputs{
        gate->inputs[0], gate->inputs[1], gate->inputs[2], gate->inputs[3],
        up->inputs[1],   up->inputs[2],   up->inputs[3],   gate->inputs[4]};
    for (const auto &n : {out, gate, up, silu})
      exclusive &= n->member == Node::kAnyMember || n->member == member;
    for (const auto &n : inputs)
      exclusive &= n->member == Node::kAnyMember || n->member == member;
    if (!exclusive)
      continue;
    std::vector<Shape> shapes;
    std::vector<DType> dtypes;
    for (const auto &n : inputs) {
      shapes.push_back(n->shape);
      dtypes.push_back(n->dtype);
    }
    KernelShapes s;
    s.inputs = shapes;
    s.input_dtypes = dtypes;
    s.output = out->shape;
    s.output_dtype = out->dtype;
    s.iattrs = gate->iattrs;
    s.device = &device;
    s.intrinsics = &sources;
    if (!pair->plan(s).workgroup_count[0])
      continue;
    for (const auto &n : out->inputs)
      --n->consumer_count;
    std::unordered_set<NodePtr> seen;
    for (const auto &n : inputs)
      if (seen.insert(n).second)
        ++n->consumer_count;
    out->set_kind(OpKind::kCustom);
    out->prim = pair;
    out->fclass = pair->fusion_class();
    out->inputs = std::move(inputs);
    out->iattrs = s.iattrs;
    ++changed;
  }
  return changed;
}
} // namespace lse::graph
