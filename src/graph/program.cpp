#include "lse/graph/program.hpp"

#include <unordered_map>

namespace lse::graph {

namespace {

void mix_u64(std::uint64_t& h, std::uint64_t v) noexcept {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

}  // namespace

void collect_reachable(const NodePtr& n, std::vector<NodePtr>& out,
                       std::unordered_set<const Node*>& seen) {
  if (!n || !seen.insert(n.get()).second) return;
  for (const NodePtr& in : n->inputs) collect_reachable(in, out, seen);
  out.push_back(n);
}

std::uint64_t program_signature(std::span<const NodePtr> order) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  mix_u64(h, order.size());
  std::unordered_map<const Node*, std::uint32_t> idx;
  idx.reserve(order.size());
  for (std::uint32_t i = 0; i < order.size(); ++i) idx[order[i].get()] = i;
  for (const NodePtr& n : order) {
    mix_u64(h, static_cast<std::uint64_t>(n->kind));
    mix_u64(h, n->shape.rank());
    for (std::size_t d = 0; d < n->shape.rank(); ++d) {
      mix_u64(h, static_cast<std::uint64_t>(n->shape.dim(d)));
    }
    for (std::int32_t a : n->iattrs) mix_u64(h, static_cast<std::uint64_t>(a));
    mix_u64(h, n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      auto it = idx.find(in.get());
      if (it != idx.end()) {
        mix_u64(h, it->second);
        continue;
      }
      mix_u64(h, 0xffffffffULL);
      if (!in) continue;
      mix_u64(h, static_cast<std::uint64_t>(in->kind));
      mix_u64(h, in->element_count());
    }
  }
  return h;
}

void Program::retain(std::span<const NodePtr> roots,
                     std::vector<Workgroup> phases,
                     std::vector<FusionGroup> groups,
                     std::span<const NodePtr> compute_order) {
  roots_.assign(roots.begin(), roots.end());
  nodes_.clear();
  std::unordered_set<const Node*> seen;
  for (const NodePtr& r : roots_) collect_reachable(r, nodes_, seen);
  phases_ = std::move(phases);
  groups_ = std::move(groups);
  compute_nodes_ = static_cast<std::uint32_t>(compute_order.size());
  sig_ = program_signature(compute_order);

  std::unordered_map<const Node*, std::uint32_t> idx;
  idx.reserve(compute_order.size());
  for (std::uint32_t i = 0; i < compute_order.size(); ++i) {
    idx[compute_order[i].get()] = i;
  }
  cuts_.clear();
  cuts_.reserve(groups_.size());
  for (const FusionGroup& g : groups_) {
    Cut c;
    c.anchor = g.anchor;
    c.anchor_class = g.anchor_class;
    c.launches = g.launches;
    c.is_phase = g.is_phase;
    for (const NodePtr& n : g.nodes) {
      auto it = idx.find(n.get());
      if (it != idx.end()) c.nodes.push_back(it->second);
    }
    for (const NodePtr& n : g.outputs) {
      auto it = idx.find(n.get());
      if (it != idx.end()) c.outputs.push_back(it->second);
    }
    cuts_.push_back(std::move(c));
  }
}

std::vector<FusionGroup> Program::remap(
    std::span<const NodePtr> compute_order) const {
  std::vector<FusionGroup> out;
  out.reserve(cuts_.size());
  for (const Cut& c : cuts_) {
    FusionGroup g;
    g.anchor = c.anchor;
    g.anchor_class = c.anchor_class;
    g.launches = c.launches;
    g.is_phase = c.is_phase;
    for (std::uint32_t i : c.nodes) {
      if (i < compute_order.size()) g.nodes.push_back(compute_order[i]);
    }
    for (std::uint32_t i : c.outputs) {
      if (i < compute_order.size()) g.outputs.push_back(compute_order[i]);
    }
    std::unordered_set<const Node*> members;
    for (const NodePtr& n : g.nodes) members.insert(n.get());
    for (const NodePtr& n : g.nodes) {
      for (const NodePtr& in : n->inputs) {
        if (!in || members.count(in.get())) continue;
        bool seen = false;
        for (const NodePtr& e : g.inputs) {
          if (e.get() == in.get()) {
            seen = true;
            break;
          }
        }
        if (!seen) g.inputs.push_back(in);
      }
    }
    if (!g.nodes.empty()) out.push_back(std::move(g));
  }
  return out;
}

void Program::destroy() noexcept {
  for (Workgroup& wg : phases_) wg.clear();
  roots_.clear();
  nodes_.clear();
  phases_.clear();
  groups_.clear();
  carries_.clear();
  cuts_.clear();
  sig_ = 0;
  compute_nodes_ = 0;
}

bool Program::holds(std::span<const NodePtr> roots) const noexcept {
  if (roots_.size() != roots.size() || roots_.empty()) return false;
  for (std::size_t i = 0; i < roots_.size(); ++i) {
    if (roots_[i].get() != roots[i].get()) return false;
  }
  return true;
}

void Program::reset_compute() noexcept {
  for (Workgroup& wg : phases_) wg.reset_compute();
  for (const NodePtr& n : nodes_) {
    if (!n) continue;
    if (n->fclass == FusionClass::kLeaf || n->kind == OpKind::kBuffer) continue;
    n->materialized = false;
  }
}

void Program::fold_carries() noexcept {
  for (Carry& c : carries_) {
    if (!c.in || !c.out || !c.out->buffer.valid()) continue;
    backend::DeviceBuffer produced = c.out->buffer;
    backend::DeviceBuffer consumed = c.in->buffer;
    c.in->buffer = produced;
    c.in->materialized = true;
    c.in->device_dirty = true;
    c.in->host_dirty = false;
    c.out->buffer = consumed;
    c.out->materialized = false;
  }
}

void Program::hold_carries() noexcept {
  for (Carry& c : carries_) {
    if (!c.in || !c.out) continue;
    c.in->materialized = true;
    c.in->device_dirty = true;
    c.in->host_dirty = false;
    c.out->materialized = false;
  }
}

}  // namespace lse::graph
