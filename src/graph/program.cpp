#include "lse/graph/program.hpp"
#include "lse/graph/view.hpp"

#include <cstdlib>
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
    mix_u64(h, n->requires_owned_storage);
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
  view_ownership_.clear();
  for (const auto& node : nodes_)
    if (node->kind == OpKind::kSlice || node->kind == OpKind::kReshape)
      view_ownership_.emplace_back(node.get(), node->requires_owned_storage);
  phases_ = std::move(phases);
  groups_ = std::move(groups);
  emissions_.clear();
  emissions_.resize(groups_.size());
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
  view_ownership_.clear();
  phases_.clear();
  groups_.clear();
  emissions_.clear();
  carries_.clear();
  cuts_.clear();
  sig_ = 0;
  compute_nodes_ = 0;
}

const EmittedKernel* Program::cached_emission(
    std::size_t index, const FusionGroup& group,
    const backend::IBackend* backend, const IKernelEmitter* emitter,
    std::string_view arch, std::uint64_t* key) const noexcept {
  if (index >= emissions_.size()) return nullptr;
  const EmissionCache& cached = emissions_[index];
  if (cached.group != &group || cached.backend != backend ||
      cached.emitter != emitter || cached.arch != arch ||
      !cached.kernel.has_value() ||
      cached.aliases != group_bindings_may_alias(group)) return nullptr;
  *key = cached.key;
  return &*cached.kernel;
}

const EmittedKernel* Program::cache_emission(
    std::size_t index, const FusionGroup& group,
    const backend::IBackend* backend, const IKernelEmitter* emitter,
    std::uint64_t key, std::string_view arch, EmittedKernel kernel) {
  if (index >= emissions_.size()) return nullptr;
  EmissionCache& cached = emissions_[index];
  cached.group = &group;
  cached.backend = backend;
  cached.emitter = emitter;
  cached.key = key;
  cached.aliases = group_bindings_may_alias(group);
  cached.arch = arch;
  cached.kernel.emplace(std::move(kernel));
  return &*cached.kernel;
}

bool Program::holds(std::span<const NodePtr> roots) const noexcept {
  if (roots_.size() != roots.size() || roots_.empty()) return false;
  for (std::size_t i = 0; i < roots_.size(); ++i) {
    if (roots_[i].get() != roots[i].get()) return false;
  }
  for (const auto& [node, owned] : view_ownership_)
    if (node->requires_owned_storage != owned) return false;
  return true;
}

void Program::reset_compute() noexcept {
  for (Workgroup& wg : phases_) wg.reset_compute();
  // Reachable materialized inputs belong to their producer, not this replay.
  // Only invalidate nodes covered by the retained execution schedule.
  for (const FusionGroup& group : groups_) {
    for (const NodePtr& n : group.nodes) {
      if (!n) continue;
      if (n->fclass == FusionClass::kLeaf || n->kind == OpKind::kBuffer) continue;
      n->materialized = false;
    }
  }
}

namespace {
void refresh_carry_view(const NodePtr& node) noexcept {
  if (!node || !is_buffer_view(*node)) return;
  refresh_carry_view(node->inputs[0]);
  (void)bind_buffer_view(*node, false);
}
Node* carry_owner(const NodePtr& node) noexcept {
  return const_cast<Node*>(buffer_allocation_owner(node.get()));
}
}  // namespace

void Program::fold_carries() noexcept {
  for (Carry& c : carries_) {
    Node* in = carry_owner(c.in);
    Node* out = carry_owner(c.out);
    if (!in || !out || in == out || !out->buffer.valid()) continue;
    std::swap(in->buffer, out->buffer);
    in->materialized = true;
    in->device_dirty = true;
    in->host_dirty = false;
    out->materialized = false;
    refresh_carry_view(c.in);
    refresh_carry_view(c.out);
  }
}

void Program::hold_carries() noexcept {
  for (Carry& c : carries_) {
    Node* in = carry_owner(c.in);
    Node* out = carry_owner(c.out);
    if (!in || !out || in == out) continue;
    in->materialized = true;
    in->device_dirty = true;
    in->host_dirty = false;
    out->materialized = false;
    refresh_carry_view(c.in);
    refresh_carry_view(c.out);
  }
}

}  // namespace lse::graph
