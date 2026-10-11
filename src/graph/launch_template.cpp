#include "lse/graph/launch_template.hpp"

#include <cstring>
#include <map>
#include <tuple>
#include <unordered_set>

#include "lse/graph/program.hpp"

namespace lse::graph {

namespace {

// Fixed-width fields, appended raw: the key is compared byte for byte, so
// every field has one encoding and the field order is fixed.
class KeyWriter {
 public:
  explicit KeyWriter(std::string& out) : out_(out) {}
  template <class T>
  void put(T v) {
    static_assert(std::is_trivially_copyable_v<T>);
    char bytes[sizeof(T)];
    std::memcpy(bytes, &v, sizeof(T));
    out_.append(bytes, sizeof(T));
  }
  void put_text(std::string_view s) {
    put<std::uint64_t>(s.size());
    out_.append(s.data(), s.size());
  }

 private:
  std::string& out_;
};

constexpr std::uint32_t kNoPosition = 0xFFFFFFFFu;

}  // namespace

std::vector<NodePtr> template_nodes(std::span<const NodePtr> roots) {
  std::vector<NodePtr> out;
  std::unordered_set<const Node*> seen;
  for (const NodePtr& r : roots) collect_reachable(r, out, seen);
  return out;
}

std::unordered_map<const Node*, std::uint32_t> node_positions(
    std::span<const NodePtr> nodes) {
  std::unordered_map<const Node*, std::uint32_t> at;
  at.reserve(nodes.size());
  for (std::uint32_t i = 0; i < nodes.size(); ++i) at.emplace(nodes[i].get(), i);
  return at;
}

std::string launch_template_key(std::span<const NodePtr> nodes,
                                std::span<const NodePtr> roots,
                                std::string_view context) {
  const auto at = node_positions(nodes);
  const auto position = [&](const Node* n) {
    if (n == nullptr) return kNoPosition - 1;
    const auto it = at.find(n);
    return it == at.end() ? kNoPosition : it->second;
  };
  std::string key;
  key.reserve(nodes.size() * 128 + context.size() + 64);
  KeyWriter w(key);
  w.put_text(context);
  w.put<std::uint64_t>(nodes.size());
  w.put<std::uint64_t>(roots.size());
  for (const NodePtr& r : roots) w.put<std::uint32_t>(position(r.get()));
  // Which allocation each buffer belongs to, as the first node holding it:
  // sharing is structure, the address is not.
  std::map<std::tuple<const void*, std::uint64_t, const void*>, std::uint32_t> owner;
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    const Node& n = *nodes[i];
    w.put<std::uint8_t>(static_cast<std::uint8_t>(n.kind));
    w.put<std::uint8_t>(static_cast<std::uint8_t>(n.fclass));
    // The primitive object is the kernel's identity for the life of the
    // process, and a template never outlives the process.
    w.put<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(n.prim));
    w.put<std::uint8_t>(static_cast<std::uint8_t>(n.dtype));
    w.put<std::uint64_t>(n.shape.rank());
    for (std::size_t d = 0; d < n.shape.rank(); ++d) w.put<std::int64_t>(n.shape.dim(d));
    for (const float a : n.attrs) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &a, sizeof bits);
      w.put<std::uint32_t>(bits);
    }
    for (const std::int32_t a : n.iattrs) w.put<std::int32_t>(a);
    w.put<std::uint64_t>(n.inputs.size());
    for (const NodePtr& in : n.inputs) w.put<std::uint32_t>(position(in.get()));
    std::uint8_t flags = 0;
    flags |= n.materialized ? 1u : 0u;
    flags |= n.requires_owned_storage ? 2u : 0u;
    flags |= n.host_dirty ? 4u : 0u;
    flags |= n.device_dirty ? 8u : 0u;
    flags |= n.proposal_only ? 16u : 0u;
    flags |= n.kv_fragments ? 32u : 0u;
    w.put<std::uint8_t>(flags);
    w.put<std::uint16_t>(n.member);
    // Read by slot planning for the nodes a step computes. A leaf's count
    // grows with every graph built over it (a weight's, request after
    // request), and nothing plans by it.
    w.put<std::uint32_t>(n.materialized ? 0u : n.consumer_count);
    // Quantized planes are a weight's, held for the life of the model.
    w.put<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(n.quant.get()));
    w.put<std::uint32_t>(position(n.quant_activation_panel.lock().get()));
    w.put<std::uint32_t>(position(n.f16_activation_panel.lock().get()));
    const backend::DeviceBuffer& b = n.buffer;
    w.put<std::uint8_t>(b.valid() ? 1u : 0u);
    if (b.valid()) {
      const auto id = std::make_tuple(static_cast<const void*>(b.storage.get()),
                                      b.storage ? std::uint64_t{0} : b.handle,
                                      b.storage ? nullptr : static_cast<const void*>(b.ptr));
      const auto [it, fresh] = owner.emplace(id, i);
      w.put<std::uint32_t>(it->second);
      w.put<std::uint64_t>(b.offset);
      w.put<std::uint64_t>(b.size_bytes);
      w.put<std::uint16_t>(b.residency.value);
      w.put<std::uint16_t>(b.member);
      w.put<std::uint8_t>(b.ptr != nullptr ? 1u : 0u);
    }
  }
  return key;
}

std::optional<FusionGroup> template_group(const LaunchTemplate::Launch& launch,
                                          std::span<const NodePtr> nodes) {
  FusionGroup g;
  const auto take = [&](const std::vector<std::uint32_t>& from,
                        std::vector<NodePtr>& to) {
    to.reserve(from.size());
    for (const std::uint32_t i : from) {
      if (i >= nodes.size()) return false;
      to.push_back(nodes[i]);
    }
    return true;
  };
  if (!take(launch.nodes, g.nodes) || !take(launch.inputs, g.inputs) ||
      !take(launch.outputs, g.outputs))
    return std::nullopt;
  g.anchor = launch.anchor;
  g.anchor_class = launch.anchor_class;
  g.launches = launch.launches;
  g.is_phase = launch.is_phase;
  return g;
}

std::optional<EmittedKernel> template_emission(const LaunchTemplate::Launch& launch,
                                               std::span<const NodePtr> nodes) {
  EmittedKernel k = launch.emitted;
  k.binding_order.clear();
  k.binding_order.reserve(launch.binding.size());
  for (const std::uint32_t i : launch.binding) {
    if (i == kNoPosition - 1) {
      k.binding_order.push_back(nullptr);
      continue;
    }
    if (i >= nodes.size()) return std::nullopt;
    k.binding_order.push_back(nodes[i]);
  }
  return k;
}

bool same_launch_group(const FusionGroup& a, const FusionGroup& b) {
  return a.nodes == b.nodes && a.inputs == b.inputs && a.outputs == b.outputs &&
         a.anchor == b.anchor && a.anchor_class == b.anchor_class &&
         a.launches == b.launches && a.is_phase == b.is_phase;
}

std::string emission_difference(const EmittedKernel& a, const EmittedKernel& b) {
  const auto same_traffic = [](const opt::TrafficModel& x, const opt::TrafficModel& y) {
    return x.read == y.read && x.written == y.written && x.work == y.work &&
           x.workgroups == y.workgroups && x.workgroup_threads == y.workgroup_threads &&
           x.stated == y.stated;
  };
  const auto same_constants = [](const ConstantsLayout& x, const ConstantsLayout& y) {
    if (x.total_bytes != y.total_bytes || x.fields.size() != y.fields.size()) return false;
    for (std::size_t i = 0; i < x.fields.size(); ++i)
      if (x.fields[i].name != y.fields[i].name || x.fields[i].offset != y.fields[i].offset ||
          x.fields[i].size != y.fields[i].size)
        return false;
    return true;
  };
  for (int d = 0; d < 3; ++d)
    if (a.dims.workgroup_count[d] != b.dims.workgroup_count[d] ||
        a.dims.workgroup_size[d] != b.dims.workgroup_size[d])
      return "launch dims";
  if (a.source != b.source) return "source";
  if (!same_traffic(a.traffic, b.traffic)) return "traffic";
  if (a.dialect != b.dialect) return "dialect";
  if (a.entry_name != b.entry_name) return "entry name (" + a.entry_name + " / " + b.entry_name + ")";
  if (a.content_addressed != b.content_addressed) return "content addressing";
  if (a.structural_entry_name != b.structural_entry_name) return "structural entry name";
  if (!same_constants(a.constants, b.constants)) return "constants layout";
  if (a.binding_order.size() != b.binding_order.size())
    return "binding count (" + std::to_string(a.binding_order.size()) + " / " +
           std::to_string(b.binding_order.size()) + ")";
  for (std::size_t i = 0; i < a.binding_order.size(); ++i)
    if (a.binding_order[i] != b.binding_order[i]) return "binding " + std::to_string(i);
  if (a.dims.subgroup_size != b.dims.subgroup_size) return "subgroup size";
  if (a.lds_bytes != b.lds_bytes) return "workgroup memory";
  if (a.scratch_bytes != b.scratch_bytes) return "scratch";
  if (a.pointer_table != b.pointer_table) return "pointer table";
  if (a.persist_grid != b.persist_grid) return "persistent grid";
  return {};
}

bool same_emission(const EmittedKernel& a, const EmittedKernel& b) {
  return emission_difference(a, b).empty();
}

}  // namespace lse::graph
