// A captured DAG plus the Workgroups that run it.
//
// The first eval of a shape builds nodes and plans launches. After that the
// Program holds those NodePtrs so the next eval of the same computation does
// not call make(). Workgroup::reset_compute() puts every non-leaf back to
// "not yet run" without freeing buffers. Carried state is folded by swapping
// the produced buffer onto the input node the next step reads.
#pragma once

#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

#include "lse/graph/graph.hpp"
#include "lse/graph/workgroup.hpp"

namespace lse::graph {

class Program {
 public:
  Program() = default;

  void retain(std::span<const NodePtr> roots, std::vector<Workgroup> phases,
              std::vector<FusionGroup> groups,
              std::span<const NodePtr> compute_order);
  void destroy() noexcept;

  [[nodiscard]] bool empty() const noexcept { return roots_.empty(); }

  // Same Node objects as last retain — the graph is still in memory.
  [[nodiscard]] bool holds(std::span<const NodePtr> roots) const noexcept;

  [[nodiscard]] std::uint64_t signature() const noexcept { return sig_; }
  [[nodiscard]] std::uint32_t node_count() const noexcept {
    return static_cast<std::uint32_t>(nodes_.size());
  }
  [[nodiscard]] std::uint32_t compute_count() const noexcept {
    return compute_nodes_;
  }

  void reset_compute() noexcept;

  [[nodiscard]] std::span<const NodePtr> roots() const noexcept { return roots_; }
  [[nodiscard]] std::span<const NodePtr> nodes() const noexcept { return nodes_; }
  [[nodiscard]] std::vector<FusionGroup>& groups() noexcept { return groups_; }
  [[nodiscard]] const std::vector<FusionGroup>& groups() const noexcept {
    return groups_;
  }
  [[nodiscard]] std::vector<Workgroup>& phases() noexcept { return phases_; }

  // Replay launch cuts onto a newly built isomorphic DAG.
  [[nodiscard]] std::vector<FusionGroup> remap(
      std::span<const NodePtr> compute_order) const;

  struct Carry {
    NodePtr in;
    NodePtr out;
  };
  void set_carries(std::vector<Carry> carries) { carries_ = std::move(carries); }
  [[nodiscard]] const std::vector<Carry>& carries() const noexcept {
    return carries_;
  }
  void fold_carries() noexcept;

 private:
  std::vector<NodePtr> roots_;
  std::vector<NodePtr> nodes_;
  std::vector<Workgroup> phases_;
  std::vector<FusionGroup> groups_;
  std::vector<Carry> carries_;
  struct Cut {
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint32_t> outputs;
    OpKind anchor = OpKind::kCustom;
    FusionClass anchor_class = FusionClass::kBarrier;
    std::uint32_t launches = 1;
    bool is_phase = false;
  };
  std::vector<Cut> cuts_;
  std::uint64_t sig_ = 0;
  std::uint32_t compute_nodes_ = 0;
};

[[nodiscard]] std::uint64_t program_signature(
    std::span<const NodePtr> order) noexcept;

void collect_reachable(const NodePtr& n, std::vector<NodePtr>& out,
                       std::unordered_set<const Node*>& seen);

}  // namespace lse::graph
