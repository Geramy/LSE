// A captured DAG plus the Workgroups that run it.
//
// The first eval of a shape builds nodes and plans launches. After that the
// Program holds those NodePtrs so the next eval of the same computation does
// not call make(). Workgroup::reset_compute() puts every non-leaf back to
// "not yet run" without freeing buffers. Carried state is folded by swapping
// the produced buffer onto the input node the next step reads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "lse/graph/codegen.hpp"
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

  // A retained group has the same nodes, shapes and source across replays.
  // Keep its identity, generated source and binding order with the Program.
  // A different backend, emitter, target or live WMMA override misses.
  [[nodiscard]] const EmittedKernel* cached_emission(
      std::size_t index, const FusionGroup& group,
      const backend::IBackend* backend, const IKernelEmitter* emitter,
      std::string_view arch, std::uint64_t* key) const noexcept;
  [[nodiscard]] const EmittedKernel* cache_emission(
      std::size_t index, const FusionGroup& group,
      const backend::IBackend* backend, const IKernelEmitter* emitter,
      std::uint64_t key, std::string_view arch, EmittedKernel kernel);

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
  // Keeps the carried inputs where they are: the pass about to run replaces the
  // one that just ran rather than following it, so the state it must start from
  // is the one already on the input node. reset_compute() un-materializes that
  // node when it is itself a produced value, which is the only reason this has
  // to be called at all.
  void hold_carries() noexcept;

 private:
  std::vector<NodePtr> roots_;
  std::vector<NodePtr> nodes_;
  std::vector<Workgroup> phases_;
  std::vector<FusionGroup> groups_;
  struct EmissionCache {
    const FusionGroup* group = nullptr;
    const backend::IBackend* backend = nullptr;
    const IKernelEmitter* emitter = nullptr;
    std::uint64_t key = 0;
    std::string arch;
    std::optional<EmittedKernel> kernel;
  };
  std::vector<EmissionCache> emissions_;
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
