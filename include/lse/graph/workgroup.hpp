// One phase's kernels, smashed into as few launches as the device allows.
//
// A Workgroup owns the members, the edges between them, the launch cuts,
// the barrier that each cut needs, and the leftover activation slots —
// a dead intermediate of size S is the next stage's buffer of size S.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/kernel_primitive.hpp"

namespace lse::graph {

class Node;
using NodePtr = std::shared_ptr<Node>;

struct WorkgroupDevice {
  std::uint32_t lds_bytes = 65536;
  std::uint32_t compute_units = 40;
  std::uint32_t wavefront = 32;
  std::uint32_t max_waves_per_cu = 32;
  std::uint32_t max_threads = 256;
  // See backend::DeviceInfo::cus_per_lds_pool. `lds_bytes / request` counts
  // workgroups per pool, so the per-CU answer is that over this.
  std::uint32_t cus_per_lds_pool = 2;

  static WorkgroupDevice from(const backend::DeviceInfo* info) noexcept;

  // The largest workgroup-scratch request that still seats one workgroup on
  // every CU — the budget a fusion decision is held to.
  [[nodiscard]] std::uint32_t resident_lds_bytes() const noexcept {
    return lds_bytes / (cus_per_lds_pool == 0 ? 1u : cus_per_lds_pool);
  }
};

// Collectives are the only hard isolate. Everything else is a stage of
// the phase Workgroup (attention, embedding, top-k included).
[[nodiscard]] bool workgroup_shareable(const Node& n) noexcept;

// A GEMV wide enough to want a grid of its own rather than a slot in a staged
// body. Matches the cut the scheduler makes when it groups sibling linears.
[[nodiscard]] bool is_wide_linear(const Node& n) noexcept;

// Workgroup scratch this node stages for itself, in bytes: its activation row,
// f32, 16-byte aligned. 0 when it stages nothing — including when the row does
// not fit the device at all, which is the emitter's own condition for reading
// the activation from global instead. The 27B's down projection is that case: a
// 17408-long row wants 69632 bytes, so it never stages and never pays.
//
// A wide linear stages this row whether or not it is fused — the solo grid GEMV
// reserves it against the whole device budget — so this is what one of these
// launches already costs, not a price fusion introduces.
[[nodiscard]] std::uint32_t staged_row_bytes(
    const Node& n, const WorkgroupDevice& dev) noexcept;

// What `nodes` cost in workgroup scratch sharing ONE launch: every DISTINCT
// staged row once. Distinct means a different activation buffer or a different
// length; the emitter hoists one panel per pair and the stages read it by name.
//
// Summed, never maximized: separate `__shared__` declarations get separate LDS
// offsets even in disjoint block scopes.
[[nodiscard]] std::uint32_t group_lds_bytes(
    std::span<const Node* const> nodes, const WorkgroupDevice& dev) noexcept;

enum class WorkgroupPhase : std::uint8_t {
  kUnknown,
  kDecode,   // few rows (M < 16): one hardware workgroup can own the token
  kPrefill,  // many rows: one grid, one workgroup per tile of rows
};

// How members depend on each other. Stages of a phase, not extra phases.
enum class WorkgroupChain : std::uint8_t {
  kNone,          // empty or a single op
  kIndependent,   // siblings: shared activation, no KP → KP edge
  kElementwise,   // each consumer only needs element i of one producer
  kStaged,        // a later KP reads a whole producer (row / buffer)
  kFork,          // a node reads two or more kernel members (diamond)
};

// What must retire before the next cut issues. Host synchronize is never
// this object's job — the stream already orders dispatches.
enum class WorkgroupSync : std::uint8_t {
  kNone,      // first cut, or independent siblings in one launch
  kBarrier,   // same launch, workgroup barrier between stages
  kStream,    // next launch; device queue visibility is enough
};

struct WorkgroupCut {
  std::vector<NodePtr> nodes;
  WorkgroupSync sync_before = WorkgroupSync::kNone;
  // Sibling decode GEMVs that share a grid, not a staged body.
  bool grid_linears = false;
};

class Workgroup {
 public:
  explicit Workgroup(WorkgroupDevice device = {});

  [[nodiscard]] bool can_add(const Node& n) const noexcept;
  bool try_add(NodePtr n);

  [[nodiscard]] WorkgroupChain chain() const noexcept;
  [[nodiscard]] WorkgroupPhase phase() const noexcept;

  // Dispatches we will actually issue (emit cuts). 1 only when a fused
  // body exists. A whole decode/prefill phase still reports ideal 1.
  [[nodiscard]] std::uint32_t launches() const;

  // A connected device phase is one launch: decode owns the token in one
  // hardware workgroup; prefill is one grid. Host cuts and a dead
  // occupancy are the only reasons this is not 1.
  [[nodiscard]] std::uint32_t ideal_launches() const;

  [[nodiscard]] bool fused() const noexcept;
  [[nodiscard]] bool emittable() const noexcept;
  [[nodiscard]] ThreadPlan plan() const noexcept;
  // The worst launch this phase would issue, not the phase: scratch sums inside
  // a cut and does not compose across cuts. Walks cuts(), so not noexcept.
  [[nodiscard]] std::uint32_t lds_bytes() const;
  [[nodiscard]] std::uint32_t occupancy() const;
  [[nodiscard]] std::uint64_t job_elems() const noexcept;

  [[nodiscard]] const std::vector<NodePtr>& members() const noexcept {
    return members_;
  }
  [[nodiscard]] const WorkgroupDevice& device() const noexcept { return device_; }

  // Smash: one cut is one launch. Staged edges become kStream between
  // cuts, or kBarrier when the emitter keeps them in one body.
  [[nodiscard]] std::vector<WorkgroupCut> cuts() const;

  // Leftovers: a slot of N bytes is reused by the next cut that needs
  // N bytes after the last reader of the previous tenant.
  void plan_slots(std::span<const NodePtr> roots);
  Status bind_slots(backend::IBackend& backend);
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  [[nodiscard]] std::uint32_t reused_slots() const noexcept { return reused_; }

  // Un-run every non-leaf. Buffers stay; the next dispatch overwrites them.
  // Leaves (weights, token ids, folded state) stay materialized.
  void reset_compute() noexcept;
  void clear() noexcept;

 private:
  [[nodiscard]] std::uint32_t kernel_count() const noexcept;
  [[nodiscard]] bool related(const Node& n) const noexcept;
  [[nodiscard]] bool all_linear_like() const noexcept;

  WorkgroupDevice device_;
  std::vector<NodePtr> members_;
  std::unordered_set<const Node*> member_set_;
  std::unordered_set<const Node*> member_input_set_;
  std::uint32_t lds_used_ = 0;
  std::uint64_t job_ = 0;
  std::uint32_t rows_ = 0;

  struct Slot {
    std::size_t bytes = 0;
    backend::DeviceBuffer buffer;
  };
  std::vector<Slot> slots_;
  std::unordered_map<const Node*, std::uint32_t> slot_of_;
  std::uint32_t reused_ = 0;
};

}  // namespace lse::graph
