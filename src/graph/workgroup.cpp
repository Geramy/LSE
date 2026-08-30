#include "lse/graph/workgroup.hpp"

#include "lse/core/dtype.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"

#include <unordered_map>
#include <unordered_set>

namespace lse::graph {

namespace {

bool is_linear_like(const Node& n) noexcept {
  if (n.kind == OpKind::kLinear) return true;
  if (n.prim == nullptr) return false;
  const auto name = n.prim->name();
  return name == "linear" || name == "linear_indexed" || name == "linear.lds" ||
         name == "linear_indexed.lds" || name == "linear.wmma";
}

// Decode / short-prefill GEMV: the LDS form owns its indexing and has
// explicit column bounds. WMMA tiles of mixed N cannot share a grid.
bool decode_gemv_shape(const Node& n) noexcept {
  if (!is_linear_like(n) || n.inputs.size() < 2) return false;
  const Shape& w = n.inputs[1]->shape;
  if (w.rank() < 2) return false;
  const std::int64_t N = w.rank() == 3 ? w.dim(1) : w.dim(0);
  if (N < 16) return false;
  const std::int64_t elems = static_cast<std::int64_t>(n.element_count());
  if (elems <= 0 || elems % N != 0) return false;
  const std::int64_t M = elems / N;
  return M > 0 && M < 16;
}

bool is_kernel_prim(const Node& n) noexcept {
  return dynamic_cast<const KernelPrimitiveBase*>(n.prim) != nullptr;
}

// The activation buffer a wide linear stages, and how many f32 of it.
struct StagedPanelKey {
  const Node* act = nullptr;
  std::uint32_t count = 0;

  friend bool operator==(const StagedPanelKey&, const StagedPanelKey&) = default;
};

StagedPanelKey staged_panel(const Node& n) noexcept {
  if (!is_wide_linear(n) || n.inputs.empty() || !n.inputs[0]) return {};
  const Shape& x = n.inputs[0]->shape;
  const std::int64_t k = x.rank() == 0 ? 0 : x.dim(x.rank() - 1);
  if (k <= 0) return {};
  return {n.inputs[0].get(), static_cast<std::uint32_t>(k)};
}

// f32 row, 16-byte aligned: kir::Lds::align of count * sizeof(float). 0 when
// the row cannot be staged at all, which is what the emitter decides by asking
// whether it fits the device budget.
std::uint32_t panel_bytes(const StagedPanelKey& p,
                          const WorkgroupDevice& dev) noexcept {
  if (p.act == nullptr) return 0;
  const std::uint32_t bytes = ((p.count * 4u) + 15u) & ~15u;
  return dev.lds_bytes != 0 && bytes > dev.lds_bytes ? 0u : bytes;
}

void collect_kp_ancestors(const Node* n,
                          const std::unordered_set<const Node*>& members,
                          std::unordered_set<const Node*>& kps,
                          std::unordered_set<const Node*>& seen) {
  if (n == nullptr || !seen.insert(n).second) return;
  if (!members.count(n)) return;
  if (is_kernel_prim(*n)) {
    kps.insert(n);
    return;
  }
  for (const NodePtr& in : n->inputs) {
    collect_kp_ancestors(in.get(), members, kps, seen);
  }
}

}  // namespace

// Nothing here counts anything itself: the arithmetic and the min over limits
// live in lse::opt so that a backend, the partitioner and this file cannot
// disagree about them.
opt::Occupancy workgroup_residency(const WorkgroupDevice& d,
                                   std::uint32_t threads,
                                   std::uint32_t lds) noexcept {
  return opt::occupancy(d.capacity(), opt::KernelDemand::counted(threads, lds));
}

opt::DeviceCapacity WorkgroupDevice::capacity() const noexcept {
  using Fact = backend::DeviceFact<std::uint32_t>;
  opt::DeviceCapacity c;
  c.wave_slots_per_simd = Fact::declared(wave_slots_per_simd);
  c.simds_per_lds_pool = Fact::declared(simds_per_lds_pool);
  c.lds_bytes_per_pool = Fact::declared(lds_bytes_per_pool);
  if (lds_alloc_granule != 0) {
    c.lds_alloc_granule_bytes = Fact::declared(lds_alloc_granule);
  }
  if (lds_bytes != 0) {
    c.lds_bytes_addressable_per_workgroup = Fact::declared(lds_bytes);
  }
  c.max_flat_workgroup_size = Fact::declared(max_threads);
  c.wavefront_size = Fact::declared(wavefront);
  return c;
}

WorkgroupDevice WorkgroupDevice::from(const backend::DeviceInfo* info) noexcept {
  WorkgroupDevice d;
  if (info == nullptr) return d;
  if (info->compute_units != 0) d.compute_units = info->compute_units;
  if (info->max_threads_per_workgroup != 0) {
    d.max_threads = info->max_threads_per_workgroup;
  }
  if (info->lds_bytes_per_workgroup != 0) {
    d.lds_bytes = info->lds_bytes_per_workgroup;
  }
  if (info->wavefront_size != 0) d.wavefront = info->wavefront_size;
  const backend::ArchFacts& f = info->arch_facts;
  if (f.wave_slots_per_simd.known() && f.wave_slots_per_simd.value != 0) {
    d.wave_slots_per_simd = f.wave_slots_per_simd.value;
  }
  if (f.simds_per_lds_pool.known() && f.simds_per_lds_pool.value != 0) {
    d.simds_per_lds_pool = f.simds_per_lds_pool.value;
  }
  if (f.lds_bytes_per_pool.known() && f.lds_bytes_per_pool.value != 0) {
    d.lds_bytes_per_pool = f.lds_bytes_per_pool.value;
  }
  // Left 0 where nothing measured it, which the model reads as "charge the
  // request unrounded and say so", not as "no rounding happens".
  d.lds_alloc_granule = f.lds_alloc_granule_bytes.value_or(0u);
  return d;
}

bool workgroup_shareable(const Node& n) noexcept {
  return n.fclass != FusionClass::kCollective;
}

// Sequence rows of a contraction. Other ops (GDN state, KV tails) do
// not set the phase — they ride with the residual's job.
std::uint32_t rows_of(const Node& n) noexcept {
  if (!is_linear_like(n) || n.inputs.size() < 2) return 0;
  const Shape& w = n.inputs[1]->shape;
  if (w.rank() < 2) return 0;
  const std::int64_t N = w.rank() == 3 ? w.dim(1) : w.dim(0);
  if (N <= 0) return 0;
  const std::int64_t elems = static_cast<std::int64_t>(n.element_count());
  if (elems <= 0 || elems % N != 0) return 0;
  return static_cast<std::uint32_t>(elems / N);
}

WorkgroupPhase phase_of_rows(std::uint32_t rows) noexcept {
  if (rows == 0) return WorkgroupPhase::kUnknown;
  return rows < 16 ? WorkgroupPhase::kDecode : WorkgroupPhase::kPrefill;
}

Workgroup::Workgroup(WorkgroupDevice device) : device_(device) {}

bool Workgroup::related(const Node& n) const noexcept {
  if (members_.empty()) return true;
  // Weights and literals bind into whichever phase they feed.
  if (n.fclass == FusionClass::kLeaf) return true;
  bool only_external = !n.inputs.empty();
  for (const NodePtr& in : n.inputs) {
    if (member_set_.count(in.get())) return true;
    // Shared activation, including a host-created constant used as x.
    // A scalar literal is not an activation.
    if (member_input_set_.count(in.get()) &&
        !(in->fclass == FusionClass::kLeaf && in->element_count() <= 1)) {
      return true;
    }
    if (!in->materialized && in->fclass != FusionClass::kLeaf) {
      only_external = false;
    }
  }
  // exp(A_log), slices of a cached mixer state, etc. only read
  // materialized buffers. They belong to this phase. A second linear
  // on a different x is a different job.
  if (only_external && !is_linear_like(n)) return true;
  if (member_input_set_.count(&n)) return true;
  if (n.fclass == FusionClass::kElementwise ||
      n.fclass == FusionClass::kStructural ||
      n.fclass == FusionClass::kReduction || n.fclass == FusionClass::kLeaf) {
    for (const NodePtr& in : n.inputs) {
      if (member_set_.count(in.get())) return true;
    }
  }
  return false;
}

bool Workgroup::can_add(const Node& n) const noexcept {
  if (n.materialized) return false;
  if (!workgroup_shareable(n)) return false;
  if (member_set_.count(&n)) return false;
  if (!members_.empty() && !related(n)) return false;

  const WorkgroupPhase incoming = phase_of_rows(rows_of(n));
  const WorkgroupPhase have = phase();
  if (have != WorkgroupPhase::kUnknown && incoming != WorkgroupPhase::kUnknown &&
      incoming != have) {
    return false;
  }

  // A Workgroup is a whole phase and `cuts()` splits it into launches, so the
  // phase-wide sum is not a quantity the hardware ever sees. What every launch
  // must clear on its own is the row of the member it holds: that is the floor
  // this admits against, and lds_bytes() reports the honest worst launch.
  const std::uint32_t row = staged_row_bytes(n, device_);
  const std::uint32_t lds = row > lds_used_ ? row : lds_used_;
  if (device_.lds_bytes != 0 && lds > device_.lds_bytes) return false;

  const std::uint32_t threads = 256u < device_.max_threads ? 256u
                                                           : device_.max_threads;
  return workgroup_residency(device_, threads == 0 ? 1 : threads, lds).seated();
}

bool Workgroup::try_add(NodePtr n) {
  if (!n || !can_add(*n)) return false;
  const std::uint32_t row = staged_row_bytes(*n, device_);
  if (row > lds_used_) lds_used_ = row;
  const std::uint64_t elems = n->element_count();
  if (elems > job_) job_ = elems;
  const std::uint32_t r = rows_of(*n);
  if (r > rows_) rows_ = r;
  member_set_.insert(n.get());
  for (const NodePtr& in : n->inputs) member_input_set_.insert(in.get());
  members_.push_back(std::move(n));
  return true;
}

void Workgroup::reset_compute() noexcept {
  for (const NodePtr& n : members_) {
    if (!n) continue;
    if (n->fclass == FusionClass::kLeaf || n->kind == OpKind::kBuffer) continue;
    n->materialized = false;
  }
}

void Workgroup::clear() noexcept {
  members_.clear();
  member_set_.clear();
  member_input_set_.clear();
  lds_used_ = 0;
  job_ = 0;
  rows_ = 0;
  slots_.clear();
  slot_of_.clear();
  reused_ = 0;
}

std::uint32_t Workgroup::kernel_count() const noexcept {
  std::uint32_t n = 0;
  for (const NodePtr& m : members_) {
    if (is_kernel_prim(*m)) ++n;
  }
  return n == 0 ? (members_.empty() ? 0u : 1u) : n;
}

WorkgroupChain Workgroup::chain() const noexcept {
  const std::uint32_t kps = kernel_count();
  if (members_.empty() || kps <= 1) return WorkgroupChain::kNone;

  std::unordered_set<const Node*> member;
  for (const NodePtr& m : members_) member.insert(m.get());

  std::uint32_t joins = 0;
  std::uint32_t staged = 0;
  for (const NodePtr& m : members_) {
    std::unordered_set<const Node*> anc;
    std::unordered_set<const Node*> seen;
    for (const NodePtr& in : m->inputs) {
      collect_kp_ancestors(in.get(), member, anc, seen);
    }
    if (anc.size() >= 2) ++joins;
    if (is_kernel_prim(*m) && !anc.empty()) ++staged;
  }
  if (joins != 0) return WorkgroupChain::kFork;
  if (staged != 0) return WorkgroupChain::kStaged;

  bool only_elem = true;
  for (const NodePtr& m : members_) {
    if (is_kernel_prim(*m) || m->fclass == FusionClass::kLeaf) continue;
    if (m->fclass != FusionClass::kElementwise &&
        m->fclass != FusionClass::kStructural) {
      only_elem = false;
    }
  }
  if (only_elem && kps == 1) return WorkgroupChain::kElementwise;
  return WorkgroupChain::kIndependent;
}

bool Workgroup::all_linear_like() const noexcept {
  bool any = false;
  for (const NodePtr& m : members_) {
    if (!is_kernel_prim(*m)) continue;
    // Indexed (MoE) linears share x and a stacked W; fusing them with
    // the QKV siblings page-faulted on lemonseed. Keep them on their
    // own launch until that body is proven.
    if (m->kind != OpKind::kLinear &&
        (m->prim == nullptr || m->prim->name() != "linear")) {
      return false;
    }
    any = true;
  }
  return any;
}

std::uint32_t Workgroup::occupancy() const {
  const std::uint32_t threads = 256u < device_.max_threads ? 256u
                                                           : device_.max_threads;
  return workgroup_residency(device_, threads == 0 ? 1 : threads, lds_bytes())
      .workgroups_per_pool;
}

std::uint32_t Workgroup::lds_bytes() const {
  // The worst launch, not the phase: within one launch the distinct staged rows
  // SUM, across launches they do not compose at all. `lds_used_` is only the
  // largest single row, which is a floor on this and the right answer whenever
  // no cut holds two.
  std::uint32_t worst = lds_used_;
  std::vector<const Node*> nodes;
  for (const WorkgroupCut& cut : cuts()) {
    nodes.clear();
    nodes.reserve(cut.nodes.size());
    for (const NodePtr& n : cut.nodes) nodes.push_back(n.get());
    const std::uint32_t need = group_lds_bytes(nodes, device_);
    if (need > worst) worst = need;
  }
  return worst;
}

std::uint64_t Workgroup::job_elems() const noexcept { return job_; }

bool Workgroup::emittable() const noexcept {
  const std::uint32_t kps = kernel_count();
  if (kps <= 1) return true;
  // Sibling decode GEMVs are the one multi-KP body the emitter can write.
  // Staged / fork chains need a grid retire or an LDS intermediate we do
  // not emit yet.
  if (chain() != WorkgroupChain::kIndependent) return false;
  if (!all_linear_like()) return false;
  for (const NodePtr& m : members_) {
    if (is_kernel_prim(*m) && !decode_gemv_shape(*m)) return false;
  }
  return true;
}

WorkgroupPhase Workgroup::phase() const noexcept {
  return phase_of_rows(rows_);
}

std::uint32_t Workgroup::ideal_launches() const {
  if (members_.empty()) return 0;
  const std::uint32_t lds = lds_bytes();
  const std::uint32_t threads = 256u < device_.max_threads ? 256u
                                                           : device_.max_threads;
  if (!workgroup_residency(device_, threads == 0 ? 1 : threads, lds).seated()) {
    return kernel_count();
  }
  if (device_.lds_bytes != 0 && lds > device_.lds_bytes) return kernel_count();
  // One phase, one launch. Decode: one hardware workgroup owns the token
  // and walks every stage. Prefill: one grid, one workgroup per row-tile.
  // Forks and staged edges are stages of that launch, not extra phases.
  return 1;
}

std::uint32_t Workgroup::launches() const {
  const std::uint32_t ideal = ideal_launches();
  if (ideal <= 1 && emittable()) return ideal == 0 ? 0u : 1u;
  return kernel_count();
}

bool Workgroup::fused() const noexcept {
  return launches() == 1 && kernel_count() > 1;
}

ThreadPlan Workgroup::plan() const noexcept {
  ThreadPlan tp;
  const std::uint32_t threads =
      256u < device_.max_threads ? 256u : device_.max_threads;
  tp.workgroup_size[0] = threads == 0 ? 1u : threads;
  tp.lds_bytes = lds_used_;
  if (job_ == 0) {
    tp.workgroup_count[0] = 1;
    return tp;
  }

  // Occupancy decides whether the members can share a launch, not how
  // many idle workgroups we invent. Cover the job; the GPU schedules it.
  const std::uint32_t cover =
      static_cast<std::uint32_t>((job_ + tp.workgroup_size[0] - 1) /
                                 tp.workgroup_size[0]);
  tp.workgroup_count[0] = cover == 0 ? 1u : cover;
  return tp;
}

namespace {

std::int64_t linear_n(const Node& n) noexcept {
  if (n.inputs.size() < 2) return 0;
  const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n.prim);
  if (kp == nullptr && n.kind != OpKind::kLinear) return 0;
  if (kp != nullptr) {
    const auto name = kp->name();
    // Keep in step with the same list in scheduler.cpp's join_wide_linear.
    // quant_linear stages a row and is priced into run_lds_bytes like the dense
    // kernels, so omitting it here is what kept every wide linear in a
    // quantized checkpoint in its own launch. quant_linear_indexed stages the
    // same row by the same rule — the expert only moves where the weight is
    // read from — so leaving it out would understate a routed layer's scratch
    // by a whole activation row.
    if (name != "linear" && name != "linear.lds" &&
        name != "linear_indexed" && name != "linear_indexed.lds" &&
        name != "quant_linear" && name != "quant_linear_indexed") {
      return 0;
    }
  }
  const Shape& w = n.inputs[1]->shape;
  if (w.rank() == 3) return w.dim(1);
  if (w.rank() >= 2) return w.dim(0);
  return 0;
}

const Node* skip_reshape(const Node* n) noexcept {
  while (n && n->kind == OpKind::kReshape && n->inputs.size() == 1) {
    n = n->inputs[0].get();
  }
  return n;
}

bool sibling_grid(const WorkgroupCut& cut, const Node& n) noexcept {
  if (!is_wide_linear(n) || n.inputs.empty() || !cut.grid_linears) return false;
  const Node* x = n.inputs[0].get();
  for (const NodePtr& m : cut.nodes) {
    if (m->inputs.empty() || m->inputs[0].get() != x) return false;
    for (const NodePtr& in : n.inputs) {
      if (in.get() == m.get()) return false;
    }
  }
  return true;
}

bool reads_large_in_cut(const WorkgroupCut& cut, const Node& n) noexcept {
  for (const NodePtr& in : n.inputs) {
    const Node* p = skip_reshape(in.get());
    if (p == nullptr || p->element_count() < 512) continue;
    for (const NodePtr& m : cut.nodes) {
      if (m.get() == p && m->kind != OpKind::kReshape) return true;
    }
  }
  return false;
}

}  // namespace

bool is_wide_linear(const Node& n) noexcept { return linear_n(n) >= 128; }

std::uint32_t staged_row_bytes(const Node& n,
                               const WorkgroupDevice& dev) noexcept {
  return panel_bytes(staged_panel(n), dev);
}

std::uint32_t group_lds_bytes(std::span<const Node* const> nodes,
                              const WorkgroupDevice& dev) noexcept {
  std::uint32_t bytes = 0;
  std::vector<StagedPanelKey> seen;
  for (const Node* n : nodes) {
    if (n == nullptr) continue;
    const StagedPanelKey p = staged_panel(*n);
    if (p.act == nullptr) continue;
    bool dup = false;
    for (const StagedPanelKey& s : seen) {
      if (s == p) dup = true;
    }
    if (dup) continue;
    seen.push_back(p);
    bytes += panel_bytes(p, dev);
  }
  return bytes;
}

std::vector<WorkgroupCut> Workgroup::cuts() const {
  std::vector<WorkgroupCut> out;
  WorkgroupCut cur;
  auto seal = [&](WorkgroupSync next) {
    if (cur.nodes.empty()) return;
    out.push_back(std::move(cur));
    cur = WorkgroupCut{};
    cur.sync_before = next;
  };
  for (const NodePtr& n : members_) {
    if (!n || n->fclass == FusionClass::kLeaf) continue;
    if (is_wide_linear(*n)) {
      if (sibling_grid(cur, *n)) {
        cur.nodes.push_back(n);
        continue;
      }
      seal(WorkgroupSync::kStream);
      cur.grid_linears = true;
      cur.nodes.push_back(n);
      continue;
    }
    if (!cur.nodes.empty() && (cur.grid_linears || reads_large_in_cut(cur, *n))) {
      seal(WorkgroupSync::kStream);
    }
    cur.nodes.push_back(n);
  }
  seal(WorkgroupSync::kNone);
  return out;
}

void Workgroup::plan_slots(std::span<const NodePtr> roots) {
  slots_.clear();
  slot_of_.clear();
  reused_ = 0;
  const auto groups = cuts();
  if (groups.empty()) return;

  std::unordered_set<const Node*> member;
  for (const NodePtr& n : members_) member.insert(n.get());

  std::unordered_map<const Node*, std::uint32_t> last_cut;
  for (std::uint32_t ci = 0; ci < groups.size(); ++ci) {
    for (const NodePtr& n : groups[ci].nodes) {
      last_cut[n.get()] = ci;
      for (const NodePtr& in : n->inputs) {
        const Node* p = skip_reshape(in.get());
        if (p && member.count(p)) last_cut[p] = ci;
      }
    }
  }
  std::unordered_set<const Node*> root_set;
  for (const NodePtr& r : roots) {
    if (r) root_set.insert(r.get());
  }
  // Keep-alive: a value with readers outside this workgroup must hold its
  // slot past the last internal use. consumer_count counts DIRECT consumers,
  // so the internal count it is compared against has to be direct too — the
  // old count looked through reshapes, so a value read internally through a
  // view tallied more readers than its consumer_count and an EXTERNAL reader
  // of the same view was invisible. A member split pushes exactly such
  // readers into a later workgroup, the slot was recycled under them, and the
  // clobber was deterministic. An escaping reshape pins the node whose bytes
  // it aliases, because that is what the outside reader reads.
  std::unordered_map<const Node*, std::uint32_t> direct_readers;
  for (const NodePtr& m : members_) {
    for (const NodePtr& in : m->inputs) {
      if (in && member.count(in.get())) ++direct_readers[in.get()];
    }
  }
  for (const NodePtr& n : members_) {
    const auto it = direct_readers.find(n.get());
    const std::uint32_t direct = it == direct_readers.end() ? 0 : it->second;
    if (root_set.count(n.get()) == 0 && n->consumer_count <= direct) continue;
    if (n->kind == OpKind::kReshape && !n->inputs.empty()) {
      const Node* owner = skip_reshape(n->inputs[0].get());
      if (owner != nullptr && member.count(owner) != 0 &&
          owner->fclass != FusionClass::kLeaf) {
        last_cut[owner] = static_cast<std::uint32_t>(groups.size());
      }
    }
    if (n->fclass != FusionClass::kLeaf) {
      last_cut[n.get()] = static_cast<std::uint32_t>(groups.size());
    }
  }

  std::unordered_map<std::size_t, std::vector<std::uint32_t>> free;
  std::vector<std::uint32_t> slot_last(0);
  // Recycling a slot across a cut boundary is safe ONLY because a launch
  // boundary is a grid-wide barrier. Fuse two cuts into one kernel and the
  // shared allocation becomes an intra-launch write-after-read between
  // workgroups, which __syncthreads cannot order -- that is the hazard
  // `a_stage_that_reads_a_broadcast_operand_is_not_a_lane_stage` pins, and it
  // is what forces lane_stage to judge a stage by its READ as well as its
  // store, which in turn is what keeps 59% of launches at one node.
  //
  // These slots are ordinary device buffers, not LDS, so declining to reuse
  // them costs memory rather than occupancy: activations are ~20 KB each.
  // LSE_NO_SLOT_REUSE=1 buys the fusion freedom with that memory.
  static const bool no_reuse = std::getenv("LSE_NO_SLOT_REUSE") != nullptr;
  auto take = [&](std::size_t bytes) -> std::uint32_t {
    auto& bin = free[bytes];
    if (!no_reuse && !bin.empty()) {
      const std::uint32_t id = bin.back();
      bin.pop_back();
      ++reused_;
      return id;
    }
    const auto id = static_cast<std::uint32_t>(slots_.size());
    slots_.push_back(Slot{bytes, {}});
    slot_last.push_back(0);
    return id;
  };

  for (std::uint32_t ci = 0; ci < groups.size(); ++ci) {
    for (const NodePtr& n : groups[ci].nodes) {
      if (n->kind == OpKind::kReshape || n->fclass == FusionClass::kLeaf) {
        continue;
      }
      if (n->prim != nullptr && n->prim->inplace_input() >= 0) continue;
      if (n->buffer.valid()) continue;
      const std::size_t bytes =
          dtype_storage_bytes(n->dtype, n->element_count());
      if (bytes == 0) continue;
      const std::uint32_t id = take(bytes);
      slot_of_[n.get()] = id;
      auto it = last_cut.find(n.get());
      slot_last[id] = it == last_cut.end() ? ci : it->second;
    }
    for (std::uint32_t id = 0; id < slot_last.size(); ++id) {
      if (slot_last[id] != ci) continue;
      free[slots_[id].bytes].push_back(id);
      slot_last[id] = ~0u;
    }
  }
}

Status Workgroup::bind_slots(backend::IBackend& backend,
                             backend::Stream stream) {
  if (slot_of_.empty()) return OkStatus();
  for (Slot& s : slots_) {
    if (s.buffer.valid()) continue;
    // Through the caller's stream: on a device spanning several GPUs the
    // stream is what places the bytes, and the default stream put every
    // member's phase activations in the primary's VRAM.
    auto buf = backend.allocate(s.bytes, backend::MemoryClass::kDevice, stream);
    if (!buf.ok()) return buf.status();
    s.buffer = buf.release();
  }
  for (const auto& [node, id] : slot_of_) {
    if (node == nullptr || id >= slots_.size()) continue;
    Node* n = const_cast<Node*>(node);
    if (n->buffer.valid()) continue;
    n->buffer = slots_[id].buffer;
    n->buffer.size_bytes = slots_[id].bytes;
  }
  for (const NodePtr& n : members_) {
    if (!n || n->prim == nullptr || n->buffer.valid()) continue;
    const int a = n->prim->inplace_input();
    if (a < 0 || static_cast<std::size_t>(a) >= n->inputs.size()) continue;
    const NodePtr& src = n->inputs[static_cast<std::size_t>(a)];
    if (!src || !src->buffer.valid()) continue;
    n->buffer = src->buffer;
  }
  for (const NodePtr& n : members_) {
    if (!n || n->kind != OpKind::kReshape || n->inputs.size() != 1) continue;
    if (n->buffer.valid()) continue;
    const Node* src = skip_reshape(n->inputs[0].get());
    if (src == nullptr || !src->buffer.valid()) continue;
    if (src->dtype != n->dtype || src->element_count() != n->element_count()) {
      continue;
    }
    n->buffer = src->buffer;
    n->buffer.size_bytes =
        dtype_storage_bytes(n->dtype, n->element_count());
    n->materialized = src->materialized;
    n->device_dirty = src->device_dirty;
    n->host_dirty = src->host_dirty;
  }
  return OkStatus();
}

}  // namespace lse::graph
