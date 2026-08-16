#include "lse/graph/workgroup.hpp"

#include "lse/backends/hrx/device_info.hpp"
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

std::uint32_t estimate_lds(const Node&) noexcept { return 0; }

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

std::uint32_t occupancy_of(const WorkgroupDevice& d, std::uint32_t threads,
                           std::uint32_t lds) noexcept {
  if (threads == 0 || d.wavefront == 0) return 0;
  if (threads > d.max_threads) return 0;
  if (d.lds_bytes != 0 && lds > d.lds_bytes) return 0;
  const std::uint32_t waves = (threads + d.wavefront - 1) / d.wavefront;
  if (waves == 0 || d.max_waves_per_cu < waves) return 0;
  const std::uint32_t wave_limit = d.max_waves_per_cu / waves;
  if (lds == 0 || d.lds_bytes == 0) return wave_limit;
  const std::uint32_t lds_limit = d.lds_bytes / lds;
  return wave_limit < lds_limit ? wave_limit : lds_limit;
}

}  // namespace

WorkgroupDevice WorkgroupDevice::from(const backend::DeviceInfo* info) noexcept {
  WorkgroupDevice d;
  if (info == nullptr) return d;
  if (info->compute_units != 0) d.compute_units = info->compute_units;
  if (info->max_threads_per_workgroup != 0) {
    d.max_threads = info->max_threads_per_workgroup;
  }
  const auto* amd = backend::device_extension<backend::AmdDeviceInfo>(*info);
  if (amd == nullptr) return d;
  if (amd->lds_bytes_per_workgroup != 0) {
    d.lds_bytes = amd->lds_bytes_per_workgroup;
  }
  if (amd->wavefront_size != 0) d.wavefront = amd->wavefront_size;
  if (amd->max_waves_per_cu != 0) d.max_waves_per_cu = amd->max_waves_per_cu;
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

  const std::uint32_t extra = estimate_lds(n);
  const std::uint32_t lds = extra > lds_used_ ? extra : lds_used_;
  if (device_.lds_bytes != 0 && lds > device_.lds_bytes) return false;

  const std::uint32_t threads = 256u < device_.max_threads ? 256u
                                                           : device_.max_threads;
  return occupancy_of(device_, threads == 0 ? 1 : threads, lds) != 0;
}

bool Workgroup::try_add(NodePtr n) {
  if (!n || !can_add(*n)) return false;
  const std::uint32_t extra = estimate_lds(*n);
  if (extra > lds_used_) lds_used_ = extra;
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

std::uint32_t Workgroup::occupancy() const noexcept {
  const std::uint32_t threads = 256u < device_.max_threads ? 256u
                                                           : device_.max_threads;
  return occupancy_of(device_, threads == 0 ? 1 : threads, lds_used_);
}

std::uint32_t Workgroup::lds_bytes() const noexcept { return lds_used_; }

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

std::uint32_t Workgroup::ideal_launches() const noexcept {
  if (members_.empty()) return 0;
  if (occupancy() == 0) return kernel_count();
  if (device_.lds_bytes != 0 && lds_used_ > device_.lds_bytes) {
    return kernel_count();
  }
  // One phase, one launch. Decode: one hardware workgroup owns the token
  // and walks every stage. Prefill: one grid, one workgroup per row-tile.
  // Forks and staged edges are stages of that launch, not extra phases.
  return 1;
}

std::uint32_t Workgroup::launches() const noexcept {
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
    if (name != "linear" && name != "linear.lds" &&
        name != "linear_indexed" && name != "linear_indexed.lds") {
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
  for (const NodePtr& n : members_) {
    if (n->fclass == FusionClass::kLeaf) continue;
    std::uint32_t readers = 0;
    for (const NodePtr& m : members_) {
      for (const NodePtr& in : m->inputs) {
        if (skip_reshape(in.get()) == n.get()) ++readers;
      }
    }
    if (root_set.count(n.get()) || n->consumer_count > readers) {
      last_cut[n.get()] = static_cast<std::uint32_t>(groups.size());
    }
  }

  std::unordered_map<std::size_t, std::vector<std::uint32_t>> free;
  std::vector<std::uint32_t> slot_last(0);
  auto take = [&](std::size_t bytes) -> std::uint32_t {
    auto& bin = free[bytes];
    if (!bin.empty()) {
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

Status Workgroup::bind_slots(backend::IBackend& backend) {
  if (slot_of_.empty()) return OkStatus();
  for (Slot& s : slots_) {
    if (s.buffer.valid()) continue;
    auto buf = backend.allocate(s.bytes, backend::MemoryClass::kDevice);
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
