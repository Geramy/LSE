#include "lse/graph/graph.hpp"

#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/workgroup.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lse::graph {

namespace {

// A kernel primitive indexes its operands itself — a matmul reads a whole row,
// a norm reads a whole row — so its inputs must stay bound buffers rather than
// values computed in registers or inlined as literals.
bool reads_inputs_as_buffers(const Node& n) noexcept {
  return dynamic_cast<const KernelPrimitiveBase*>(n.prim) != nullptr ||
         n.fclass == FusionClass::kBarrier;
}

bool is_kernel_prim(const Node& n) noexcept {
  return dynamic_cast<const KernelPrimitiveBase*>(n.prim) != nullptr;
}

// Attention, MoE, embedding, top-k, 2-D matmul, collectives. linear / rms /
// conv / gdn / slice are shareable and go through Workgroup instead.
bool hard_barrier(const Node& n) noexcept {
  return (n.fclass == FusionClass::kBarrier ||
          n.fclass == FusionClass::kCollective) &&
         !workgroup_shareable(n);
}

bool only_kernels_and_leaves(const FusionGroup& g) noexcept {
  for (const NodePtr& m : g.nodes) {
    if (is_kernel_prim(*m) || m->fclass == FusionClass::kLeaf) continue;
    return false;
  }
  return true;
}

std::uint32_t kernel_prims_in(const FusionGroup& g) noexcept {
  std::uint32_t n = 0;
  for (const NodePtr& m : g.nodes) {
    if (is_kernel_prim(*m)) ++n;
  }
  return n;
}

}  // namespace

namespace {

// Recomputing a producer in each consumer beats materializing it below this.
constexpr std::uint64_t kRecomputeThreshold = 1024;

void topo_visit(const NodePtr& n, std::unordered_set<const Node*>& seen,
                std::vector<NodePtr>& order) {
  if (!n || n->materialized || seen.count(n.get())) return;
  seen.insert(n.get());
  for (const NodePtr& in : n->inputs) topo_visit(in, seen, order);
  order.push_back(n);
}

// A run of siblings shares one launch, so every one of their outputs is live
// at once and plan_slots cannot recycle between them.
//
// This used to be four because the LDS budget made it four: every stage staged
// its own copy of the shared activation row and the compiler summed them, so
// `run * 4 * K` had to fit a quarter of the workgroup budget. The emitter now
// stages the shared row once for the whole run and prices the run's scratch by
// its distinct rows, so nothing about workgroup scratch scales with the run
// length any more — and a run whose rows genuinely do not fit is refused there
// rather than mis-emitted here.
//
// What binds instead is the pool of candidates. Six is the ceiling lemonseed
// can present — the MoE block has two shared-expert and four routed wide
// linears off one normed activation, and nothing else in the model offers a
// seventh — so this is a pool limit, not a resource limit.
//
// Measured, `-n 32` on one gfx1151, same tree, only this constant changed: caps
// 6, 8 and 12 emit a byte-identical program (sha256 over all 62 dumped kernels
// agrees), 12661 launches and `cse=92 lds_fold=0 dce=0` in all three. Raising
// it past six is therefore unmeasurable here and is not done on the strength of
// a model nobody has run. What the earlier cap did buy, for the record: run 4
// gives 99.5 tok/s at 395 groups per decode token, run 6 gives 100.5 at 375.
constexpr std::size_t kMaxSiblingRun = 6;

// Rows the linear covers. One is the decode shape, where every wide-linear
// kind is a wave-per-column GEMV.
std::int64_t linear_rows(const Node& n) noexcept {
  if (n.inputs.size() < 2) return 0;
  const Shape& w = n.inputs[1]->shape;
  const std::int64_t cols =
      w.rank() == 3 ? w.dim(1) : (w.rank() >= 2 ? w.dim(0) : 0);
  const auto elems = static_cast<std::int64_t>(n.shape.elem_count());
  return cols > 0 && elems % cols == 0 ? elems / cols : 0;
}

// Length of the activation row a wide linear reads. Every sibling in a run
// shares inputs[0], so this is one number for the whole run.
std::int64_t activation_k(const Node& n) noexcept {
  if (n.inputs.empty()) return 0;
  const Shape& x = n.inputs[0]->shape;
  return x.rank() == 0 ? 0 : x.dim(x.rank() - 1);
}

// Make independent wide linears that share an activation adjacent, so the
// scheduler's join_wide_linear can put them in one launch and the emitter's
// sibling path can emit one body per output.
//
// join_wide_linear only ever looks at the group it just emitted, and the DFS
// post-order separates the pairs that would join: `x, gate, silu, up, mul`
// puts silu between the MoE gate and up. Pulling the later one forward is the
// whole pass.
//
// Legality is one condition: every input of the node being moved must already
// be behind the cut. Post-order puts a node's inputs before it, so if all of
// its *direct* inputs are behind the cut then its whole transitive cone is,
// and the order stays topological.
void group_sibling_linears(std::vector<NodePtr>& order,
                           const WorkgroupDevice& dev) {
  std::unordered_set<const Node*> in_order;
  std::unordered_set<const Node*> emitted;
  in_order.reserve(order.size() * 2);
  emitted.reserve(order.size() * 2);
  for (const NodePtr& n : order) in_order.insert(n.get());

  std::vector<std::size_t> take;
  std::vector<const Node*> run;
  std::vector<const Node*> probe;
  for (std::size_t p = 0; p < order.size(); ++p) {
    emitted.insert(order[p].get());
    if (!is_wide_linear(*order[p]) || order[p]->inputs.empty()) continue;

    const Node* x = order[p]->inputs[0].get();
    const std::int64_t k = activation_k(*order[p]);
    if (k <= 0) continue;

    const std::size_t cap = kMaxSiblingRun;

    run.assign(1, order[p].get());
    take.clear();

    // What the run may spend on workgroup scratch.
    //
    // THE RULE: a merged run must seat at least as many workgroups on one
    // scratch pool as the member would seat launching alone. Not "does the sum
    // fit the per-workgroup cap" — fitting is not the question. Two panels
    // summing to 64000 bytes fit a 65536-byte cap and seat 2 workgroups per
    // pool where 32000 bytes seats 4, and that halving is what a fitting test
    // cannot see.
    //
    // The common case is admitted by the same rule rather than by an exception
    // to it: the emitter hoists ONE panel per distinct (activation, length), so
    // a run whose members share an activation asks for exactly what any one of
    // them asks for alone. Equal residency, and `prefer` admits equality.
    const std::uint32_t threads = dev.launch_threads();
    const opt::Occupancy solo =
        workgroup_residency(dev, threads, staged_row_bytes(*order[p], dev));
    auto run_fits = [&](const Node& candidate) {
      probe.assign(run.begin(), run.end());
      for (std::size_t t : take) probe.push_back(order[t].get());
      probe.push_back(&candidate);
      const std::uint32_t need = group_lds_bytes(probe, dev);
      return opt::prefer(workgroup_residency(dev, threads, need), solo);
    };

    for (std::size_t j = p + 1;
         j < order.size() && run.size() + take.size() < cap; ++j) {
      const Node& c = *order[j];
      if (!is_wide_linear(c) || c.inputs.empty()) continue;
      if (c.inputs[0].get() != x || activation_k(c) != k) continue;
      if (!run_fits(c)) continue;
      // Mixing kinds in one run is legal only where every kind has a
      // self-indexing form. `linear` gains a matrix-core form at M >= 16;
      // `linear_indexed` has only the wave-per-column GEMV, which needs a
      // decode-shaped M. Put them in one group at M >= 16 and the emitter
      // refuses the sibling path — the stages no longer cover the group — and
      // drops the whole group to the per-element scaffold, which took the two
      // shared-expert linears off the matrix core and moved the logits by
      // 1.6e-2 (measured, T=16 and T=32, fused vs unfused). At one row every
      // wide-linear kind is a GEMV and the mix costs nothing.
      if (c.kind != order[p]->kind && linear_rows(*order[p]) != 1) continue;
      bool ok = true;
      for (const NodePtr& in : c.inputs) {
        if (in_order.count(in.get()) && !emitted.count(in.get())) ok = false;
        // A sibling that reads another member of the run is not independent of
        // it: they cannot share a launch, so moving it buys nothing.
        for (const Node* m : run) {
          if (in.get() == m) ok = false;
        }
        for (std::size_t t : take) {
          if (in.get() == order[t].get()) ok = false;
        }
        if (!ok) break;
      }
      if (ok) take.push_back(j);
    }
    // Rotating in increasing source order leaves the later indices where they
    // are: everything between the destination and the source shifts right by
    // one, and nothing past the source moves.
    for (std::size_t t : take) {
      ++p;
      std::rotate(order.begin() + static_cast<std::ptrdiff_t>(p),
                  order.begin() + static_cast<std::ptrdiff_t>(t),
                  order.begin() + static_cast<std::ptrdiff_t>(t) + 1);
      emitted.insert(order[p].get());
    }
  }
}

}  // namespace

bool Partitioner::can_fuse(const Node& producer, const Node& consumer) noexcept {
  if (producer.materialized) return false;

  const FusionClass pc = producer.fclass;
  const FusionClass cc = consumer.fclass;

  const bool p_barrier = pc == FusionClass::kBarrier || pc == FusionClass::kCollective;
  const bool c_barrier = cc == FusionClass::kBarrier || cc == FusionClass::kCollective;

  if (p_barrier) {
    if (cc != FusionClass::kElementwise) return false;
    // Only when the barrier's kernel actually has an epilogue slot; otherwise
    // the fused op would be emitted into a kernel that never runs it.
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(producer.prim);
    if (kp == nullptr || !kp->supports_epilogue()) return false;
    // The group iterates over its largest output, and the barrier's device
    // function is called at every one of those indices. If the epilogue is
    // wider than the barrier's own output — a broadcast, as in
    // value * sigmoid(router_logits) — the kernel indexes its operands far past
    // their end and faults the device. Broadcasting the barrier's value into
    // the epilogue is not something the emitter can express, so decline.
    return producer.shape.elem_count() == consumer.shape.elem_count();
  }
  if (c_barrier) return false;
  // Same reason as the barrier case: a kernel primitive cannot read a value
  // that only exists in a register, so nothing fuses into one.
  if (reads_inputs_as_buffers(consumer)) return false;

  // A leaf is folded into whatever reads it, so it never forces a split and
  // its fan-out count is irrelevant.
  if (pc == FusionClass::kLeaf) return true;

  // Fanning out to several consumers means the value has to be real somewhere,
  // unless recomputing it is cheaper than the round trip.
  if (producer.consumer_count > 1 &&
      producer.recompute_cost() > kRecomputeThreshold) {
    return false;
  }

  if (pc == FusionClass::kStructural || cc == FusionClass::kStructural) return true;

  if (pc == FusionClass::kElementwise) {
    if (cc == FusionClass::kElementwise) {
      return producer.shape == consumer.shape ||
             producer.shape.is_broadcastable_to(consumer.shape);
    }
    // Elementwise folds into a reduction as its prologue.
    return cc == FusionClass::kReduction;
  }

  if (pc == FusionClass::kReduction && cc == FusionClass::kElementwise) {
    // Only when the reduced value broadcasts back over the consumer, which is
    // the RMSNorm/softmax shape. Otherwise the iteration spaces disagree.
    return producer.shape.is_broadcastable_to(consumer.shape);
  }

  return false;
}

std::vector<FusionGroup> Partitioner::partition(std::span<const NodePtr> roots) {
  return partition(roots, nullptr);
}

std::vector<NodePtr> Partitioner::unmaterialized(std::span<const NodePtr> roots) {
  std::vector<NodePtr> order;
  std::unordered_set<const Node*> seen;
  for (const NodePtr& r : roots) topo_visit(r, seen, order);
  return order;
}

std::vector<Workgroup> Partitioner::phases(std::span<const NodePtr> roots) {
  return phases(roots, nullptr);
}

std::vector<Workgroup> Partitioner::phases(
    std::span<const NodePtr> roots, const backend::DeviceInfo* device) {
  const WorkgroupDevice dev = WorkgroupDevice::from(device);

  std::vector<NodePtr> order;
  std::unordered_set<const Node*> seen;
  for (const NodePtr& r : roots) topo_visit(r, seen, order);
  group_sibling_linears(order, dev);

  std::vector<Workgroup> out;
  for (const NodePtr& n : order) {
    if (!out.empty() && out.back().try_add(n)) continue;
    if (std::getenv("LSE_DEBUG_PHASES") != nullptr && !out.empty()) {
      std::fprintf(stderr, "phase split after %zu members at %s%s\n",
                   out.back().members().size(),
                   std::string(to_string(n->kind)).c_str(),
                   n->shape.to_string().c_str());
    }
    Workgroup wg(dev);
    (void)wg.try_add(n);
    out.push_back(std::move(wg));
  }
  return out;
}

FusionGroup Partitioner::phase_group(const Workgroup& wg,
                                     std::span<const NodePtr> roots) {
  FusionGroup g;
  g.is_phase = true;
  g.launches = 1;
  g.anchor_class = FusionClass::kBarrier;
  std::unordered_set<const Node*> members;
  std::unordered_set<const Node*> is_root;
  for (const NodePtr& r : roots) is_root.insert(r.get());
  for (const NodePtr& n : wg.members()) {
    members.insert(n.get());
    if (n->fclass == FusionClass::kLeaf) continue;
    g.nodes.push_back(n);
    if (g.anchor == OpKind::kCustom) g.anchor = n->kind;
  }
  std::unordered_set<const Node*> added_in;
  for (const NodePtr& n : g.nodes) {
    for (const NodePtr& in : n->inputs) {
      if (members.count(in.get()) && in->fclass != FusionClass::kLeaf) continue;
      if (added_in.insert(in.get()).second) g.inputs.push_back(in);
    }
  }
  for (const NodePtr& n : g.nodes) {
    if (is_root.count(n.get())) g.outputs.push_back(n);
  }
  if (g.outputs.empty() && !g.nodes.empty()) g.outputs.push_back(g.nodes.back());
  return g;
}

std::vector<FusionGroup> Partitioner::phase_chunks(const FusionGroup& phase,
                                                  std::size_t max_bindings) {
  std::vector<FusionGroup> out;
  if (max_bindings < 4) max_bindings = 4;
  FusionGroup cur;
  cur.is_phase = true;
  cur.launches = 1;
  cur.anchor_class = FusionClass::kBarrier;
  std::unordered_set<const Node*> in_chunk;
  auto binds_if_add = [&](const NodePtr& n) {
    std::size_t b = in_chunk.size();
    auto count = [&](const Node* p) {
      while (p && p->kind == OpKind::kReshape && p->inputs.size() == 1) {
        p = p->inputs[0].get();
      }
      if (p && !in_chunk.count(p)) ++b;
    };
    count(n.get());
    for (const NodePtr& in : n->inputs) count(in.get());
    return b;
  };
  auto resolve = [](NodePtr p) {
    while (p && p->kind == OpKind::kReshape && p->inputs.size() == 1) {
      p = p->inputs[0];
    }
    return p;
  };
  auto seal = [&] {
    if (cur.nodes.empty()) return;
    std::unordered_set<const Node*> members;
    for (const NodePtr& n : cur.nodes) members.insert(n.get());
    std::unordered_set<const Node*> added;
    for (const NodePtr& n : cur.nodes) {
      for (const NodePtr& in : n->inputs) {
        NodePtr p = resolve(in);
        if (!p || members.count(p.get()) || !added.insert(p.get()).second) {
          continue;
        }
        cur.inputs.push_back(p);
      }
    }
    cur.outputs.push_back(cur.nodes.back());
    out.push_back(std::move(cur));
    cur = FusionGroup{};
    cur.is_phase = true;
    cur.launches = 1;
    cur.anchor_class = FusionClass::kBarrier;
    in_chunk.clear();
  };
  for (const NodePtr& n : phase.nodes) {
    if (!cur.nodes.empty() && binds_if_add(n) > max_bindings) seal();
    cur.nodes.push_back(n);
    auto add = [&](const Node* p) {
      while (p && p->kind == OpKind::kReshape && p->inputs.size() == 1) {
        p = p->inputs[0].get();
      }
      if (p) in_chunk.insert(p);
    };
    add(n.get());
    for (const NodePtr& in : n->inputs) add(in.get());
  }
  seal();
  return out;
}

std::vector<FusionGroup> Partitioner::partition(
    std::span<const NodePtr> roots, const backend::DeviceInfo* device) {
  std::vector<NodePtr> order;
  std::unordered_set<const Node*> seen;
  for (const NodePtr& r : roots) topo_visit(r, seen, order);

  std::unordered_set<const Node*> is_root;
  for (const NodePtr& r : roots) is_root.insert(r.get());

  std::vector<FusionGroup> groups;
  std::vector<Workgroup> wgs;
  std::unordered_map<const Node*, std::size_t> group_of;
  const WorkgroupDevice wg_dev = WorkgroupDevice::from(device);

  auto inputs_ordered = [&](const Node& n, std::size_t gi) {
    for (const NodePtr& other : n.inputs) {
      if (other->fclass == FusionClass::kLeaf) continue;
      auto oit = group_of.find(other.get());
      if (oit != group_of.end() && oit->second > gi) return false;
    }
    return true;
  };

  for (const NodePtr& n : order) {
    std::size_t target = groups.size();

    const bool n_barrier = n->fclass == FusionClass::kBarrier ||
                           n->fclass == FusionClass::kCollective;
    if (!n_barrier) {
      for (const NodePtr& in : n->inputs) {
        auto it = group_of.find(in.get());
        if (it == group_of.end()) continue;
        if (!can_fuse(*in, *n)) continue;

        bool ordered = true;
        for (const NodePtr& other : n->inputs) {
          if (other.get() == in.get()) continue;
          if (other->fclass == FusionClass::kLeaf) continue;
          auto oit = group_of.find(other.get());
          if (oit != group_of.end() && oit->second > it->second) {
            ordered = false;
            break;
          }
        }
        if (!ordered) continue;

        const bool n_is_kernel = is_kernel_prim(*n);
        bool width_ok = true;
        for (const NodePtr& g : groups[it->second].nodes) {
          if (!is_kernel_prim(*g) && !n_is_kernel) continue;
          if (g->element_count() != n->element_count()) {
            width_ok = false;
            break;
          }
        }
        if (!width_ok) continue;

        target = it->second;
        break;
      }
    }

    // linear / rms / conv / gdn / slice share a Workgroup. Rebuild the
    // trial from the group's live nodes — a parallel Workgroup can miss
    // a member if try_add failed, and an empty trial would accept anyone.
    if (target == groups.size() && !hard_barrier(*n) &&
        workgroup_shareable(*n) && is_kernel_prim(*n)) {
      for (std::size_t gi = groups.size(); gi-- > 0;) {
        if (kernel_prims_in(groups[gi]) == 0) continue;
        if (!only_kernels_and_leaves(groups[gi])) continue;
        if (!inputs_ordered(*n, gi)) continue;
        Workgroup trial(wg_dev);
        bool ok = true;
        for (const NodePtr& m : groups[gi].nodes) {
          if (!trial.try_add(m)) {
            ok = false;
            break;
          }
        }
        if (!ok || !trial.try_add(n) || !trial.emittable() ||
            trial.launches() != 1) {
          continue;
        }
        if (gi < wgs.size()) wgs[gi] = std::move(trial);
        target = gi;
        break;
      }
    }

    if (target == groups.size()) {
      FusionGroup g;
      g.anchor = n->kind;
      g.anchor_class = n->fclass;
      groups.push_back(std::move(g));
      Workgroup wg(wg_dev);
      (void)wg.try_add(n);
      wgs.push_back(std::move(wg));
    } else {
      // The anchor names what shaped the group, so a leaf yields to any real
      // op and an elementwise op yields to a reduction or barrier.
      const FusionClass cur = groups[target].anchor_class;
      const bool promote = cur == FusionClass::kLeaf ||
                           (cur != FusionClass::kBarrier &&
                            (n->fclass == FusionClass::kReduction || n_barrier));
      if (promote) {
        groups[target].anchor = n->kind;
        groups[target].anchor_class = n->fclass;
      }
      if (target < wgs.size()) (void)wgs[target].try_add(n);
    }

    groups[target].nodes.push_back(n);
    group_of[n.get()] = target;
  }

  // A group of nothing but leaves emits a kernel that only materializes
  // literals. Fold it into its unique consumer group so the constant is inlined
  // there instead of costing a launch.
  {
    std::vector<int> merge_into(groups.size(), -1);
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      bool all_leaf = true;
      for (const NodePtr& n : groups[gi].nodes) {
        all_leaf = all_leaf && n->fclass == FusionClass::kLeaf;
      }
      if (!all_leaf) continue;

      std::size_t consumer = groups.size();
      bool unique = true;
      for (const auto& [node, other] : group_of) {
        if (other == gi) continue;
        for (const NodePtr& in : node->inputs) {
          if (group_of.count(in.get()) == 0 || group_of[in.get()] != gi) continue;
          if (consumer == groups.size()) consumer = other;
          else if (consumer != other) unique = false;
        }
      }
      // A collective moves buffers rather than generating source, so nothing
      // can be inlined into it. Barrier groups are generated now — the emitter
      // wraps the kernel primitive — so a constant folds into them fine.
      bool consumer_generates =
          consumer < groups.size() &&
          groups[consumer].anchor_class != FusionClass::kCollective;

      // A barrier indexes its operands as buffers — a matmul reads a whole row,
      // not element i — so an operand of one must stay a binding. Inlining it
      // as a literal would leave the kernel reading a pointer never bound.
      // Leaves feeding the *epilogue* around that barrier still fold in.
      if (consumer_generates) {
        for (const NodePtr& c : groups[consumer].nodes) {
          if (!reads_inputs_as_buffers(*c)) continue;
          for (const NodePtr& in : c->inputs) {
            auto git = group_of.find(in.get());
            if (git != group_of.end() && git->second == gi) {
              consumer_generates = false;
            }
          }
        }
      }

      if (unique && consumer_generates && consumer > gi) {
        merge_into[gi] = static_cast<int>(consumer);
      }
    }

    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      if (merge_into[gi] < 0) continue;
      auto& dst = groups[static_cast<std::size_t>(merge_into[gi])];
      dst.nodes.insert(dst.nodes.begin(), groups[gi].nodes.begin(),
                       groups[gi].nodes.end());
      for (const NodePtr& n : groups[gi].nodes) {
        group_of[n.get()] = static_cast<std::size_t>(merge_into[gi]);
      }
      groups[gi].nodes.clear();
    }
    std::vector<FusionGroup> kept;
    std::vector<std::size_t> remap(groups.size());
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      remap[gi] = kept.size();
      if (!groups[gi].nodes.empty()) kept.push_back(std::move(groups[gi]));
    }
    for (auto& [node, gi] : group_of) gi = remap[gi];
    groups = std::move(kept);
  }

  // Linked SwiGLU/RMS merge is parked until the staged kernel is safe on
  // lemonseed shapes. The matcher still lives in the HRX linked emitter.
  constexpr bool kLinkFuse = false;
  if (kLinkFuse) {
    auto is_lin = [](const Node* n) {
      if (n == nullptr) return false;
      if (n->kind == OpKind::kLinear) return true;
      return n->prim != nullptr && (n->prim->name() == "linear" ||
                                    n->prim->name() == "linear_indexed");
    };
    auto skip = [](const Node* n) {
      while (n != nullptr && n->inputs.size() == 1 &&
             (n->kind == OpKind::kReshape ||
              (n->kind == OpKind::kSlice &&
               n->element_count() == n->inputs[0]->element_count()))) {
        n = n->inputs[0].get();
      }
      return n;
    };
    auto merge_ids = [&](std::vector<std::size_t> ids) {
      if (ids.size() < 2) return;
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      const std::size_t dst = ids.front();
      for (std::size_t i = 1; i < ids.size(); ++i) {
        const std::size_t src = ids[i];
        if (src >= groups.size() || groups[src].nodes.empty()) continue;
        for (const NodePtr& n : groups[src].nodes) {
          groups[dst].nodes.push_back(n);
          group_of[n.get()] = dst;
        }
        groups[src].nodes.clear();
        if (groups[dst].anchor_class != FusionClass::kBarrier) {
          groups[dst].anchor = groups[src].anchor;
          groups[dst].anchor_class = FusionClass::kBarrier;
        }
      }
    };

    for (FusionGroup& g : groups) {
      for (const NodePtr& n : g.nodes) {
        if (!is_lin(n.get()) || n->inputs.size() < 2) continue;
        const Node* h = skip(n->inputs[0].get());
        if (h != nullptr && h->kind == OpKind::kMul && h->inputs.size() == 2) {
          const Node* a = skip(h->inputs[0].get());
          const Node* b = skip(h->inputs[1].get());
          const Node* silu = nullptr;
          const Node* u = nullptr;
          if (a != nullptr && a->kind == OpKind::kSiLU) {
            silu = a;
            u = b;
          } else if (b != nullptr && b->kind == OpKind::kSiLU) {
            silu = b;
            u = a;
          }
          if (silu != nullptr && !silu->inputs.empty()) {
            const Node* gate = skip(silu->inputs[0].get());
            u = skip(u);
            if (is_lin(gate) && is_lin(u) &&
                gate->inputs[0].get() == u->inputs[0].get()) {
              std::vector<std::size_t> ids;
              const Node* parts[] = {gate, u, silu, h, n.get()};
              for (const Node* p : parts) {
                auto it = group_of.find(p);
                if (it != group_of.end()) ids.push_back(it->second);
              }
              merge_ids(std::move(ids));
              continue;
            }
          }
        }
        const Node* act = skip(n->inputs[0].get());
        if (act != nullptr && act->kind == OpKind::kRMS &&
            act->consumer_count == 1) {
          std::vector<std::size_t> ids;
          const Node* parts[] = {act, n.get()};
          for (const Node* p : parts) {
            auto it = group_of.find(p);
            if (it != group_of.end()) ids.push_back(it->second);
          }
          merge_ids(std::move(ids));
        }
      }
    }

    std::vector<FusionGroup> kept;
    std::vector<std::size_t> remap(groups.size());
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      remap[gi] = kept.size();
      if (!groups[gi].nodes.empty()) kept.push_back(std::move(groups[gi]));
    }
    for (auto& [node, gi] : group_of) {
      if (gi < remap.size()) gi = remap[gi];
    }
    groups = std::move(kept);
  }

  // Inputs are values read from outside the group; outputs are values a node in
  // another group, or a root, needs.
  for (std::size_t gi = 0; gi < groups.size(); ++gi) {
    FusionGroup& g = groups[gi];
    std::unordered_set<const Node*> members;
    for (const NodePtr& n : g.nodes) members.insert(n.get());

    std::unordered_set<const Node*> added_in;
    for (const NodePtr& n : g.nodes) {
      for (const NodePtr& in : n->inputs) {
        if (members.count(in.get())) continue;
        if (added_in.insert(in.get()).second) g.inputs.push_back(in);
      }
    }

    for (const NodePtr& n : g.nodes) {
      bool escapes = is_root.count(n.get()) != 0;
      if (!escapes) {
        for (const auto& [node, other] : group_of) {
          if (other == gi) continue;
          const Node* consumer = node;
          for (const NodePtr& in : consumer->inputs) {
            if (in.get() == n.get()) {
              escapes = true;
              break;
            }
          }
          if (escapes) break;
        }
      }
      if (escapes) g.outputs.push_back(n);
    }

    Workgroup wg(wg_dev);
    for (const NodePtr& n : g.nodes) (void)wg.try_add(n);
    g.launches = wg.launches();
  }

  return groups;
}

}  // namespace lse::graph
