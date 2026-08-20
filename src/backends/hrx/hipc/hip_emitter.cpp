#include "lse/backends/hrx/hipc/hip_emitter.hpp"

#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ir/lower.hpp"
#include "lse/ir/pass/pass.hpp"
#include "lse/kernels/lds_linear.hpp"
#include "lse/kernels/linked.hpp"
#include "lse/opt/fusion.hpp"

namespace lse::backend {

using namespace lse::graph;

namespace {

void mix_name(std::uint64_t& h, std::string_view name) {
  h ^= 0x9e3779b97f4a7c15ull;
  for (char c : name) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    h *= 1099511628211ull;
  }
}

struct IndexedStage {
  NodePtr node;
  const KernelPrimitiveBase* prim = nullptr;
  // The plan this stage would launch under ALONE: its own grid and the
  // workgroup scratch its own body declares with nothing hoisted for it. The
  // scratch half is the unfused arrangement's price for this member, and it is
  // the same quantity as the fused body's — both are what the primitive says
  // it will declare, asked with and without the run's panels.
  ThreadPlan plan;
  // The activation panel this stage puts in workgroup scratch, as the stage
  // itself declares it. Asked at specialize time so the run can share it.
  KernelPrimitiveBase::StagedRow row;
  // The int8 spec this stage would quantize that row under, invalid when it
  // takes no integer path. A run hoists the int8 form only where every member
  // agrees on this.
  kernels::DotStagingPlan dot_plan;
  // The operand shapes, owned, so the stage can be re-asked with a panel
  // described without the caller rebuilding them.
  std::vector<Shape> in_shapes;
  std::vector<DType> in_dtypes;

  [[nodiscard]] KernelShapes shapes(const DeviceInfo& device,
                                    const kir::TypeTable& types,
                                    const DialectSourceTable& spellings) const {
    KernelShapes s;
    s.inputs = in_shapes;
    s.input_dtypes = in_dtypes;
    s.output = node->shape;
    s.output_dtype = node->dtype;
    s.attrs = node->attrs;
    s.iattrs = node->iattrs;
    s.device = &device;
    s.types = types;
    s.intrinsics = &spellings;
    return s;
  }
};

// One activation row, staged once for every stage of a run that reads it.
struct RowPanel {
  const Node* act = nullptr;
  std::uint32_t count = 0;
  std::uint32_t rows = 0;
  std::size_t members = 0;
  // The array in the merged body, once the prologue has declared it. Empty
  // while the row belongs to a single stage, which stages it itself.
  std::string name;

  // The int8 form of the row, hoisted beside it when every member would have
  // quantized it the same way. Empty names mean each member stages its own.
  kernels::DotStagingPlan quant_plan;
  bool hoist_blocked = false;
  kernels::StagedQuantNames quant;

  // Whether the merged body hoists this panel at all, and whether it hoists
  // the int8 form beside it. A panel only one stage reads is left to that
  // stage, so it is not hoisted and costs the run nothing extra.
  bool hoisted = false;
  bool quant_hoisted = false;
};

constexpr std::size_t kNoPanel = static_cast<std::size_t>(-1);

// Which stages of a run stage the same row.
//
// The predicate is deliberately narrow: the same activation *node* — hence the
// same pointer at runtime, since bindings are keyed on the node — at the same
// element count, over the same number of rows, in f32. Nothing is inferred from
// shapes that merely agree, and nothing is inferred about what a body reads:
// staged_row() is the stage's own statement that row `workgroup_id_y` of that
// operand is all it takes from it.
//
// Two stages over different buffers, different K, or different row counts land
// in different panels and are staged separately, which is the correct answer
// rather than a missed opportunity.
std::vector<RowPanel> row_panels(const std::vector<IndexedStage>& stages,
                                 std::vector<std::size_t>& panel_of) {
  std::vector<RowPanel> panels;
  panel_of.assign(stages.size(), kNoPanel);
  for (std::size_t si = 0; si < stages.size(); ++si) {
    const IndexedStage& st = stages[si];
    if (st.row.count == 0 || st.row.rows == 0) continue;
    if (st.row.input >= st.node->inputs.size()) continue;
    const Node* act = st.node->inputs[st.row.input].get();
    // The panel is f32; a narrower activation buffer would need the fill to
    // widen, which is not what the panel promises.
    if (act->dtype != DType::kF32) continue;
    std::size_t at = panels.size();
    for (std::size_t p = 0; p < panels.size(); ++p) {
      if (panels[p].act == act && panels[p].count == st.row.count &&
          panels[p].rows == st.row.rows) {
        at = p;
        break;
      }
    }
    if (at == panels.size()) {
      panels.push_back(RowPanel{act, st.row.count, st.row.rows, 0, {}, {}, false, {}});
    }
    ++panels[at].members;
    panel_of[si] = at;
  }
  return panels;
}

// A name for a panel that does not exist yet. The primitives read
// `staged.name` to BUILD the tile and `staged.count` to decide whether the
// panel is theirs; pricing asks only the second question, and the body gets the
// real name the prologue returned.
constexpr std::string_view kPricedPanel = "?panel";

// What one arrangement of a run costs in workgroup scratch: the fused body's
// declaration and the tightest of the solo bodies'.
//
// THE SAME QUANTITY ON BOTH SIDES, which is the whole point. Each figure is the
// sum of the workgroup-shared arrays the body that would be printed declares,
// asked of the primitives themselves — the fused side with the run's panels
// described, the solo side with nothing described. Anything else makes the
// comparison meaningless: pricing the UNFUSED arrangement as the FUSED one's
// hoisted f32 panel, which is what this used to do, made both sides equal on
// every run in every model and the residency gate could not fire at all. On the
// 27B the true figures for one 5120-wide decode run are 28480 fused against
// 8000 solo — 4 workgroups per pool against 8.
//
// An ESTIMATE, in the sense that no kernel has been compiled: it is what the
// emitter is about to write, not what the assembler emitted. It is exact by
// construction against Body::workgroup_bytes() on the merged body, and a test
// pins the two together so a drift shows up at build time.
struct ScratchCounts {
  std::uint32_t fused = 0;
  std::uint32_t worst_solo = 0;
};

// Which panels the merged body would hoist, and the int8 spec each would hoist
// beside its row. Fills the panels in place because the emit path needs the
// same answers to write the prologue, and two rules that could disagree is the
// defect this replaces.
void plan_hoisting(const std::vector<IndexedStage>& stages,
                   std::vector<RowPanel>& panels,
                   const std::vector<std::size_t>& panel_of,
                   std::uint32_t run_block, std::uint32_t lds_budget) {
  // A panel hoists the int8 form only when every stage reading the row would
  // have produced the same one. One member on the float codec, or on another
  // group size, and the codes would be wrong for it -- so the panel keeps the
  // row alone and each member stages for itself, exactly as before.
  for (std::size_t si = 0; si < stages.size(); ++si) {
    if (panel_of[si] == kNoPanel) continue;
    RowPanel& panel = panels[panel_of[si]];
    const kernels::DotStagingPlan& plan = stages[si].dot_plan;
    if (!plan.valid() || plan.count != panel.count) {
      panel.quant_plan = {};
      panel.hoist_blocked = true;
    } else if (!panel.hoist_blocked) {
      if (panel.quant_plan.valid() && !(panel.quant_plan == plan)) {
        panel.quant_plan = {};
        panel.hoist_blocked = true;
      } else {
        panel.quant_plan = plan;
      }
    }
  }
  // A row only one stage wants is left to that stage. Hoisting it would put
  // the fill under `row < rows` alone, and every workgroup of the run would
  // then pay for it, including the ones covering tiles that stage does not
  // have.
  std::uint32_t used = 0;
  for (RowPanel& panel : panels) {
    panel.hoisted = panel.members >= 2;
    panel.quant_hoisted = false;
    if (!panel.hoisted) continue;
    used += kir::Lds::align(panel.count * kir::pack_elem_bytes<kir::f32>());
    if (!panel.quant_plan.valid()) continue;
    const std::uint32_t q = kernels::staged_dot_bytes(
        panel.quant_plan.count, panel.quant_plan.group_size, run_block);
    // emit_staged_dot_acts declines what will not fit beside the row, and a
    // declined hoist leaves every member staging for itself.
    if (q == 0 || (lds_budget != 0 && used + q > lds_budget)) continue;
    panel.quant_hoisted = true;
    used += q;
  }
}

ScratchCounts count_run_scratch(const std::vector<IndexedStage>& stages,
                                const std::vector<RowPanel>& panels,
                                const std::vector<std::size_t>& panel_of,
                                const DeviceInfo& device,
                                std::uint32_t run_block) {
  const DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();
  ScratchCounts out;
  for (const RowPanel& panel : panels) {
    if (!panel.hoisted) continue;
    out.fused += kir::Lds::align(panel.count * kir::pack_elem_bytes<kir::f32>());
    if (panel.quant_hoisted) {
      out.fused += kernels::staged_dot_bytes(
          panel.quant_plan.count, panel.quant_plan.group_size, run_block);
    }
  }
  for (std::size_t si = 0; si < stages.size(); ++si) {
    const IndexedStage& st = stages[si];
    // The unfused arrangement's price for this member: what its own body
    // declares with nothing hoisted for it. The tightest of these is what the
    // arrangement's residency is, since it is the launch that seats fewest.
    if (st.plan.lds_bytes > out.worst_solo) out.worst_solo = st.plan.lds_bytes;

    KernelShapes s = st.shapes(device, type_table, spellings);
    const RowPanel* panel =
        panel_of[si] == kNoPanel ? nullptr : &panels[panel_of[si]];
    if (panel != nullptr && panel->hoisted) {
      s.staged = graph::StagedPanel{kPricedPanel, panel->count};
      if (panel->quant_hoisted) {
        s.staged_quant = graph::StagedQuantPanel{
            kPricedPanel,
            kPricedPanel,
            kPricedPanel,
            panel->quant_plan.count,
            panel->quant_plan.group_size,
            panel->quant_plan.bits};
      }
    }
    out.fused += st.prim->plan(s).lds_bytes;
  }
  return out;
}

// Identity of the `__device__` helper the per-element scaffold emits for a
// kernel primitive.
//
// `entry_name()` is a constant per primitive — every `linear` is
// "lse_linear" — while `emit_kernel` bakes the extents, the operand dtypes
// and `iattrs[0]` (the expert slot of `linear_indexed`) in as literals. Two
// nodes of the same primitive are therefore the same function only when all
// of those agree; deduplicating on the name alone let a sibling with N=128
// call a body specialized for N=1024 and read past the end of its weight,
// and let the slot-1 expert run the slot-0 body.
std::string device_fn_name(const KernelPrimitiveBase& kp, const Node& n) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&h](std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  auto mix_shape = [&](const Shape& s, DType dt) {
    mix(s.rank());
    for (std::size_t i = 0; i < s.rank(); ++i) {
      mix(static_cast<std::uint64_t>(s.dim(i)));
    }
    mix(static_cast<std::uint64_t>(dt));
  };
  for (const NodePtr& in : n.inputs) mix_shape(in->shape, in->dtype);
  mix_shape(n.shape, n.dtype);
  for (std::int32_t v : n.iattrs) mix(static_cast<std::uint64_t>(v));
  for (float f : n.attrs) {
    std::uint32_t bits = 0;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    mix(bits);
  }
  return std::string(kp.entry_name()) + "_" + std::to_string(h);
}

std::vector<IndexedStage> indexed_stages(std::span<const NodePtr> nodes,
                                         const DeviceInfo& device) {
  const graph::DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();
  std::vector<IndexedStage> out;
  for (const NodePtr& n : nodes) {
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    if (kp == nullptr) continue;
    IndexedStage st;
    st.node = n;
    st.in_shapes.reserve(n->inputs.size());
    st.in_dtypes.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      st.in_shapes.push_back(in->shape);
      st.in_dtypes.push_back(in->dtype);
    }
    const KernelShapes probe = st.shapes(device, type_table, spellings);
    const KernelPrimitiveBase* chosen = kp->specialize(probe);
    if (chosen == nullptr || !chosen->owns_indexing()) continue;
    st.prim = chosen;
    // Nothing is described to it, so this is what the stage declares ALONE.
    st.plan = chosen->plan(probe);
    st.row = chosen->staged_row(probe);
    st.dot_plan = kernels::dot_staging_plan(
        probe, chosen->name() == "quant_linear_indexed");
    out.push_back(std::move(st));
  }
  return out;
}

// The kernel signature, with the launch geometry the object will be dispatched
// at stated to the compiler.
//
// WHY IT IS SAID AT ALL. Without a bound every emitted object reports
// max_flat_workgroup_size 1024 — the AMDGPU default, which means the compiler
// was told nothing and had to assume the widest workgroup the target allows.
// The register allocator then sizes a wave's registers so that SIXTEEN waves
// of one workgroup could co-reside, for a workgroup this engine never
// launches, and the compiler's own occupancy figure is derived from that
// assumption rather than from the dispatch. The engine knows the geometry at
// emit time — it is the plan the dispatch will use — so it states it.
//
// MEASURED, gfx1151, Qwen3.5-0.8B-4bit, 109 kernels compared before and after:
// 108 of them move from max_flat_workgroup_size 1024 to their real 256; the
// remaining one really is a 1024-thread scaffold. Register allocation is
// unchanged on 104, moves by one or two on four, and moves 113 -> 165 on one
// staged body, which is the allocator taking the room the true bound gives it.
// Nothing spills either way.
//
// LANDMINE: this is a PROMISE, not a hint. A dispatch with more threads than
// the bound fails outright, so `flat_threads` must be the product of the same
// LaunchDims::workgroup_size the emitted kernel is launched with and nothing
// else. 0 states nothing, which is what an emitter that cannot know it does.
std::string kernel_signature(std::string_view entry,
                             std::uint32_t flat_threads) {
  std::string out = "extern \"C\" __global__ ";
  if (flat_threads != 0) {
    out += "__launch_bounds__(" + std::to_string(flat_threads) + ") ";
  }
  out += "void ";
  out += entry;
  out += "(\n";
  return out;
}

std::uint32_t flat_of(const std::uint32_t (&wg)[3]) noexcept {
  const std::uint64_t n =
      static_cast<std::uint64_t>(wg[0]) * wg[1] * wg[2];
  return n == 0 || n > 0xFFFFFFFFull ? 0u : static_cast<std::uint32_t>(n);
}

// The entry names the launches this run would replace are compiled as.
//
// A member launching alone is a one-node group — its own inputs, itself as the
// single output — which is exactly what the scheduler builds for a wide linear
// that joins no group before it. The signature is therefore reproducible here
// without emitting anything, and a measurement recorded against it last run
// scores the unfused side of the comparison.
std::vector<std::string> solo_entry_names(
    const std::vector<IndexedStage>& stages) {
  std::vector<std::string> out;
  out.reserve(stages.size());
  for (const IndexedStage& st : stages) {
    FusionGroup one;
    one.nodes.push_back(st.node);
    one.outputs.push_back(st.node);
    one.inputs = st.node->inputs;
    out.push_back("lse_fused_" + std::to_string(one.signature()));
  }
  return out;
}

// Whether the emitter CAN write this run as one body. Whether it SHOULD is the
// engine's answer, taken separately: this function is lowering, and the
// residency arithmetic that used to sit in its first line is not.
bool sibling_stages(const FusionGroup& group,
                    const std::vector<IndexedStage>& stages) {
  if (stages.size() < 2) return false;
  std::unordered_set<const Node*> stage_set;
  std::unordered_set<const Node*> outputs;
  for (const IndexedStage& s : stages) stage_set.insert(s.node.get());
  for (const NodePtr& o : group.outputs) outputs.insert(o.get());
  // The stages must cover the group. The emit loop walks `stages`, so a member
  // that did not specialize to a self-indexing form would get no body, no
  // binding and no store — yet the scheduler marks every group.nodes member
  // materialized and device-dirty, so the next group reads a buffer nothing
  // ever wrote. Refusing costs one launch; guessing costs an argmax.
  for (const NodePtr& n : group.nodes) {
    if (!stage_set.count(n.get())) return false;
  }
  for (const IndexedStage& s : stages) {
    if (!outputs.count(s.node.get())) return false;
    for (const NodePtr& in : s.node->inputs) {
      if (stage_set.count(in.get())) return false;
    }
    // Concatenated bodies share one workgroup shape, and a stage bakes its own
    // block size in as a literal — the GEMV strides its LDS staging loop by it.
    // Widening the block under such a body walks the extra threads off the end
    // of the scratch it reserved, so a disagreement is not composable.
    for (int d = 0; d < 3; ++d) {
      if (s.plan.workgroup_size[d] != stages.front().plan.workgroup_size[d]) {
        return false;
      }
    }
  }
  return true;
}

// Must agree with hip_types.cpp: bf16 is the compiler's own __bf16, which
// converts to and from float with a plain cast. Spelling it __hip_bfloat16
// here and __bf16 there is a comgr compile error, not a wrong answer.
std::string_view device_scalar(DType dt) noexcept {
  switch (dt) {
    case DType::kF16: return "_Float16";
    case DType::kBF16: return "__bf16";
    case DType::kI32: return "int";
    case DType::kI8: return "signed char";
    case DType::kU8: return "unsigned char";
    case DType::kU32: return "unsigned int";
    default: return "float";
  }
}

// Loads always widen to float and stores narrow back, so the generated body is
// written once in float regardless of storage dtype.
std::string load_expr(std::string_view buffer, std::string_view index, DType dt) {
  (void)dt;
  return "(float)" + std::string(buffer) + "[" + std::string(index) + "]";
}

std::string store_stmt(std::string_view buffer, std::string_view index, DType dt,
                       std::string_view value) {
  std::string out(buffer);
  out += "[";
  out += index;
  out += "] = (";
  out += device_scalar(dt);
  out += ")(";
  out += value;
  out += ");";
  return out;
}

// The text form of BroadcastMap::apply, built from the same map the host
// interpreter uses so the two index the same element.
std::string broadcast_index_expr(const Shape& src, const Shape& out,
                                 std::string_view flat) {
  const BroadcastMap m = BroadcastMap::build(src, out);
  if (m.identity) return std::string(flat);
  if (m.scalar) return "0";

  std::string expr = "(";
  bool first = true;
  for (std::size_t i = m.gap; i < m.rank; ++i) {
    const std::size_t si = i - m.gap;
    if (m.src_stride[si] == 0) continue;
    if (!first) expr += " + ";
    first = false;
    expr += "((" + std::string(flat) + " / " + std::to_string(m.out_stride[i]) +
            ") % " + std::to_string(m.out_dim[i]) + ") * " +
            std::to_string(m.src_stride[si]);
  }
  if (first) expr += "0";
  expr += ")";
  return expr;
}

std::string_view builtin_expr(OpKind k) noexcept {
  switch (k) {
    case OpKind::kCast: return "$0";
    case OpKind::kReshape: return "$0";
    default: return {};
  }
}

}  // namespace

// The launch geometry is the engine's to choose: graph::choose_launch_dims
// counts elements and residency, neither of which is a property of HIP.
LaunchDims HipEmitter::choose_dims(const FusionGroup& group,
                                   const DeviceInfo& device,
                                   std::uint32_t lds_bytes) {
  return graph::choose_launch_dims(group, device, lds_bytes);
}

std::string HipEmitter::constants_decl(const graph::ConstantsLayout& layout) {
  std::ostringstream os;
  os << "struct LseConstants {\n";
  for (const graph::ConstantsLayout::Field& f : layout.fields) {
    os << "  " << (f.size == 4 ? "unsigned int" : "unsigned long long") << " "
       << f.name << ";\n";
  }
  // An empty push-constant block is not a legal struct.
  if (layout.fields.empty()) os << "  unsigned int _unused;\n";
  os << "};\n";
  return os.str();
}

graph::DialectSourceTable HipEmitter::sources() const noexcept {
  return hip_sources();
}

std::uint64_t HipEmitter::cache_key(const FusionGroup& group,
                                    const DeviceInfo& device) const {
  std::uint64_t h = group.signature();
  const KernelPrimitiveBase* self = nullptr;
  if (kernels::linked_bindings(group).ok) {
    KernelShapes dummy;
    self = kernels::linked_kernel_for(group, dummy);
  }
  // A matched-but-declined linked pipeline must not erase the specialize
  // probe: emit() falls back to the specialized primitive, so the key must
  // name the same kernel or replays serve mismatched dims.
  if (self == nullptr) {
    const graph::DialectSourceTable spellings = hip_sources();
    const kir::TypeTable type_table = hip_types();
    std::vector<Shape> storage;
    std::vector<DType> dtypes;
    for (const NodePtr& n : group.nodes) {
      const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
      if (kp == nullptr) continue;
      storage.clear();
      dtypes.clear();
      for (const NodePtr& in : n->inputs) {
        storage.push_back(in->shape);
        dtypes.push_back(in->dtype);
      }
      KernelShapes probe;
      probe.inputs = storage;
      probe.input_dtypes = dtypes;
      probe.output = n->shape;
      probe.output_dtype = n->dtype;
      probe.attrs = n->attrs;
      probe.iattrs = n->iattrs;
      probe.device = &device;
      probe.types = type_table;
      probe.intrinsics = &spellings;
      const KernelPrimitiveBase* chosen = kp->specialize(probe);
      if (chosen == nullptr || !chosen->owns_indexing()) continue;
      if (group.outputs.size() != 1) break;
      self = chosen;
      break;
    }
  }
  if (self) mix_name(h, self->name());
  h ^= device.wavefront_size;
  h *= 1099511628211ull;
  // Persist-grid source (gbar, lse_grid_sync, grid-stride walks) is a
  // different kernel from the one-workgroup walk. Disk and emit caches
  // key on this, so a 1-CU box cannot reuse a 40-CU object.
  if (group.is_phase && device.compute_units >= 2) {
    h ^= 2;
    h *= 1099511628211ull;
  }
  return h;
}

std::string_view HipEmitter::prelude() const noexcept {
  static constexpr std::string_view kPrelude =
      "#include <hip/hip_runtime.h>\n"
      "#include <hip/hip_bf16.h>\n\n";
  return kPrelude;
}

graph::IKernelEmitter::RunScratch HipEmitter::run_scratch(
    std::span<const graph::NodePtr> run, const DeviceInfo& device) const {
  RunScratch out;
  if (run.size() < 2) return out;
  // The group the scheduler would build for this run, member for member and
  // input for input, so the entry name below is the one the merged kernel is
  // actually compiled as and a previous compile of it can be found.
  FusionGroup g;
  g.nodes.assign(run.begin(), run.end());
  g.outputs.assign(run.begin(), run.end());
  // The first member's inputs verbatim and each later member's deduplicated
  // against them, which is exactly how the group grows there — including the
  // detail that a first member naming one buffer twice keeps it twice. The
  // signature is over this list, so a difference here is a different entry
  // name and a measurement that is never found.
  g.inputs = run.front()->inputs;
  for (std::size_t i = 1; i < run.size(); ++i) {
    for (const NodePtr& in : run[i]->inputs) {
      bool seen = false;
      for (const NodePtr& e : g.inputs) seen = seen || e.get() == in.get();
      if (!seen) g.inputs.push_back(in);
    }
  }
  g.anchor = run.front()->kind;
  g.anchor_class = run.front()->fclass;

  const std::uint64_t key = g.signature();
  if (const auto it = run_scratch_cache_.find(key);
      it != run_scratch_cache_.end()) {
    return it->second;
  }

  const std::vector<IndexedStage> stages = indexed_stages(run, device);
  // Not a run this emitter would write as one body: nothing to report, which
  // the engine reads as no answer rather than as a free merge.
  if (!sibling_stages(g, stages)) return out;

  std::vector<std::size_t> panel_of;
  std::vector<RowPanel> panels = row_panels(stages, panel_of);
  const std::uint32_t threads = stages.front().plan.workgroup_size[0];
  plan_hoisting(stages, panels, panel_of, threads,
                workgroup_lds_bytes(&device));
  const ScratchCounts bytes =
      count_run_scratch(stages, panels, panel_of, device, threads);
  out.threads = threads;
  out.fused = bytes.fused;
  out.worst_solo = bytes.worst_solo;
  out.fused_entry = "lse_fused_" + std::to_string(key);
  out.solo_entries = solo_entry_names(stages);
  run_scratch_cache_.emplace(key, out);
  return out;
}

Result<graph::EmittedKernel> HipEmitter::emit(const FusionGroup& group,
                                              const DeviceInfo& device) const {
  if (group.nodes.empty()) {
    return LSE_ERROR(kInvalidArgument, "cannot emit an empty fusion group");
  }
  if (group.is_phase) {
    const std::uint64_t key = cache_key(group, device);
    if (const auto it = emit_cache_.find(key); it != emit_cache_.end()) {
      EmittedKernel out;
      out.source = it->second.source;
      out.entry_name = it->second.entry_name;
      out.dims = it->second.dims;
      out.pointer_table = false;
      out.constants.add("count", 4);
      out.lds_bytes = it->second.lds_bytes;
      out.scratch_bytes = it->second.scratch_bytes;
      out.persist_grid = it->second.persist_grid;
      bind_phase(group, out);
      return out;
    }
    auto emitted = emit_phase(group, device);
    if (emitted.ok()) {
      emit_cache_[key] =
          CachedEmit{emitted->source,        emitted->entry_name,
                     emitted->dims,          emitted->lds_bytes,
                     emitted->scratch_bytes, emitted->persist_grid};
    }
    return emitted;
  }
  if (group.anchor_class == FusionClass::kCollective) {
    return LSE_ERROR(kUnimplemented, "group anchored on ",
                     std::string(to_string(group.anchor)),
                     " is a transport operation, not generated source");
  }

  // The spans point into `storage` and `dtypes`, which the caller must keep
  // alive.
  const graph::DialectSourceTable spellings = sources();
  const kir::TypeTable type_table = hip_types();
  auto shapes_for = [&](const NodePtr& n, std::vector<Shape>& storage,
                        std::vector<DType>& dtypes) {
    storage.clear();
    dtypes.clear();
    storage.reserve(n->inputs.size());
    dtypes.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      storage.push_back(in->shape);
      dtypes.push_back(in->dtype);
    }
    KernelShapes s;
    s.inputs = storage;
    s.input_dtypes = dtypes;
    s.output = n->shape;
    s.output_dtype = n->dtype;
    s.attrs = n->attrs;
    s.iattrs = n->iattrs;
    s.device = &device;
    s.types = type_table;
    s.intrinsics = &spellings;
    return s;
  };

  const auto stages = indexed_stages(group.nodes, device);
  std::vector<std::size_t> panel_of;
  std::vector<RowPanel> panels = row_panels(stages, panel_of);
  const std::uint32_t lds_budget = workgroup_lds_bytes(&device);
  const std::string run_entry =
      "lse_fused_" + std::to_string(group.signature());

  // Can the emitter write it, and should the engine ask for it. The second
  // question is residency, which is counted from bytes and threads and is
  // therefore not this file's to answer.
  bool fuse = sibling_stages(group, stages);
  ScratchCounts run_bytes;
  if (fuse) {
    const std::uint32_t block = stages.front().plan.workgroup_size[0];
    plan_hoisting(stages, panels, panel_of, block, lds_budget);
    run_bytes = count_run_scratch(stages, panels, panel_of, device, block);
    opt::FusionCandidate candidate;
    candidate.threads = stages.front().plan.workgroup_size[0];
    candidate.fused_scratch_bytes = run_bytes.fused;
    candidate.worst_solo_scratch_bytes = run_bytes.worst_solo;
    candidate.fused_entry = run_entry;
    const std::vector<std::string> solo = solo_entry_names(stages);
    candidate.solo_entries = solo;
    fuse = opt::admit_fusion(opt::DeviceCapacity::of(device), candidate).admit;
  }
  if (fuse) {
    EmittedKernel out;
    out.entry_name = run_entry;

    std::unordered_map<const Node*, std::size_t> binding_of;
    auto bind = [&](const NodePtr& n) {
      if (binding_of.emplace(n.get(), out.binding_order.size()).second) {
        out.binding_order.push_back(n);
      }
    };
    for (const IndexedStage& st : stages) {
      for (const NodePtr& in : st.node->inputs) bind(in);
    }
    for (const NodePtr& in : group.inputs) bind(in);
    std::unordered_set<const Node*> output_set;
    for (const IndexedStage& st : stages) {
      if (binding_of.emplace(st.node.get(), out.binding_order.size()).second) {
        out.binding_order.push_back(st.node);
      }
      output_set.insert(st.node.get());
    }
    out.constants.add("count", 4);

    // What the run means to move. Stages that share a staged row share the
    // activation traffic too — the panel is filled once for all of them — so
    // charging every stage for it would restate the very saving the sharing
    // exists to make. Weight, scale and output are per stage and never shared.
    {
      std::vector<Shape> tstore;
      std::vector<DType> tdtypes;
      std::unordered_set<std::size_t> charged;
      opt::TrafficModel run;
      run.stated = !stages.empty();
      for (std::size_t i = 0; i < stages.size(); ++i) {
        KernelShapes ts = shapes_for(stages[i].node, tstore, tdtypes);
        opt::TrafficModel stage = stages[i].prim->traffic(ts);
        if (!stage.stated) {
          run = opt::TrafficModel{};
          break;
        }
        const std::size_t panel = panel_of[i];
        if (panel < panels.size() && !charged.insert(panel).second) {
          stage.read[static_cast<std::size_t>(
              opt::OperandClass::kActivation)] = 0;
        }
        run += stage;
      }
      out.traffic = run;
    }

    std::uint64_t sig = group.signature();
    for (const IndexedStage& st : stages) mix_name(sig, st.prim->name());
    if (const auto it = lds_refused_.find(sig); it != lds_refused_.end()) {
      return LSE_ERROR(kOutOfMemory, "fused run needs ",
                       std::to_string(it->second),
                       " bytes of workgroup scratch, device allows ",
                       std::to_string(lds_budget));
    }
    if (const auto it = emit_cache_.find(sig); it != emit_cache_.end()) {
      out.source = it->second.source;
      out.entry_name = it->second.entry_name;
      out.dims = it->second.dims;
      // The measured total of the cached text, not today's prediction of it.
      out.lds_bytes = it->second.lds_bytes;
      return out;
    }

    std::ostringstream body;
    body << "#include <hip/hip_runtime.h>\n"
         << "#include <hip/hip_bf16.h>\n\n"
         << constants_decl(out.constants) << "\n"
         << kernel_signature(out.entry_name,
                             flat_of(stages.front().plan.workgroup_size));
    for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
      const NodePtr& b = out.binding_order[i];
      const bool is_out = output_set.count(b.get()) != 0;
      body << "    " << (is_out ? "" : "const ") << device_scalar(b->dtype)
           << "* __restrict__ b" << i << ",\n";
    }
    body << "    LseConstants k) {\n"
         << "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n";

    // One IR body for the whole run, not N texts concatenated. That is the
    // whole reason the middle end exists: the stages are ordinary sibling ops
    // once they are in one body, and a pass can see across them.
    //
    // Each stage records against the run's real binding names (`b3`, not
    // `in1`), so two stages reading one buffer name the same IR value, and
    // against a name prefix, so their generated locals cannot collide once the
    // per-stage braces are gone.
    ir::Body merged(type_table, spellings);

    // The shared rows come first, one staging each, so the barrier behind each
    // fill dominates every stage read spliced after it. Stages then reference
    // the array by name — splice interns symbols, so all of them name the one
    // allocation the prologue declared.
    //
    // A row only one stage wants is left to that stage. Hoisting it would put
    // the fill under `row < rows` alone, and every workgroup of the run would
    // then pay for it, including the ones covering tiles that stage does not
    // have.
    const std::uint32_t run_block = stages.front().plan.workgroup_size[0];
    for (std::size_t p = 0; p < panels.size(); ++p) {
      RowPanel& panel = panels[p];
      if (!panel.hoisted) continue;
      const auto ait = binding_of.find(panel.act);
      if (ait == binding_of.end()) {
        return LSE_ERROR(kInternal,
                         "the row a sibling run shares is not bound in it");
      }
      const std::string act_buf = "b" + std::to_string(ait->second);
      const std::string prefix = "p" + std::to_string(p) + "_";
      const ir::RecordOptions opts{{}, {}, prefix};
      const ir::KernelBody::Recording rec(opts);
      ir::KernelBody::Capture cap;
      {
        kir::KernelBody kb(type_table, spellings, lds_budget);
        const kir::Buffer<kir::f32> act(&kb, &kb.types(), act_buf);
        panel.name = kernels::emit_staged_row(kb, act, panel.count,
                                                  panel.rows, run_block);
        if (!panel.name.empty() && panel.quant_hoisted) {
          panel.quant = kernels::emit_staged_dot_acts(
              kb, panel.name, panel.count, panel.quant_plan.group_size,
              panel.rows, run_block);
        }
      }
      if (panel.name.empty() || !cap.has()) {
        return LSE_ERROR(kInternal, "the shared activation row did not stage");
      }
      merged.splice(cap.body(), merged.entry());
    }

    ThreadPlan unified;
    unified.workgroup_size[0] = 1;
    unified.workgroup_count[0] = 1;
    // The signature above already promised the compiler a flat size, taken
    // from the first stage because sibling_stages required every stage to
    // agree; `unified` below is the max over the same stages and so must come
    // out the same. A dispatch wider than the bound fails outright, so the two
    // are checked against each other rather than assumed equal.
    const std::uint32_t promised_threads =
        flat_of(stages.front().plan.workgroup_size);
    for (std::size_t si = 0; si < stages.size(); ++si) {
      const IndexedStage& st = stages[si];
      std::vector<Shape> storage;
      std::vector<DType> dtypes;
      KernelShapes shapes = shapes_for(st.node, storage, dtypes);
      const auto out_it = binding_of.find(st.node.get());
      if (out_it == binding_of.end()) {
        return LSE_ERROR(kInternal, "sibling stage has no output binding");
      }
      const std::string out_buf = "b" + std::to_string(out_it->second);
      const DType out_dt = st.node->dtype;
      shapes.store = [&, out_buf, out_dt](std::string_view index,
                                          std::string_view value) {
        return store_stmt(out_buf, index, out_dt, value) + "\n";
      };

      std::vector<std::string> input_names;
      input_names.reserve(st.node->inputs.size());
      for (const NodePtr& in : st.node->inputs) {
        const auto iit = binding_of.find(in.get());
        if (iit == binding_of.end()) {
          return LSE_ERROR(kInternal, std::string(st.prim->name()),
                           ": an input is not bound in this group");
        }
        input_names.push_back("b" + std::to_string(iit->second));
      }
      const std::string prefix = "s" + std::to_string(si) + "_";
      if (panel_of[si] != kNoPanel) {
        const RowPanel& panel = panels[panel_of[si]];
        if (!panel.name.empty()) shapes.staged = {panel.name, panel.count};
        if (!panel.quant.codes.empty()) {
          shapes.staged_quant = {panel.quant.codes, panel.quant.scale,
                                 panel.quant.sum, panel.quant_plan.count,
                                 panel.quant_plan.group_size,
                                 panel.quant_plan.bits};
        }
      }

      std::string stage_text;
      {
        const ir::RecordOptions opts{input_names, {}, prefix};
        const ir::KernelBody::Recording rec(opts);
        ir::KernelBody::Capture cap;
        stage_text = st.prim->emit_kernel(shapes);
        if (stage_text.empty() || !cap.has()) {
          return LSE_ERROR(kUnimplemented, "primitive '",
                           std::string(st.prim->name()),
                           "' declined to emit for this invocation");
        }
        // Sibling bodies share one thread index, so an early return in one
        // retires threads the next still needs — silently, as unwritten
        // output. A stage that wants to opt out of work must predicate, not
        // return. Asked of the IR now, not of the text.
        if (cap.body().contains(ir::OpKind::kReturn)) {
          return LSE_ERROR(kInternal, "primitive '",
                           std::string(st.prim->name()),
                           "' returns early and cannot be a fused sibling");
        }
        merged.splice(cap.body(), merged.entry());
      }

      // A workgroup barrier between stages, carrying no data dependence: the
      // shared row was published by the prologue and nothing here writes it.
      // It is a convoy for the weight stream. Left to themselves the waves of a
      // workgroup drift onto different stages, so the workgroup has one weight
      // matrix per stage in flight at once instead of one in total, and a
      // DRAM-bound GEMV pays for it. Measured, lemonseed decode at `-n 128` on
      // gfx1151, mean of 8 runs: 93.4 tok/s with, 92.6 without.
      //
      // This is why removing the per-stage barriers that `lds_fold` used to
      // leave behind was not the win it looked like.
      if (si + 1 < stages.size()) {
        const ir::RecordOptions opts{{}, {}, "c" + std::to_string(si) + "_"};
        const ir::KernelBody::Recording rec(opts);
        ir::KernelBody::Capture cap;
        {
          kir::KernelBody kb(type_table, spellings, 0);
          kb.barrier();
        }
        if (!cap.has()) {
          return LSE_ERROR(kInternal, "sibling convoy barrier did not record");
        }
        merged.splice(cap.body(), merged.entry());
      }

      const ThreadPlan& tp = st.plan;
      for (int d = 0; d < 3; ++d) {
        if (tp.workgroup_size[d] > unified.workgroup_size[d]) {
          unified.workgroup_size[d] = tp.workgroup_size[d];
        }
        if (tp.workgroup_count[d] > unified.workgroup_count[d]) {
          unified.workgroup_count[d] = tp.workgroup_count[d];
        }
      }
    }

    if (flat_of(unified.workgroup_size) != promised_threads) {
      return LSE_ERROR(kInternal,
                       "the fused run's launch geometry does not match the "
                       "bound its signature declares");
    }

    std::vector<ir::PassStat> pass_stats;
    if (const Status s = ir::default_pipeline().run(merged, &pass_stats);
        !s.ok()) {
      return LSE_ERROR(kInternal, "fused sibling body: ", s.message());
    }
    ir::record_pass_totals(pass_stats);

    // The run's real workgroup scratch, read off the body that is about to be
    // printed: every `__shared__` array it still declares, summed. Nothing here
    // depends on lds_fold having collapsed anything — an array it left standing
    // is still an array the compiler charges for, and it is counted.
    const std::uint32_t lds_emitted = merged.workgroup_bytes();
    if (lds_budget != 0 && lds_emitted > lds_budget) {
      lds_refused_.emplace(sig, lds_emitted);
      return LSE_ERROR(kOutOfMemory, "fused run needs ",
                       std::to_string(lds_emitted),
                       " bytes of workgroup scratch, device allows ",
                       std::to_string(lds_budget));
    }
    body << ir::lower(merged) << "\n}\n";
    out.source = body.str();
    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = unified.workgroup_size[d];
      out.dims.workgroup_count[d] = unified.workgroup_count[d];
    }
    out.lds_bytes = lds_emitted;
    out.dims.subgroup_size = device.wavefront_size;
    emit_cache_[sig] =
        CachedEmit{out.source, out.entry_name, out.dims, lds_emitted};
    return out;
  }

  // A primitive that maps threads itself writes the entire kernel body, so the
  // per-element scaffolding below does not apply. It still takes an epilogue:
  // it stores through a hook, so trailing elementwise work runs on the value in
  // register instead of in a second launch.
  //
  // This is also where a primitive gets to swap in a device-specific form, as
  // the matrix-core GEMM does — here and not earlier because the device is not
  // known until now.
  const KernelPrimitiveBase* self_indexed = nullptr;
  NodePtr anchor;
  std::vector<Shape> si_storage;
  std::vector<DType> si_dtypes;
  KernelShapes si_shapes;
  for (const NodePtr& n : group.nodes) {
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    if (kp == nullptr) continue;
    KernelShapes probe = shapes_for(n, si_storage, si_dtypes);
    const KernelPrimitiveBase* chosen = kp->specialize(probe);
    if (chosen == nullptr || !chosen->owns_indexing()) continue;
    // The hook stores one value at one index, so a group producing two live
    // outputs keeps the per-element path instead.
    if (group.outputs.size() != 1) break;
    self_indexed = chosen;
    anchor = n;
    si_shapes = probe;
    break;
  }

  const auto linked = kernels::linked_bindings(group);
  // Active only when the linked pipeline supplies a kernel. A matched-but-
  // declined pipeline (linked_kernel_for -> nullptr) must not clobber the
  // specialize() choice above: that turned the decode lm_head GEMV back into
  // the per-element scaffold on every replayed token.
  bool linked_active = false;
  if (linked.ok) {
    KernelShapes probe;
    if (const KernelPrimitiveBase* lk =
            kernels::linked_kernel_for(group, probe)) {
      linked_active = true;
      self_indexed = lk;
      for (const NodePtr& n : group.nodes) {
        if (n.get() == linked.sink) anchor = n;
      }
      si_storage.clear();
      si_dtypes.clear();
      for (const Node* in : linked.inputs) {
        si_storage.push_back(in->shape);
        si_dtypes.push_back(in->dtype);
      }
      si_shapes.inputs = si_storage;
      si_shapes.input_dtypes = si_dtypes;
      si_shapes.output = linked.sink != nullptr ? linked.sink->shape : Shape{};
      si_shapes.output_dtype =
          linked.sink != nullptr ? linked.sink->dtype : DType::kF32;
      si_shapes.attrs = linked.attrs;
      si_shapes.iattrs = linked.iattrs;
      si_shapes.device = &device;
      si_shapes.types = type_table;
      si_shapes.intrinsics = &spellings;
    }
  }

  EmittedKernel out;
  out.entry_name = "lse_fused_" + std::to_string(group.signature());
  if (self_indexed != nullptr) out.traffic = self_indexed->traffic(si_shapes);

  std::unordered_map<const Node*, std::size_t> binding_of;
  auto bind = [&](const NodePtr& n) {
    if (binding_of.emplace(n.get(), out.binding_order.size()).second) {
      out.binding_order.push_back(n);
    }
  };
  // A self-indexing primitive names its operands in0..inN in its own order, so
  // they are bound first and an epilogue's extra inputs cannot shift them.
  if (self_indexed != nullptr) {
    if (linked_active) {
      for (const Node* want : linked.inputs) {
        NodePtr found;
        for (const NodePtr& n : group.nodes) {
          if (n.get() == want) found = n;
          for (const NodePtr& in_n : n->inputs) {
            if (in_n.get() == want) found = in_n;
          }
        }
        // Skipping would shift every later slot while the kernel text keeps
        // its own numbering — a silent misbind, so it must be fatal.
        if (!found) {
          return LSE_ERROR(kInternal,
                           "linked pipeline input is outside the group closure");
        }
        bind(found);
      }
    } else if (anchor) {
      for (const NodePtr& in : anchor->inputs) bind(in);
    }
  }
  for (const NodePtr& in : group.inputs) bind(in);
  const std::size_t input_count = out.binding_order.size();

  std::unordered_set<const Node*> output_set;
  for (const NodePtr& o : group.outputs) {
    if (binding_of.emplace(o.get(), out.binding_order.size()).second) {
      out.binding_order.push_back(o);
    }
    output_set.insert(o.get());
  }

  out.constants.add("count", 4);

  // specialize() picks a different body (LDS vs WMMA vs scalar) for the same
  // graph node; the group hash only sees the generic primitive name.
  std::uint64_t sig = group.signature();
  if (self_indexed != nullptr) {
    sig ^= 0x9e3779b97f4a7c15ull;
    for (char c : self_indexed->name()) {
      sig ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
      sig *= 1099511628211ull;
    }
  }
  if (const auto it = emit_cache_.find(sig); it != emit_cache_.end()) {
    out.source = it->second.source;
    out.entry_name = it->second.entry_name;
    out.dims = it->second.dims;
    return out;
  }

  // A kernel primitive indexes its inputs itself, so those bindings must not
  // be pre-loaded as scalars at the output index — the shapes do not line up.
  std::unordered_set<const Node*> pointer_inputs;
  for (const NodePtr& n : group.nodes) {
    // repeat reads element `src(i)` of its input, not element i, so the
    // scaffold must not widen it at the launch index the way it does for an
    // elementwise operand. It loads for itself, below.
    if (n->kind == OpKind::kRepeat) {
      for (const NodePtr& in : n->inputs) pointer_inputs.insert(in.get());
      continue;
    }
    if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) == nullptr) continue;
    for (const NodePtr& in : n->inputs) pointer_inputs.insert(in.get());
  }

  // Deduplicated device preambles: several primitives from one source file
  // share helpers, so the file is emitted once.
  std::ostringstream preamble;
  std::unordered_set<std::string> seen_preambles;
  if (self_indexed != nullptr) {
    // The specialized primitive is the whole kernel; the node's original one
    // contributes nothing, not even a preamble.
    const std::string_view text = self_indexed->device_preamble();
    if (!text.empty()) preamble << text << "\n";
  } else {
    for (const NodePtr& n : group.nodes) {
      if (n->prim == nullptr) continue;
      const std::string_view text = n->prim->device_preamble();
      if (text.empty()) continue;
      if (!seen_preambles.emplace(n->prim->preamble_id()).second) continue;
      preamble << text << "\n";
    }

    for (const NodePtr& n : group.nodes) {
      const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
      if (kp == nullptr) continue;
      // A self-indexing primitive is the whole kernel body, not a helper the
      // loop calls, so it gets no device function.
      if (kp->owns_indexing()) continue;
      const std::string fn = device_fn_name(*kp, *n);
      if (!seen_preambles.emplace(fn).second) continue;

      std::vector<Shape> in_shapes;
      std::vector<DType> in_dtypes;
      const KernelShapes shapes = shapes_for(n, in_shapes, in_dtypes);

      preamble << "__device__ float " << fn
               << "(unsigned int i";
      for (std::size_t a = 0; a < n->inputs.size(); ++a) {
        preamble << ", const " << device_scalar(n->inputs[a]->dtype)
                 << "* __restrict__ in" << a;
      }
      preamble << ") {\n" << kp->emit_kernel(shapes) << "\n}\n\n";
    }
  }

  if (self_indexed != nullptr) {
    // Expands one of the primitive's stores into the fused epilogue followed by
    // the store. With nothing fused this is just the store, so the primitive is
    // written the same way either way.
    Status epilogue_error;
    bool stored = false;
    const NodePtr& sink = group.outputs.front();
    auto epilogue_store = [&](std::string_view index,
                              std::string_view value) -> std::string {
      stored = true;
      std::ostringstream s;
      std::unordered_map<const Node*, std::string> value_of;
      value_of[anchor.get()] = std::string(value);

      // Everything the epilogue reads is indexed at the *stored* element, not
      // at the thread id: a wave's lane owns a tile position, not element i.
      for (std::size_t j = 0; j < input_count; ++j) {
        const NodePtr& n = out.binding_order[j];
        if (pointer_inputs.count(n.get()) != 0) continue;
        const std::string var = "ep" + std::to_string(j);
        s << "const float " << var << " = "
          << load_expr("in" + std::to_string(j),
                       broadcast_index_expr(n->shape, sink->shape, index),
                       n->dtype)
          << ";\n";
        value_of[n.get()] = var;
      }

      std::size_t temp = 0;
      for (const NodePtr& n : group.nodes) {
        if (n.get() == anchor.get() || value_of.count(n.get())) continue;
        // A second kernel (GDN state, another GEMV) is not an epilogue op.
        if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) != nullptr) {
          continue;
        }

        if (n->kind == OpKind::kConstant) {
          const std::string var = "es" + std::to_string(temp++);
          s << "const float " << var << " = " << float_literal(n->attrs[0]) << ";\n";
          value_of[n.get()] = var;
          continue;
        }

        std::vector<std::string> args;
        args.reserve(n->inputs.size());
        for (const NodePtr& in : n->inputs) {
          auto it = value_of.find(in.get());
          if (it == value_of.end()) {
            epilogue_error = LSE_ERROR(kInternal, "value for an input of ",
                                        std::string(to_string(n->kind)),
                                        " is not available in the epilogue");
            return {};
          }
          args.push_back(it->second);
        }

        const std::string var = "es" + std::to_string(temp++);
        s << "float " << var << ";\n";
        if (n->prim != nullptr) {
          EmitContext ctx;
          ctx.inputs = args;
          ctx.out = var;
          ctx.device = &device;
          ctx.attrs = n->attrs;
          ctx.iattrs = n->iattrs;
          ctx.dialect = graph::Dialect::kHip;
          ctx.sources = &spellings;
          const std::string body = n->prim->emit_device(ctx);
          if (body.empty()) {
            epilogue_error = LSE_ERROR(kUnimplemented, "primitive '",
                                        std::string(n->prim->name()),
                                        "' has no HIP source");
            return {};
          }
          s << body << "\n";
        } else {
          const std::string_view tmpl = builtin_expr(n->kind);
          if (tmpl.empty()) {
            epilogue_error = LSE_ERROR(kUnimplemented, "no source template for ",
                                        std::string(to_string(n->kind)));
            return {};
          }
          s << var << " = " << substitute(tmpl, args) << ";\n";
        }
        value_of[n.get()] = var;
      }

      auto it = value_of.find(sink.get());
      if (it == value_of.end()) {
        epilogue_error = LSE_ERROR(kInternal, "group output was never computed");
        return {};
      }
      s << store_stmt("out", index, sink->dtype, it->second) << "\n";
      return s.str();
    };

    si_shapes.store = epilogue_store;
    const KernelShapes& shapes = si_shapes;
    const std::string self_body = self_indexed->emit_kernel(shapes);
    if (!epilogue_error.ok()) return epilogue_error;
    // An empty body would compile into a kernel that silently writes nothing.
    // Checked before the hook, since declining also means never storing.
    if (self_body.empty()) {
      return LSE_ERROR(kUnimplemented, "primitive '",
                       std::string(self_indexed->name()),
                       "' declined to emit for this invocation");
    }
    // A primitive that wrote the output buffer itself would drop the epilogue
    // on the floor and produce quietly wrong results, so require the hook.
    if (!stored) {
      return LSE_ERROR(kInternal, "primitive '",
                       std::string(self_indexed->name()),
                       "' owns its indexing but never stored through the hook");
    }

    // The thread map is the primitive's, not choose_dims': a tile-per-workgroup
    // layout has nothing to do with the output element count. Asked here and
    // not after the body, because the signature states it.
    const ThreadPlan tp = self_indexed->plan(shapes);

    std::ostringstream body;
    body << "#include <hip/hip_runtime.h>\n"
         << "#include <hip/hip_bf16.h>\n\n"
         << constants_decl(out.constants) << "\n"
         << preamble.str() << "\n"
         << kernel_signature(out.entry_name, flat_of(tp.workgroup_size));
    std::unordered_set<const Node*> group_members;
    for (const NodePtr& n : group.nodes) group_members.insert(n.get());
    const int inplace =
        (anchor && anchor->prim) ? anchor->prim->inplace_input() : -1;
    const Node* inplace_src =
        (inplace >= 0 && anchor &&
         static_cast<std::size_t>(inplace) < anchor->inputs.size())
            ? anchor->inputs[static_cast<std::size_t>(inplace)].get()
            : nullptr;
    for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
      const NodePtr& b = out.binding_order[i];
      const bool is_out = output_set.count(b.get()) != 0;
      // Linked scratch (the SwiGLU hidden) lives in the group and is written
      // by the staged kernel, so it cannot be a const binding.
      const bool scratch =
          linked_active && !is_out && group_members.count(b.get());
      const bool aliased_in = inplace_src != nullptr && b.get() == inplace_src;
      const bool writable = is_out || scratch || aliased_in;
      body << "    " << (writable ? "" : "const ")
           << device_scalar(b->dtype)
           << (aliased_in || (is_out && inplace_src != nullptr) ? "* "
                                                                : "* __restrict__ ")
           << (is_out ? "out" : "in" + std::to_string(i)) << ",\n";
    }
    body << "    LseConstants k) {\n"
         << "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
         << self_body << "\n}\n";
    out.source = body.str();

    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = tp.workgroup_size[d];
      out.dims.workgroup_count[d] = tp.workgroup_count[d];
    }
    // The primitive's own text, plus whatever the epilogue spliced in. A plan
    // that under-declared its scratch would be believed if this were tp's
    // number, and the epilogue is not in the plan at all.
    auto declared = shared_bytes(out.source);
    if (!declared.ok()) return declared.status();
    if (lds_budget != 0 && *declared > lds_budget) {
      return LSE_ERROR(kOutOfMemory, "'", std::string(self_indexed->name()),
                       "' declares ", std::to_string(*declared),
                       " bytes of workgroup scratch, device allows ",
                       std::to_string(lds_budget));
    }
    out.lds_bytes = *declared;
    out.dims.subgroup_size = device.wavefront_size;
    emit_cache_[sig] =
        CachedEmit{out.source, out.entry_name, out.dims, *declared};
    return out;
  }

  const Shape& out_shape = group.nodes.back()->shape;
  // The launch covers the widest thing in the group — choose_dims takes the
  // largest output, the walk below indexes against the last node — so every
  // narrower member has to be guarded against the same bound.
  std::size_t launch_elems = out_shape.elem_count();
  for (const NodePtr& o : group.outputs) {
    launch_elems = std::max(launch_elems, o->element_count());
  }

  // Two halves, joined once the launch geometry is known: choose_dims reads
  // the scratch the finished text declares, and the signature states what
  // choose_dims answered. Neither can be written before the other, so the
  // signature is the last thing assembled — no declaration lives in it, so
  // the scratch count is the same either way.
  std::ostringstream head;
  head << "#include <hip/hip_runtime.h>\n"
       << "#include <hip/hip_bf16.h>\n\n"
       << constants_decl(out.constants) << "\n"
       << preamble.str() << "\n";

  std::ostringstream src;
  for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
    const NodePtr& n = out.binding_order[i];
    const bool is_out = output_set.count(n.get()) != 0;
    src << "    " << (is_out ? "" : "const ") << device_scalar(n->dtype) << "* __restrict__ b"
        << i << ",\n";
  }
  src << "    LseConstants k) {\n"
      << "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      << "  if (i >= k.count) return;\n";

  // Values already computed in this kernel, keyed by node.
  std::unordered_map<const Node*, std::string> value_of;
  for (std::size_t i = 0; i < input_count; ++i) {
    const NodePtr& n = out.binding_order[i];
    if (pointer_inputs.count(n.get()) != 0) continue;
    const std::string var = "in" + std::to_string(i);
    const std::string idx = broadcast_index_expr(n->shape, out_shape, "i");
    src << "  const float " << var << " = "
        << load_expr("b" + std::to_string(i), idx, n->dtype) << ";\n";
    value_of[n.get()] = var;
  }

  std::size_t temp = 0;
  for (const NodePtr& n : group.nodes) {
    if (value_of.count(n.get())) continue;

    if (n->kind == OpKind::kConstant) {
      const std::string var = "t" + std::to_string(temp++);
      src << "  const float " << var << " = " << float_literal(n->attrs[0]) << ";\n";
      value_of[n.get()] = var;
      continue;
    }

    if (const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim)) {
      const std::string var = "t" + std::to_string(temp++);
      // The helper turns the flat thread id into a row and a column of *its*
      // output, so calling it past its own extent walks the activation off the
      // end and faults the device — guarding only the store is not enough.
      // A member narrower than the launch contributes nothing at those threads:
      // can_fuse keeps a kernel primitive's consumers the same width, so the
      // value is either stored under the same guard or never read.
      const std::size_t elems = n->element_count();
      const bool guarded = elems < launch_elems;
      src << "  const float " << var << " = ";
      if (guarded) src << "i < " << elems << "u ? ";
      src << device_fn_name(*kp, *n) << "(i";
      for (const NodePtr& in : n->inputs) {
        auto it = binding_of.find(in.get());
        if (it == binding_of.end()) {
          return LSE_ERROR(kInternal, std::string(kp->name()),
                           ": an input is not bound in this group");
        }
        src << ", b" << it->second;
      }
      src << ")";
      if (guarded) src << " : 0.0f";
      src << ";\n";
      value_of[n.get()] = var;
      continue;
    }

    if (n->kind == OpKind::kRepeat) {
      // Same map as repeat_stage in phase_emit: the axis is spread by `count`
      // and the elements under it ride along unchanged. Without this the op
      // has no per-element template at all and the whole group goes to the
      // host -- 864 of them per prefill pass on a 64 layer model, because
      // this is how grouped-query attention widens 4 key heads to 24.
      const Shape& sh = n->inputs[0]->shape;
      const auto axis = static_cast<std::size_t>(n->iattrs[0]);
      const auto count = static_cast<std::uint32_t>(n->iattrs[1]);
      if (axis >= sh.rank() || count == 0) {
        return LSE_ERROR(kInvalidArgument, "repeat has no axis to spread");
      }
      std::uint32_t inner = 1;
      std::uint32_t in_axis = 1;
      for (std::size_t d = 0; d < sh.rank(); ++d) {
        const auto dim = static_cast<std::uint32_t>(sh.dim(d));
        if (d > axis) inner *= dim;
        else if (d == axis) in_axis = dim;
      }
      if (inner == 0 || in_axis == 0) {
        return LSE_ERROR(kInvalidArgument, "repeat has an empty axis");
      }
      auto it = binding_of.find(n->inputs[0].get());
      if (it == binding_of.end()) {
        return LSE_ERROR(kInternal, "repeat input is not bound in this group");
      }
      const std::string var = "t" + std::to_string(temp++);
      const std::string ib = "b" + std::to_string(it->second);
      const std::string sv = var + "_src";
      // The launch covers the widest member of the group, so a thread past
      // this node's own extent must not map an index into its input: the map
      // is not monotone in i and lands well outside the buffer rather than
      // just past it.
      const auto elems = static_cast<std::uint32_t>(n->element_count());
      const bool guarded = elems < launch_elems;
      src << "  const unsigned int " << sv << " = ";
      if (guarded) src << "i < " << elems << "u ? ";
      src << "((i / " << (in_axis * count * inner) << "u) * " << in_axis
          << "u + ((i % " << (in_axis * count * inner) << "u) / " << inner
          << "u) / " << count << "u) * " << inner << "u + (i % " << inner
          << "u)";
      if (guarded) src << " : 0u";
      src << ";\n"
          << "  const float " << var << " = "
          << load_expr(ib, sv, n->inputs[0]->dtype) << ";\n";
      value_of[n.get()] = var;
      continue;
    }

    std::vector<std::string> args;
    args.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      auto it = value_of.find(in.get());
      if (it == value_of.end()) {
        return LSE_ERROR(kInternal, "value for an input of ",
                         std::string(to_string(n->kind)),
                         " is not available in this group");
      }
      args.push_back(it->second);
    }

    const std::string var = "t" + std::to_string(temp++);
    src << "  float " << var << ";\n";

    if (n->prim != nullptr) {
      if (!n->prim->has_device_impl()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(n->prim->name()),
                         "' has no device implementation");
      }
      EmitContext ctx;
      ctx.inputs = args;
      ctx.out = var;
      ctx.device = &device;
      ctx.attrs = n->attrs;
      ctx.iattrs = n->iattrs;
      ctx.dialect = graph::Dialect::kHip;
      const graph::DialectSourceTable table = sources();
      ctx.sources = &table;
      const std::string body = n->prim->emit_device(ctx);
      if (body.empty()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(n->prim->name()),
                         "' has no HIP source");
      }
      src << "  " << body << "\n";
    } else {
      const std::string_view tmpl = builtin_expr(n->kind);
      if (tmpl.empty()) {
        return LSE_ERROR(kUnimplemented, "no source template for ",
                         std::string(to_string(n->kind)));
      }
      src << "  " << var << " = " << substitute(tmpl, args) << ";\n";
    }
    value_of[n.get()] = var;
  }

  for (std::size_t i = input_count; i < out.binding_order.size(); ++i) {
    const NodePtr& n = out.binding_order[i];
    auto it = value_of.find(n.get());
    if (it == value_of.end()) {
      return LSE_ERROR(kInternal, "group output was never computed");
    }
    // The launch covers the largest output in the group, so a smaller one has
    // to be guarded: writing it at every i walks off the end of its buffer and
    // faults the device. Its extent is a literal — shapes are in the cache key.
    const std::size_t elems = n->element_count();
    const bool needs_guard = elems < launch_elems;
    if (needs_guard) src << "  if (i < " << elems << "u) ";
    else src << "  ";
    src << store_stmt("b" + std::to_string(i), "i", n->dtype, it->second)
        << "\n";
  }

  src << "}\n";

  // Measured, not left at whatever `out.lds_bytes` was initialized to: this is
  // the one place choose_dims reads it, and it used to be structurally 0 here
  // because every path that assigns it returns before reaching this line.
  auto declared = shared_bytes(head.str() + src.str());
  if (!declared.ok()) return declared.status();
  if (lds_budget != 0 && *declared > lds_budget) {
    return LSE_ERROR(kOutOfMemory, "element scaffold declares ",
                     std::to_string(*declared),
                     " bytes of workgroup scratch, device allows ",
                     std::to_string(lds_budget));
  }
  out.lds_bytes = *declared;
  out.dims = choose_dims(group, device, out.lds_bytes);
  out.source = head.str() +
               kernel_signature(out.entry_name, flat_of(out.dims.workgroup_size)) +
               src.str();
  emit_cache_[sig] =
      CachedEmit{out.source, out.entry_name, out.dims, *declared};
  return out;
}

}  // namespace lse::backend
