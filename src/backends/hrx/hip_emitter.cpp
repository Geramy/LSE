#include "lse/backends/hrx/hip_emitter.hpp"

#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hip_types.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ir/lower.hpp"
#include "lse/ir/pass/pass.hpp"
#include "lse/backends/hrx/kernels/lds_linear.hpp"
#include "lse/backends/hrx/kernels/linked.hpp"

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
  ThreadPlan plan;
  // The activation panel this stage puts in workgroup scratch, as the stage
  // itself declares it. Asked at specialize time so the run can share it.
  KernelPrimitiveBase::StagedRow row;
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
      panels.push_back(RowPanel{act, st.row.count, st.row.rows, 0, {}});
    }
    ++panels[at].members;
    panel_of[si] = at;
  }
  return panels;
}

// What the run's workgroup scratch actually costs: every distinct row once,
// plus whatever a stage that shares nothing takes on its own.
std::uint32_t run_lds_bytes(const std::vector<IndexedStage>& stages,
                            const std::vector<RowPanel>& panels) {
  std::uint32_t bytes = 0;
  for (const RowPanel& p : panels) {
    bytes += kir::Lds::align(p.count * kir::pack_elem_bytes<kir::f32>());
  }
  for (const IndexedStage& st : stages) {
    if (st.row.count == 0) bytes += kir::Lds::align(st.plan.lds_bytes);
  }
  return bytes;
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

std::vector<IndexedStage> indexed_stages(const FusionGroup& group,
                                         const DeviceInfo& device) {
  const graph::DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();
  std::vector<IndexedStage> out;
  std::vector<Shape> storage;
  std::vector<DType> dtypes;
  for (const NodePtr& n : group.nodes) {
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    if (kp == nullptr) continue;
    storage.clear();
    dtypes.clear();
    storage.reserve(n->inputs.size());
    dtypes.reserve(n->inputs.size());
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
    out.push_back(
        IndexedStage{n, chosen, chosen->plan(probe), chosen->staged_row(probe)});
  }
  return out;
}

bool sibling_stages(const FusionGroup& group,
                    const std::vector<IndexedStage>& stages,
                    std::uint32_t lds_needed, std::uint32_t lds_budget) {
  if (stages.size() < 2) return false;
  // One launch, one workgroup scratch allocation, so the run's rows are summed
  // and not maximized. Nothing checked this before the rows were shared by
  // construction, because the fold pass reduced them to one afterwards and one
  // always fit; a run whose distinct rows do not fit cannot be launched at all,
  // so it goes back to a group per node the same way an uncovered run does.
  if (lds_budget != 0 && lds_needed > lds_budget) return false;
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

LaunchDims HipEmitter::choose_dims(const FusionGroup& group,
                                   const DeviceInfo& device,
                                   std::uint32_t lds_bytes) {
  std::size_t elements = 1;
  for (const NodePtr& out : group.outputs) {
    elements = std::max(elements, out->element_count());
  }
  if (group.outputs.empty() && !group.nodes.empty()) {
    elements = group.nodes.back()->element_count();
  }

  // Pick the workgroup size with the best occupancy among wavefront multiples
  // the device actually permits.
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(device);
  const std::uint32_t wave =
      device.wavefront_size != 0 ? device.wavefront_size : 64;
  const std::uint32_t cap =
      device.max_threads_per_workgroup ? device.max_threads_per_workgroup : 256;

  // Every legal workgroup size keeps the same number of threads resident on a
  // CU — occupancy_per_cu counts *workgroups*, and workgroups x threads is
  // constant — so maximizing it just picks the smallest size every time. That
  // is how a 254M-element fill came to launch 7.9M workgroups of 32 threads.
  // What actually differs is the launch count, so take the largest size that
  // still fits, bounded by the work there is to do.
  std::uint32_t needed = wave;
  while (needed < elements && needed < cap) needed *= 2;

  std::uint32_t best = wave;
  for (std::uint32_t threads = wave; threads <= cap && threads <= needed;
       threads *= 2) {
    const std::uint32_t occ =
        amd != nullptr ? occupancy_per_cu(device, threads, lds_bytes) : 1;
    if (occ > 0) best = threads;
  }

  LaunchDims dims;
  dims.workgroup_size[0] = best;
  dims.workgroup_count[0] =
      static_cast<std::uint32_t>((elements + best - 1) / best);
  dims.subgroup_size = wave;
  return dims;
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
  if (hrx_kernels::linked_bindings(group).ok) {
    KernelShapes dummy;
    self = hrx_kernels::linked_kernel_for(group, dummy);
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
      out.scratch_bytes = it->second.scratch_bytes;
      out.persist_grid = it->second.persist_grid;
      bind_phase(group, out);
      return out;
    }
    auto emitted = emit_phase(group, device);
    if (emitted.ok()) {
      emit_cache_[key] = CachedEmit{emitted->source, emitted->entry_name,
                                    emitted->dims, emitted->scratch_bytes,
                                    emitted->persist_grid};
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

  const auto stages = indexed_stages(group, device);
  std::vector<std::size_t> panel_of;
  std::vector<RowPanel> panels = row_panels(stages, panel_of);
  const std::uint32_t lds_budget = workgroup_lds_bytes(&device);
  const std::uint32_t lds_needed = run_lds_bytes(stages, panels);
  if (sibling_stages(group, stages, lds_needed, lds_budget)) {
    EmittedKernel out;
    out.entry_name = "lse_fused_" + std::to_string(group.signature());

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

    std::uint64_t sig = group.signature();
    for (const IndexedStage& st : stages) mix_name(sig, st.prim->name());
    if (const auto it = emit_cache_.find(sig); it != emit_cache_.end()) {
      out.source = it->second.source;
      out.entry_name = it->second.entry_name;
      out.dims = it->second.dims;
      out.lds_bytes = lds_needed;
      return out;
    }

    std::ostringstream body;
    body << "#include <hip/hip_runtime.h>\n"
         << "#include <hip/hip_bf16.h>\n\n"
         << constants_decl(out.constants) << "\n"
         << "extern \"C\" __global__ void " << out.entry_name << "(\n";
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
      if (panel.members < 2) continue;
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
        panel.name = hrx_kernels::emit_staged_row(kb, act, panel.count,
                                                  panel.rows, run_block);
      }
      if (panel.name.empty() || !cap.has()) {
        return LSE_ERROR(kInternal, "the shared activation row did not stage");
      }
      merged.splice(cap.body(), merged.entry());
    }

    ThreadPlan unified;
    unified.workgroup_size[0] = 1;
    unified.workgroup_count[0] = 1;
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

    std::vector<ir::PassStat> pass_stats;
    if (const Status s = ir::default_pipeline().run(merged, &pass_stats);
        !s.ok()) {
      return LSE_ERROR(kInternal, "fused sibling body: ", s.message());
    }
    ir::record_pass_totals(pass_stats);
    body << ir::lower(merged) << "\n}\n";
    out.source = body.str();
    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = unified.workgroup_size[d];
      out.dims.workgroup_count[d] = unified.workgroup_count[d];
    }
    // The run's rows, each counted once — not the widest stage's, which is what
    // a max over the stage plans would have said.
    out.lds_bytes = lds_needed;
    out.dims.subgroup_size = device.wavefront_size;
    emit_cache_[sig] = CachedEmit{out.source, out.entry_name, out.dims};
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

  const auto linked = hrx_kernels::linked_bindings(group);
  // Active only when the linked pipeline supplies a kernel. A matched-but-
  // declined pipeline (linked_kernel_for -> nullptr) must not clobber the
  // specialize() choice above: that turned the decode lm_head GEMV back into
  // the per-element scaffold on every replayed token.
  bool linked_active = false;
  if (linked.ok) {
    KernelShapes probe;
    if (const KernelPrimitiveBase* lk =
            hrx_kernels::linked_kernel_for(group, probe)) {
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

    std::ostringstream body;
    body << "#include <hip/hip_runtime.h>\n"
         << "#include <hip/hip_bf16.h>\n\n"
         << constants_decl(out.constants) << "\n"
         << preamble.str() << "\n"
         << "extern \"C\" __global__ void " << out.entry_name << "(\n";
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

    // The thread map is the primitive's, not choose_dims': a tile-per-workgroup
    // layout has nothing to do with the output element count.
    const ThreadPlan tp = self_indexed->plan(shapes);
    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = tp.workgroup_size[d];
      out.dims.workgroup_count[d] = tp.workgroup_count[d];
    }
    out.lds_bytes = tp.lds_bytes;
    out.dims.subgroup_size = device.wavefront_size;
    emit_cache_[sig] = CachedEmit{out.source, out.entry_name, out.dims};
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

  std::ostringstream src;
  src << "#include <hip/hip_runtime.h>\n"
      << "#include <hip/hip_bf16.h>\n\n"
      << constants_decl(out.constants) << "\n"
      << preamble.str() << "\n"
      << "extern \"C\" __global__ void " << out.entry_name << "(\n";

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

  out.source = src.str();
  out.dims = choose_dims(group, device, out.lds_bytes);
  emit_cache_[sig] = CachedEmit{out.source, out.entry_name, out.dims};
  return out;
}

}  // namespace lse::backend
