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
#include "kernels/linked.hpp"

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
};

std::vector<IndexedStage> indexed_stages(const FusionGroup& group,
                                         const DeviceInfo& device) {
  const graph::DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();
  std::vector<IndexedStage> out;
  std::vector<Shape> storage;
  for (const NodePtr& n : group.nodes) {
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    if (kp == nullptr) continue;
    storage.clear();
    storage.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) storage.push_back(in->shape);
    KernelShapes probe;
    probe.inputs = storage;
    probe.output = n->shape;
    probe.attrs = n->attrs;
    probe.iattrs = n->iattrs;
    probe.device = &device;
    probe.types = type_table;
    probe.intrinsics = &spellings;
    const KernelPrimitiveBase* chosen = kp->specialize(probe);
    if (chosen == nullptr || !chosen->owns_indexing()) continue;
    out.push_back(IndexedStage{n, chosen});
  }
  return out;
}

bool sibling_stages(const FusionGroup& group,
                    const std::vector<IndexedStage>& stages) {
  if (stages.size() < 2) return false;
  std::unordered_set<const Node*> stage_set;
  std::unordered_set<const Node*> outputs;
  for (const IndexedStage& s : stages) stage_set.insert(s.node.get());
  for (const NodePtr& o : group.outputs) outputs.insert(o.get());
  for (const IndexedStage& s : stages) {
    if (!outputs.count(s.node.get())) return false;
    for (const NodePtr& in : s.node->inputs) {
      if (stage_set.count(in.get())) return false;
    }
  }
  return true;
}

std::string_view device_scalar(DType dt) noexcept {
  switch (dt) {
    case DType::kF16: return "_Float16";
    case DType::kBF16: return "__hip_bfloat16";
    case DType::kI32: return "int";
    case DType::kI8: return "signed char";
    case DType::kU8: return "unsigned char";
    default: return "float";
  }
}

// Loads always widen to float and stores narrow back, so the generated body is
// written once in float regardless of storage dtype.
std::string load_expr(std::string_view buffer, std::string_view index, DType dt) {
  std::string out;
  switch (dt) {
    case DType::kBF16:
      out = "__bfloat162float(" + std::string(buffer) + "[" + std::string(index) + "])";
      break;
    case DType::kF16:
      out = "(float)" + std::string(buffer) + "[" + std::string(index) + "]";
      break;
    default:
      out = "(float)" + std::string(buffer) + "[" + std::string(index) + "]";
      break;
  }
  return out;
}

std::string store_stmt(std::string_view buffer, std::string_view index, DType dt,
                       std::string_view value) {
  std::string out(buffer);
  out += "[";
  out += index;
  out += "] = ";
  switch (dt) {
    case DType::kBF16:
      out += "__float2bfloat16(" + std::string(value) + ")";
      break;
    case DType::kF16:
      out += "(_Float16)(" + std::string(value) + ")";
      break;
    default:
      out += "(" + std::string(device_scalar(dt)) + ")(" + std::string(value) + ")";
      break;
  }
  out += ";";
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
      (amd != nullptr && amd->wavefront_size != 0) ? amd->wavefront_size : 64;
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
  } else {
    const graph::DialectSourceTable spellings = hip_sources();
    const kir::TypeTable type_table = hip_types();
    std::vector<Shape> storage;
    for (const NodePtr& n : group.nodes) {
      const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
      if (kp == nullptr) continue;
      storage.clear();
      for (const NodePtr& in : n->inputs) storage.push_back(in->shape);
      KernelShapes probe;
      probe.inputs = storage;
      probe.output = n->shape;
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
  if (const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(device)) {
    h ^= amd->wavefront_size;
    h *= 1099511628211ull;
  }
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

  // The span points into `storage`, which the caller must keep alive.
  const graph::DialectSourceTable spellings = sources();
  const kir::TypeTable type_table = hip_types();
  auto shapes_for = [&](const NodePtr& n, std::vector<Shape>& storage) {
    storage.clear();
    storage.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) storage.push_back(in->shape);
    KernelShapes s;
    s.inputs = storage;
    s.output = n->shape;
    s.attrs = n->attrs;
    s.iattrs = n->iattrs;
    s.device = &device;
    s.types = type_table;
    s.intrinsics = &spellings;
    return s;
  };

  const auto stages = indexed_stages(group, device);
  if (sibling_stages(group, stages)) {
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

    ThreadPlan unified;
    unified.workgroup_size[0] = 1;
    unified.workgroup_count[0] = 1;
    for (const IndexedStage& st : stages) {
      std::vector<Shape> storage;
      KernelShapes shapes = shapes_for(st.node, storage);
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
      const std::string stage_body = st.prim->emit_kernel(shapes);
      if (stage_body.empty()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(st.prim->name()),
                         "' declined to emit for this invocation");
      }
      body << "  {\n";
      for (std::size_t a = 0; a < st.node->inputs.size(); ++a) {
        auto iit = binding_of.find(st.node->inputs[a].get());
        if (iit == binding_of.end()) {
          return LSE_ERROR(kInternal, std::string(st.prim->name()),
                           ": an input is not bound in this group");
        }
        body << "    const " << device_scalar(st.node->inputs[a]->dtype)
             << "* in" << a << " = b" << iit->second << ";\n";
      }
      body << stage_body << "\n  }\n";

      const ThreadPlan tp = st.prim->plan(shapes);
      for (int d = 0; d < 3; ++d) {
        if (tp.workgroup_size[d] > unified.workgroup_size[d]) {
          unified.workgroup_size[d] = tp.workgroup_size[d];
        }
        if (tp.workgroup_count[d] > unified.workgroup_count[d]) {
          unified.workgroup_count[d] = tp.workgroup_count[d];
        }
      }
      if (tp.lds_bytes > unified.lds_bytes) unified.lds_bytes = tp.lds_bytes;
    }
    body << "}\n";
    out.source = body.str();
    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = unified.workgroup_size[d];
      out.dims.workgroup_count[d] = unified.workgroup_count[d];
    }
    out.lds_bytes = unified.lds_bytes;
    if (const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(device)) {
      out.dims.subgroup_size = amd->wavefront_size;
    }
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
  KernelShapes si_shapes;
  for (const NodePtr& n : group.nodes) {
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    if (kp == nullptr) continue;
    KernelShapes probe = shapes_for(n, si_storage);
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
  if (linked.ok) {
    KernelShapes probe;
    self_indexed = hrx_kernels::linked_kernel_for(group, probe);
    for (const NodePtr& n : group.nodes) {
      if (n.get() == linked.sink) anchor = n;
    }
    si_storage.clear();
    for (const Node* in : linked.inputs) si_storage.push_back(in->shape);
    si_shapes.inputs = si_storage;
    si_shapes.output = linked.sink != nullptr ? linked.sink->shape : Shape{};
    si_shapes.attrs = linked.attrs;
    si_shapes.iattrs = linked.iattrs;
    si_shapes.device = &device;
    si_shapes.types = type_table;
    si_shapes.intrinsics = &spellings;
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
    if (linked.ok) {
      for (const Node* want : linked.inputs) {
        NodePtr found;
        for (const NodePtr& n : group.nodes) {
          if (n.get() == want) found = n;
          for (const NodePtr& in_n : n->inputs) {
            if (in_n.get() == want) found = in_n;
          }
        }
        if (found) bind(found);
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
      if (!seen_preambles.emplace(std::string(kp->entry_name())).second) continue;

      std::vector<Shape> in_shapes;
      const KernelShapes shapes = shapes_for(n, in_shapes);

      preamble << "__device__ float " << kp->entry_name()
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
      const bool scratch = linked.ok && !is_out && group_members.count(b.get());
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
    if (const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(device)) {
      out.dims.subgroup_size = amd->wavefront_size;
    }
    emit_cache_[sig] = CachedEmit{out.source, out.entry_name, out.dims};
    return out;
  }

  const Shape& out_shape = group.nodes.back()->shape;

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
      src << "  const float " << var << " = " << kp->entry_name() << "(i";
      for (const NodePtr& in : n->inputs) {
        auto it = binding_of.find(in.get());
        if (it == binding_of.end()) {
          return LSE_ERROR(kInternal, std::string(kp->name()),
                           ": an input is not bound in this group");
        }
        src << ", b" << it->second;
      }
      src << ");\n";
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
    const bool needs_guard = elems < out_shape.elem_count();
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
