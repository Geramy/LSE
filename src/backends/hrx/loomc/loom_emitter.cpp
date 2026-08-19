#include "lse/backends/hrx/loomc/loom_emitter.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/linked.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"

namespace lse::backend {

using namespace lse::graph;

namespace {

kir::Scalar elem_of(DType dt) {
  switch (dt) {
    case DType::kF16: return kir::Scalar::kF16;
    case DType::kBF16: return kir::Scalar::kBF16;
    case DType::kI32: return kir::Scalar::kI32;
    case DType::kI8: return kir::Scalar::kI8;
    case DType::kU8: return kir::Scalar::kU8;
    case DType::kU32: return kir::Scalar::kU32;
    default: return kir::Scalar::kF32;
  }
}

// A buffer dtype this dialect can view. The block-quantized tags are a layout
// rather than an element type and have no Loom view; a group that binds one
// declines rather than being viewed as bytes it is not.
bool viewable(DType dt) {
  switch (dt) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kI32:
    case DType::kI8:
    case DType::kU8:
    case DType::kU32:
      return true;
    default:
      return false;
  }
}

// Mints the names the emitter itself introduces. One counter per kernel so a
// name cannot repeat, which in SSA is not a shadow but a redefinition error.
class Mint {
 public:
  std::string operator()(std::string_view stem) {
    return "%k" + std::string(stem) + std::to_string(n_++);
  }

 private:
  std::uint32_t n_ = 0;
};

// A value the graph carries as f32, from a buffer that may be narrower. Loads
// widen and stores narrow, exactly as the HIP emitter's load_expr/store_stmt
// do, so a body is written once in f32 regardless of storage.
std::string widen_to_f32(std::string value, DType dt, Mint& mint,
                         std::string& out) {
  switch (dt) {
    case DType::kF32:
      return value;
    case DType::kF16:
    case DType::kBF16: {
      const std::string r = mint("wide");
      out += "  " + r + " = scalar.extf " + value + " : " +
             std::string(loom_storage_type(elem_of(dt))) + " to f32\n";
      return r;
    }
    case DType::kI8:
    case DType::kU8: {
      const std::string w = mint("wide");
      const bool sgn = dt == DType::kI8;
      out += "  " + w + " = scalar." + (sgn ? "extsi" : "extui") + " " + value +
             " : i8 to i32\n";
      const std::string r = mint("wide");
      out += "  " + r + " = scalar." + (sgn ? "sitofp" : "uitofp") + " " + w +
             " : i32 to f32\n";
      return r;
    }
    case DType::kI32:
    case DType::kU32: {
      const std::string r = mint("wide");
      out += "  " + r + " = scalar." +
             (dt == DType::kI32 ? "sitofp" : "uitofp") + " " + value +
             " : i32 to f32\n";
      return r;
    }
    default:
      return value;
  }
}

std::string narrow_from_f32(std::string value, DType dt, Mint& mint,
                            std::string& out) {
  switch (dt) {
    case DType::kF32:
      return value;
    case DType::kF16:
    case DType::kBF16: {
      const std::string r = mint("narrow");
      out += "  " + r + " = scalar.fptrunc " + value + " : f32 to " +
             std::string(loom_storage_type(elem_of(dt))) + "\n";
      return r;
    }
    case DType::kI8:
    case DType::kU8: {
      const bool sgn = dt == DType::kI8;
      const std::string w = mint("narrow");
      out += "  " + w + " = scalar." + (sgn ? "fptosi" : "fptoui") + " " +
             value + " : f32 to i32\n";
      const std::string r = mint("narrow");
      out += "  " + r + " = scalar.trunci " + w + " : i32 to i8\n";
      return r;
    }
    case DType::kI32:
    case DType::kU32: {
      const std::string r = mint("narrow");
      out += "  " + r + " = scalar." +
             (dt == DType::kI32 ? "fptosi" : "fptoui") + " " + value +
             " : f32 to i32\n";
      return r;
    }
    default:
      return value;
  }
}

// The address a Loom memory op reads, with the range the caller is required to
// have honoured stated where Loom can use it. A view is bounded and every
// access is proven against that bound; our index arithmetic runs on a signed
// carrier and proves nothing on its own. The C kernel has the same
// precondition and does not say it — an index past the extent there is a wild
// store rather than a diagnostic — so this moves where the contract is
// written, not what it is.
std::string bounded(const std::string& index, std::uint64_t extent,
                    Mint& mint, std::string& out) {
  const std::string r = mint("at");
  out += "  " + r + " = index.assume " + index + " [range(" + index + ", 0, " +
         std::to_string(extent == 0 ? 0 : extent - 1) + ")] : index\n";
  return r;
}

// BroadcastMap::apply, as Loom index arithmetic. The same map the host
// interpreter uses, so the two index the same element; the HIP emitter builds
// the same expression out of infix operators.
std::string broadcast_index(const Shape& src, const Shape& out_shape,
                            const std::string& flat, Mint& mint,
                            std::string& out) {
  const BroadcastMap m = BroadcastMap::build(src, out_shape);
  if (m.identity) return flat;
  if (m.scalar) {
    const std::string z = mint("bz");
    out += "  " + z + " = index.constant 0 : index\n";
    return z;
  }
  std::string sum;
  for (std::size_t i = m.gap; i < m.rank; ++i) {
    const std::size_t si = i - m.gap;
    if (m.src_stride[si] == 0) continue;
    const std::string cs = mint("bs");
    const std::string cd = mint("bd");
    const std::string ct = mint("bt");
    out += "  " + cs + " = index.constant " +
           std::to_string(m.out_stride[i]) + " : index\n";
    out += "  " + cd + " = index.constant " + std::to_string(m.out_dim[i]) +
           " : index\n";
    out += "  " + ct + " = index.constant " + std::to_string(m.src_stride[si]) +
           " : index\n";
    const std::string q = mint("bq");
    const std::string r = mint("br");
    const std::string p = mint("bp");
    out += "  " + q + " = index.div " + flat + ", " + cs + " : index\n";
    out += "  " + r + " = index.rem " + q + ", " + cd + " : index\n";
    out += "  " + p + " = index.mul " + r + ", " + ct + " : index\n";
    if (sum.empty()) {
      sum = p;
    } else {
      const std::string a = mint("ba");
      out += "  " + a + " = index.add " + sum + ", " + p + " : index\n";
      sum = a;
    }
  }
  if (sum.empty()) {
    const std::string z = mint("bz");
    out += "  " + z + " = index.constant 0 : index\n";
    sum = z;
  }
  return sum;
}

// Workgroup size and launch count for a group with no primitive of its own.
//
// This is HipEmitter::choose_dims' policy, restated: the two toolchains must
// choose the same launch for the same group or a differential comparison is
// comparing two different kernels, and the occupancy reasoning behind it is
// device policy rather than a property of either language. It is duplicated
// rather than shared because the shared half of the two emitters has not been
// extracted yet; a test pins the two answers together so a drift shows up at
// build time.
LaunchDims choose_dims(const FusionGroup& group, const DeviceInfo& device,
                       std::uint32_t lds_bytes) {
  std::size_t elements = 1;
  for (const NodePtr& out : group.outputs) {
    elements = std::max(elements, out->element_count());
  }
  if (group.outputs.empty() && !group.nodes.empty()) {
    elements = group.nodes.back()->element_count();
  }
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(device);
  const std::uint32_t wave =
      device.wavefront_size != 0 ? device.wavefront_size : 64;
  const std::uint32_t cap =
      device.max_threads_per_workgroup ? device.max_threads_per_workgroup : 256;
  std::uint32_t needed = wave;
  while (needed < elements && needed < cap) needed *= 2;
  std::uint32_t best = wave;
  for (std::uint32_t threads = wave; threads <= cap && threads <= needed;
       threads *= 2) {
    const std::uint32_t occ =
        amd != nullptr ? occupancy_per_lds_pool(device, threads, lds_bytes) : 1;
    if (occ > 0) best = threads;
  }
  LaunchDims dims;
  dims.workgroup_size[0] = best;
  dims.workgroup_count[0] =
      static_cast<std::uint32_t>((elements + best - 1) / best);
  dims.subgroup_size = wave;
  return dims;
}

// The kernel header: the config region that states the launch, and the
// parameter list that IS the kernarg layout. Declaration order is the layout —
// buffers in binding order, then one by_value per ConstantsLayout field, both
// packed with no padding on our side — so this and EmittedKernel must agree by
// construction rather than by convention.
std::string kernel_header(const EmittedKernel& out,
                          const std::vector<std::string>& params) {
  std::string s;
  s += "kernel.def export(\"" + out.entry_name + "\") @" + out.entry_name +
       "() {\n";
  s += "  %cfg_one = index.constant 1 : index\n";
  for (int d = 0; d < 3; ++d) {
    s += "  %cfg_wg" + std::to_string(d) + " = index.constant " +
         std::to_string(std::max(1u, out.dims.workgroup_size[d])) +
         " : index\n";
    s += "  %cfg_n" + std::to_string(d) + " = index.constant " +
         std::to_string(std::max(1u, out.dims.workgroup_count[d])) +
         " : index\n";
  }
  s += "  kernel.launch.config workgroups(%cfg_n0, %cfg_n1, %cfg_n2) "
       "workgroup_size(%cfg_wg0, %cfg_wg1, %cfg_wg2) : index\n";
  s += "} launch(";
  for (std::size_t i = 0; i < params.size(); ++i) {
    if (i != 0) s += ", ";
    s += params[i];
  }
  s += ") {\n";
  return s;
}

}  // namespace

std::uint64_t LoomEmitter::cache_key(const FusionGroup& group,
                                     const DeviceInfo& device) const {
  // Same group, same device, different language: the emit cache and the JIT
  // cache both key on this, so the dialect has to be in it or the two
  // toolchains serve each other's objects.
  std::uint64_t h = IKernelEmitter::cache_key(group, device);
  h ^= static_cast<std::uint64_t>(Dialect::kLoom) + 0x9e3779b97f4a7c15ull;
  h *= 1099511628211ull;
  return h;
}

Result<EmittedKernel> LoomEmitter::emit(const FusionGroup& group,
                                        const DeviceInfo& device) const {
  if (group.nodes.empty()) {
    return LSE_ERROR(kInvalidArgument, "cannot emit an empty fusion group");
  }
  if (group.anchor_class == FusionClass::kCollective) {
    return LSE_ERROR(kUnimplemented, "group anchored on ",
                     std::string(to_string(group.anchor)),
                     " is a transport operation, not generated source");
  }
  if (group.is_phase) {
    // A phase body concatenates staged bodies under one launch and is the
    // shape that becomes a resident grid the moment `persist` in
    // hipc/phase_emit.cpp stops being hardwired false. Loom refuses grid-wide
    // synchronization by design and answers with a grid-contract diagnostic,
    // so this dialect can never carry the persistent-grid path — the decline
    // is structural, not a gap waiting to be filled.
    return LSE_ERROR(kUnimplemented,
                     "a staged phase body has no Loom form: its dependent "
                     "stages need a grid-wide barrier, which Loom's grid "
                     "contract refuses");
  }

  const DialectSourceTable spellings = sources();
  const kir::TypeTable type_table = loom_types();

  std::vector<Shape> storage;
  std::vector<DType> dtypes;
  auto shapes_for = [&](const NodePtr& n, std::vector<Shape>& shp,
                        std::vector<DType>& dts) {
    shp.clear();
    dts.clear();
    shp.reserve(n->inputs.size());
    dts.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      shp.push_back(in->shape);
      dts.push_back(in->dtype);
    }
    KernelShapes s;
    s.inputs = shp;
    s.input_dtypes = dts;
    s.output = n->shape;
    s.output_dtype = n->dtype;
    s.attrs = n->attrs;
    s.iattrs = n->iattrs;
    s.device = &device;
    s.types = type_table;
    s.intrinsics = &spellings;
    return s;
  };

  if (kernels::linked_bindings(group).ok) {
    KernelShapes probe;
    if (kernels::linked_kernel_for(group, probe) != nullptr) {
      return LSE_ERROR(kUnimplemented,
                       "the linked pipeline stages its activation in workgroup "
                       "scratch across a barrier and carries accumulators "
                       "across a loop; neither has a Loom form here yet");
    }
  }

  // A primitive that maps threads itself writes the whole body and stores
  // through the hook, exactly as in the HIP emitter.
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
    if (group.outputs.size() != 1) break;
    self_indexed = chosen;
    anchor = n;
    si_shapes = probe;
    break;
  }

  EmittedKernel out;
  out.dialect = Dialect::kLoom;
  out.entry_name = "lse_loom_" + std::to_string(group.signature());

  std::unordered_map<const Node*, std::size_t> binding_of;
  auto bind = [&](const NodePtr& n) {
    if (binding_of.emplace(n.get(), out.binding_order.size()).second) {
      out.binding_order.push_back(n);
    }
  };
  // A self-indexing primitive names its operands in0..inN in its own order, so
  // they are bound first and an epilogue's extra inputs cannot shift them.
  if (self_indexed != nullptr && anchor) {
    for (const NodePtr& in : anchor->inputs) bind(in);
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

  // The launch parameters, and the views the body reads through. Parameter
  // names follow the emitter convention the recorder already binds against:
  // `in0..inN` and `out` for a self-indexed body, `b0..bN` for the per-element
  // scaffold, which is what hipc names them too.
  const bool si = self_indexed != nullptr;
  std::vector<std::string> params;
  std::vector<std::string> names;
  std::unordered_map<std::string, LoomBufferView> views;
  std::string prologue;
  for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
    const NodePtr& n = out.binding_order[i];
    if (!viewable(n->dtype)) {
      return LSE_ERROR(kUnimplemented, "binding ", std::to_string(i),
                       " has dtype ", std::string(to_string(n->dtype)),
                       ", which is a block layout rather than a Loom element "
                       "type");
    }
    const bool is_out = output_set.count(n.get()) != 0;
    const std::string name =
        si ? (is_out ? std::string("out") : "in" + std::to_string(i))
           : "b" + std::to_string(i);
    names.push_back(name);
    params.push_back("%" + name + ": buffer");
    views[name] = LoomBufferView{elem_of(n->dtype), n->element_count(),
                                 "%" + name + "_view"};
  }
  for (const ConstantsLayout::Field& f : out.constants.fields) {
    params.push_back("%" + f.name + ": " + (f.size == 4 ? "i32" : "i64"));
  }

  // The launch is decided before the body is printed: the thread id's own
  // bound comes from it, and this target constrains every address to 32 bits
  // (`amdgpu.address.u32`), which nothing in `workgroup_id * workgroup_size +
  // lane` proves on its own.
  if (si) {
    const ThreadPlan tp = self_indexed->plan(si_shapes);
    for (int d = 0; d < 3; ++d) {
      out.dims.workgroup_size[d] = tp.workgroup_size[d];
      out.dims.workgroup_count[d] = tp.workgroup_count[d];
    }
  } else {
    out.dims = choose_dims(group, device, 0);
  }
  out.dims.subgroup_size = device.wavefront_size;
  const std::uint64_t grid =
      static_cast<std::uint64_t>(std::max(1u, out.dims.workgroup_size[0])) *
      std::max(1u, out.dims.workgroup_count[0]);

  Mint mint;
  prologue += "  %kbase = index.constant 0 : offset\n";
  // buffer.assume.noalias is a promise, not a decoration: an in-place primitive
  // binds one buffer as both an input and the output, so asserting it would be
  // a miscompile. When any binding aliases, none of them is asserted — the
  // same conservative answer the HIP signature reaches by dropping __restrict__
  // on the pair.
  const int inplace =
      (si && anchor && anchor->prim) ? anchor->prim->inplace_input() : -1;
  const bool aliased = inplace >= 0;
  if (!aliased && out.binding_order.size() > 1) {
    std::string lhs;
    std::string rhs;
    std::string ty;
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (i != 0) {
        lhs += ", ";
        rhs += ", ";
        ty += ", ";
      }
      lhs += "%" + names[i] + "_na";
      rhs += "%" + names[i];
      ty += "buffer";
    }
    prologue += "  " + lhs + " = buffer.assume.noalias " + rhs + " : " + ty +
                "\n";
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    const NodePtr& n = out.binding_order[i];
    const std::string src =
        (!aliased && out.binding_order.size() > 1) ? "%" + names[i] + "_na"
                                                   : "%" + names[i];
    prologue += "  %" + names[i] + "_view = buffer.view " + src + "[%kbase] : "
                "buffer -> " +
                loom_view_type(elem_of(n->dtype), n->element_count()) + "\n";
  }
  // blockIdx.x * blockDim.x + threadIdx.x, said once and shared by the guard
  // and by every body spliced below.
  prologue += "  %ki_wg = kernel.workgroup.id<x> : index\n";
  prologue += "  %ki_size = kernel.workgroup.size<x> : index\n";
  prologue += "  %ki_lane = kernel.workitem.id<x> : index\n";
  prologue += "  %ki_flat = index.madd %ki_wg, %ki_size, %ki_lane : index\n";
  prologue += "  %i = index.assume %ki_flat [range(%ki_flat, 0, " +
              std::to_string(grid == 0 ? 0 : grid - 1) +
              ")] : index\n";

  if (si) {
    // ---- a primitive that owns its indexing -------------------------------
    const NodePtr& sink = group.outputs.front();
    Status epilogue_error;
    bool stored = false;
    Mint hook_mint;
    auto epilogue_store = [&](std::string_view index,
                              std::string_view value) -> std::string {
      stored = true;
      std::string s;
      std::unordered_map<const Node*, std::string> value_of;
      value_of[anchor.get()] = std::string(value);
      const std::string idx(index);

      for (std::size_t j = 0; j < input_count; ++j) {
        const NodePtr& n = out.binding_order[j];
        // A kernel primitive indexes its own operands; they must not be
        // pre-loaded at the output index, because the shapes do not line up.
        bool is_operand = false;
        for (const NodePtr& m : group.nodes) {
          if (dynamic_cast<const KernelPrimitiveBase*>(m->prim) == nullptr) {
            continue;
          }
          for (const NodePtr& in : m->inputs) {
            if (in.get() == n.get()) is_operand = true;
          }
        }
        if (is_operand) continue;
        const std::string at = bounded(
            broadcast_index(n->shape, sink->shape, idx, hook_mint, s),
            n->element_count(), hook_mint, s);
        const std::string raw = hook_mint("ep");
        s += "  " + raw + " = view.load %" + names[j] + "_view[" + at + "] : " +
             loom_view_type(elem_of(n->dtype), n->element_count()) + " -> " +
             std::string(loom_storage_type(elem_of(n->dtype))) + "\n";
        value_of[n.get()] = widen_to_f32(raw, n->dtype, hook_mint, s);
      }

      for (const NodePtr& n : group.nodes) {
        if (n.get() == anchor.get() || value_of.count(n.get())) continue;
        if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) != nullptr) {
          continue;
        }
        if (n->kind == graph::OpKind::kConstant) {
          const std::string var = hook_mint("es");
          s += "  " + var + " = scalar.constant " +
               loom_float_literal(n->attrs[0]) + " : f32\n";
          value_of[n.get()] = var;
          continue;
        }
        std::vector<std::string> args;
        args.reserve(n->inputs.size());
        for (const NodePtr& in : n->inputs) {
          const auto it = value_of.find(in.get());
          if (it == value_of.end()) {
            epilogue_error = LSE_ERROR(kInternal, "value for an input of ",
                                        std::string(to_string(n->kind)),
                                        " is not available in the epilogue");
            return {};
          }
          args.push_back(it->second);
        }
        const std::string var = hook_mint("es");
        if (n->prim == nullptr) {
          if (n->kind != graph::OpKind::kCast &&
              n->kind != graph::OpKind::kReshape) {
            epilogue_error = LSE_ERROR(kUnimplemented, "no Loom template for ",
                                        std::string(to_string(n->kind)));
            return {};
          }
          value_of[n.get()] = args.empty() ? std::string{} : args[0];
          continue;
        }
        const std::string_view tmpl = spellings.find(n->prim->name());
        if (tmpl.empty()) {
          epilogue_error = LSE_ERROR(kUnimplemented, "primitive '",
                                      std::string(n->prim->name()),
                                      "' has no Loom source");
          return {};
        }
        std::vector<std::string> attr_names;
        for (float f : n->attrs) {
          const std::string a = hook_mint("ea");
          s += "  " + a + " = scalar.constant " + loom_float_literal(f) +
               " : f32\n";
          attr_names.push_back(a);
        }
        const std::string body =
            loom_splice(tmpl, args, attr_names, var, var + "_t");
        for (std::size_t at = 0; at <= body.size();) {
          const std::size_t nl = body.find('\n', at);
          const std::string one = body.substr(
              at, nl == std::string::npos ? body.size() - at : nl - at);
          if (!one.empty()) s += "  " + one + "\n";
          if (nl == std::string::npos) break;
          at = nl + 1;
        }
        value_of[n.get()] = var;
      }

      const auto it = value_of.find(sink.get());
      if (it == value_of.end()) {
        epilogue_error = LSE_ERROR(kInternal, "group output was never computed");
        return {};
      }
      const std::string narrowed =
          narrow_from_f32(it->second, sink->dtype, hook_mint, s);
      const std::string dst =
          bounded(idx, sink->element_count(), hook_mint, s);
      s += "  view.store " + narrowed + ", %out_view[" + dst + "] : " +
           std::string(loom_storage_type(elem_of(sink->dtype))) + ", " +
           loom_view_type(elem_of(sink->dtype), sink->element_count()) + "\n";
      return s;
    };

    si_shapes.store = epilogue_store;
    si_shapes.types = type_table;
    si_shapes.intrinsics = &spellings;

    ir::KernelBody::Capture cap;
    const std::string text = self_indexed->emit_kernel(si_shapes);
    if (!epilogue_error.ok()) return epilogue_error;
    if (text.empty() || !cap.has()) {
      return LSE_ERROR(kUnimplemented, "primitive '",
                       std::string(self_indexed->name()),
                       "' declined to emit for this invocation");
    }
    if (!stored) {
      return LSE_ERROR(kInternal, "primitive '",
                       std::string(self_indexed->name()),
                       "' owns its indexing but never stored through the hook");
    }

    LoomPrintOptions popts;
    popts.buffers = views;
    popts.name_prefix = "s";
    auto printed = loom_print(cap.body(), popts);
    if (!printed.ok()) {
      return LSE_ERROR(kUnimplemented, "primitive '",
                       std::string(self_indexed->name()), "': ",
                       printed.status().message());
    }

    out.lds_bytes = cap.body().workgroup_bytes();
    const std::uint32_t budget = workgroup_lds_bytes(&device);
    if (budget != 0 && out.lds_bytes > budget) {
      return LSE_ERROR(kOutOfMemory, "'", std::string(self_indexed->name()),
                       "' declares ", std::to_string(out.lds_bytes),
                       " bytes of workgroup scratch, device allows ",
                       std::to_string(budget));
    }
    out.source = kernel_header(out, params) + prologue + printed->text +
                 "  kernel.return\n}\n";
    return out;
  }

  // ---- the per-element scaffold -----------------------------------------
  const Shape& out_shape = group.nodes.back()->shape;
  std::size_t launch_elems = out_shape.elem_count();
  for (const NodePtr& o : group.outputs) {
    launch_elems = std::max(launch_elems, o->element_count());
  }

  std::unordered_set<const Node*> pointer_inputs;
  for (const NodePtr& n : group.nodes) {
    if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) == nullptr) continue;
    for (const NodePtr& in : n->inputs) pointer_inputs.insert(in.get());
  }

  std::string body;
  // `if (i >= k.count) return;` in HIP. Loom will not take the dispatch's word
  // for the count, so the bound it is guaranteed to hold to is stated where it
  // can be checked — which is also what lets the loads below be proven in
  // range.
  body += "  %klimit0 = index.cast %count : i32 to index\n";
  body += "  %klimit = index.assume %klimit0 [range(%klimit0, 0, " +
          std::to_string(launch_elems) + ")] : index\n";
  body += "  %klive = index.cmp ult, %i, %klimit : index\n";
  body += "  scf.if %klive {\n";

  std::string inner;
  std::unordered_map<const Node*, std::string> value_of;
  for (std::size_t i = 0; i < input_count; ++i) {
    const NodePtr& n = out.binding_order[i];
    if (pointer_inputs.count(n.get()) != 0) continue;
    const std::string at =
        bounded(broadcast_index(n->shape, out_shape, "%i", mint, inner),
                n->element_count(), mint, inner);
    const std::string raw = mint("in");
    inner += "  " + raw + " = view.load %" + names[i] + "_view[" + at + "] : " +
             loom_view_type(elem_of(n->dtype), n->element_count()) + " -> " +
             std::string(loom_storage_type(elem_of(n->dtype))) + "\n";
    value_of[n.get()] = widen_to_f32(raw, n->dtype, mint, inner);
  }

  std::size_t sub = 0;
  for (const NodePtr& n : group.nodes) {
    if (value_of.count(n.get())) continue;

    if (n->kind == graph::OpKind::kConstant) {
      const std::string var = mint("c");
      inner += "  " + var + " = scalar.constant " +
               loom_float_literal(n->attrs[0]) + " : f32\n";
      value_of[n.get()] = var;
      continue;
    }

    if (const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim)) {
      // The per-element form of a kernel primitive. HIP puts it in a
      // `__device__ float` helper and calls it; here the body is spliced
      // inline, because Loom's func.def would have to take the operand views
      // as parameters and the dedup a helper bought was only source size.
      if (n->element_count() < launch_elems) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(kp->name()),
                         "' is narrower than the launch, and a guarded splice "
                         "would need a value on the untaken side");
      }
      std::vector<std::string> input_names;
      input_names.reserve(n->inputs.size());
      for (const NodePtr& in : n->inputs) {
        const auto it = binding_of.find(in.get());
        if (it == binding_of.end()) {
          return LSE_ERROR(kInternal, std::string(kp->name()),
                           ": an input is not bound in this group");
        }
        input_names.push_back(names[it->second]);
      }
      const std::string prefix = "n" + std::to_string(sub) + "_";
      std::vector<Shape> shp;
      std::vector<DType> dts;
      KernelShapes shapes = shapes_for(n, shp, dts);
      const KernelPrimitiveBase* chosen = kp->specialize(shapes);
      if (chosen == nullptr) chosen = kp;
      if (chosen->owns_indexing()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(chosen->name()),
                         "' owns its indexing but is not the group's only "
                         "output");
      }
      std::string text;
      ir::KernelBody::Capture cap;
      {
        const ir::RecordOptions ropts{input_names, {}, prefix};
        const ir::KernelBody::Recording rec(ropts);
        text = chosen->emit_kernel(shapes);
      }
      if (text.empty() || !cap.has()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(chosen->name()),
                         "' declined to emit for this invocation");
      }
      LoomPrintOptions popts;
      popts.buffers = views;
      popts.name_prefix = prefix;
      popts.value_prefix = prefix;
      popts.indent = 2;
      auto printed = loom_print(cap.body(), popts);
      if (!printed.ok()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(chosen->name()), "': ",
                         printed.status().message());
      }
      if (printed->result.empty()) {
        return LSE_ERROR(kUnimplemented, "primitive '",
                         std::string(chosen->name()),
                         "' produced no per-element value");
      }
      inner += printed->text;
      value_of[n.get()] = printed->result;
      ++sub;
      continue;
    }

    std::vector<std::string> args;
    args.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      const auto it = value_of.find(in.get());
      if (it == value_of.end()) {
        return LSE_ERROR(kInternal, "value for an input of ",
                         std::string(to_string(n->kind)),
                         " is not available in this group");
      }
      args.push_back(it->second);
    }

    if (n->prim == nullptr) {
      if (n->kind != graph::OpKind::kCast && n->kind != graph::OpKind::kReshape) {
        return LSE_ERROR(kUnimplemented, "no Loom template for ",
                         std::string(to_string(n->kind)));
      }
      value_of[n.get()] = args.empty() ? std::string{} : args[0];
      continue;
    }
    if (!n->prim->has_device_impl()) {
      return LSE_ERROR(kUnimplemented, "primitive '",
                       std::string(n->prim->name()),
                       "' has no device implementation");
    }
    const std::string_view tmpl = spellings.find(n->prim->name());
    if (tmpl.empty()) {
      return LSE_ERROR(kUnimplemented, "primitive '",
                       std::string(n->prim->name()), "' has no Loom source");
    }
    std::vector<std::string> attr_names;
    for (float f : n->attrs) {
      const std::string a = mint("a");
      inner += "  " + a + " = scalar.constant " + loom_float_literal(f) +
               " : f32\n";
      attr_names.push_back(a);
    }
    const std::string var = mint("t");
    const std::string spliced =
        loom_splice(tmpl, args, attr_names, var, var + "_t");
    for (std::size_t at = 0; at <= spliced.size();) {
      const std::size_t nl = spliced.find('\n', at);
      const std::string one = spliced.substr(
          at, nl == std::string::npos ? spliced.size() - at : nl - at);
      if (!one.empty()) inner += "  " + one + "\n";
      if (nl == std::string::npos) break;
      at = nl + 1;
    }
    value_of[n.get()] = var;
  }

  for (std::size_t i = input_count; i < out.binding_order.size(); ++i) {
    const NodePtr& n = out.binding_order[i];
    const auto it = value_of.find(n.get());
    if (it == value_of.end()) {
      return LSE_ERROR(kInternal, "group output was never computed");
    }
    std::string guard_close;
    if (n->element_count() < launch_elems) {
      // The launch covers the largest output, so a narrower one is guarded:
      // writing it at every i walks off the end of its buffer.
      const std::string lim = mint("g");
      const std::string ok = mint("g");
      inner += "  " + lim + " = index.constant " +
               std::to_string(n->element_count()) + " : index\n";
      inner += "  " + ok + " = index.cmp ult, %i, " + lim + " : index\n";
      inner += "  scf.if " + ok + " {\n";
      guard_close = "  }\n";
    }
    const std::string narrowed =
        narrow_from_f32(it->second, n->dtype, mint, inner);
    const std::string dst = bounded("%i", n->element_count(), mint, inner);
    inner += "  view.store " + narrowed + ", %" + names[i] + "_view[" + dst +
             "] : " +
             std::string(loom_storage_type(elem_of(n->dtype))) + ", " +
             loom_view_type(elem_of(n->dtype), n->element_count()) + "\n";
    inner += guard_close;
  }

  // The scaffold's own statements are written at one level and land inside the
  // launch guard, so they are shifted once here rather than threaded through
  // every append above.
  for (std::size_t at = 0; at <= inner.size();) {
    const std::size_t nl = inner.find('\n', at);
    if (nl == std::string::npos) break;
    body += "  " + inner.substr(at, nl - at + 1);
    at = nl + 1;
  }
  body += "  }\n";

  out.lds_bytes = 0;
  out.source =
      kernel_header(out, params) + prologue + body + "  kernel.return\n}\n";
  return out;
}

}  // namespace lse::backend
