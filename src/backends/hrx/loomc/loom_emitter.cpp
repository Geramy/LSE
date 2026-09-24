#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/graph/epilogue_input.hpp"
#include "lse/kernels/int8_policy.hpp"
#include "lse/kernels/quant_operand_policy.hpp"
#include "lse/kernels/quant_operand_cache.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/linked.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/gdn_pair.hpp"
#include "lse/kernels/gdn.hpp"
#include "lse/graph/ops.hpp"

namespace lse::backend {

using namespace lse::graph;

namespace {

// Full structural identity, independent of graph addresses. The graph's
// display signature omits edges and aliasing; those change argument indices
// and cannot identify reusable source or JIT objects.
std::string emission_identity(const FusionGroup& group, const DeviceInfo& device) {
  std::string key;
  key.reserve(1024);
  auto number = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
      key.push_back(static_cast<char>((value >> (byte * 8)) & 255u));
    }
  };
  auto text = [&](std::string_view value) {
    number(value.size());
    key.append(value);
  };
  text("loom");
  number(3);  // identity includes the shared compute-operand policy
  number(kernels::activation_int8_enabled());
  number(kernels::quant_operand_cache_key(0));
  number(kernels::quant_operand_specialization_key(0, group, device,
                                                  loom_types(), loom_sources()));
  text(device.arch);
  // Kernel selection overrides also distinguish persistent JIT identities.
  // FLASH_SDPA and WMMA_MIN_M are latched by their kernels for the process.
  for (const char* name : {"LSE_WMMA", "LSE_FLASH_SDPA", "LSE_WMMA_MIN_M"}) {
    const char* value = std::getenv(name);
    text(value != nullptr ? std::string_view(value) : std::string_view{});
  }
  number(device.lds_bytes_per_workgroup);
  number(device.compute_units);
  number(device.max_threads_per_workgroup);
  number(device.wavefront_size);
  number(device.max_waves_per_cu);
  number(device.cus_per_lds_pool);
  auto fact = [&](const auto& value) {
    number(static_cast<std::uint64_t>(value.source));
    number(value.value);
  };
  const auto& f = device.arch_facts;
  fact(f.vector_registers_per_simd);
  fact(f.vector_register_alloc_granule);
  fact(f.vector_registers_addressable_per_wave);
  fact(f.scalar_registers_per_simd);
  fact(f.scalar_register_alloc_granule);
  fact(f.scalar_registers_addressable_per_wave);
  fact(f.lds_bytes_addressable_per_workgroup);
  fact(f.lds_banks);
  fact(f.max_flat_workgroup_size);
  fact(f.wave_slots_per_simd);
  fact(f.simds_per_lds_pool);
  fact(f.lds_bytes_per_pool);
  fact(f.lds_alloc_granule_bytes);
  number(static_cast<std::uint64_t>(device.residency_bandwidth.source));
  for (auto p : device.residency_bandwidth.percent) number(p);
  text(device.extension_id);
  const auto* amd = device_extension<AmdDeviceInfo>(device);
  number(amd != nullptr);
  if (amd != nullptr) {
    number(amd->l2_cache_bytes);
    number(amd->clock_khz);
    number(static_cast<std::uint64_t>(amd->matrix_core));
    number(amd->has_bf16_arith);
    number(amd->matrix_core_bf16);
    number(amd->has_dot4_i8);
    number(amd->has_dot4_iu8);
    number(amd->max_load_bytes);
    number(amd->max_store_bytes);
  }
  number(static_cast<std::uint64_t>(group.anchor));
  number(static_cast<std::uint64_t>(group.anchor_class));
  number(group.is_phase);
  std::unordered_map<const Node*, std::uint64_t> ids;
  std::vector<const Node*> nodes;
  auto identify = [&](const NodePtr& node) {
    if (!node) return std::uint64_t{0};
    const auto [it, inserted] = ids.emplace(node.get(), ids.size() + 1);
    if (inserted) nodes.push_back(node.get());
    return it->second;
  };
  auto sequence = [&](const auto& list) {
    number(list.size());
    for (const auto& node : list) number(identify(node));
  };
  sequence(group.inputs);
  sequence(group.nodes);
  sequence(group.outputs);
  // Edges of computed nodes, including any boundary input absent from the
  // caller's input list. Boundary nodes' own producers do not enter the kernel.
  for (const auto& node : group.nodes) sequence(node->inputs);
  number(nodes.size());
  for (const Node* node : nodes) {
    number(static_cast<std::uint64_t>(node->kind));
    number(static_cast<std::uint64_t>(node->fclass));
    number(static_cast<std::uint64_t>(node->dtype));
    number(node->shape.rank());
    for (std::size_t i = 0; i < node->shape.rank(); ++i) {
      number(static_cast<std::uint64_t>(node->shape.dim(i)));
    }
    for (auto value : node->iattrs) number(static_cast<std::uint64_t>(value));
    for (float value : node->attrs) number(std::bit_cast<std::uint32_t>(value));
    text(node->prim ? node->prim->name() : std::string_view{});
  }
  return key;
}

std::uint64_t identity_hash(std::string_view key) {
  std::uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : key) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

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
  return identity_hash(emission_identity(group, device));
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

  GdnPair gdn_pair;
  if (group.nodes.size() == 2 && group.outputs.size() == 2) {
    gdn_pair = exact_gdn_pair(group.nodes[0], group.nodes[1]);
    if (gdn_pair && (!std::ranges::count(group.outputs, gdn_pair.output) ||
                     !std::ranges::count(group.outputs, gdn_pair.state))) gdn_pair = {};
  }
  if (!gdn_pair && kernels::linked_bindings(group).ok) {
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

  std::vector<NodePtr> logical_inputs;
  if (gdn_pair) {
    self_indexed = kernels::gdn_pair_kernel();
    anchor = gdn_pair.output;
    logical_inputs = anchor->inputs;
    logical_inputs.push_back(gdn_pair.state);
    si_storage.clear(); si_dtypes.clear();
    for (const auto& input : logical_inputs) {
      si_storage.push_back(input->shape);
      si_dtypes.push_back(input->dtype);
    }
    si_shapes = shapes_for(anchor, storage, dtypes);
    si_shapes.inputs = si_storage;
    si_shapes.input_dtypes = si_dtypes;
  } else if (anchor) logical_inputs = anchor->inputs;

  EmittedKernel out;
  out.dialect = Dialect::kLoom;
  const std::string identity = emission_identity(group, device);
  out.entry_name = "lse_loom_" + std::to_string(identity_hash(identity));

  std::unordered_map<const Node*, std::size_t> binding_of;
  auto bind = [&](const NodePtr& n) {
    if (binding_of.emplace(n.get(), out.binding_order.size()).second) {
      out.binding_order.push_back(n);
    }
  };
  // A self-indexing primitive names its operands in0..inN in its own order, so
  // they are bound first and an epilogue's extra inputs cannot shift them.
  if (self_indexed != nullptr && anchor) {
    for (const NodePtr& in : logical_inputs) bind(in);
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

  // Keep only source and launch metadata. The bindings above always come from
  // this invocation, including a rebuilt graph after a request restart.
  {
    const std::lock_guard lock(cache_mutex_);
    if (const auto it = emit_cache_.find(identity); it != emit_cache_.end()) {
      auto bindings = std::move(out.binding_order);
      out = it->second;
      out.binding_order = std::move(bindings);
      ++cache_hits_;
      return out;
    }
    ++cache_misses_;
  }
  auto remember = [&] {
    EmittedKernel saved = out;
    saved.binding_order.clear();
    const std::lock_guard lock(cache_mutex_);
    constexpr std::size_t kSourceBudget = 64u * 1024u * 1024u;
    if (saved.source.size() > kSourceBudget) return;
    if (emit_cache_.size() >= 1024 ||
        cache_bytes_ + saved.source.size() > kSourceBudget) {
      emit_cache_.clear();
      cache_bytes_ = 0;
    }
    if (emit_cache_.emplace(identity, std::move(saved)).second) {
      cache_bytes_ += out.source.size();
    }
  };

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
    const bool is_out = gdn_pair ? n == gdn_pair.output : output_set.count(n.get()) != 0;
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
    out.dims = graph::choose_launch_dims(group, device, 0);
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
  const bool aliased = inplace >= 0 || bool(gdn_pair);
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
    const NodePtr& sink = gdn_pair ? gdn_pair.output : group.outputs.front();
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
        bool is_operand = std::ranges::count(logical_inputs, n) != 0;
        for (const NodePtr& m : group.nodes) {
          if (dynamic_cast<const KernelPrimitiveBase*>(m->prim) == nullptr) {
            continue;
          }
          for (const NodePtr& in : m->inputs) {
            if (in.get() == n.get()) is_operand = true;
          }
        }
        if (is_operand && !graph::needs_elementwise_input(group, n.get())) continue;
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

    // Binding order deduplicates aliases. Record the logical argument names
    // through that map too, so a repeated input cannot shift later operands.
    std::vector<std::string> input_names;
    for (const NodePtr& input : logical_inputs) {
      input_names.push_back(names[binding_of.at(input.get())]);
    }
    ir::KernelBody::Capture cap;
    std::string text;
    {
      const ir::RecordOptions options{input_names, {}, {}};
      const ir::KernelBody::Recording recording(options);
      text = self_indexed->emit_kernel(si_shapes);
    }
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
    remember();
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
    if (dynamic_cast<const KernelPrimitiveBase*>(n->prim) == nullptr &&
        n->kind != graph::OpKind::kRepeat) continue;
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
  // Shape-only copies must preserve integer bits and floating-point payloads.
  std::unordered_map<const Node*, std::string> raw_value_of;
  for (std::size_t i = 0; i < input_count; ++i) {
    const NodePtr& n = out.binding_order[i];
    if (pointer_inputs.count(n.get()) != 0 &&
        !graph::needs_elementwise_input(group, n.get())) continue;
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
      // A multi-output RMS group cannot use the single-output store hook.
      // Keep its original per-element implementation, as the HIP scaffold does.
      if (n->kind == graph::OpKind::kRMS && chosen->owns_indexing() &&
          !kp->owns_indexing()) chosen = kp;
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

    if (n->kind == graph::OpKind::kRepeat) {
      if (n->inputs.size() != 1 || n->iattrs[1] <= 0) {
        return LSE_ERROR(kInvalidArgument, "repeat requires one input and a positive count");
      }
      const NodePtr& input = n->inputs[0];
      const Shape& sh = input->shape;
      const auto axis = static_cast<std::size_t>(n->iattrs[0]);
      const auto count = static_cast<std::uint64_t>(n->iattrs[1]);
      if (axis >= sh.rank() || sh.rank() != n->shape.rank() ||
          input->dtype != n->dtype) {
        return LSE_ERROR(kInvalidArgument, "repeat shape, axis, or dtype mismatch");
      }
      std::uint64_t inner_size = 1, total = 1;
      for (std::size_t d = 0; d < sh.rank(); ++d) {
        const auto dim = sh.dim(d);
        if (dim <= 0 || total > std::numeric_limits<std::uint32_t>::max() /
                                  static_cast<std::uint64_t>(dim)) {
          return LSE_ERROR(kInvalidArgument, "repeat requires nonempty u32-sized shapes");
        }
        total *= static_cast<std::uint64_t>(dim);
        if (d > axis) inner_size *= static_cast<std::uint64_t>(dim);
        const std::uint64_t expected = static_cast<std::uint64_t>(dim) *
                                       (d == axis ? count : 1);
        if (n->shape.dim(d) <= 0 ||
            static_cast<std::uint64_t>(n->shape.dim(d)) != expected) {
          return LSE_ERROR(kInvalidArgument, "repeat output shape does not match count");
        }
      }
      if (total > std::numeric_limits<std::uint32_t>::max() / count) {
        return LSE_ERROR(kInvalidArgument, "repeat output exceeds u32 indexing");
      }
      const auto binding = binding_of.find(input.get());
      if (binding == binding_of.end()) {
        return LSE_ERROR(kInternal, "repeat input is not bound in this group");
      }
      auto constant = [&](std::uint64_t value) {
        const std::string id = mint("repeat");
        inner += "  " + id + " = index.constant " + std::to_string(value) + " : index\n";
        return id;
      };
      auto binary = [&](std::string_view op, const std::string& a,
                        const std::string& b) {
        const std::string id = mint("repeat");
        inner += "  " + id + " = index." + std::string(op) + " " + a + ", " + b + " : index\n";
        return id;
      };
      const auto axis_size = static_cast<std::uint64_t>(sh.dim(axis));
      const auto stride = constant(inner_size);
      const auto copies = constant(count);
      const auto input_axis = constant(axis_size);
      const auto span = constant(axis_size * count * inner_size);
      // A wider sibling output can launch extra lanes. Map those lanes into
      // this input too; the existing per-output store guard discards them.
      std::string flat = "%i";
      if (total * count < launch_elems) {
        flat = binary("rem", flat, constant(total * count));
      }
      const auto outer = binary("div", flat, span);
      const auto within = binary("rem", flat, span);
      const auto position = binary("div", binary("div", within, stride), copies);
      const auto source_axis = binary("add", binary("mul", outer, input_axis), position);
      const auto source = binary("add", binary("mul", source_axis, stride),
                                 binary("rem", flat, stride));
      const auto at = bounded(source, total, mint, inner);
      const auto raw = mint("repeat_value");
      inner += "  " + raw + " = view.load %" + names[binding->second] + "_view[" + at +
               "] : " + loom_view_type(elem_of(n->dtype), total) + " -> " +
               std::string(loom_storage_type(elem_of(n->dtype))) + "\n";
      raw_value_of[n.get()] = raw;
      value_of[n.get()] = widen_to_f32(raw, n->dtype, mint, inner);
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
    const auto raw = raw_value_of.find(n.get());
    const std::string narrowed = raw != raw_value_of.end() ? raw->second :
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
  remember();
  return out;
}

LoomEmitter::CacheStats LoomEmitter::cache_stats() const {
  const std::lock_guard lock(cache_mutex_);
  return {cache_hits_, cache_misses_, emit_cache_.size(), cache_bytes_};
}

}  // namespace lse::backend
