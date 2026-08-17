#include "lse/backends/hrx/hipc/hip_emitter.hpp"

#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/math.hpp"
#include "lse/backends/hrx/kernels/lds_linear.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vector>

namespace lse::backend {

using namespace lse::graph;

namespace {

// Must agree with hip_types.cpp and hip_emitter.cpp: __bf16, the compiler's
// own type, which casts to and from float directly.
std::string_view device_scalar(DType dt) noexcept {
  switch (dt) {
    case DType::kF16: return "_Float16";
    case DType::kBF16: return "__bf16";
    case DType::kI32: return "int";
    case DType::kU32: return "unsigned int";
    default: return "float";
  }
}

std::string typed_ptr(const std::string& buf, DType dt) {
  return "((" + std::string(device_scalar(dt)) + "*)(" + buf + "))";
}

std::string load_elem(const std::string& buf, const std::string& index,
                      DType dt) {
  return "(float)(" + typed_ptr(buf, dt) + "[" + index + "])";
}

std::string store_elem(const std::string& buf, const std::string& index,
                       DType dt, const std::string& value) {
  return typed_ptr(buf, dt) + "[" + index + "] = (" +
         std::string(device_scalar(dt)) + ")(" + value + ");";
}

std::string broadcast_index(const Shape& src, const Shape& out,
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

bool is_linear_name(std::string_view n) {
  return n == "linear" || n == "linear.lds" || n == "linear_indexed" ||
         n == "linear_indexed.lds";
}

bool is_linked_name(std::string_view n) {
  return n.find("linked") != std::string_view::npos;
}

// The primitive the phase actually stages. WMMA owns a grid we cannot share
// and a linked pipeline owns the whole kernel; the scalar body is the same
// function and fits the staged walk.
const KernelPrimitiveBase* phase_spec(const KernelPrimitiveBase* kp,
                                      const KernelShapes& sh) {
  if (kp == nullptr) return nullptr;
  const KernelPrimitiveBase* spec = kp->specialize(sh);
  if (spec == nullptr) return kp;
  if (spec != kp &&
      (spec->name() == "linear.wmma" || is_linked_name(spec->name()))) {
    return kp;
  }
  return spec;
}

struct GemvShape {
  bool ok = false;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  std::uint32_t m = 0;
};

GemvShape gemv_shape_of(const Node& node) noexcept {
  GemvShape g;
  if (node.inputs.size() < 2) return g;
  const Shape& x = node.inputs[0]->shape;
  const Shape& w = node.inputs[1]->shape;
  if (x.rank() == 0 || w.rank() < 2) return g;
  const auto n = static_cast<std::uint32_t>(
      w.rank() == 3 ? w.dim(1) : w.dim(0));
  const auto k = static_cast<std::uint32_t>(w.dim(w.rank() - 1));
  const auto elems = static_cast<std::uint32_t>(node.element_count());
  if (n == 0 || k == 0 || elems == 0 || elems % n != 0) return g;
  if (static_cast<std::uint32_t>(x.dim(x.rank() - 1)) != k) return g;
  g.n = n;
  g.k = k;
  g.m = elems / n;
  g.ok = true;
  return g;
}

// One thread per output column walks a whole K-row, so an N=8 K=1024 GEMV
// runs on 8 lanes that are 4KB apart — 12 us for eight dot products. emit_gemv
// gives each column a wave with the lanes on consecutive K: same arithmetic,
// 32x the threads, one coalesced stream. Worth it only while the columns
// cannot fill the group on their own. Plain linear only: an indexed one needs
// the expert slot validated, which is the specialization's job.
bool wave_gemv_beats_lanes(const Node& node, const GemvShape& g) noexcept {
  return g.ok && node.inputs.size() == 2 && g.m <= 4 && g.n * g.m <= 256 &&
         g.k >= 128;
}

// Independent work items the stage can spend. A primitive that owns its
// indexing sizes its own grid (a decode gdn_chunk_scan wants 16384 threads
// for a 512-element output), so its plan is the only honest answer.
std::uint32_t stage_thread_count(const Node& node, const KernelShapes& sh,
                            const KernelPrimitiveBase* spec) {
  if (spec != nullptr && spec->owns_indexing() &&
      !is_linear_name(spec->name())) {
    const ThreadPlan tp = spec->plan(sh);
    const std::uint32_t t = tp.workgroup_count[0] * tp.workgroup_size[0];
    if (t != 0) return t;
  }
  const auto elems = static_cast<std::uint32_t>(node.element_count());
  return elems == 0 ? 1u : elems;
}

// Records the same C++ GEMV as linear.lds. HIP spelling comes from kir.
// Phase bindings are all declared float*, so every access casts to the node's
// real storage type first — `wdt` is what makes a bf16 weight read as bf16
// rather than reinterpret two of them as one float.
std::string gemv_stage(const std::string& x, const std::string& w,
                       const std::string& out, const std::string& idx,
                       std::uint32_t keep, std::uint32_t slot, std::uint32_t N,
                       std::uint32_t K, std::uint32_t M, DType odt, DType wdt,
                       bool grid, bool persist, std::uint32_t persist_wgs,
                       const DeviceInfo& device,
                       const kir::TypeTable& types,
                       const DialectSourceTable& spellings) {
  return hrx_kernels::with_elem(wdt, [&]<class W>() -> std::string {
    kir::KernelBody body(types, spellings, workgroup_lds_bytes(&device));
    body.set_store([&](std::string_view index, std::string_view value) {
      return store_elem(out, std::string(index), odt, std::string(value));
    });
    const kir::Buffer<kir::f32> xb(&body, &body.types(), x);
    const kir::Buffer<W> wb(&body, &body.types(), typed_ptr(w, wdt));
    kir::Buffer<kir::f32> ib;
    const kir::Buffer<kir::f32>* idxp = nullptr;
    if (!idx.empty()) {
      ib = kir::Buffer<kir::f32>(&body, &body.types(), idx);
      idxp = &ib;
    }
    std::uint32_t wave = 32;
    if (device.wavefront_size == 32 || device.wavefront_size == 64) {
      wave = device.wavefront_size;
    }
    hrx_kernels::emit_gemv<W>(body, xb, wb, idxp, keep, slot, N, K, M,
                              hrx_kernels::device_load_bytes(&device), grid,
                              wave, persist, persist_wgs);
    if (!body.lds().ok()) return {};
    return "    {\n" + body.str() + "\n    }\n";
  });
}

std::string elem_loop(std::uint32_t count, const std::string& body, bool grid,
                      bool persist, std::uint32_t persist_wgs) {
  std::ostringstream s;
  s << "    {\n      const unsigned n = " << count << "u;\n";
  if (persist) {
    if (persist_wgs == 0) persist_wgs = 1;
    s << "      const unsigned stride = " << persist_wgs << "u * 256u;\n"
      << "      for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n; "
         "i += stride) {\n"
      << body << "      }\n    }\n";
  } else if (grid) {
    s << "      const unsigned i = blockIdx.x * 256u + threadIdx.x;\n"
      << "      if (i < n) {\n"
      << body << "      }\n    }\n";
  } else {
    s << "      for (unsigned i = threadIdx.x; i < n; i += 256u) {\n"
      << body << "      }\n    }\n";
  }
  return s.str();
}

// Repeat is host-only in the library. Stage the index map so a GQA GDN
// does not punch a host hole in the phase.
std::string repeat_stage(const Node& n, const std::string& in,
                         const std::string& out, bool grid, bool persist,
                         std::uint32_t persist_wgs) {
  const Shape& src = n.inputs[0]->shape;
  const auto axis = static_cast<std::size_t>(n.iattrs[0]);
  const auto count = static_cast<std::uint32_t>(n.iattrs[1]);
  if (axis >= src.rank() || count == 0) return {};
  std::uint32_t inner = 1;
  std::uint32_t in_axis = 1;
  for (std::size_t i = 0; i < src.rank(); ++i) {
    const auto d = static_cast<std::uint32_t>(src.dim(i));
    if (i > axis) inner *= d;
    else if (i == axis) in_axis = d;
  }
  const auto out_axis = in_axis * count;
  if (inner == 0 || out_axis == 0) return {};
  std::ostringstream body;
  body << "        const unsigned span = " << (out_axis * inner) << "u;\n"
       << "        const unsigned o = i / span;\n"
       << "        const unsigned rem = i % span;\n"
       << "        const unsigned a = rem / " << inner << "u;\n"
       << "        const unsigned ii = rem % " << inner << "u;\n"
       << "        const unsigned src_a = a / " << count << "u;\n"
       << "        const unsigned src = (o * " << in_axis << "u + src_a) * "
       << inner << "u + ii;\n"
       << "        " << store_elem(out, "i", n.dtype, load_elem(in, "src", n.inputs[0]->dtype))
       << "\n";
  return elem_loop(static_cast<std::uint32_t>(n.element_count()), body.str(),
                   grid, persist, persist_wgs);
}

const Node* skip_view(const Node* n) noexcept {
  while (n != nullptr && n->kind == OpKind::kReshape && n->inputs.size() == 1) {
    n = n->inputs[0].get();
  }
  return n;
}

// How a stage touches memory. Lane: thread i only reads/writes element i.
// Gather: some thread reads (or writes) an element another thread produced.
enum class LaneUse : std::uint8_t { kLane, kGather };

struct StageUse {
  LaneUse write = LaneUse::kLane;
  LaneUse read = LaneUse::kLane;
  std::uint32_t elems = 0;
};

StageUse stage_use(const Node& n) noexcept {
  StageUse u;
  u.elems = static_cast<std::uint32_t>(n.element_count());
  if (is_elementwise(n.kind) || n.kind == OpKind::kCast ||
      n.kind == OpKind::kBroadcast) {
    return u;
  }
  const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n.prim);
  if (kp != nullptr && kp->owns_indexing()) {
    u.write = LaneUse::kGather;
    u.read = LaneUse::kGather;
    return u;
  }
  // rms / softmax / slice / rope / transpose / repeat: store at i, gather
  // inputs. Anything we cannot prove is per-lane is treated as a gather.
  u.read = LaneUse::kGather;
  if (is_reduction(n.kind)) u.write = LaneUse::kGather;
  return u;
}

// Any recorded RAW/WAR/WAW needs a barrier. Bindings are __restrict__, so
// two nodes that share a slot look like distinct pointers; skipping the
// barrier lets the compiler reorder those accesses. What the StageUse decides
// is not whether to order but how wide the barrier has to be — see
// lane_aligned_edge.
bool needs_sync(const StageUse&, const StageUse&) noexcept { return true; }

bool needs_sync_war(const StageUse&, const StageUse&) noexcept { return true; }

bool needs_sync_waw(const StageUse&, const StageUse&) noexcept { return true; }

const Node* written_buf(const Node& n) noexcept {
  if (n.prim != nullptr) {
    const int a = n.prim->inplace_input();
    if (a >= 0 && static_cast<std::size_t>(a) < n.inputs.size()) {
      return skip_view(n.inputs[static_cast<std::size_t>(a)].get());
    }
  }
  return &n;
}

// Thread i of this stage touches element i and nothing else, and its store
// lands in its own binding rather than back into an input.
bool lane_stage_node(const Node& n) noexcept {
  const StageUse u = stage_use(n);
  if (u.write != LaneUse::kLane || u.read != LaneUse::kLane) return false;
  if (u.elems == 0 || written_buf(n) != &n) return false;
  // stage_use answers kLane for every elementwise/cast/broadcast node without
  // looking at its operands, but the elementwise stage indexes each input
  // through broadcast_index(in->shape, n->shape) — so a broadcast operand is
  // read at a computed index, not at `i`. Build the same map the emitter will
  // and take the answer from it: an operand that is not 1:1 makes this thread
  // touch an element it did not write, which is the one thing a lane chunk
  // promises does not happen. Slot recycling then has a cross-workgroup WAR
  // that __syncthreads cannot order.
  for (const NodePtr& in : n.inputs) {
    if (!in) continue;
    if (!BroadcastMap::build(in->shape, n.shape).identity) return false;
  }
  return true;
}

// `consumer` reads `producer` at the index the producing thread wrote, over
// one element count, so the read-after-write between the two never leaves the
// thread. __syncthreads is then the whole barrier: the stages can share a fat
// grid instead of buying grid-wide ordering they never use with a launch
// boundary. Anything gathering, reducing or owning its indexing fails here and
// still pays for the split.
bool lane_aligned_edge(const Node& producer, const Node& consumer) noexcept {
  if (!lane_stage_node(producer) || !lane_stage_node(consumer)) return false;
  if (producer.element_count() != consumer.element_count()) return false;
  return BroadcastMap::build(producer.shape, consumer.shape).identity;
}

bool node_can_stage(const Node& n) noexcept {
  if (n.kind == OpKind::kReshape || n.kind == OpKind::kConstant ||
      n.kind == OpKind::kBuffer || n.kind == OpKind::kCast ||
      n.kind == OpKind::kBroadcast || n.kind == OpKind::kRepeat) {
    return true;
  }
  if (is_elementwise(n.kind)) return true;
  if (n.prim == nullptr) return false;
  const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n.prim);
  if (kp == nullptr) return false;
  const auto name = kp->name();
  if (is_linked_name(name) || name == "linear.wmma") return false;
  if (is_linear_name(name)) return true;
  if (name == "overwrite_slice" || name == "kv_page_write" ||
      name == "gdn_chunk_scan" || name == "gdn_chunk_scan.pair") {
    return true;
  }
  return !kp->owns_indexing();
}

std::uint32_t node_stage_threads(const Node& n, const DeviceInfo& device) {
  if (n.kind == OpKind::kReshape || n.kind == OpKind::kConstant ||
      n.kind == OpKind::kBuffer) {
    return 0;
  }
  const DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();
  std::vector<Shape> storage;
  std::vector<DType> dtypes;
  storage.reserve(n.inputs.size());
  dtypes.reserve(n.inputs.size());
  for (const NodePtr& in : n.inputs) {
    storage.push_back(in->shape);
    dtypes.push_back(in->dtype);
  }
  KernelShapes sh;
  sh.inputs = storage;
  sh.input_dtypes = dtypes;
  sh.output = n.shape;
  sh.output_dtype = n.dtype;
  sh.attrs = n.attrs;
  sh.iattrs = n.iattrs;
  sh.device = &device;
  sh.types = type_table;
  sh.intrinsics = &spellings;
  const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n.prim);
  return stage_thread_count(n, sh, phase_spec(kp, sh));
}

}  // namespace

Result<std::uint32_t> HipEmitter::shared_bytes(std::string_view source) {
  const kir::TypeTable types = hip_types();
  // Both spellings come from the same tables the lowering prints with, so this
  // reads what was written rather than what someone believed was written.
  const std::string_view kMark = hip_sources().find("shared");
  if (types.scalar == nullptr || kMark.empty()) {
    return LSE_ERROR(kInternal, "the HIP dialect cannot spell workgroup memory");
  }
  std::uint32_t total = 0;
  for (std::size_t at = 0;
       (at = source.find(kMark, at)) != std::string_view::npos;) {
    const std::size_t decl = at;
    at += kMark.size();
    while (at < source.size() && (source[at] == ' ' || source[at] == '\t' ||
                                  source[at] == '\n')) {
      ++at;
    }
    // Longest spelling wins, so "unsigned int" is not read as the "int" row.
    std::size_t width = 0;
    kir::Scalar elem = kir::Scalar::kF32;
    for (std::size_t i = 0; i <= static_cast<std::size_t>(kir::Scalar::kBool);
         ++i) {
      const auto s = static_cast<kir::Scalar>(i);
      const std::string_view spelling = types.scalar(s);
      if (spelling.empty() || spelling.size() <= width) continue;
      if (source.compare(at, spelling.size(), spelling) == 0) {
        width = spelling.size();
        elem = s;
      }
    }
    if (width == 0) {
      return LSE_ERROR(kInternal, "workgroup declaration at offset ",
                       std::to_string(decl),
                       " has no element type this dialect can spell");
    }
    const std::size_t open = source.find('[', at + width);
    const std::size_t close = open == std::string_view::npos
                                  ? std::string_view::npos
                                  : source.find(']', open);
    if (open == std::string_view::npos || close == std::string_view::npos) {
      return LSE_ERROR(kInternal, "workgroup declaration at offset ",
                       std::to_string(decl),
                       " is not an array with a literal extent");
    }
    std::uint64_t count = 0;
    bool digits = false;
    for (std::size_t i = open + 1; i < close; ++i) {
      const char c = source[i];
      if (c == ' ' || c == '\t') continue;
      if (c < '0' || c > '9') {
        digits = false;
        break;
      }
      digits = true;
      count = count * 10 + static_cast<std::uint64_t>(c - '0');
    }
    if (!digits) {
      return LSE_ERROR(kInternal, "workgroup declaration at offset ",
                       std::to_string(decl),
                       " has a non-literal extent");
    }
    const auto bytes =
        static_cast<std::uint32_t>(count * kir::scalar_bytes(elem));
    total += (bytes + 15u) & ~15u;
    at = close;
  }
  return total;
}

void HipEmitter::bind_phase(const FusionGroup& group, EmittedKernel& out) {
  out.binding_order.clear();
  out.scratch_bytes = 0;
  std::unordered_set<const Node*> bound;
  auto bind = [&](const NodePtr& n) {
    if (!n || !bound.insert(n.get()).second) return;
    out.binding_order.push_back(n);
  };
  for (const NodePtr& in : group.inputs) bind(in);
  for (const NodePtr& n : group.nodes) {
    if (n->kind == OpKind::kReshape) continue;
    bind(n);
  }
}

bool HipEmitter::can_stage(const Node& n) const noexcept {
  return node_can_stage(n);
}

std::uint32_t HipEmitter::stage_threads(const Node& n,
                                        const DeviceInfo& device) const {
  return node_stage_threads(n, device);
}

bool HipEmitter::lane_stage(const Node& n) const noexcept {
  return lane_stage_node(n);
}

bool HipEmitter::lane_aligned(const Node& producer,
                              const Node& consumer) const noexcept {
  return lane_aligned_edge(producer, consumer);
}

Result<EmittedKernel> HipEmitter::emit_phase(const FusionGroup& group,
                                             const DeviceInfo& device) {
  if (group.nodes.empty()) {
    return LSE_ERROR(kInvalidArgument, "empty phase group");
  }
  bool any_compute = false;
  for (const NodePtr& n : group.nodes) {
    if (n->kind != OpKind::kReshape && n->kind != OpKind::kConstant &&
        n->kind != OpKind::kBuffer) {
      any_compute = true;
      break;
    }
  }
  if (!any_compute) {
    return LSE_ERROR(kUnimplemented, "phase group is views only");
  }
  for (const NodePtr& n : group.nodes) {
    if (!node_can_stage(*n)) {
      return LSE_ERROR(kUnimplemented, "phase kernel cannot stage ",
                       std::string(to_string(n->kind)));
    }
  }

  const DialectSourceTable spellings = hip_sources();
  const kir::TypeTable type_table = hip_types();

  EmittedKernel out;
  out.pointer_table = false;
  out.entry_name = "lse_phase_" + std::to_string(group.signature());
  out.constants.add("count", 4);
  bind_phase(group, out);

  std::unordered_map<const Node*, std::size_t> binding_of;
  for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
    const Node* p = skip_view(out.binding_order[i].get());
    if (p != nullptr) binding_of.emplace(p, i);
  }

  auto bname = [&](const Node* n) -> std::string {
    n = skip_view(n);
    if (n == nullptr) return {};
    auto it = binding_of.find(n);
    if (it == binding_of.end()) return {};
    return "b" + std::to_string(it->second);
  };

  std::vector<Shape> storage;
  std::vector<DType> dtypes;
  auto shapes_for = [&](const NodePtr& n) {
    storage.clear();
    dtypes.clear();
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

  std::unordered_set<const Node*> members;
  for (const NodePtr& n : group.nodes) members.insert(n.get());
  bool only_linears = true;
  bool dependent = false;
  // Every stage pure per-lane over one element count, every in-group read at
  // the index its own thread wrote. Then no dependence in the body crosses a
  // thread, let alone a workgroup, and the chain can keep the fat grid.
  bool lane_chunk = true;
  std::uint32_t lane_elems = 0;
  std::uint32_t max_n = 0;
  std::uint32_t max_m = 1;
  std::uint32_t max_threads = 0;
  std::uint32_t compute = 0;
  for (const NodePtr& n : group.nodes) {
    if (n->kind == OpKind::kConstant || n->kind == OpKind::kBuffer ||
        n->kind == OpKind::kReshape) {
      continue;
    }
    ++compute;
    const auto elems = static_cast<std::uint32_t>(n->element_count());
    {
      KernelShapes sh = shapes_for(n);
      const auto* kpn = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
      const std::uint32_t want = stage_thread_count(*n, sh, phase_spec(kpn, sh));
      if (want > max_threads) max_threads = want;
    }
    if (!lane_stage_node(*n)) lane_chunk = false;
    if (lane_elems == 0) lane_elems = elems;
    if (elems != lane_elems) lane_chunk = false;
    for (const NodePtr& in : n->inputs) {
      const Node* p = in.get();
      while (p && p->kind == OpKind::kReshape && p->inputs.size() == 1) {
        p = p->inputs[0].get();
      }
      if (p && members.count(p) && p->kind != OpKind::kConstant &&
          p->kind != OpKind::kBuffer && p->kind != OpKind::kReshape) {
        dependent = true;
        if (!lane_aligned_edge(*p, *n)) lane_chunk = false;
      }
    }
    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    const bool lin =
        kp != nullptr && is_linear_name(kp->name()) && n->inputs.size() >= 2;
    if (!lin) {
      only_linears = false;
      continue;
    }
    const Shape& wsh = n->inputs[1]->shape;
    const auto N = static_cast<std::uint32_t>(
        wsh.rank() == 3 ? wsh.dim(1) : wsh.dim(0));
    const auto M = N == 0 ? 1u : elems / N;
    if (N > max_n) max_n = N;
    if (M > max_m) max_m = M;
  }
  // Independent stages can use a fat grid, and so does a lane chunk: its
  // dependences are intra-thread, so __syncthreads carries them at any grid
  // width. Every other dependent group shares one workgroup, where
  // __syncthreads is the whole grid. A multi-WG software barrier deadlocks
  // here: this launch path does not keep the whole grid resident.
  const bool persist = false;
  const bool grid_gemv =
      !persist && only_linears && compute > 0 && max_n >= 256;
  const bool grid_elem = !persist && !grid_gemv &&
                         (!dependent || lane_chunk) && max_threads >= 512;
  // A grid launch orders independent stages by having nothing to order. Only
  // the lane chunk brings a dependence into one, and only its barriers are
  // workgroup-wide facts; grid_gemv keeps the no-barrier form it always had.
  const bool lane_fused = grid_elem && dependent;
  // Every stage of a grid launch walks the same flat thread space, so the
  // stride is one literal for the whole kernel.
  const std::uint32_t grid_wgs =
      max_threads == 0 ? 1u : (max_threads + 255u) / 256u;
  std::uint32_t persist_wgs = 1;
  if (persist) {
    const std::uint32_t tiles = max_n == 0 ? 1u : (max_n + 7u) / 8u;
    const std::uint32_t gemv_wgs = tiles * (max_m == 0 ? 1u : max_m);
    const std::uint32_t elem_wgs = grid_wgs;
    const std::uint32_t work = gemv_wgs > elem_wgs ? gemv_wgs : elem_wgs;
    const std::uint32_t cap =
        device.compute_units == 0 ? 1u : device.compute_units;
    persist_wgs = work < cap ? work : cap;
    if (persist_wgs == 0) persist_wgs = 1;
  }

  std::ostringstream preamble;
  std::ostringstream stages;
  std::uint32_t stage_id = 0;

  std::vector<const Node*> staged;
  std::vector<StageUse> uses;
  for (const NodePtr& n : group.nodes) {
    if (n->kind == OpKind::kConstant || n->kind == OpKind::kBuffer ||
        n->kind == OpKind::kReshape) {
      continue;
    }
    staged.push_back(n.get());
    uses.push_back(stage_use(*n));
  }
  std::unordered_map<const Node*, int> last_writer;
  std::unordered_map<const Node*, int> last_reader;
  int visible_through = -1;
  int stage_i = 0;
  const bool grid_stages = grid_gemv || grid_elem;

  auto sync_before = [&]() {
    if ((grid_stages && !lane_fused) ||
        stage_i >= static_cast<int>(staged.size())) {
      return;
    }
    const StageUse& me = uses[static_cast<std::size_t>(stage_i)];
    const Node* dest =
        written_buf(*staged[static_cast<std::size_t>(stage_i)]);
    bool need = false;
    for (const NodePtr& in : staged[static_cast<std::size_t>(stage_i)]->inputs) {
      const Node* p = skip_view(in.get());
      const auto it = last_writer.find(p);
      if (it == last_writer.end()) continue;
      if (it->second > visible_through &&
          needs_sync(uses[static_cast<std::size_t>(it->second)], me)) {
        need = true;
        break;
      }
    }
    if (!need && dest != nullptr) {
      if (const auto rd = last_reader.find(dest); rd != last_reader.end() &&
          rd->second > visible_through &&
          needs_sync_war(uses[static_cast<std::size_t>(rd->second)], me)) {
        need = true;
      } else if (const auto wr = last_writer.find(dest);
                 wr != last_writer.end() && wr->second > visible_through &&
                 needs_sync_waw(uses[static_cast<std::size_t>(wr->second)],
                                me)) {
        need = true;
      }
    }
    if (!need) return;
    if (persist) {
      stages << "    lse_grid_sync(gbar);\n";
    } else {
      stages << "    __syncthreads();\n";
    }
    visible_through = stage_i - 1;
  };

  auto record_write = [&]() {
    if (stage_i >= static_cast<int>(staged.size())) return;
    const Node* n = staged[static_cast<std::size_t>(stage_i)];
    last_writer[n] = stage_i;
    const Node* dest = written_buf(*n);
    if (dest != nullptr) last_writer[dest] = stage_i;
    for (const NodePtr& in : n->inputs) {
      const Node* p = skip_view(in.get());
      if (p != nullptr) last_reader[p] = stage_i;
    }
    ++stage_i;
  };

  for (const NodePtr& n : group.nodes) {
    if (n->kind == OpKind::kConstant || n->kind == OpKind::kBuffer) continue;
    if (n->kind == OpKind::kReshape) continue;

    sync_before();

    if (n->kind == OpKind::kRepeat) {
      if (n->inputs.size() != 1) {
        return LSE_ERROR(kInvalidArgument, "repeat stage missing input");
      }
      const std::string in = bname(n->inputs[0].get());
      const std::string ob = bname(n.get());
      if (in.empty() || ob.empty()) {
        return LSE_ERROR(kInternal, "repeat stage is not bound");
      }
      const std::string body =
          repeat_stage(*n, in, ob, grid_elem, persist, persist_wgs);
      if (body.empty()) {
        return LSE_ERROR(kUnimplemented, "repeat stage has a bad shape");
      }
      stages << body;
      record_write();
      continue;
    }

    const auto* kp = dynamic_cast<const KernelPrimitiveBase*>(n->prim);
    const KernelPrimitiveBase* spec = kp;
    if (kp != nullptr) {
      KernelShapes sh = shapes_for(n);
      spec = phase_spec(kp, sh);
    }

    const bool spec_gemv = spec != nullptr && spec->owns_indexing() &&
                           is_linear_name(spec->name());
    const GemvShape gsh =
        kp != nullptr && is_linear_name(kp->name()) ? gemv_shape_of(*n)
                                                    : GemvShape{};
    if (spec_gemv || wave_gemv_beats_lanes(*n, gsh)) {
      if (n->inputs.size() < 2) {
        return LSE_ERROR(kInvalidArgument, "linear stage missing operands");
      }
      if (!gsh.ok) {
        return LSE_ERROR(kInvalidArgument, "linear stage has a bad shape");
      }
      const std::uint32_t N = gsh.n;
      const std::uint32_t K = gsh.k;
      const std::uint32_t M = gsh.m;

      std::string idx;
      std::uint32_t keep = 0;
      std::uint32_t slot = 0;
      if (n->inputs.size() >= 3) {
        idx = bname(n->inputs[2].get());
        if (idx.empty()) {
          return LSE_ERROR(kInternal, "indexed linear missing idx binding");
        }
        keep = static_cast<std::uint32_t>(
            n->inputs[2]->shape.dim(n->inputs[2]->shape.rank() - 1));
        slot = static_cast<std::uint32_t>(n->iattrs[0]);
      }
      const std::string xb = bname(n->inputs[0].get());
      const std::string wb = bname(n->inputs[1].get());
      const std::string ob = bname(n.get());
      if (xb.empty() || wb.empty() || ob.empty()) {
        return LSE_ERROR(kInternal, "linear stage is not bound");
      }
      const std::string body =
          gemv_stage(xb, wb, ob, idx, keep, slot, N, K, M, n->dtype,
                     n->inputs[1]->dtype, grid_gemv, persist, persist_wgs,
                     device, type_table, spellings);
      if (body.empty()) {
        return LSE_ERROR(kInternal, "linear stage exceeded LDS");
      }
      // The workgroup walk covers the whole GEMV by itself. In a grid launch
      // sized for some other stage, every workgroup would redo it and race on
      // the same addresses, so only one runs it.
      if (grid_elem) stages << "    if (blockIdx.x == 0u)\n";
      stages << body;
      record_write();
      continue;
    }

    if (spec != nullptr && spec->owns_indexing()) {
      KernelShapes sh = shapes_for(n);
      const std::string ob = bname(n.get());
      if (ob.empty()) {
        return LSE_ERROR(kInternal, std::string(spec->name()),
                         ": output is not bound");
      }
      bool stored = false;
      sh.store = [&](std::string_view index, std::string_view value) {
        stored = true;
        return store_elem(ob, std::string(index), n->dtype, std::string(value));
      };
      std::string body = spec->emit_kernel(sh);
      if (body.empty() || !stored) {
        return LSE_ERROR(kUnimplemented, "phase primitive '",
                         std::string(spec->name()), "' declined");
      }
      for (std::size_t pos = 0;
           (pos = body.find("return;", pos)) != std::string::npos;) {
        body.replace(pos, 7, "continue;");
        pos += 9;
      }
      const std::uint32_t nthreads = stage_thread_count(*n, sh, spec);
      stages << "    {\n";
      for (std::size_t a = 0; a < n->inputs.size(); ++a) {
        const std::string b = bname(n->inputs[a].get());
        if (b.empty()) {
          return LSE_ERROR(kInternal, std::string(spec->name()),
                           ": an input is not bound");
        }
        stages << "      float* in" << a << " = " << b << ";\n";
      }
      stages << "      const unsigned n = " << nthreads << "u;\n";
      if (persist) {
        stages << "      const unsigned stride = " << persist_wgs
               << "u * 256u;\n"
               << "      for (unsigned i = blockIdx.x * 256u + threadIdx.x; "
                  "i < n; i += stride) {\n"
               << body << "      }\n    }\n";
      } else if (grid_elem) {
        // Loop form, not `if (i < n)`: the body spells an early-out as
        // `continue`, which needs a loop to continue out of.
        stages << "      for (unsigned i = blockIdx.x * 256u + threadIdx.x; "
                  "i < n; i += "
               << (grid_wgs * 256u) << "u) {\n"
               << body << "      }\n    }\n";
      } else {
        stages << "      for (unsigned i = threadIdx.x; i < n; i += 256u) {\n"
               << body << "      }\n    }\n";
      }
      record_write();
      continue;
    }

    if (spec != nullptr && !spec->owns_indexing()) {
      KernelShapes sh = shapes_for(n);
      const std::string body = spec->emit_kernel(sh);
      if (body.empty()) {
        return LSE_ERROR(kUnimplemented, "phase primitive '",
                         std::string(spec->name()), "' declined");
      }
      // Stage id is part of the symbol. Two slices with the same count
      // and axis but different begins must not share a body — extents
      // are literals inside emit_kernel.
      const std::string fn =
          std::string(spec->entry_name()) + "_" + std::to_string(stage_id++);
      preamble << "__device__ float " << fn << "(unsigned int i";
      for (std::size_t a = 0; a < n->inputs.size(); ++a) {
        preamble << ", const " << device_scalar(n->inputs[a]->dtype)
                 << "* __restrict__ in" << a;
      }
      preamble << ") {\n" << body << "\n}\n\n";

      const auto count = static_cast<std::uint32_t>(n->element_count());
      const std::string ob = bname(n.get());
      if (ob.empty()) {
        return LSE_ERROR(kInternal, fn, ": output is not bound");
      }
      std::ostringstream call;
      call << "        const float v = " << fn << "(i";
      for (const NodePtr& in : n->inputs) {
        const std::string b = bname(in.get());
        if (b.empty()) {
          return LSE_ERROR(kInternal, fn, ": an input is not bound");
        }
        call << ", (" << device_scalar(in->dtype) << "*)" << b;
      }
      call << ");\n        " << store_elem(ob, "i", n->dtype, "v") << "\n";
      stages << elem_loop(count, call.str(), grid_elem, persist, persist_wgs);
      record_write();
      continue;
    }

    const auto count = static_cast<std::uint32_t>(n->element_count());
    std::vector<std::string> args;
    args.reserve(n->inputs.size());
    for (const NodePtr& in : n->inputs) {
      const std::string b = bname(in.get());
      if (b.empty()) {
        return LSE_ERROR(kUnimplemented, "phase elementwise missing input");
      }
      args.push_back(load_elem(b, broadcast_index(in->shape, n->shape, "i"),
                               in->dtype));
    }
    const std::string ob = bname(n.get());
    if (ob.empty()) {
      return LSE_ERROR(kInternal, "phase elementwise output is not bound");
    }
    if (n->prim != nullptr) {
      EmitContext ctx;
      ctx.inputs = args;
      ctx.out = "v";
      ctx.device = &device;
      ctx.attrs = n->attrs;
      ctx.iattrs = n->iattrs;
      ctx.dialect = Dialect::kHip;
      ctx.sources = &spellings;
      const std::string body = n->prim->emit_device(ctx);
      if (body.empty()) {
        return LSE_ERROR(kUnimplemented, "phase has no source for '",
                         std::string(n->prim->name()), "'");
      }
      std::ostringstream loop;
      loop << "        float v; " << body << "\n        "
           << store_elem(ob, "i", n->dtype, "v") << "\n";
      stages << elem_loop(count, loop.str(), grid_elem, persist, persist_wgs);
      record_write();
      continue;
    }
    if (n->kind == OpKind::kCast || n->kind == OpKind::kBroadcast) {
      std::ostringstream loop;
      loop << "        "
           << store_elem(ob, "i", n->dtype, args.empty() ? "0.0f" : args[0])
           << "\n";
      stages << elem_loop(count, loop.str(), grid_elem, persist, persist_wgs);
      record_write();
      continue;
    }
    return LSE_ERROR(kUnimplemented, "phase kernel cannot stage ",
                     std::string(to_string(n->kind)));
  }

  std::ostringstream src;
  src << "#include <hip/hip_runtime.h>\n"
      << "#include <hip/hip_bf16.h>\n\n"
      << HipEmitter::constants_decl(out.constants) << "\n"
      << preamble.str();
  if (persist) {
    src << "__device__ void lse_grid_sync(unsigned* ctr) {\n"
        << "  __syncthreads();\n"
        << "  if (threadIdx.x == 0) {\n"
        << "    const unsigned ticket = atomicAdd(ctr, 1u);\n"
        << "    const unsigned nx = " << persist_wgs << "u;\n"
        << "    const unsigned goal = ((ticket / nx) + 1u) * nx;\n"
        // atomicCAS cannot fold into a cached load. HIP gridDim is also
        // not filled by this launch path, so nx is a source literal.
        << "    while (atomicCAS(ctr, goal, goal) < goal) ;\n"
        << "  }\n"
        << "  __syncthreads();\n"
        << "}\n\n";
  }
  src << "extern \"C\" __global__ void " << out.entry_name << "(\n";
  for (std::size_t i = 0; i < out.binding_order.size(); ++i) {
    src << "    float* __restrict__ b" << i << ",\n";
  }
  if (persist) src << "    unsigned* gbar,\n";
  src << "    LseConstants k) {\n"
      << "  (void)k;\n"
      << stages.str() << "}\n";
  out.source = src.str();
  out.persist_grid = persist;
  out.dims.workgroup_size[0] = 256;
  if (persist) {
    out.dims.workgroup_count[0] = persist_wgs;
    out.dims.workgroup_count[1] = 1;
  } else if (grid_gemv) {
    out.dims.workgroup_count[0] = (max_n + 7u) / 8u;
    out.dims.workgroup_count[1] = max_m == 0 ? 1u : max_m;
  } else if (grid_elem) {
    out.dims.workgroup_count[0] = grid_wgs;
    out.dims.workgroup_count[1] = 1;
  } else {
    out.dims.workgroup_count[0] = 1;
    out.dims.workgroup_count[1] = 1;
  }

  // What the phase actually declares, not a constant per geometry. Every stage
  // here builds its body against the whole device budget and none of them can
  // see the others, so the only place the total exists is the assembled text:
  // two staged GEMVs at K=2176 declare 17408 bytes between them where this used
  // to report 1024, and a phase that overruns the budget does not fail to
  // launch — it fails to compile, and the group silently falls to the host
  // interpreter for the rest of the run.
  auto declared = shared_bytes(out.source);
  if (!declared.ok()) return declared.status();
  const std::uint32_t budget = workgroup_lds_bytes(&device);
  if (budget != 0 && *declared > budget) {
    return LSE_ERROR(kOutOfMemory, "phase declares ", std::to_string(*declared),
                     " bytes of workgroup scratch, device allows ",
                     std::to_string(budget));
  }
  out.lds_bytes = *declared;
  out.dims.subgroup_size = device.wavefront_size;
  return out;
}

}  // namespace lse::backend
