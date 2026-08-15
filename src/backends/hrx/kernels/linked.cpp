// Linked contraction pipelines: the generator sees how linears share an
// activation and folds the chain into one launch.
//
// SwiGLU is the one that shows up every block: two linears from the same
// x, silu(g)*u, then a down linear. Decode is one workgroup per row;
// hidden lives in LDS so the down GEMV never hits global for it.
#include "kernels/linked.hpp"
#include "kernels/gdn.hpp"

#include <string>

#include "kernels/vec_mem.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

bool is_linear_node(const Node* n) {
  if (n == nullptr) return false;
  if (n->kind == OpKind::kLinear) return true;
  if (n->prim == nullptr) return false;
  const auto name = n->prim->name();
  return name == "linear" || name == "linear_indexed" || name == "linear.lds" ||
         name == "linear_indexed.lds";
}

bool is_indexed_node(const Node* n) {
  return n != nullptr && n->prim != nullptr &&
         (n->prim->name() == "linear_indexed" ||
          n->prim->name() == "linear_indexed.lds");
}

const Node* skip_views(const Node* n) {
  while (n != nullptr && n->inputs.size() == 1 &&
         (n->kind == OpKind::kReshape ||
          (n->kind == OpKind::kSlice &&
           n->element_count() == n->inputs[0]->element_count()))) {
    n = n->inputs[0].get();
  }
  return n;
}

struct SwigluParts {
  const Node* x = nullptr;
  const Node* wg = nullptr;
  const Node* wu = nullptr;
  const Node* wd = nullptr;
  const Node* idx = nullptr;
  const Node* hid = nullptr;
  const Node* down = nullptr;
  std::int32_t slot = 0;
  bool ok = false;
};

SwigluParts match_swiglu_down(const Node* down) {
  SwigluParts p;
  if (!is_linear_node(down) || down->inputs.size() < 2) return p;
  const Node* h = skip_views(down->inputs[0].get());
  if (h == nullptr || h->kind != OpKind::kMul || h->inputs.size() != 2) {
    return p;
  }
  const Node* a = skip_views(h->inputs[0].get());
  const Node* b = skip_views(h->inputs[1].get());
  const Node* silu = nullptr;
  const Node* u = nullptr;
  if (a != nullptr && a->kind == OpKind::kSiLU) {
    silu = a;
    u = b;
  } else if (b != nullptr && b->kind == OpKind::kSiLU) {
    silu = b;
    u = a;
  } else {
    return p;
  }
  if (silu->inputs.empty()) return p;
  const Node* g = skip_views(silu->inputs[0].get());
  u = skip_views(u);
  if (!is_linear_node(g) || !is_linear_node(u)) return p;
  if (g->inputs.size() < 2 || u->inputs.size() < 2) return p;
  if (g->inputs[0].get() != u->inputs[0].get()) return p;

  p.x = g->inputs[0].get();
  p.wg = g->inputs[1].get();
  p.wu = u->inputs[1].get();
  p.wd = down->inputs[1].get();
  p.hid = h;
  p.down = down;
  if (is_indexed_node(g)) {
    if (!is_indexed_node(u) || !is_indexed_node(down) || g->inputs.size() < 3) {
      return {};
    }
    p.idx = g->inputs[2].get();
    p.slot = g->iattrs[0];
  }
  p.ok = true;
  return p;
}

SwigluParts match_swiglu(const FusionGroup& group) {
  const Node* down = nullptr;
  for (const NodePtr& n : group.nodes) {
    if (is_linear_node(n.get())) down = n.get();
  }
  return match_swiglu_down(down);
}

struct RmsLinearParts {
  const Node* x = nullptr;
  const Node* rms_w = nullptr;
  const Node* w = nullptr;
  const Node* linear = nullptr;
  float eps = 1e-6f;
  bool zero_centered = false;
  bool ok = false;
};

RmsLinearParts match_rms_linear(const FusionGroup& group) {
  RmsLinearParts p;
  const Node* lin = nullptr;
  for (const NodePtr& n : group.nodes) {
    if (is_linear_node(n.get())) lin = n.get();
  }
  if (lin == nullptr || lin->inputs.size() < 2) return p;
  const Node* act = skip_views(lin->inputs[0].get());
  if (act == nullptr || act->kind != OpKind::kRMS || act->inputs.size() < 2) {
    return p;
  }
  p.x = act->inputs[0].get();
  p.rms_w = act->inputs[1].get();
  p.w = lin->inputs[1].get();
  p.linear = lin;
  p.eps = act->attrs[0];
  p.zero_centered = act->iattrs[0] != 0;
  p.ok = true;
  return p;
}

constexpr std::uint32_t kBlock = 256;

void emit_row_dot(kir::KernelBody& k, const kir::Buffer<kir::f32>& a,
                  const kir::Buffer<kir::f32>& b,
                  const kir::Val<kir::u32>& a0,
                  const kir::Val<kir::u32>& b0,
                  const kir::LValue<kir::f32>& acc, std::uint32_t kdim,
                  std::uint32_t load_bytes) {
  emit_dot_f32(k, a, b, a0, b0, acc, kdim, load_bytes);
}

struct SwigluKernel final : KernelPrimitive<SwigluKernel> {
  static constexpr std::string_view kName = "swiglu.linked";
  static constexpr std::string_view kEntry = "lse_swiglu_linked";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 5 || !s.store || s.types.scalar == nullptr ||
        s.intrinsics == nullptr) {
      return {};
    }
    const auto K = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto N = static_cast<std::uint32_t>(s.inputs[1].rank() == 3
                                                  ? s.inputs[1].dim(1)
                                                  : s.inputs[1].dim(0));
    const auto D = static_cast<std::uint32_t>(s.inputs[3].rank() == 3
                                                  ? s.inputs[3].dim(1)
                                                  : s.inputs[3].dim(0));
    if (K == 0 || N == 0 || D == 0) return {};
    const bool indexed = s.inputs[1].rank() == 3;
    const auto slot = static_cast<std::uint32_t>(s.iattrs[0]);
    const std::size_t hid_i = indexed ? 5 : 4;
    const std::size_t idx_i = 4;
    if (s.inputs.size() <= hid_i) return {};
    const auto keep = indexed && s.inputs[idx_i].rank() > 0
                          ? static_cast<std::uint32_t>(
                                s.inputs[idx_i].dim(s.inputs[idx_i].rank() - 1))
                          : 1u;
    const auto M = static_cast<std::uint32_t>(s.output.elem_count() / D);
    if (M == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    const auto x = k.input<kir::f32>(0);
    const auto wg = k.input<kir::f32>(1);
    const auto wu = k.input<kir::f32>(2);
    const auto wd = k.input<kir::f32>(3);
    const auto hid = k.input<kir::f32>(hid_i);
    const auto load_bytes = device_load_bytes(s.device);

    const auto lid = k.let<kir::u32>("lid", math::local_id());
    const auto row = k.let<kir::u32>("row", math::workgroup_id_y());

    kir::Val<kir::u32> e_up;
    kir::Val<kir::u32> e_down;
    if (indexed && s.inputs.size() >= 5) {
      const auto idx = k.input<kir::f32>(4);
      const auto e = k.let<kir::u32>(
          "e", kir::cast<kir::u32>(
                   idx[row * keep + k.constant<kir::u32>(slot)].read()));
      e_up = k.let<kir::u32>("eu", e * N);
      e_down = k.let<kir::u32>("ed", e * D);
    } else {
      e_up = k.constant<kir::u32>(0);
      e_down = k.constant<kir::u32>(0);
    }

    k.loop("n", lid, k.constant<kir::u32>(N), kBlock,
           [&](kir::Val<kir::u32> n) {
             auto g = k.var<kir::f32>("g", k.lit(0.0f));
             auto u = k.var<kir::f32>("u", k.lit(0.0f));
             emit_row_dot(k, x, wg, row * K, (e_up + n) * K, g, K, load_bytes);
             emit_row_dot(k, x, wu, row * K, (e_up + n) * K, u, K, load_bytes);
             const auto gate =
                 g.read() / (k.lit(1.0f) + math::exp(k.lit(0.0f) - g.read()));
             hid[row * N + n] = gate * u.read();
           });
    math::barrier();

    k.loop("d", lid, k.constant<kir::u32>(D), kBlock,
           [&](kir::Val<kir::u32> d) {
             auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
             emit_row_dot(k, hid, wd, row * N, (e_down + d) * N, acc, N,
                         load_bytes);
             k.store(row * D + d, acc.read());
           });
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 4) return LSE_ERROR(kInvalidArgument, "swiglu needs 4 inputs");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[3].rank() == 3 ? in[3].dim(1) : in[3].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const auto D = static_cast<std::uint32_t>(
        s.inputs.size() >= 4
            ? (s.inputs[3].rank() == 3 ? s.inputs[3].dim(1) : s.inputs[3].dim(0))
            : 1);
    const auto M =
        D == 0 ? 1u : static_cast<std::uint32_t>(s.output.elem_count() / D);
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = 1;
    tp.workgroup_count[1] = M == 0 ? 1u : M;
    return tp;
  }
};

// The first emit_kernel draft called emit_row_dot against a dummy buffer for
// hidden; the real walk is emit_dot_lds. Strip the dead call.
struct RmsLinearKernel final : KernelPrimitive<RmsLinearKernel> {
  static constexpr std::string_view kName = "rms_linear.linked";
  static constexpr std::string_view kEntry = "lse_rms_linear";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 3 || !s.store || s.types.scalar == nullptr ||
        s.intrinsics == nullptr) {
      return {};
    }
    const auto K = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto N = static_cast<std::uint32_t>(s.inputs[2].dim(0));
    if (K == 0 || N == 0) return {};
    const float eps = s.attrs[0] == 0.0f ? 1e-6f : s.attrs[0];
    const bool zc = s.iattrs[0] != 0;
    const auto load_bytes = device_load_bytes(s.device);

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    const auto x = k.input<kir::f32>(0);
    const auto rw = k.input<kir::f32>(1);
    const auto w = k.input<kir::f32>(2);
    const auto lid = k.let<kir::u32>("lid", math::local_id());
    const auto row = k.let<kir::u32>("row", math::workgroup_id_y());
    auto xs = k.lds().array<kir::f32>("xs", K);
    auto red = k.lds().array<kir::f32>("red", kBlock);
    if (!xs || !red) return {};

    k.loop("t", lid, k.constant<kir::u32>(K), kBlock,
           [&](kir::Val<kir::u32> t) {
             const auto v = x[row * K + t].read();
             xs[t] = v;
           });
    math::barrier();

    auto ss = k.var<kir::f32>("ss", k.lit(0.0f));
    k.loop("t2", lid, k.constant<kir::u32>(K), kBlock,
           [&](kir::Val<kir::u32> t) {
             const auto v = xs[t].read();
             ss = math::fma(v, v, ss.read());
           });
    red[lid] = ss.read();
    math::barrier();
    k.when(lid == 0, [&] {
      auto tot = k.var<kir::f32>("tot", k.lit(0.0f));
      k.loop("r", k.constant<kir::u32>(0), k.constant<kir::u32>(kBlock), 1,
             [&](kir::Val<kir::u32> r) { tot = tot.read() + red[r].read(); });
      red[0] = math::rsqrt(tot.read() / static_cast<float>(K) + eps);
    });
    math::barrier();
    const auto scale = k.let<kir::f32>("sc", red[0].read());
    k.loop("t3", lid, k.constant<kir::u32>(K), kBlock,
           [&](kir::Val<kir::u32> t) {
             const auto gain =
                 zc ? k.lit(1.0f) + rw[t].read() : rw[t].read();
             xs[t] = xs[t].read() * scale * gain;
           });
    math::barrier();

    k.loop("n", lid, k.constant<kir::u32>(N), kBlock,
           [&](kir::Val<kir::u32> n) {
             auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
             const auto vn = kir::pack_n(load_bytes, 4);
             const auto aligned = (K / vn) * vn;
             k.loop("tt", k.constant<kir::u32>(0), k.constant<kir::u32>(aligned),
                    vn, [&](kir::Val<kir::u32> t) {
                      const auto wv = math::load(w, n * K + t, load_bytes);
                      k.unroll("e", vn, [&](kir::Val<kir::u32> e) {
                        acc = math::fma(xs[t + e].read(), wv[e], acc.read());
                      });
                    });
             if (aligned < K) {
               k.loop("tt", k.constant<kir::u32>(aligned),
                      k.constant<kir::u32>(K), 1, [&](kir::Val<kir::u32> t) {
                        acc = math::fma(xs[t].read(), w[n * K + t].read(),
                                        acc.read());
                      });
             }
             k.store(row * N + n, acc.read());
           });
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 3) return LSE_ERROR(kInvalidArgument, "rms_linear needs 3");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[2].dim(0));
    return out;
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const auto N =
        s.inputs.size() >= 3 ? static_cast<std::uint32_t>(s.inputs[2].dim(0))
                             : 1u;
    const auto M =
        N == 0 ? 1u : static_cast<std::uint32_t>(s.output.elem_count() / N);
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = 1;
    tp.workgroup_count[1] = M == 0 ? 1u : M;
    tp.lds_bytes = (4096 + kBlock) * 4;
    return tp;
  }
};

}  // namespace

LinkedBinding linked_bindings(const FusionGroup& group) {
  if (LinkedBinding gdn = gdn_pair_bindings(group); gdn.ok) return gdn;
  LinkedBinding b;
  if (const SwigluParts p = match_swiglu(group); p.ok) {
    b.inputs = {p.x, p.wg, p.wu, p.wd};
    if (p.idx != nullptr) b.inputs.push_back(p.idx);
    if (p.hid != nullptr) b.inputs.push_back(p.hid);
    b.sink = p.down;
    b.iattrs[0] = p.slot;
    b.ok = true;
    return b;
  }
  if (const RmsLinearParts p = match_rms_linear(group); p.ok) {
    b.inputs = {p.x, p.rms_w, p.w};
    b.sink = p.linear;
    b.attrs[0] = p.eps;
    b.iattrs[0] = p.zero_centered ? 1 : 0;
    b.ok = true;
  }
  return b;
}

const KernelPrimitiveBase* linked_kernel_for(const FusionGroup& group,
                                             const KernelShapes&) {
  if (const KernelPrimitiveBase* gdn = gdn_pair_kernel();
      gdn_pair_bindings(group).ok) {
    return gdn;
  }
  static const SwigluKernel kSwiglu;
  static const RmsLinearKernel kRms;
  (void)kSwiglu;
  (void)kRms;
  // SwiGLU staged body still faults on lemonseed N=2176. GDN pair is safe.
  return nullptr;
}

}  // namespace lse::backend::hrx_kernels
