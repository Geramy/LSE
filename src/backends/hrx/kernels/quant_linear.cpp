// `quant_linear`: x [.., K] against a group-affine weight [N, K] that is never
// materialized. One wave owns one output column and walks the packed plane in
// chunks; each chunk's codes become weights in register through
// quant::dequant_chunk and are consumed by the accumulator immediately. The
// only global traffic is the packed lanes and one scale/bias pair per group,
// which is the whole reason the weight is not widened at load.
//
// `quant_linear_indexed`: the same contraction against one matrix of a stack
// [E, N, K], picked per token by `idx[row, slot]`. It is the quantized twin of
// linear_indexed and shares every line of the body below — the expert only
// moves where the plane is read from, exactly as it does in lds_linear.cpp,
// where `linear` and `linear_indexed` differ by a single base term. MLX's
// SwitchGLU stores 256 experts as one stacked tensor per projection, so
// without this op a routed FFN has no contraction at all.
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;
// Codes decoded per lane iteration. Past this the straight-line block stops
// paying for itself in registers.
constexpr std::uint32_t kMaxUnrolledCodes = 32;

std::uint32_t wave_of(const DeviceInfo* device) {
  if (device == nullptr) return 32;
  const std::uint32_t wave = device->wavefront_size;
  return (wave == 32 || wave == 64) ? wave : 32u;
}

struct QuantDims {
  quant::GroupAffine spec{};
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int64_t lanes = 0;   // packed u32 per weight row
  std::int64_t groups = 0;  // scale/bias entries per weight row
  std::int64_t experts = 0;  // 0 when the weight is a single matrix
  std::uint32_t keep = 0;    // width of the index row
  std::uint32_t slot = 0;
  bool valid = false;
};

// `indexed` selects the stacked form: five inputs, rank-3 planes, and the
// geometry one place further along the int attrs because iattrs[0] is the
// expert slot there. Every one of those attrs is mixed into the emitter's
// device_fn_name, so a 6-bit expert and an 8-bit router in the same checkpoint
// get different device functions without this file doing anything about it.
QuantDims dims_of(const KernelShapes& s, bool indexed) {
  QuantDims d;
  const std::size_t want = indexed ? 5u : 4u;
  if (s.inputs.size() != want || s.input_dtypes.size() < want) return d;
  if (s.input_dtypes[1] != DType::kU32) return d;
  if (s.input_dtypes[2] != s.input_dtypes[3]) return d;
  if (indexed && s.input_dtypes[4] != DType::kF32) return d;
  auto spec = indexed ? quant::GroupAffine::make(s.iattrs[1], s.iattrs[2])
                      : quant::GroupAffine::make(s.iattrs[0], s.iattrs[1]);
  if (!spec.ok()) return d;
  d.spec = *spec;

  const std::size_t rank = indexed ? 3u : 2u;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  const Shape& sc = s.inputs[2];
  const Shape& bi = s.inputs[3];
  if (x.rank() == 0 || w.rank() != rank || sc.rank() != rank ||
      bi.rank() != rank) {
    return d;
  }
  d.k = x.dim(x.rank() - 1);
  if (indexed) {
    d.experts = w.dim(0);
    d.n = w.dim(1);
    d.lanes = w.dim(2);
    d.groups = sc.dim(2);
    if (d.experts <= 0) return d;
    if (sc.dim(0) != d.experts || bi.dim(0) != d.experts) return d;
    if (sc.dim(1) != d.n || bi.dim(1) != d.n || bi.dim(2) != d.groups) return d;
  } else {
    d.n = w.dim(0);
    d.lanes = w.dim(1);
    d.groups = sc.dim(1);
    if (sc.dim(0) != d.n || bi.dim(0) != d.n || bi.dim(1) != d.groups) return d;
  }
  if (d.k <= 0 || d.n <= 0) return d;
  // The plane must describe exactly this K at exactly this bit width; a shape
  // that solves to another width is a different tensor, not a slower path.
  if (d.lanes * 32 != d.k * d.spec.bits) return d;
  if (d.groups * d.spec.group_size != d.k) return d;

  const auto elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  if (d.m <= 0) return d;

  if (indexed) {
    const Shape& ix = s.inputs[4];
    if (ix.rank() == 0) return d;
    d.keep = static_cast<std::uint32_t>(ix.dim(ix.rank() - 1));
    d.slot = static_cast<std::uint32_t>(s.iattrs[0]);
    if (d.keep == 0 || d.slot >= d.keep) return d;
    // One index row per output row, or `row * keep` walks off the end of a
    // buffer whose extent nothing else here constrains.
    if (static_cast<std::int64_t>(ix.elem_count()) != d.m * d.keep) return d;
    // Buffer subscripts are u32 the whole way down. The 35B's down_proj stack
    // is 2.7e8 lanes, comfortably inside; a stack that is not would silently
    // wrap into another expert's codes.
    const std::int64_t plane_elems = d.experts * d.n * d.lanes;
    if (plane_elems > static_cast<std::int64_t>(0xffffffffll)) return d;
  }
  d.valid = true;
  return d;
}

bool device_fits(const KernelShapes& s) {
  if (s.device == nullptr || s.intrinsics == nullptr) return false;
  if (s.intrinsics->find("wave.shfl_xor").empty()) return false;
  for (std::string_view sym : quant::kGroupAffineSymbols) {
    if (s.intrinsics->find(sym).empty()) return false;
  }
  return device_extension<AmdDeviceInfo>(*s.device) != nullptr;
}

// No row cap, unlike linear.lds: that one hands shapes above a tile to WMMA,
// and a packed plane has no matrix-core operand form to hand off to. Declining
// here would put the contraction on the host, so the wave-per-column schedule
// covers every row count — one workgroup row per token, which is a GEMV per
// token during prefill. A tiled form that shares a decoded chunk across
// several tokens is the improvement; correctness does not wait for it.
bool shape_ok(const QuantDims& d) { return d.valid; }

// Chunks a lane decodes per iteration. Consecutive weight rows start at
// multiples of `lanes`, so the run has to divide that too or some row's block
// straddles its natural alignment.
std::uint32_t chunks_per_step(const QuantDims& d, std::uint32_t max_bytes) {
  const auto words = static_cast<std::uint32_t>(d.spec.words_per_chunk());
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t cap =
      row_pack(static_cast<std::uint32_t>(d.lanes), max_bytes, 4);
  std::uint32_t cpl = 1;
  if ((words & (words - 1)) == 0 && words <= cap) cpl = cap / words;
  while (cpl > 1 && cpl * vals > kMaxUnrolledCodes) cpl >>= 1;
  return cpl;
}

ThreadPlan gemv_plan(const QuantDims& d, std::uint32_t wave,
                     std::uint32_t lds_budget) {
  if (wave != 32 && wave != 64) wave = 32;
  const std::uint32_t waves = kBlock / wave;
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  tp.workgroup_count[0] =
      static_cast<std::uint32_t>((d.n + waves - 1) / waves);
  tp.workgroup_count[1] = static_cast<std::uint32_t>(d.m > 0 ? d.m : 1);
  tp.workgroup_count[2] = 1;
  const std::uint32_t need = kir::Lds::align(
      static_cast<std::uint32_t>(d.k) * kir::pack_elem_bytes<kir::f32>());
  if (lds_budget == 0 || need <= lds_budget) tp.lds_bytes = need;
  return tp;
}

template <class S>
struct QuantLinearArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::Out<kir::f32, env::Emit> out;
};

template <class S>
struct QuantLinearIndexedArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<std::uint32_t, env::Emit> packed;
  env::In<S, env::Emit> scales;
  env::In<S, env::Emit> biases;
  env::In<kir::f32, env::Emit> idx;
  env::Out<kir::f32, env::Emit> out;
};

template <class A>
concept Indexed = requires(A& a) { a.idx; };

// acc += sum over the codes of one chunk. `xs` is the activation row staged in
// LDS, or null when it did not fit and the row is read from global.
template <class A>
void emit_chunk(env::Emit& e, const A& a, const kir::Tile<kir::f32>* xs,
                const quant::GroupAffine& spec,
                const kir::Val<kir::u32>& row_base,
                const kir::Val<kir::u32>& scale_base,
                const kir::Val<kir::u32>& x_base,
                const kir::Val<kir::u32>& chunk,
                const kir::LValue<kir::f32>& acc,
                std::uint32_t chunks_per_group) {
  const auto vals = static_cast<std::uint32_t>(spec.values_per_chunk());
  const auto words = static_cast<std::uint32_t>(spec.words_per_chunk());
  const auto group = e.let(chunk / chunks_per_group);
  const auto scale = e.let(math::widen(a.scales[scale_base + group]));
  const auto bias = e.let(math::widen(a.biases[scale_base + group]));
  const auto k_base = e.let(chunk * vals);
  quant::dequant_chunk(
      e, a.packed, spec, e.let(row_base + chunk * words), scale, bias,
      [&](int c, const kir::Val<kir::f32>& w) {
        const auto idx = e.let(k_base + static_cast<std::uint32_t>(c));
        const auto xe = xs != nullptr ? (*xs)[idx].read() : a.x[x_base + idx];
        acc = math::fma(xe, w, acc.read());
      });
}

template <class A>
std::string emit_body(const KernelShapes& s, const QuantDims& d) {
  const auto n = static_cast<std::uint32_t>(d.n);
  const auto k = static_cast<std::uint32_t>(d.k);
  const auto m = static_cast<std::uint32_t>(d.m);
  const auto lanes = static_cast<std::uint32_t>(d.lanes);
  const auto groups = static_cast<std::uint32_t>(d.groups);
  const auto vals = static_cast<std::uint32_t>(d.spec.values_per_chunk());
  const std::uint32_t nchunks = k / vals;
  const std::uint32_t chunks_per_group =
      static_cast<std::uint32_t>(d.spec.group_size) / vals;
  const std::uint32_t wave = wave_of(s.device);
  const std::uint32_t waves = kBlock / wave;
  const std::uint32_t cpl = chunks_per_step(d, device_load_bytes(s.device));
  const std::uint32_t span = wave * cpl;
  const std::uint32_t aligned = (nchunks / span) * span;
  const std::uint32_t ntiles = (n + waves - 1) / waves;

  kir::KernelBody kb(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  kb.set_store(s.store);
  A a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto tile = e.let(math::workgroup_id_x());
  const auto row = e.let(math::workgroup_id_y());
  const auto col = e.let(tile * waves + wave_id);

  // A fused run stages the shared row once, ahead of every sibling's body;
  // `s.staged` is that panel. Otherwise this body owns the staging.
  kir::Tile<kir::f32> xs;
  bool fill = false;
  if (s.staged.count == k && !s.staged.name.empty()) {
    xs = kir::Tile<kir::f32>(&kb, &kb.types(), std::string(s.staged.name), k);
  } else if (e.lds_fits<kir::f32>(k)) {
    xs = e.lds<kir::f32>(k);
    fill = true;
  }
  const bool stage = static_cast<bool>(xs);

  auto acc = e.var(0.0f);
  if (auto in_grid = e.when(tile < ntiles && row < m)) {
    if (fill) {
      for (auto t : e.range(lid, e.u32(k), kBlock)) {
        xs[t] = a.x[row * k + t];
      }
      e.barrier();
    }
    if (auto in_cols = e.when(col < n)) {
      // The expert's matrix within the stack. `linear` and `linear_indexed`
      // differ by exactly this term too; nothing downstream of it knows which
      // op it is in. topk writes float ids, and 256 experts are exact in f32.
      auto plane = e.u32(0);
      if constexpr (Indexed<A>) {
        plane = e.let(
            kir::cast<kir::u32>(a.idx[row * d.keep + e.u32(d.slot)]) * n);
      }
      const auto row_base = e.let((plane + col) * lanes);
      const auto scale_base = e.let((plane + col) * groups);
      // Zero on the staged arm: the panel is already row-relative, so adding
      // the row back would make every row read row workgroup_id_y * K of it.
      // On the global arm the row term is the only thing separating token r
      // from token 0 — dropping it produces fluent, uniformly wrong output.
      const auto x_base = e.let(stage ? e.u32(0) : row * k);
      for (auto c0 : e.range(0u, aligned, span)) {
        const auto chunk0 = e.let(c0 + lane * cpl);
        for (std::uint32_t u = 0; u < cpl; ++u) {
          emit_chunk<A>(e, a, stage ? &xs : nullptr, d.spec, row_base,
                        scale_base, x_base, e.let(chunk0 + u), acc,
                        chunks_per_group);
        }
      }
      if (aligned < nchunks) {
        for (auto chunk : e.range(e.u32(aligned) + lane, e.u32(nchunks), wave)) {
          emit_chunk<A>(e, a, stage ? &xs : nullptr, d.spec, row_base,
                        scale_base, x_base, chunk, acc, chunks_per_group);
        }
      }
    }
  }

  // Outside every guard: shfl_xor is wave-cooperative, and the guards above are
  // workgroup-uniform, so a wave that owns no column still has to take part.
  for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
    acc = acc.read() + math::shfl_xor(acc.read(), e.u32(bit));
  }
  if (auto lane0 = e.when(lane == 0 && col < n && row < m)) {
    e.store(row * n + col, acc.read());
  }
  if (!kb.lds().ok()) return {};
  return kb.str();
}

}  // namespace

struct QuantLinearKernel final : KernelPrimitive<QuantLinearKernel> {
  static constexpr std::string_view kName = "quant_linear";
  static constexpr std::string_view kEntry = "lse_quant_linear";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }

  StagedRow staged_row(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, false);
    if (!shape_ok(d) || !device_fits(s)) return {};
    return {0, static_cast<std::uint32_t>(d.k),
            static_cast<std::uint32_t>(d.m)};
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, false);
    if (!shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<QuantLinearArgs<S>>(s, d);
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4 || in[1].rank() != 2) {
      return LSE_ERROR(kInvalidArgument,
                       "quant_linear takes x, packed[N, lanes], scales, biases");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }

  // The codes carry the weight and the scale only places it, so the product is
  // an activation whatever the planes are stored as.
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gemv_plan(dims_of(s, false), wave_of(s.device),
                     workgroup_lds_bytes(s.device));
  }
};

LSE_REGISTER_PRIMITIVE(QuantLinearKernel);

// out[t, j] = x[t] · dequant(W[idx[t, slot]])[j], W stacked [E, N, K] as three
// planes. The stack is read in place — no expert is ever gathered out of it,
// and no code is widened outside a register.
struct QuantLinearIndexedKernel final
    : KernelPrimitive<QuantLinearIndexedKernel> {
  static constexpr std::string_view kName = "quant_linear_indexed";
  static constexpr std::string_view kEntry = "lse_quant_linear_indexed";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }

  StagedRow staged_row(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, true);
    if (!shape_ok(d) || !device_fits(s)) return {};
    return {0, static_cast<std::uint32_t>(d.k),
            static_cast<std::uint32_t>(d.m)};
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s, true);
    if (!shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<QuantLinearIndexedArgs<S>>(s, d);
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || in[1].rank() != 3) {
      return LSE_ERROR(kInvalidArgument,
                       "quant_linear_indexed takes x, packed[E, N, lanes], "
                       "scales, biases, idx");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(1));
    return out;
  }

  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    return gemv_plan(dims_of(s, true), wave_of(s.device),
                     workgroup_lds_bytes(s.device));
  }
};

LSE_REGISTER_PRIMITIVE(QuantLinearIndexedKernel);

}  // namespace lse::backend::hrx_kernels
