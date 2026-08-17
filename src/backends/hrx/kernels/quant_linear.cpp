// `quant_linear`: x [.., K] against a group-affine weight [N, K] that is never
// materialized. One wave owns one output column and walks the packed plane in
// chunks; each chunk's codes become weights in register through
// quant::dequant_chunk and are consumed by the accumulator immediately. The
// only global traffic is the packed lanes and one scale/bias pair per group,
// which is the whole reason the weight is not widened at load.
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
  bool valid = false;
};

QuantDims dims_of(const KernelShapes& s) {
  QuantDims d;
  if (s.inputs.size() != 4 || s.input_dtypes.size() < 4) return d;
  if (s.input_dtypes[1] != DType::kU32) return d;
  if (s.input_dtypes[2] != s.input_dtypes[3]) return d;
  auto spec = quant::GroupAffine::make(s.iattrs[0], s.iattrs[1]);
  if (!spec.ok()) return d;
  d.spec = *spec;

  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  const Shape& sc = s.inputs[2];
  const Shape& bi = s.inputs[3];
  if (x.rank() == 0 || w.rank() != 2 || sc.rank() != 2 || bi.rank() != 2) {
    return d;
  }
  d.k = x.dim(x.rank() - 1);
  d.n = w.dim(0);
  d.lanes = w.dim(1);
  d.groups = sc.dim(1);
  if (d.k <= 0 || d.n <= 0) return d;
  if (sc.dim(0) != d.n || bi.dim(0) != d.n || bi.dim(1) != d.groups) return d;
  // The plane must describe exactly this K at exactly this bit width; a shape
  // that solves to another width is a different tensor, not a slower path.
  if (d.lanes * 32 != d.k * d.spec.bits) return d;
  if (d.groups * d.spec.group_size != d.k) return d;

  const auto elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
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

ThreadPlan gemv_plan(const QuantDims& d, std::uint32_t wave) {
  if (wave != 32 && wave != 64) wave = 32;
  const std::uint32_t waves = kBlock / wave;
  ThreadPlan tp;
  tp.workgroup_size[0] = kBlock;
  tp.workgroup_count[0] =
      static_cast<std::uint32_t>((d.n + waves - 1) / waves);
  tp.workgroup_count[1] = static_cast<std::uint32_t>(d.m > 0 ? d.m : 1);
  tp.workgroup_count[2] = 1;
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

// acc += sum over the codes of one chunk. `xs` is the activation row staged in
// LDS, or null when it did not fit and the row is read from global.
template <class S>
void emit_chunk(env::Emit& e, const QuantLinearArgs<S>& a,
                const kir::Tile<kir::f32>* xs, const quant::GroupAffine& spec,
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

template <class S>
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
  QuantLinearArgs<S> a;
  if (!env::bind(kb, a, s)) return {};
  env::Emit e{&kb};

  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto tile = e.let(math::workgroup_id_x());
  const auto row = e.let(math::workgroup_id_y());
  const auto col = e.let(tile * waves + wave_id);

  kir::Tile<kir::f32> xs;
  const bool stage = e.lds_fits<kir::f32>(k);
  if (stage) xs = e.lds<kir::f32>(k);

  auto acc = e.var(0.0f);
  if (auto in_grid = e.when(tile < ntiles && row < m)) {
    if (stage) {
      for (auto t : e.range(lid, e.u32(k), kBlock)) {
        xs[t] = a.x[row * k + t];
      }
      e.barrier();
    }
    if (auto in_cols = e.when(col < n)) {
      const auto row_base = e.let(col * lanes);
      const auto scale_base = e.let(col * groups);
      const auto x_base = e.let(stage ? e.u32(0) : row * k);
      for (auto c0 : e.range(0u, aligned, span)) {
        const auto chunk0 = e.let(c0 + lane * cpl);
        for (std::uint32_t u = 0; u < cpl; ++u) {
          emit_chunk<S>(e, a, stage ? &xs : nullptr, d.spec, row_base,
                        scale_base, x_base, e.let(chunk0 + u), acc,
                        chunks_per_group);
        }
      }
      if (aligned < nchunks) {
        for (auto chunk : e.range(e.u32(aligned) + lane, e.u32(nchunks), wave)) {
          emit_chunk<S>(e, a, stage ? &xs : nullptr, d.spec, row_base,
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

  std::string emit_kernel(const KernelShapes& s) const override {
    const QuantDims d = dims_of(s);
    if (!shape_ok(d) || !device_fits(s) || s.types.scalar == nullptr ||
        !s.store) {
      return {};
    }
    return with_elem(s.input_dtypes[2], [&]<class S>() -> std::string {
      return emit_body<S>(s, d);
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
    return gemv_plan(dims_of(s), wave_of(s.device));
  }
};

LSE_REGISTER_PRIMITIVE(QuantLinearKernel);

}  // namespace lse::backend::hrx_kernels
