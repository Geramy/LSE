// `linear` on the RDNA matrix cores.
//
// One wave owns one 16x16 output tile and walks K in steps of 16, accumulating
// in the v_wmma_f32_16x16x16_f16 fragment. The operand layout the instruction
// wants is the layout the checkpoints already have: lane L supplies row L of x
// and row L of w, both contiguous in k, which is exactly x[row*K+k] and
// w[col*K+k].
//
// The per-lane fragment mapping was measured on gfx1151 rather than taken from
// documentation, because it is not guessable and the two RDNA generations
// disagree. With a_frag[e] = a[16*lane+e] and b_frag[e] = b[16*lane+e]:
//
//     D[2*e + lane/16][lane%16] == c_frag[e]        256/256 slots, max err 5e-07
//
// gfx12 (RDNA4) is a different builtin name *and* a different operand width
// (v8f16, so k splits across the half-waves), so this kernel declines there
// rather than emit a layout nothing has run.
//
// Operands are f16: that is what the instruction multiplies. Against f32
// weights a linear's error moves from ~1e-6 to ~1e-3. Default is on when
// the live device has WMMA; LSE_WMMA=0 keeps the f32 scalar oracle.
#include "lse/backends/hrx/kernels/wmma.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::int64_t kTile = 16;

// LSE_WMMA=0 forces the scalar loop. Anything else, including unset, follows
// the device: gfx11 wave32 with matrix cores uses WMMA.
bool wmma_forced_off() {
  const char* v = std::getenv("LSE_WMMA");
  return v != nullptr && std::strcmp(v, "0") == 0;
}

struct LinearDims {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  bool valid = false;
};

LinearDims dims_of(const KernelShapes& s) {
  LinearDims d;
  if (s.inputs.size() != 2) return d;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  if (x.rank() == 0 || w.rank() != 2) return d;
  d.k = x.dim(x.rank() - 1);
  d.n = w.dim(0);
  if (d.k <= 0 || d.n <= 0 || w.dim(1) != d.k) return d;
  const std::int64_t elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

// The measured layout is the gfx11 wave32 form. Anything else keeps the scalar
// kernel: a matmul that is subtly wrong is worse than one that is merely slow.
bool device_supported(const KernelShapes& s) {
  if (s.device == nullptr) return false;
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(*s.device);
  if (amd == nullptr) return false;
  if (amd->matrix_core != MatrixCore::kWMMA) return false;
  if (amd->wavefront_size != 32) return false;
  const ArchFamily family = arch_family(s.device->arch);
  return family == ArchFamily::kRdna3 || family == ArchFamily::kRdna35;
}

template <class E>
struct WmmaLinearArgs {
  env::In<kir::f32, E> x;
  env::In<kir::f32, E> w;
  // Unused directly: owns_indexing stores go through the emitter hook, but the
  // binding contract still names the output slot.
  env::Out<kir::f32, E> out;
};

struct WmmaLinearKernel final : KernelPrimitive<WmmaLinearKernel> {
  static constexpr std::string_view kName = "linear.wmma";
  static constexpr std::string_view kEntry = "lse_linear_wmma";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }

  // The kernel, in C++. Nothing below names a HIP type or a builtin: types come
  // from s.types and the matrix instruction from s.intrinsics, so retargeting
  // this is two table entries rather than an edit here.
  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of(s);
    if (!d.valid || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store) {
      return {};
    }

    const auto M = static_cast<std::uint32_t>(d.m);
    const auto N = static_cast<std::uint32_t>(d.n);
    const auto K = static_cast<std::uint32_t>(d.k);
    const std::uint32_t tiles_n = (N + kTile - 1) / kTile;
    const std::uint32_t tiles = ((M + kTile - 1) / kTile) * tiles_n;

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    WmmaLinearArgs<env::Emit> a;
    env::bind(k, a);
    env::Emit e{&k};

    const auto wave = e.let(e.thread_id() / 32u);
    (void)e.ret_if(wave >= tiles);

    const auto lane = e.let(e.thread_id() % 32u);
    const auto l16 = e.let(lane % 16u);
    const auto half = e.let(lane / 16u);
    const auto m0 = e.let((wave / tiles_n) * kTile);
    const auto n0 = e.let((wave % tiles_n) * kTile);

    // Lane L supplies row L of x and row L of w. Both are contiguous in k, so
    // the fragment load is the checkpoint's own layout.
    const auto arow = e.let(m0 + l16);
    const auto bcol = e.let(n0 + l16);

    const auto acc = e.local<kir::f32, 8>();
    for (auto z : e.unroll(8)) acc[z] = 0.0f;

    const auto load_bytes = device_load_bytes(s.device);
    for (auto k0 : e.range(0u, K, kTile)) {
      const auto af = e.local<lse::f16, 16>();
      const auto bf = e.local<lse::f16, 16>();
      const auto zero = cast<lse::f16>(e.f32(0.0f));
      for (auto z : e.unroll(kTile)) {
        af[z] = zero;
        bf[z] = zero;
      }
      const auto step =
          kir::pack_n(load_bytes, kir::pack_elem_bytes<kir::f32>());
      const auto chunks = static_cast<std::uint32_t>(kTile) / step;
      for (auto c : e.unroll(chunks)) {
        const auto kk = e.let(k0 + c * step);
        const auto kend = e.let(kk + step);
        if (auto ga = e.when(arow < M && kend <= K)) {
          const auto xv = e.load(a.x, arow * K + kk, load_bytes);
          for (auto v : e.unroll(step)) {
            af[c * step + v] = cast<lse::f16>(xv[v]);
          }
        }
        if (auto gb = e.when(bcol < N && kend <= K)) {
          const auto wv = e.load(a.w, bcol * K + kk, load_bytes);
          for (auto v : e.unroll(step)) {
            bf[c * step + v] = cast<lse::f16>(wv[v]);
          }
        }
      }

      acc = lse::math::wmma_f32_16x16x16(af.value(), bf.value(), acc.value());
    }

    const auto col = e.let(n0 + l16);
    (void)e.ret_if(col >= N);
    for (auto z : e.unroll(8)) {
      const auto row = e.let(m0 + z * 2u + half);
      // Storing through the hook, not into `out`: this is where any fused
      // silu/mul/add runs, on the accumulator still in register.
      if (auto gr = e.when(row < M)) e.store(row * N + col, acc[z].read());
    }

    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) return LSE_ERROR(kInvalidArgument, "linear takes 2 inputs");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }

  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  // A tile is a wave, not an output element, so the launch is sized in waves.
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const LinearDims d = dims_of(s);
    const std::int64_t tiles_n = (d.n + kTile - 1) / kTile;
    const std::int64_t tiles = ((d.m + kTile - 1) / kTile) * tiles_n;

    const std::uint32_t cap =
        s.device != nullptr && s.device->max_threads_per_workgroup >= 256
            ? 256u : 64u;
    const std::uint32_t waves = cap / 32u;
    tp.workgroup_size[0] = waves * 32u;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(
        (tiles + waves - 1) / waves);
    return tp;
  }
};

}  // namespace

const KernelPrimitiveBase* wmma_linear_for(const KernelShapes& s) {
  static const WmmaLinearKernel kKernel;
  if (wmma_forced_off()) return nullptr;
  if (!device_supported(s)) return nullptr;
  const LinearDims d = dims_of(s);
  if (!d.valid || d.k < kTile || d.n < kTile) return nullptr;
  return &kKernel;
}

}  // namespace lse::backend::hrx_kernels
