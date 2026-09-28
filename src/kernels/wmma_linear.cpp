#include "lse/kernels/wmma.hpp"
#include "lse/dispatch/quant.hpp"
#include "lse/dispatch/attention.hpp"

#include <cstring>
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

// These name device facts, which the backend supplies.

using namespace lse::graph;
namespace math = lse::math;

namespace {
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
template <math::MatrixTarget... Live, class F>
[[nodiscard]] std::string for_live_target(math::MatrixTarget t, F&& fn) {
  std::string out;
  const auto one = [&]<math::MatrixTarget L>() {
    if (t == L) out = fn.template operator()<L>();
  };
  (one.template operator()<Live>(), ...);
  return out;
}

struct LinearDims {
  std::int64_t m = 0;
  std::int64_t n = 0;
  // Row length in *buffer* elements. Equal to K for an unpacked operand; K/4
  // for int8 riding four to an i32.
  std::int64_t kb = 0;
  bool valid = false;
};

LinearDims dims_of(const KernelShapes& s) {
  LinearDims d;
  if (s.inputs.size() != 2) return d;
  const Shape& x = s.inputs[0];
  const Shape& w = s.inputs[1];
  if (x.rank() == 0 || w.rank() != 2) return d;
  d.kb = x.dim(x.rank() - 1);
  d.n = w.dim(0);
  if (d.kb <= 0 || d.n <= 0 || w.dim(1) != d.kb) return d;
  const std::int64_t elems = static_cast<std::int64_t>(s.output.elem_count());
  if (elems <= 0 || elems % d.n != 0) return d;
  d.m = elems / d.n;
  d.valid = d.m > 0;
  return d;
}

const math::MatrixCoreRow* row_for(const KernelShapes& s) {
  return dispatch::linear_matrix_row(s);
}

template <class E, class X = kir::f32, class W = kir::f32>
struct MatrixLinearArgs {
  env::In<X, E> x;
  env::In<W, E> w;
  // Unused directly: owns_indexing stores go through the emitter hook, but the
  // binding contract still names the output slot.
  env::Out<kir::f32, E> out;
};
template <math::MatrixTarget G, math::MatrixElem A, math::MatrixElem T>
struct LinearTile
    : MatrixTile<LinearTile<G, A, T>, G, A, T, kTileM, kTileN, kTileK> {
  std::uint32_t cols = 0;

  void emit_element(env::Emit& e, const kir::Val<kir::u32>& row,
                    const kir::Val<kir::u32>& col,
                    const kir::Val<kir::f32>& v) const {
    e.store(row * cols + col, v);
  }
};

struct MatrixLinearKernel final : KernelPrimitive<MatrixLinearKernel> {
  static constexpr std::string_view kName = "linear.wmma";
  static constexpr std::string_view kEntry = "lse_linear_wmma";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    const LinearDims d = dims_of(s);
    const math::MatrixCoreRow* row = row_for(s);
    if (!d.valid || row == nullptr || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store) {
      return {};
    }
    return for_live_target<math::MatrixTarget::kRdna3,
                                   math::MatrixTarget::kRdna4>(
        row->target, [&]<math::MatrixTarget G>() -> std::string {
          return with_matrix_operand<std::string>(
              s.input_dtypes[1],
              [&]<class X, class W, math::MatrixElem A, math::MatrixElem T>()
                  -> std::string { return emit_body<G, X, W, A, T>(s, d); });
        });
  }
  template <math::MatrixTarget G, class X, class W, math::MatrixElem A,
            math::MatrixElem T>
  std::string emit_body(const KernelShapes& s, const LinearDims& d) const {
    const auto M = static_cast<std::uint32_t>(d.m);
    const auto N = static_cast<std::uint32_t>(d.n);
    const auto KB = static_cast<std::uint32_t>(d.kb);

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    MatrixLinearArgs<env::Emit, X, W> a;
    if (!env::bind(k, a, s)) {
      return {};
    }
    env::Emit e{&k};

    LinearTile<G, A, T> tile;
    tile.cols = N;
    tile.run(e, a.x, a.w, M, N, KB, device_load_bytes(s.device));
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
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const math::MatrixCoreRow* row = row_for(s);
    if (row == nullptr) return tp;  // never selected; nothing to size
    const LinearDims d = dims_of(s);
    const auto lanes = static_cast<std::uint32_t>(row->wave);
    const std::int64_t tiles_n = (d.n + row->n - 1) / row->n;
    const std::int64_t tiles = ((d.m + row->m - 1) / row->m) * tiles_n;

    const std::uint32_t cap = dispatch::scalar_threads(s);
    const std::uint32_t waves = cap / lanes;
    tp.workgroup_size[0] = waves * lanes;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(
        (tiles + waves - 1) / waves);
    return tp;
  }
};

}  // namespace

const KernelPrimitiveBase* wmma_linear_for(const KernelShapes& s) {
  static const MatrixLinearKernel kKernel;
  const math::MatrixCoreRow* row = row_for(s);
  if (row == nullptr) return nullptr;
  const LinearDims d = dims_of(s);
  if (!d.valid) return nullptr;
  if (d.kb < row->k_step / row->pack || d.n < row->n) return nullptr;
  return &kKernel;
}

}  // namespace lse::kernels
