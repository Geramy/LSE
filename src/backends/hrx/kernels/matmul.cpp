#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <string>

#include "lse/backends/hrx/kernels/lds_linear.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/backends/hrx/kernels/wmma.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

// The value leaves through the wrapper's per-element return; the Out member
// only names the slot the binding contract requires.
template <class E, class Y = kir::f32>
struct MatMulArgs {
  env::In<kir::f32, E> x;
  env::In<Y, E> y;
  env::Out<kir::f32, E> out;
};

// Device implementation of the matmul barrier op. Lives with the backend, not
// with the graph: the graph layer decides *that* a matmul runs, this decides
// *how* it runs on this device.
//
// One thread per output element; correct but untiled, pending the WMMA
// version. The emitter wraps this body and appends any fused epilogue, so a
// linear followed by silu/mul/add is one launch, not four.
struct MatMulKernel final : KernelPrimitive<MatMulKernel> {
  static constexpr std::string_view kName = "matmul";
  static constexpr std::string_view kEntry = "lse_matmul";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }

  // Extents are baked in as literals rather than read from a constant: they are
  // already part of the JIT cache key, so specializing costs nothing and lets
  // the compiler size the loop.
  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 2 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.input_dtypes.size() < 2) {
      return {};
    }
    const Shape& a = s.inputs[0];
    const Shape& b = s.inputs[1];
    const auto kdim = static_cast<std::uint32_t>(a.dim(a.rank() - 1));
    const auto cols = static_cast<std::uint32_t>(b.dim(b.rank() - 1));

    return with_elem(s.input_dtypes[1], [&]<class Y>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics);
      MatMulArgs<env::Emit, Y> args;
      if (!env::bind(k, args, s)) return {};
      env::Emit e{&k};
      const auto row = e.let(e.thread_id() / cols);
      const auto col = e.let(e.thread_id() % cols);
      auto acc = e.var(0.0f);
      // B is [K, cols] in the usual matmul, so the inner walk on y is strided
      // and cannot be a wide load. x is contiguous in k.
      for (auto t : e.range(kdim)) {
        acc = math::fma(args.x[row * kdim + t],
                        math::widen(args.y[t * cols + col]), acc);
      }
      e.ret(acc.read());
      return k.str();
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) return LSE_ERROR(kInvalidArgument, "matmul takes 2 inputs");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(in[1].rank() - 1));
    return out;
  }

  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = (elems + threads - 1) / threads;
    return tp;
  }
};

LSE_REGISTER_PRIMITIVE(MatMulKernel);

template <class E, class W = kir::f32>
struct LinearArgs {
  env::In<kir::f32, E> x;
  env::In<W, E> w;
  env::Out<kir::f32, E> out;
};

// x [.., K] against w [N, K] -> [.., N]. The weight is stored transposed, the
// layout every checkpoint in this engine uses, so the inner loop walks both
// operands contiguously.
struct LinearKernel final : KernelPrimitive<LinearKernel> {
  static constexpr std::string_view kName = "linear";
  static constexpr std::string_view kEntry = "lse_linear";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }

  // Hands over to the matrix cores where they exist and the layout has been
  // verified; everywhere else this scalar loop stands.
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    // W is [N, K], x is [.., K]: both rows are contiguous in K, which is
    // the WMMA A/B layout. M below one tile goes to the wave-per-column
    // GEMV first: the padded 16x16 path masks 15/16 rows and its per-lane
    // K-strided weight loads measured ~9 GB/s on the M=1 vocab head, vs a
    // coalesced stream from the GEMV. lds_linear_for gates itself on m<16,
    // so WMMA keeps every shape with full tiles.
    if (const KernelPrimitiveBase* lds = lds_linear_for(s)) return lds;
    if (const KernelPrimitiveBase* wmma = wmma_linear_for(s)) return wmma;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 2 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.input_dtypes.size() < 2) {
      return {};
    }
    const auto kdim = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));

    return with_elem(s.input_dtypes[1], [&]<class W>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics);
      LinearArgs<env::Emit, W> args;
      if (!env::bind(k, args, s)) return {};
      env::Emit e{&k};
      const auto row = e.let(e.thread_id() / n);
      const auto col = e.let(e.thread_id() % n);
      auto acc = e.var(0.0f);
      emit_dot<W>(e, args.x, args.w, row * kdim, col * kdim, acc, kdim,
                  device_load_bytes(s.device));
      e.ret(acc.read());
      return k.str();
    });
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
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = (elems + threads - 1) / threads;
    return tp;
  }
};

LSE_REGISTER_PRIMITIVE(LinearKernel);

}  // namespace lse::backend::hrx_kernels
