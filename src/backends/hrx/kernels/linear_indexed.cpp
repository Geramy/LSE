#include "lse/graph/kernel_primitive.hpp"

#include <string>

#include "kernels/lds_linear.hpp"
#include "kernels/vec_mem.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

// out[t, n] = x[t] · W[idx[t, slot], n]. W is stacked [E, N, K].
struct LinearIndexedKernel final : KernelPrimitive<LinearIndexedKernel> {
  static constexpr std::string_view kName = "linear_indexed";
  static constexpr std::string_view kEntry = "lse_linear_indexed";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    if (const KernelPrimitiveBase* lds = lds_linear_indexed_for(s)) return lds;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 3 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[1].rank() != 3 ||
        s.inputs[2].rank() == 0) {
      return {};
    }
    const auto kdim = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto nout = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    const auto keep = static_cast<std::uint32_t>(
        s.inputs[2].dim(s.inputs[2].rank() - 1));
    const auto slot = static_cast<std::uint32_t>(s.iattrs[0]);
    if (kdim == 0 || nout == 0 || keep == 0 || slot >= keep) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto w = k.input<kir::f32>(1);
    const auto idx = k.input<kir::f32>(2);
    const auto row = k.let<kir::u32>("row", k.thread_id() / nout);
    const auto col = k.let<kir::u32>("col", k.thread_id() % nout);
    const auto e = k.let<kir::u32>(
        "e", cast<kir::u32>(
                 idx[row * keep + k.constant<kir::u32>(slot)].read()));
    const auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
    const auto base =
        k.let<kir::u32>("base", (e * nout + col) * k.constant<kir::u32>(kdim));
    emit_dot_f32(k, x, w, row * kdim, base, acc, kdim,
                 device_load_bytes(s.device));
    k.ret(acc.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 || in[1].rank() != 3) {
      return LSE_ERROR(kInvalidArgument, "linear_indexed takes x, W[E,N,K], idx");
    }
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i) out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(1));
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
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(LinearIndexedKernel);

}  // namespace lse::backend::hrx_kernels
