#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

// Output axis d reads source axis iattrs[d]. Rank > 4 has no slot in iattrs
// and declines, so the host reference keeps covering those.
struct TransposeKernel final : KernelPrimitive<TransposeKernel> {
  static constexpr std::string_view kName = "transpose";
  static constexpr std::string_view kEntry = "lse_transpose";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr) {
      return {};
    }
    const Shape& in = s.inputs[0];
    const Shape& out = s.output;
    if (in.rank() != out.rank() || in.rank() == 0 || in.rank() > 4) return {};

    const auto in_st = in.strides();
    const auto out_st = out.strides();

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto i = k.thread_id();
    auto src = k.var<kir::u32>("src", k.constant<kir::u32>(0));
    for (std::size_t d = 0; d < out.rank(); ++d) {
      const auto perm = static_cast<std::size_t>(s.iattrs[d]);
      if (perm >= in.rank()) return {};
      const auto coord = k.let<kir::u32>(
          "c" + std::to_string(d),
          (i / static_cast<std::uint32_t>(out_st[d])) %
              static_cast<std::uint32_t>(out.dim(d)));
      src = src.read() +
            coord * static_cast<std::uint32_t>(in_st[perm]);
    }
    k.ret(x[src.read()].read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) {
      return LSE_ERROR(kInvalidArgument, "transpose takes 1 input");
    }
    return in[0];
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

LSE_REGISTER_PRIMITIVE(TransposeKernel);

}  // namespace lse::backend::hrx_kernels
