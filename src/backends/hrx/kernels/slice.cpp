#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

// Device copy of a slice. Stays on the device so a last-token take or a conv
// tail is a D2D launch, not a host bounce. The general (outer, axis, inner)
// index is baked in as literals; shapes are already in the JIT key.
struct SliceKernel final : KernelPrimitive<SliceKernel> {
  static constexpr std::string_view kName = "slice";
  static constexpr std::string_view kEntry = "lse_slice";
  static constexpr std::string_view kSource = {};
  static constexpr FusionClass kClass = FusionClass::kStructural;

  std::size_t arity() const noexcept override { return 1; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr) {
      return {};
    }
    const Shape& in = s.inputs[0];
    const auto axis = static_cast<std::size_t>(s.iattrs[0]);
    if (axis >= in.rank()) return {};
    const auto begin = static_cast<std::uint32_t>(s.iattrs[1]);
    std::uint32_t inner = 1;
    std::uint32_t in_axis = 1;
    std::uint32_t out_axis = 1;
    for (std::size_t i = 0; i < in.rank(); ++i) {
      const auto d = static_cast<std::uint32_t>(in.dim(i));
      if (i > axis) inner *= d;
      else if (i == axis) in_axis = d;
    }
    if (axis < s.output.rank()) {
      out_axis = static_cast<std::uint32_t>(s.output.dim(axis));
    }
    if (inner == 0 || out_axis == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto i = k.thread_id();
    const auto span = k.constant<kir::u32>(out_axis * inner);
    const auto o = k.let<kir::u32>("o", i / span);
    const auto rem = k.let<kir::u32>("rem", i % span);
    const auto a = k.let<kir::u32>("a", rem / inner);
    const auto ii = k.let<kir::u32>("ii", rem % inner);
    const auto src = k.let<kir::u32>(
        "src", (o * in_axis + begin + a) * inner + ii);
    k.ret(x[src].read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "slice takes 1 input");
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

LSE_REGISTER_PRIMITIVE(SliceKernel);

}  // namespace lse::backend::hrx_kernels
