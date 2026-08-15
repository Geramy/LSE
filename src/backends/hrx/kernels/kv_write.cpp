#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

// Writes src [.., T, inner] into dst [.., Cap, inner] at a runtime begin.
// Threads cover src only; dest bytes the window does not touch stay put.
struct OverwriteSliceKernel final : KernelPrimitive<OverwriteSliceKernel> {
  static constexpr std::string_view kName = "overwrite_slice";
  static constexpr std::string_view kEntry = "lse_overwrite_slice";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 3 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store) {
      return {};
    }
    const Shape& dst = s.inputs[0];
    const Shape& src = s.inputs[1];
    const auto axis = static_cast<std::size_t>(s.iattrs[0]);
    if (axis >= dst.rank() || axis >= src.rank()) return {};

    std::uint32_t inner = 1;
    std::uint32_t src_axis = 1;
    std::uint32_t dst_axis = 1;
    for (std::size_t i = 0; i < dst.rank(); ++i) {
      const auto d = static_cast<std::uint32_t>(dst.dim(i));
      if (i > axis) inner *= d;
      else if (i == axis) dst_axis = d;
    }
    if (axis < src.rank()) {
      src_axis = static_cast<std::uint32_t>(src.dim(axis));
    }
    const auto src_n = static_cast<std::uint32_t>(src.elem_count());
    if (inner == 0 || src_axis == 0 || dst_axis == 0 || src_n == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    const auto in = k.input<kir::f32>(1);
    const auto begin = k.input<kir::f32>(2);
    const auto i = k.thread_id();
    k.ret_if(i >= src_n);

    const auto span = k.constant<kir::u32>(src_axis * inner);
    const auto o = k.let<kir::u32>("o", i / span);
    const auto rem = k.let<kir::u32>("rem", i % span);
    const auto a = k.let<kir::u32>("a", rem / inner);
    const auto ii = k.let<kir::u32>("ii", rem % inner);
    const auto pos = k.let<kir::u32>(
        "pos", kir::cast<kir::u32>(begin[k.constant<kir::u32>(0)].read()));
    const auto dest = k.let<kir::u32>(
        "dest", (o * dst_axis + pos + a) * inner + ii);
    k.store(dest, in[i].read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3) {
      return LSE_ERROR(kInvalidArgument, "overwrite_slice takes 3 inputs");
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
    const auto elems = s.inputs.size() > 1
                           ? static_cast<std::uint32_t>(s.inputs[1].elem_count())
                           : 1u;
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(OverwriteSliceKernel);

}  // namespace lse::backend::hrx_kernels
