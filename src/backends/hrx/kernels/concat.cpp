#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

// Pack along one axis. Split points are literals; the thread picks which
// input owns its output coordinate and copies that element.
struct ConcatKernel final : KernelPrimitive<ConcatKernel> {
  static constexpr std::string_view kName = "concat";
  static constexpr std::string_view kEntry = "lse_concat";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 2 || s.inputs.size() > 4 ||
        s.types.scalar == nullptr || s.intrinsics == nullptr) {
      return {};
    }
    const auto axis = static_cast<std::size_t>(s.iattrs[0]);
    if (axis >= s.output.rank()) return {};

    std::uint32_t inner = 1;
    std::uint32_t out_axis = 1;
    for (std::size_t i = 0; i < s.output.rank(); ++i) {
      const auto d = static_cast<std::uint32_t>(s.output.dim(i));
      if (i > axis) inner *= d;
      else if (i == axis) out_axis = d;
    }
    if (inner == 0 || out_axis == 0) return {};

    std::uint32_t lens[4] = {};
    for (std::size_t p = 0; p < s.inputs.size(); ++p) {
      if (axis >= s.inputs[p].rank()) return {};
      lens[p] = static_cast<std::uint32_t>(s.inputs[p].dim(axis));
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto i = k.thread_id();
    const auto span = k.constant<kir::u32>(out_axis * inner);
    const auto o = k.let<kir::u32>("o", i / span);
    const auto rem = k.let<kir::u32>("rem", i % span);
    const auto a = k.let<kir::u32>("a", rem / inner);
    const auto ii = k.let<kir::u32>("ii", rem % inner);

    const auto v = k.var<kir::f32>("v", k.lit(0.0f));
    std::uint32_t off = 0;
    for (std::size_t p = 0; p < s.inputs.size(); ++p) {
      const auto src = k.input<kir::f32>(p);
      const auto lo = k.constant<kir::u32>(off);
      const auto hi = k.constant<kir::u32>(off + lens[p]);
      const auto local = k.let<kir::u32>("c" + std::to_string(p), a - lo);
      const auto idx = k.let<kir::u32>(
          "idx" + std::to_string(p), (o * lens[p] + local) * inner + ii);
      k.when(a >= lo && a < hi, [&] { v = src[idx].read(); });
      off += lens[p];
    }
    k.ret(v.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 2) return LSE_ERROR(kInvalidArgument, "concat needs 2+ inputs");
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

LSE_REGISTER_PRIMITIVE(ConcatKernel);

}  // namespace lse::backend::hrx_kernels
