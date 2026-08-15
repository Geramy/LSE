#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

// Interleaved pairs: (x0,x1) -> (x0 c - x1 s, x1 c + x0 s). cos/sin are
// [max_T, D]; position is offset + (row % seq).
struct RopeKernel final : KernelPrimitive<RopeKernel> {
  static constexpr std::string_view kName = "rope";
  static constexpr std::string_view kEntry = "lse_rope";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if ((s.inputs.size() != 3 && s.inputs.size() != 4) ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() < 2) {
      return {};
    }
    const Shape& x = s.inputs[0];
    const auto dim = static_cast<std::uint32_t>(x.dim(x.rank() - 1));
    const auto seq = static_cast<std::uint32_t>(x.dim(x.rank() - 2));
    const auto baked_off = static_cast<std::uint32_t>(s.iattrs[0]);
    const bool live_off = s.inputs.size() >= 4;
    if (dim < 2 || seq == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto in = k.input<kir::f32>(0);
    const auto cos = k.input<kir::f32>(1);
    const auto sin = k.input<kir::f32>(2);
    const auto i = k.thread_id();
    const auto d = k.let<kir::u32>("d", i % dim);
    const auto r = k.let<kir::u32>("r", i / dim);
    kir::Val<kir::u32> offset = k.constant<kir::u32>(baked_off);
    if (live_off) {
      const auto offb = k.input<kir::f32>(3);
      offset = k.let<kir::u32>(
          "off", kir::cast<kir::u32>(offb[k.constant<kir::u32>(0)].read()));
    }
    const auto t = k.let<kir::u32>("t", offset + (r % seq));
    const auto pair = k.let<kir::u32>("pair", (d / 2u) * 2u);
    const auto base = k.let<kir::u32>("base", r * dim + pair);
    const auto ang = k.let<kir::u32>("ang", t * dim + pair);
    const auto a = k.let<kir::f32>("a", in[base].read());
    const auto b = k.let<kir::f32>("b", in[base + 1u].read());
    const auto c = k.let<kir::f32>("c", cos[ang].read());
    const auto s0 = k.let<kir::f32>("s0", sin[ang].read());
    k.ret(select((d % 2u) == 0u, a * c - b * s0, b * c + a * s0));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 && in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "rope takes 3 or 4 inputs");
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
LSE_REGISTER_PRIMITIVE(RopeKernel);

}  // namespace lse::backend::hrx_kernels
