#include "lse/graph/kernel_primitive.hpp"

#include <string>

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

namespace {

template <class E>
struct ScatterAddArgs {
  env::In<kir::f32, E> base;
  env::In<kir::f32, E> rows;
  env::In<kir::f32, E> values;
  // Per-element kernel: the value leaves through k.ret, never this slot, but
  // bind requires the output declared.
  env::Out<kir::f32, E> out;
};

}  // namespace

// out = base; out[row[s]] += values[s]. Two experts can land on one token, so
// this accumulates. The scatter list is short (tokens routed to one expert),
// so each output thread scans it rather than taking atomics.
struct ScatterAddKernel final : KernelPrimitive<ScatterAddKernel> {
  static constexpr std::string_view kName = "scatter";
  static constexpr std::string_view kEntry = "lse_scatter";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 3 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.output.rank() == 0) {
      return {};
    }
    const auto width =
        static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 1));
    const auto count = static_cast<std::uint32_t>(s.inputs[1].elem_count());
    if (width == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    ScatterAddArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto dst_row = e.let(i / width);
    const auto col = e.let(i % width);
    auto v = e.var(a.base[i]);
    for (std::uint32_t sidx = 0; sidx < count; ++sidx) {
      const auto src_row = e.let(cast<kir::u32>(a.rows[sidx]));
      if (auto g = e.when(src_row == dst_row)) {
        v = v.read() + a.values[sidx * width + col];
      }
    }
    e.ret(v.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3) {
      return LSE_ERROR(kInvalidArgument, "scatter takes 3 inputs");
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
LSE_REGISTER_PRIMITIVE(ScatterAddKernel);

}  // namespace lse::backend::hrx_kernels
