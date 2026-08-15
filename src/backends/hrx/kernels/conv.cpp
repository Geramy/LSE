#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

template <class E>
struct CausalConv1dArgs {
  env::In<kir::f32, E> x;
  env::In<kir::f32, E> w;
  env::In<kir::f32, E> bias;
  env::Out<kir::f32, E> out;
};

// Depthwise causal conv: out[b,t,c] = bias[c] + sum_j w[c,j] * x[b,t-(K-1-j),c]
// with zeros before t=0. K is a literal; lemonseed uses a small kernel.
struct CausalConv1dKernel final : KernelPrimitive<CausalConv1dKernel> {
  static constexpr std::string_view kName = "causal_conv1d";
  static constexpr std::string_view kEntry = "lse_causal_conv1d";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 3 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[0].rank() < 2) {
      return {};
    }
    const Shape& x = s.inputs[0];
    const auto channels = static_cast<std::uint32_t>(x.dim(x.rank() - 1));
    const auto seq = static_cast<std::uint32_t>(x.dim(x.rank() - 2));
    const auto kernel = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    if (channels == 0 || seq == 0 || kernel == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    CausalConv1dArgs<env::Emit> a;
    env::bind(k, a);
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto c = e.let(i % channels);
    const auto t = e.let((i / channels) % seq);
    const auto bi = e.let(i / (channels * seq));
    auto acc = e.var(0.0f);
    acc = a.bias[c];
    for (std::uint32_t j = 0; j < kernel; ++j) {
      const auto back = kernel - 1 - j;
      const auto src_t = e.let(t - back);
      const auto src = e.let((bi * seq + src_t) * channels + c);
      if (auto in_range = e.when(t >= back)) {
        acc = math::fma(a.x[src], a.w[c * kernel + j], acc);
      }
    }
    e.ret(acc.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3) {
      return LSE_ERROR(kInvalidArgument, "causal_conv1d takes 3 inputs");
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
LSE_REGISTER_PRIMITIVE(CausalConv1dKernel);

}  // namespace lse::backend::hrx_kernels
