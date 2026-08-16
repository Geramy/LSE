#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

// `W` is the filter's storage element; the weight and the bias come from the
// same checkpoint, so one parameter covers both. A mismatch makes bind refuse.
template <class E, class W = kir::f32>
struct CausalConv1dArgs {
  env::In<kir::f32, E> x;
  env::In<W, E> w;
  env::In<W, E> bias;
  env::In<kir::f32, E> tail;  // optional 4th input: previous K-1 columns
  env::Out<kir::f32, E> out;
};

// Depthwise causal conv: out[b,t,c] = bias[c] + sum_j w[c,j] * x[b,t-(K-1-j),c]
// with zeros before t=0, or the tail [B,K-1,C] when the 4th input is present.
// K is a literal; lemonseed uses a small kernel.
struct CausalConv1dKernel final : KernelPrimitive<CausalConv1dKernel> {
  static constexpr std::string_view kName = "causal_conv1d";
  static constexpr std::string_view kEntry = "lse_causal_conv1d";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if ((s.inputs.size() != 3 && s.inputs.size() != 4) ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() < 2 || s.input_dtypes.size() < 3) {
      return {};
    }
    const Shape& x = s.inputs[0];
    const auto channels = static_cast<std::uint32_t>(x.dim(x.rank() - 1));
    const auto seq = static_cast<std::uint32_t>(x.dim(x.rank() - 2));
    const auto kernel = static_cast<std::uint32_t>(s.inputs[1].dim(1));
    if (channels == 0 || seq == 0 || kernel == 0) return {};
    const bool has_tail = s.inputs.size() == 4;
    if (has_tail) {
      const Shape& tl = s.inputs[3];
      if (tl.rank() < 2 || tl.dim(tl.rank() - 1) != channels ||
          tl.dim(tl.rank() - 2) != kernel - 1) {
        return {};
      }
    }

    return with_elem(s.input_dtypes[1], [&]<class W>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics);
      CausalConv1dArgs<env::Emit, W> a;
      if (!env::bind(k, a, s)) return {};
      env::Emit e{&k};
      const auto i = e.thread_id();
      const auto c = e.let(i % channels);
      const auto t = e.let((i / channels) % seq);
      const auto bi = e.let(i / (channels * seq));
      auto acc = e.var(0.0f);
      acc = math::widen(a.bias[c]);
      for (std::uint32_t j = 0; j < kernel; ++j) {
        const auto back = kernel - 1 - j;
        const auto src_t = e.let(t - back);
        const auto src = e.let((bi * seq + src_t) * channels + c);
        const auto tap = e.let(math::widen(a.w[c * kernel + j]));
        if (auto in_range = e.when(t >= back)) {
          acc = math::fma(a.x[src], tap, acc);
        }
        if (has_tail && back > 0) {
          // t < back <= K-1 puts t+j inside the K-1 tail columns.
          const auto tsrc = e.let((bi * (kernel - 1) + t + j) * channels + c);
          if (auto in_tail = e.when(t < back)) {
            acc = math::fma(a.tail[tsrc], tap, acc);
          }
        }
      }
      e.ret(acc.read());
      return k.str();
    });
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 3 && in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "causal_conv1d takes 3 or 4 inputs");
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

template <class E>
struct ConvTailArgs {
  env::In<kir::f32, E> tail;
  env::In<kir::f32, E> x;
  // Ret-style kernel: the element value is returned, not stored; the slot
  // exists to satisfy the binding contract.
  env::Out<kir::f32, E> out;
};

// The advanced conv window: out [B,L,C] holds the last L columns of
// tail ++ x. One copy in place of the concat + slice pair.
struct ConvTailShiftKernel final : KernelPrimitive<ConvTailShiftKernel> {
  static constexpr std::string_view kName = "conv_tail";
  static constexpr std::string_view kEntry = "lse_conv_tail";
  static constexpr std::string_view kSource = {};
  static constexpr FusionClass kClass = FusionClass::kStructural;

  std::size_t arity() const noexcept override { return 2; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 2 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || s.inputs[1].rank() < 2 ||
        s.output.rank() < 2) {
      return {};
    }
    const Shape& x = s.inputs[1];
    const auto channels = static_cast<std::uint32_t>(x.dim(x.rank() - 1));
    const auto seq = static_cast<std::uint32_t>(x.dim(x.rank() - 2));
    const auto keep =
        static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 2));
    if (channels == 0 || seq == 0 || keep == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    ConvTailArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto c = e.let(i % channels);
    const auto p = e.let((i / channels) % keep);
    const auto bi = e.let(i / (channels * keep));
    if (seq >= keep) {
      e.ret(a.x[(bi * seq + (seq - keep) + p) * channels + c]);
    } else {
      // Position seq + p in tail ++ x: the first keep-seq slots come from
      // the old tail shifted left, the rest from x.
      const auto shift = keep - seq;
      if (auto from_tail = e.when(p < shift)) {
        e.ret(a.tail[(bi * keep + seq + p) * channels + c]);
      }
      e.ret(a.x[(bi * seq + p - shift) * channels + c]);
    }
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) {
      return LSE_ERROR(kInvalidArgument, "conv_tail takes 2 inputs");
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
LSE_REGISTER_PRIMITIVE(ConvTailShiftKernel);

}  // namespace lse::backend::hrx_kernels
