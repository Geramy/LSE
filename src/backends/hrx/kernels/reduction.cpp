// Row-wise normalizations, as device kernels.
//
// These are the ops that kept the whole model on the host: 41 RMSNorms plus
// every softmax and l2-normalize. Each one reduces over the last axis and
// produces a tensor the same shape as its input, so it fits the one-thread-
// per-output-element contract without needing cross-thread cooperation.
//
// Each thread re-reduces its own row rather than sharing a partial sum through
// LDS. That is `dim` redundant reads per element, which is a real cost worth
// removing later — but it is correct, it needs no new machinery, and it is far
// cheaper than the host round trip it replaces.
#include <string>

#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

std::int64_t last_dim(const Shape& s) { return s.dim(s.rank() - 1); }

bool usable(const KernelShapes& s) {
  return !s.inputs.empty() && s.types.scalar != nullptr &&
         s.intrinsics != nullptr && last_dim(s.inputs[0]) > 0;
}

// Sum of squares over the row this thread's element belongs to. Extents are
// literals: shapes are already part of the JIT cache key.
kir::LValue<kir::f32> sum_of_squares(kir::KernelBody& k,
                                     const kir::Buffer<kir::f32>& x,
                                     const kir::Val<kir::u32>& row,
                                     std::uint32_t d) {
  const auto acc = k.var<kir::f32>("acc", k.lit(0.0f));
  k.loop("t", k.constant<kir::u32>(0), k.constant<kir::u32>(d), 1,
         [&](kir::Val<kir::u32> t) {
           const auto v = k.let<kir::f32>("v", x[row + t].read());
           acc = math::fma(v, v, acc.read());
         });
  return acc;
}

}  // namespace

// scale = rsqrt(mean(x^2) + eps); out = x * scale * (bias + w[col]).
// The host reference accumulates in fp64 and this in fp32; over 1024 terms
// that stays inside the 1e-5 relative bound the differential tests hold to.
struct RmsNormKernel final : KernelPrimitive<RmsNormKernel> {
  static constexpr std::string_view kName = "rms_norm";
  static constexpr std::string_view kEntry = "lse_rms_norm";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kReduction;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!usable(s)) return {};
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto g = k.input<kir::f32>(1);
    const auto row = k.let<kir::u32>("row", (k.thread_id() / d) * d);
    const auto col = k.let<kir::u32>("col", k.thread_id() % d);
    const auto acc = sum_of_squares(k, x, row, d);
    const auto scale = k.let<kir::f32>(
        "scale", math::rsqrt(acc.read() / static_cast<float>(d) + s.attrs[0]));
    // iattrs[0] selects the zero-centered form, where the stored weight is an
    // offset from 1 rather than the scale itself.
    const auto w = s.iattrs[0] != 0 ? k.lit(1.0f) + g[col].read()
                                    : g[col].read();
    k.ret(x[k.thread_id()].read() * scale * w);
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) return LSE_ERROR(kInvalidArgument, "rms_norm takes 2 inputs");
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
    tp.workgroup_count[0] = (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(RmsNormKernel);

// out = x / max(||x||, eps). eps floors the norm, it does not sit under the
// sqrt — see the note in the host implementation.
struct L2NormKernel final : KernelPrimitive<L2NormKernel> {
  static constexpr std::string_view kName = "l2_normalize";
  static constexpr std::string_view kEntry = "lse_l2_normalize";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kReduction;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!usable(s)) return {};
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto row = k.let<kir::u32>("row", (k.thread_id() / d) * d);
    const auto acc = sum_of_squares(k, x, row, d);
    const auto inv = k.let<kir::f32>(
        "inv", k.lit(1.0f) / math::max(math::sqrt(acc.read()),
                                       k.lit(s.attrs[0])));
    k.ret(x[k.thread_id()].read() * inv);
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "l2_normalize takes 1 input");
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    return RmsNormKernel::plan_impl(s);
  }
};
LSE_REGISTER_PRIMITIVE(L2NormKernel);

// Max-shifted softmax over the last axis. Any other axis declines, and the
// group falls back to the host rather than emitting the wrong reduction.
struct SoftmaxKernel final : KernelPrimitive<SoftmaxKernel> {
  static constexpr std::string_view kName = "softmax";
  static constexpr std::string_view kEntry = "lse_softmax";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kReduction;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!usable(s)) return {};
    if (static_cast<std::size_t>(s.iattrs[0]) + 1 != s.inputs[0].rank()) {
      return {};
    }
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    kir::KernelBody k(s.types, *s.intrinsics);
    const auto x = k.input<kir::f32>(0);
    const auto row = k.let<kir::u32>("row", (k.thread_id() / d) * d);

    const auto m = k.var<kir::f32>("m", math::neg_inf());
    k.loop("t", k.constant<kir::u32>(0), k.constant<kir::u32>(d), 1,
           [&](kir::Val<kir::u32> t) {
             m = math::max(m.read(), x[row + t].read());
           });

    const auto denom = k.var<kir::f32>("denom", k.lit(0.0f));
    k.loop("t", k.constant<kir::u32>(0), k.constant<kir::u32>(d), 1,
           [&](kir::Val<kir::u32> t) {
             denom = denom.read() + math::exp(x[row + t].read() - m.read());
           });

    k.ret(math::exp(x[k.thread_id()].read() - m.read()) / denom.read());
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "softmax takes 1 input");
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    return RmsNormKernel::plan_impl(s);
  }
};
LSE_REGISTER_PRIMITIVE(SoftmaxKernel);

}  // namespace lse::backend::hrx_kernels
