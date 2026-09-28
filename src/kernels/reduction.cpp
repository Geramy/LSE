// Row-wise normalization with cooperative and scalar implementations.
#include <string>
#include <array>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/dispatch/attention.hpp"

#include "lse/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

std::int64_t last_dim(const Shape& s) { return s.dim(s.rank() - 1); }

// Sum of squares over the row this thread's element belongs to. Extents are
// literals: shapes are already part of the JIT cache key.
kir::LValue<kir::f32> sum_of_squares(env::Emit& e,
                                     const env::In<kir::f32, env::Emit>& x,
                                     const kir::Val<kir::u32>& row,
                                     std::uint32_t d) {
  auto acc = e.var(0.0f);
  for (auto t : e.range(d)) {
    const auto v = e.let(x[row + t]);
    acc = math::fma(v, v, acc.read());
  }
  return acc;
}

}  // namespace

// Scalar kernels return an element through the emitter store hook.
template <class E, class G = kir::f32>
struct RmsNormArgs {
  env::In<kir::f32, E> x;
  env::In<G, E> g;
  env::Out<kir::f32, E> out;
};

namespace {
constexpr std::uint32_t kRmsBlock = 256;
constexpr std::uint32_t kRmsScratch = kRmsBlock * sizeof(float);

}  // namespace

struct CooperativeRmsNormKernel final : KernelPrimitive<CooperativeRmsNormKernel> {
  static constexpr std::string_view kName = "rms_norm.cooperative.v2";
  static constexpr std::string_view kEntry = "lse_rms_norm_cooperative_v2";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 2; }
  FusionClass fusion_class() const noexcept override { return FusionClass::kReduction; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::cooperative_rms_supported(s) || !s.store) return {};
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));
    return with_elem(s.input_dtypes[1], [&]<class G>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics, backend::workgroup_lds_bytes(s.device));
      k.set_store(s.store);
      RmsNormArgs<env::Emit, G> a;
      if (!env::bind(k, a, s)) return {};
      env::Emit e{&k};
      const auto lane = e.let(math::local_id());
      const auto row = e.let(math::workgroup_id_x() * d);
      auto partial = e.var(0.0f);
      for (auto col : e.range(lane, e.u32(d), kRmsBlock)) {
        const auto value = e.let(a.x[row + col]);
        partial = math::fma(value, value, partial.read());
      }
      auto sums = e.lds<kir::f32>(kRmsBlock);
      if (!sums) return {};
      sums[lane] = partial.read();
      e.barrier();
      // Preserve the original 256-lane FP32 tree inside the first wave.
      if (auto first_wave = e.when(lane < 32u)) {
        const auto a0 = e.let(sums[lane].read() + sums[lane + 128u].read());
        const auto a1 = e.let(sums[lane + 64u].read() + sums[lane + 192u].read());
        const auto a2 = e.let(sums[lane + 32u].read() + sums[lane + 160u].read());
        const auto a3 = e.let(sums[lane + 96u].read() + sums[lane + 224u].read());
        auto total = e.let((a0 + a1) + (a2 + a3));
        for (std::uint32_t offset = 16; offset > 0; offset >>= 1)
          total = e.let(total + math::shfl_xor(total, e.u32(offset)));
        if (auto lead = e.when(lane == 0u)) sums[0] = total;
      }
      e.barrier();
      // All row reads precede stores; the hook preserves fused epilogues.
      const auto scale = e.let(math::rsqrt(
          sums[0].read() / static_cast<float>(d) + s.attrs[0]));
      for (auto col : e.range(lane, e.u32(d), kRmsBlock)) {
        const auto gain = math::widen(a.g[col]);
        const auto weight = s.iattrs[0] != 0 ? e.f32(1.0f) + gain : gain;
        const auto index = e.let(row + col);
        e.store(index, a.x[index] * scale * weight);
      }
      if (!k.lds().ok()) return {};
      return k.str();
    });
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
    tp.workgroup_size[0] = kRmsBlock;
    tp.workgroup_count[0] = static_cast<std::uint32_t>(
        s.output.elem_count() / static_cast<std::size_t>(last_dim(s.inputs[0])));
    tp.lds_bytes = kRmsScratch;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(CooperativeRmsNormKernel);

// FP32 RMS reduction; zero-centered gains add one before scaling.
struct RmsNormKernel final : KernelPrimitive<RmsNormKernel> {
  static constexpr std::string_view kName = "rms_norm";
  static constexpr std::string_view kEntry = "lse_rms_norm";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kReduction;
  }
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    static const CooperativeRmsNormKernel cooperative;
    if (dispatch::cooperative_rms_supported(s)) return &cooperative;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::reduction_row_supported(s) || s.input_dtypes.size() < 2) return {};
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    return with_elem(s.input_dtypes[1], [&]<class G>() -> std::string {
      kir::KernelBody k(s.types, *s.intrinsics);
      RmsNormArgs<env::Emit, G> a;
      if (!env::bind(k, a, s)) return {};
      env::Emit e{&k};
      const auto row = e.let((e.thread_id() / d) * d);
      const auto col = e.let(e.thread_id() % d);
      const auto acc = sum_of_squares(e, a.x, row, d);
      const auto scale =
          e.let(math::rsqrt(acc.read() / static_cast<float>(d) + s.attrs[0]));
      // iattrs[0] selects the zero-centered form, where the stored weight is an
      // offset from 1 rather than the scale itself.
      const auto gain = math::widen(a.g[col]);
      const auto w = s.iattrs[0] != 0 ? e.f32(1.0f) + gain : gain;
      e.ret(a.x[e.thread_id()] * scale * w);
      return k.str();
    });
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
        dispatch::scalar_threads(s);
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(RmsNormKernel);

template <class E>
struct L2NormArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

namespace {
constexpr std::uint32_t kL2Wave = 32;
constexpr std::uint32_t kL2Head = 128;
constexpr std::uint32_t kL2Block = 128;
constexpr std::uint32_t kL2Rows = 16;

}  // namespace

// Each complete wave owns one head. Read all four elements per lane before
// storing so input/output aliasing cannot expose a partially rewritten row.
struct Wave32L2NormKernel final : KernelPrimitive<Wave32L2NormKernel> {
  static constexpr std::string_view kName = "l2_normalize.wave32.d128.m1.v1";
  static constexpr std::string_view kEntry = "lse_l2_normalize_wave32_d128_m1_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  FusionClass fusion_class() const noexcept override { return FusionClass::kReduction; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::wave32_l2_supported(s) || !s.store) return {};
    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    L2NormArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto local = e.let(math::local_id());
    const auto lane = e.let(local % kL2Wave);
    const auto head = e.let(math::workgroup_id_x() * (kL2Block / kL2Wave) +
                           local / kL2Wave);
    const auto row = e.let(head * kL2Head);
    std::array<kir::Val<kir::f32>, kL2Head / kL2Wave> values;
    auto partial = e.var(0.0f);
    for (std::uint32_t j = 0; j < values.size(); ++j) {
      values[j] = e.let(a.x[row + lane + j * kL2Wave]);
      partial = math::fma(values[j], values[j], partial.read());
    }
    auto sum = e.let(partial.read());
    for (std::uint32_t mask = 1; mask < kL2Wave; mask <<= 1) {
      sum = e.let(sum + math::shfl_xor(sum, e.u32(mask)));
    }
    const auto inv = e.let(
        e.f32(1.0f) / math::max(math::sqrt(sum), e.f32(s.attrs[0])));
    for (std::uint32_t j = 0; j < values.size(); ++j) {
      e.store(e.let(row + lane + j * kL2Wave), values[j] * inv);
    }
    return k.str();
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "l2_normalize takes 1 input");
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes&) {
    ThreadPlan tp;
    tp.workgroup_size[0] = kL2Block;
    tp.workgroup_count[0] = kL2Rows / (kL2Block / kL2Wave);
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(Wave32L2NormKernel);

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
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    static const Wave32L2NormKernel wave32;
    if (dispatch::wave32_l2_supported(s)) return &wave32;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!dispatch::reduction_row_supported(s)) return {};
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    kir::KernelBody k(s.types, *s.intrinsics);
    L2NormArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto row = e.let((e.thread_id() / d) * d);
    const auto acc = sum_of_squares(e, a.x, row, d);
    const auto inv = e.let(
        e.f32(1.0f) / math::max(math::sqrt(acc.read()), e.f32(s.attrs[0])));
    e.ret(a.x[e.thread_id()] * inv);
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

template <class E>
struct SoftmaxArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

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
    if (!dispatch::reduction_row_supported(s)) return {};
    if (static_cast<std::size_t>(s.iattrs[0]) + 1 != s.inputs[0].rank()) {
      return {};
    }
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));

    kir::KernelBody k(s.types, *s.intrinsics);
    SoftmaxArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto row = e.let((e.thread_id() / d) * d);

    // env vars lift only float literals; seed, then overwrite with the
    // recorded -inf before any read.
    auto m = e.var(0.0f);
    m = math::neg_inf();
    for (auto t : e.range(d)) {
      m = math::max(m.read(), a.x[row + t]);
    }

    auto denom = e.var(0.0f);
    for (auto t : e.range(d)) {
      denom = denom.read() + math::exp(a.x[row + t] - m.read());
    }

    e.ret(math::exp(a.x[e.thread_id()] - m.read()) / denom.read());
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

}  // namespace lse::kernels
