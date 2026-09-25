// Row-wise normalizations shared by device backends. RMSNorm specializes to
// one cooperative workgroup per row where the target supports the required
// scratch and barriers; the scalar forms remain the general fallback.
#include <string>
#include <limits>

#include "lse/backends/hrx/device_info.hpp"

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

bool usable(const KernelShapes& s) {
  return !s.inputs.empty() && s.types.scalar != nullptr &&
         s.intrinsics != nullptr && last_dim(s.inputs[0]) > 0;
}

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

// These kernels are not self-indexing: they hand their element back through
// `ret` and the emitter stores it, so `out` is bound but never written here.
template <class E, class G = kir::f32>
struct RmsNormArgs {
  env::In<kir::f32, E> x;
  env::In<G, E> g;
  env::Out<kir::f32, E> out;
};

namespace {
constexpr std::uint32_t kRmsBlock = 256;
constexpr std::uint32_t kRmsScratch = kRmsBlock * sizeof(float);

bool cooperative_rms_usable(const KernelShapes& s) {
  if (s.inputs.size() != 2 || s.inputs[0].rank() == 0 || !usable(s) ||
      s.input_dtypes.size() != 2 || s.inputs[1].rank() != 1 ||
      s.output != s.inputs[0] || s.input_dtypes[0] != DType::kF32 ||
      s.output_dtype != DType::kF32 || s.device == nullptr ||
      s.device->max_threads_per_workgroup < kRmsBlock ||
      backend::workgroup_lds_bytes(s.device) < kRmsScratch ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty()) return false;
  const auto dtype = s.input_dtypes[1];
  if (dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16)
    return false;
  const auto d = last_dim(s.inputs[0]);
  if (d < 32 || s.inputs[1].elem_count() != static_cast<std::size_t>(d) ||
      s.output.elem_count() == 0 ||
      s.output.elem_count() > std::numeric_limits<std::uint32_t>::max() / sizeof(float))
    return false;
  for (std::string_view op : {"thread.local_id", "thread.workgroup_id.x",
                              "barrier", "fma", "rsqrt"}) {
    if (s.intrinsics->find(op).empty()) return false;
  }
  return true;
}
}  // namespace

struct CooperativeRmsNormKernel final : KernelPrimitive<CooperativeRmsNormKernel> {
  static constexpr std::string_view kName = "rms_norm.cooperative.v1";
  static constexpr std::string_view kEntry = "lse_rms_norm_cooperative_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 2; }
  FusionClass fusion_class() const noexcept override { return FusionClass::kReduction; }
  bool owns_indexing() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!cooperative_rms_usable(s) || !s.store) return {};
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
      for (std::uint32_t offset = kRmsBlock / 2; offset > 0; offset >>= 1) {
        if (auto active = e.when(lane < offset)) {
          sums[lane] = sums[lane].read() + sums[lane + offset].read();
        }
        e.barrier();
      }
      // Every lane has finished reading the row before any output store.
      // Stores pass through the emitter hook so trailing elementwise work
      // keeps its original index and the normal buffer-alias contract.
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
  const KernelPrimitiveBase* specialize(const KernelShapes& s) const override {
    static const CooperativeRmsNormKernel cooperative;
    if (cooperative_rms_usable(s)) return &cooperative;
    return this;
  }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (!usable(s) || s.input_dtypes.size() < 2) return {};
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
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
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
    if (!usable(s)) return {};
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
