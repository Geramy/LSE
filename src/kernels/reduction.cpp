// Row-wise normalization with cooperative and scalar implementations.
#include <algorithm>
#include <cmath>
#include <string>
#include <array>
#include <vector>

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
// Register-held rows: up to 32 values per lane, for up to 8 rows (a verify).
constexpr std::uint32_t kRmsCachedPerLane = 32, kRmsCachedRows = 8;
// Rows no wider than four wave32 lanes' worth (a head's 128) take a wave
// each, eight to a workgroup, instead of a 256-lane workgroup each.
constexpr std::uint32_t kNarrowRows = kRmsBlock / 32u;
bool narrow_rows(const KernelShapes& s, std::uint32_t d) {
  return s.device && s.device->wavefront_size == 32 && d <= 4u * 32u &&
         s.intrinsics && !s.intrinsics->find("wave.shfl_xor").empty();
}

}  // namespace

struct CooperativeRmsNormKernel final : KernelPrimitive<CooperativeRmsNormKernel> {
  static constexpr std::string_view kName = "rms_norm.cooperative.v3";
  static constexpr std::string_view kEntry = "lse_rms_norm_cooperative_v3";
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
      if (narrow_rows(s, d)) {
        // A wave per row. Lane l holds the squares of columns l, l+32, l+64
        // and l+96, which are exactly the four 256-lane partials the wide
        // tree adds for it (every other partial of a row this narrow is
        // zero), added in the same order; the xor butterfly then leaves
        // every lane with lane 0's total.
        const auto lid = e.let(math::local_id());
        const auto lane = e.let(lid % 32u);
        const auto row_index = e.let(math::workgroup_id_x() * kNarrowRows + lid / 32u);
        const auto rows = static_cast<std::uint32_t>(s.output.elem_count() / d);
        if (auto live = e.when(row_index < rows)) {
          const auto row = e.let(row_index * d);
          std::array<kir::Val<kir::f32>, 4> squares;
          std::array<kir::LValue<kir::f32>, 4> values{e.var(0.0f), e.var(0.0f), e.var(0.0f), e.var(0.0f)};
          for (std::uint32_t c = 0; c < 4u; ++c) {
            auto square = e.var(0.0f);
            if (auto in = e.when(lane + c * 32u < d)) {
              values[c] = a.x[row + lane + c * 32u];
              square = math::fma(values[c].read(), values[c].read(), e.f32(0.0f));
            }
            squares[c] = e.let(square.read());
          }
          const auto zero = e.f32(0.0f);
          const auto a0 = e.let(squares[0] + zero);
          const auto a1 = e.let(squares[2] + zero);
          const auto a2 = e.let(squares[1] + zero);
          const auto a3 = e.let(squares[3] + zero);
          auto total = e.let((a0 + a1) + (a2 + a3));
          for (std::uint32_t offset = 16; offset > 0; offset >>= 1)
            total = e.let(total + math::shfl_xor(total, e.u32(offset)));
          const auto scale = e.let(math::rsqrt(total / static_cast<float>(d) + s.attrs[0]));
          for (std::uint32_t c = 0; c < 4u; ++c) {
            const auto col = e.let(lane + c * 32u);
            if (auto in = e.when(col < d)) {
              const auto gain = math::widen(a.g[col]);
              const auto weight = s.iattrs[0] != 0 ? e.f32(1.0f) + gain : gain;
              e.store(row + col, values[c].read() * scale * weight);
            }
          }
        }
        return k.str();
      }
      const auto lane = e.let(math::local_id());
      const auto row = e.let(math::workgroup_id_x() * d);
      auto partial = e.var(0.0f);
      // Rows of a decode or verify step keep their values in registers, the
      // loads issued together, instead of a dependent load loop; the sums and
      // their order are unchanged. Long prefill chunks keep the loop.
      const bool cache_row = d % kRmsBlock == 0 && d / kRmsBlock <= kRmsCachedPerLane &&
          s.output.elem_count() <= static_cast<std::size_t>(kRmsCachedRows) * d &&
          s.device && s.device->wavefront_size == 32;
      std::vector<kir::Val<kir::f32>> row_values;
      if (cache_row) {
        // Retain decode inputs across the reduction without changing its order.
        for (std::uint32_t col = 0; col < d; col += kRmsBlock) {
          const auto value = e.let(a.x[row + lane + col]);
          row_values.push_back(value);
          partial = math::fma(value, value, partial.read());
        }
      } else {
        for (auto col : e.range(lane, e.u32(d), kRmsBlock)) {
          const auto value = e.let(a.x[row + col]);
          partial = math::fma(value, value, partial.read());
        }
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
      if (cache_row) {
        for (std::uint32_t offset = 0; offset < d; offset += kRmsBlock) {
          const auto col = e.let(lane + offset);
          const auto gain = math::widen(a.g[col]);
          const auto weight = s.iattrs[0] != 0 ? e.f32(1.0f) + gain : gain;
          e.store(row + col, row_values[offset / kRmsBlock] * scale * weight);
        }
      } else {
        for (auto col : e.range(lane, e.u32(d), kRmsBlock)) {
          const auto gain = math::widen(a.g[col]);
          const auto weight = s.iattrs[0] != 0 ? e.f32(1.0f) + gain : gain;
          const auto index = e.let(row + col);
          e.store(index, a.x[index] * scale * weight);
        }
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
    const auto d = static_cast<std::uint32_t>(last_dim(s.inputs[0]));
    const auto rows = static_cast<std::uint32_t>(
        s.output.elem_count() / static_cast<std::size_t>(d));
    if (narrow_rows(s, d)) {
      tp.workgroup_count[0] = (rows + kNarrowRows - 1u) / kNarrowRows;
      return tp;
    }
    tp.workgroup_count[0] = rows;
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
    const auto rows = static_cast<std::uint32_t>(s.inputs[0].elem_count() / kL2Head);
    const auto row = e.let(head * kL2Head);
    // A grid that overruns the rows parks the surplus waves on the last row:
    // they compute and store the same values, so no guard splits the body.
    const auto safe_row = rows % (kL2Block / kL2Wave) == 0
                              ? row
                              : e.let(select(head < rows, row, e.u32((rows - 1u) * kL2Head)));
    std::array<kir::Val<kir::f32>, kL2Head / kL2Wave> values;
    auto partial = e.var(0.0f);
    for (std::uint32_t j = 0; j < values.size(); ++j) {
      values[j] = e.let(a.x[safe_row + lane + j * kL2Wave]);
      partial = math::fma(values[j], values[j], partial.read());
    }
    auto sum = e.let(partial.read());
    for (std::uint32_t mask = 1; mask < kL2Wave; mask <<= 1) {
      sum = e.let(sum + math::shfl_xor(sum, e.u32(mask)));
    }
    const auto inv = e.let(
        e.f32(1.0f) / math::max(math::sqrt(sum), e.f32(s.attrs[0])));
    for (std::uint32_t j = 0; j < values.size(); ++j) {
      e.store(e.let(safe_row + lane + j * kL2Wave), values[j] * inv);
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
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const auto rows = s.inputs.empty()
                          ? kL2Rows
                          : static_cast<std::uint32_t>(s.inputs[0].elem_count() / kL2Head);
    constexpr std::uint32_t per_block = kL2Block / kL2Wave;
    tp.workgroup_size[0] = kL2Block;
    tp.workgroup_count[0] = (rows + per_block - 1u) / per_block;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(Wave32L2NormKernel);

// out = x / max(||x||, eps). eps floors the norm, it does not sit under the
// sqrt — see the note in the host implementation.
// SiLU over a column window of a row-major [.., C] tensor, then the L2 norm of
// each D-wide head of it: what slice -> silu -> reshape -> l2_normalize (and
// an optional constant scale) compute, in one dispatch instead of two. The
// SiLU is the elementwise "silu" spelling (x / (1 + exp(-x))) and the norm is
// the per-thread row scan of l2_normalize, so the values are those of the
// unfused graph. attrs: column offset, eps, scale (1 = none).
template <class E>
struct SiluL2Args {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};
constexpr std::uint32_t kSiluL2Threads = 256;
struct SiluL2NormKernel final : KernelPrimitive<SiluL2NormKernel> {
  static constexpr std::string_view kName = "gdn.silu_l2norm.v1";
  static constexpr std::string_view kEntry = "lse_gdn_silu_l2norm_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.inputs[0].rank() < 2 || s.output.rank() != s.inputs[0].rank() + 1 ||
        !s.device || !s.store || !s.types.scalar || !s.intrinsics ||
        s.input_dtypes[0] != DType::kF32 || s.output_dtype != DType::kF32) return {};
    for (const auto symbol : {"neg", "exp", "sqrt", "max", "fma", "barrier"})
      if (s.intrinsics->find(symbol).empty()) return {};
    const auto c = static_cast<std::uint32_t>(last_dim(s.inputs[0]));
    const auto d = static_cast<std::uint32_t>(last_dim(s.output));
    const auto heads = static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 2));
    const auto offset = static_cast<std::uint32_t>(s.attrs[0]);
    if (d == 0 || kSiluL2Threads % d != 0 || offset + heads * d > c) return {};
    kir::KernelBody k(s.types, *s.intrinsics, backend::workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SiluL2Args<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    // One head row per d threads: each thread computes its element's SiLU
    // once into workgroup memory, then scans the row's values in order.
    const auto rows = static_cast<std::uint32_t>(s.output.elem_count() / d);
    const auto per_group = kSiluL2Threads / d;
    const auto lid = e.let(math::local_id());
    const auto row = e.let(math::workgroup_id_x() * per_group + lid / d);
    const auto col = e.let(lid % d);
    const auto values = e.lds<kir::f32>(per_group * d);
    const auto slot = e.let((lid / d) * d);
    auto mine = e.var(0.0f);
    if (auto live = e.when(row < rows)) {
      const auto base = e.let((row / heads) * c + offset + (row % heads) * d);
      const auto x = e.let(a.x[base + col]);
      mine = x / (e.f32(1.0f) + math::exp(math::neg(x)));
      values[e.let(slot + col)] = mine.read();
    }
    e.barrier();
    if (auto live = e.when(row < rows)) {
      auto acc = e.var(0.0f);
      for (auto t : e.range(d)) {
        const auto v = e.let(values[e.let(slot + t)].read());
        acc = math::fma(v, v, acc.read());
      }
      const auto inv = e.let(
          e.f32(1.0f) / math::max(math::sqrt(acc.read()), e.f32(s.attrs[1])));
      auto value = e.let(mine.read() * inv);
      if (s.attrs[2] != 1.0f) value = e.let(value * e.f32(s.attrs[2]));
      e.store(e.let(row * d + col), value);
    }
    return k.str();
  }
  // The host reference, in the device body's order: SiLU per element, then
  // the head's sum of squares accumulated left to right with fma.
  bool has_typed_host_impl() const noexcept override { return true; }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4>& attrs,
                        const std::array<std::int32_t, 4>&) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 || out.dtype != DType::kF32 ||
        in[0].shape.rank() < 2 || out.shape.rank() != in[0].shape.rank() + 1)
      return LSE_ERROR(kInvalidArgument, "gdn.silu_l2norm.v1: invalid host storage");
    const auto c = static_cast<std::size_t>(last_dim(in[0].shape));
    const auto d = static_cast<std::size_t>(last_dim(out.shape));
    const auto heads = static_cast<std::size_t>(out.shape.dim(out.shape.rank() - 2));
    const auto offset = static_cast<std::size_t>(attrs[0]);
    const std::size_t rows = d ? out.shape.elem_count() / d : 0;
    if (d == 0 || heads == 0 || offset + heads * d > c ||
        in[0].bytes.size() < in[0].shape.elem_count() * 4 ||
        out.bytes.size() < out.shape.elem_count() * 4 ||
        rows / heads * c > in[0].shape.elem_count())
      return LSE_ERROR(kInvalidArgument, "gdn.silu_l2norm.v1: head window outside its input");
    const auto* x = reinterpret_cast<const float*>(in[0].bytes.data());
    auto* y = reinterpret_cast<float*>(out.bytes.data());
    std::vector<float> values(d);
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t base = (row / heads) * c + offset + (row % heads) * d;
      float acc = 0.0f;
      for (std::size_t t = 0; t < d; ++t) {
        const float v = x[base + t];
        values[t] = v / (1.0f + std::exp(-v));
        acc = std::fma(values[t], values[t], acc);
      }
      const float inv = 1.0f / std::max(std::sqrt(acc), attrs[1]);
      for (std::size_t t = 0; t < d; ++t) {
        float value = values[t] * inv;
        if (attrs[2] != 1.0f) value *= attrs[2];
        y[row * d + t] = value;
      }
    }
    return OkStatus();
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    // The head geometry is the caller's: gdn_silu_l2norm sets the shape.
    if (in.size() != 1) return LSE_ERROR(kInvalidArgument, "silu_l2norm takes 1 input");
    return in[0];
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const auto d = static_cast<std::uint32_t>(last_dim(s.output));
    const auto per_group = d ? kSiluL2Threads / d : 1u;
    const auto rows = static_cast<std::uint32_t>(d ? s.output.elem_count() / d : 0);
    tp.workgroup_size[0] = kSiluL2Threads;
    tp.workgroup_count[0] = (rows + per_group - 1) / per_group;
    tp.lds_bytes = kSiluL2Threads * static_cast<std::uint32_t>(sizeof(float));
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SiluL2NormKernel);

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
