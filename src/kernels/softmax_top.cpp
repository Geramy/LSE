// The largest softmax probability of each row, in two launches, without the
// row's probabilities ever being written: max(softmax(x)) = 1 / sum(exp(x -
// max(x))).
//
// The partial stage gives one workgroup per chunk of a row: the workgroup
// reduces its chunk's maximum through LDS, then the sum of exp(x - that
// maximum), and lane 0 writes (maximum, sum) for the chunk. The final stage is
// one thread per row folding the pairs against the row's maximum and
// returning the inverse of the total. A row of -inf has no softmax; it comes
// out NaN, for the caller to refuse.
//
// An MTP draft reads its proposal's probability this way: the argmax is the
// proposal, and this is how sure the module was of it.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using backend::workgroup_lds_bytes;
using namespace lse::graph;
namespace math = lse::math;

namespace {

constexpr std::uint32_t kBlock = 256;
// One workgroup reduces one chunk of a row.
constexpr std::int64_t kChunk = 4096;

std::int64_t chunks_of(std::int64_t n) { return (n + kChunk - 1) / kChunk; }

// (maximum, sum of exp(x - maximum)) of one chunk.
std::pair<float, float> chunk_mass(const float* x, std::int64_t n) {
  float top = -std::numeric_limits<float>::infinity();
  for (std::int64_t i = 0; i < n; ++i) top = std::max(top, x[i]);
  const float shift = top == -std::numeric_limits<float>::infinity() ? 0.0f : top;
  float sum = 0.0f;
  for (std::int64_t i = 0; i < n; ++i) sum += std::exp(x[i] - shift);
  return {top, sum};
}

template <class E>
struct SoftmaxTopPartialArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

template <class E>
struct SoftmaxTopFinalArgs {
  env::In<kir::f32, E> x;
  env::Out<kir::f32, E> out;
};

}  // namespace

// [rows, n] -> [rows, chunks, 2].
struct SoftmaxTopPartialKernel final : KernelPrimitive<SoftmaxTopPartialKernel> {
  static constexpr std::string_view kName = "softmax_top.partial";
  static constexpr std::string_view kEntry = "lse_softmax_top_partial";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        !s.store || s.inputs[0].rank() != 2 || s.output.rank() != 3 ||
        s.input_dtypes.size() != 1 || s.input_dtypes[0] != DType::kF32)
      return {};
    const auto n = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto nchunks = static_cast<std::uint32_t>(s.output.dim(1));
    if (n == 0 || nchunks != chunks_of(n) || s.output.dim(2) != 2) return {};

    kir::KernelBody k(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
    k.set_store(s.store);
    SoftmaxTopPartialArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};

    const auto lid = e.let(math::local_id());
    const auto c = e.let(math::workgroup_id_x());
    const auto row = e.let(math::workgroup_id_y());
    const auto start = e.let(c * static_cast<std::uint32_t>(kChunk));
    auto stop = e.var(start + e.u32(static_cast<std::uint32_t>(kChunk)));
    if (auto clip = e.when(stop.read() > e.u32(n))) {
      stop = e.u32(n);
    }

    auto red = e.lds<kir::f32>(kBlock);
    if (!red) return {};
    auto top = e.var(0.0f);
    top = math::neg_inf();
    for (auto t : e.range(start + lid, stop.read(), kBlock)) {
      top = math::max(top.read(), a.x[row * n + t]);
    }
    red[lid] = top.read();
    e.barrier();
    for (std::uint32_t off = kBlock / 2; off > 0; off >>= 1) {
      if (auto low = e.when(lid < off)) {
        red[lid] = math::max(red[lid].read(), red[lid + off].read());
      }
      e.barrier();
    }
    const auto chunk_top = e.let(red[0].read());
    e.barrier();
    const auto shift = e.let(select(chunk_top == math::neg_inf(), e.f32(0.0f), chunk_top));
    auto sum = e.var(0.0f);
    for (auto t : e.range(start + lid, stop.read(), kBlock)) {
      sum = sum.read() + math::exp(a.x[row * n + t] - shift);
    }
    red[lid] = sum.read();
    e.barrier();
    for (std::uint32_t off = kBlock / 2; off > 0; off >>= 1) {
      if (auto low = e.when(lid < off)) {
        red[lid] = red[lid].read() + red[lid + off].read();
      }
      e.barrier();
    }
    if (auto lead = e.when(lid == 0)) {
      const auto slot = e.let((row * nchunks + c) * e.u32(2));
      e.store(slot, chunk_top);
      e.store(slot + e.u32(1), red[0].read());
    }
    if (!k.lds().ok()) return {};
    return k.str();
  }

  void eval_cpu(std::span<const float* const> in, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto n = static_cast<std::int64_t>(attrs[0]);
    const auto nchunks = chunks_of(n);
    const auto rows = static_cast<std::int64_t>(count) / (2 * nchunks);
    for (std::int64_t r = 0; r < rows; ++r)
      for (std::int64_t c = 0; c < nchunks; ++c) {
        const auto first = c * kChunk;
        const auto [top, sum] = chunk_mass(in[0] + r * n + first, std::min(kChunk, n - first));
        out[(r * nchunks + c) * 2] = top;
        out[(r * nchunks + c) * 2 + 1] = sum;
      }
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != 2 || in[0].dim(1) <= 0)
      return LSE_ERROR(kInvalidArgument, "softmax_top.partial takes [rows, n]");
    return Shape{in[0].dim(0), chunks_of(in[0].dim(1)), 2};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = s.output.rank() == 3 ? static_cast<std::uint32_t>(s.output.dim(1)) : 1u;
    tp.workgroup_count[1] = s.output.rank() == 3 ? static_cast<std::uint32_t>(s.output.dim(0)) : 1u;
    tp.lds_bytes = kBlock * static_cast<std::uint32_t>(sizeof(float));
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SoftmaxTopPartialKernel);

// [rows, chunks, 2] -> [rows]: the largest softmax probability of each row.
struct SoftmaxTopFinalKernel final : KernelPrimitive<SoftmaxTopFinalKernel> {
  static constexpr std::string_view kName = "softmax_top.final";
  static constexpr std::string_view kEntry = "lse_softmax_top_final";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 1; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 1 || s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() != 3 || s.inputs[0].dim(2) != 2)
      return {};
    const auto nchunks = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    if (nchunks == 0) return {};
    kir::KernelBody k(s.types, *s.intrinsics);
    SoftmaxTopFinalArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto base = e.let(e.thread_id() * (2 * nchunks));
    auto top = e.var(0.0f);
    top = math::neg_inf();
    for (auto c : e.range(nchunks)) {
      top = math::max(top.read(), a.x[base + c * 2u]);
    }
    auto sum = e.var(0.0f);
    for (auto c : e.range(nchunks)) {
      const auto m = e.let(a.x[base + c * 2u]);
      // A chunk of -inf adds nothing (its sum is zero); exp(-inf - -inf)
      // would be NaN.
      if (auto live = e.when(m > math::neg_inf())) {
        sum = sum.read() + a.x[base + c * 2u + 1u] * math::exp(m - top.read());
      }
    }
    e.ret(e.f32(1.0f) / sum.read());
    return k.str();
  }

  void eval_cpu(std::span<const float* const> in, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto nchunks = static_cast<std::size_t>(attrs[0]);
    for (std::size_t r = 0; r < count; ++r) {
      const float* x = in[0] + r * nchunks * 2;
      float top = -std::numeric_limits<float>::infinity();
      for (std::size_t c = 0; c < nchunks; ++c) top = std::max(top, x[c * 2]);
      float sum = 0.0f;
      for (std::size_t c = 0; c < nchunks; ++c)
        if (x[c * 2] > -std::numeric_limits<float>::infinity())
          sum += x[c * 2 + 1] * std::exp(x[c * 2] - top);
      out[r] = 1.0f / sum;
    }
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != 3 || in[0].dim(2) != 2)
      return LSE_ERROR(kInvalidArgument, "softmax_top.final takes [rows, chunks, 2]");
    return Shape{in[0].dim(0)};
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 64 ? 64u : 1u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SoftmaxTopFinalKernel);

}  // namespace lse::kernels
