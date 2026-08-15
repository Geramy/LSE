#include "lse/graph/kernel_primitive.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;

namespace {

// out[s, c] = src[row[s], c]. rows are f32 indices, matching the rest of the
// graph. Width is the last axis of the output.
std::string emit_row_gather(const KernelShapes& s) {
  if (s.inputs.size() != 2 || s.types.scalar == nullptr ||
      s.intrinsics == nullptr || s.output.rank() == 0) {
    return {};
  }
  const auto width =
      static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 1));
  if (width == 0) return {};

  kir::KernelBody k(s.types, *s.intrinsics);
  const auto src = k.input<kir::f32>(0);
  const auto rows = k.input<kir::f32>(1);
  const auto i = k.thread_id();
  const auto slot = k.let<kir::u32>("s", i / width);
  const auto col = k.let<kir::u32>("col", i % width);
  const auto row = k.let<kir::u32>("row", cast<kir::u32>(rows[slot].read()));
  k.ret(src[row * width + col].read());
  return k.str();
}

ThreadPlan plan_by_elems(const KernelShapes& s) {
  ThreadPlan tp;
  const std::uint32_t threads =
      s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
  const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
  tp.workgroup_size[0] = threads;
  tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
  return tp;
}

}  // namespace

struct GatherKernel final : KernelPrimitive<GatherKernel> {
  static constexpr std::string_view kName = "gather";
  static constexpr std::string_view kEntry = "lse_gather";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  std::string emit_kernel(const KernelShapes& s) const override {
    return emit_row_gather(s);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) return LSE_ERROR(kInvalidArgument, "gather takes 2 inputs");
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) { return plan_by_elems(s); }
};
LSE_REGISTER_PRIMITIVE(GatherKernel);

struct EmbeddingKernel final : KernelPrimitive<EmbeddingKernel> {
  static constexpr std::string_view kName = "embedding";
  static constexpr std::string_view kEntry = "lse_embedding";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 2; }
  std::string emit_kernel(const KernelShapes& s) const override {
    return emit_row_gather(s);
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 2) {
      return LSE_ERROR(kInvalidArgument, "embedding takes 2 inputs");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) { return plan_by_elems(s); }
};
LSE_REGISTER_PRIMITIVE(EmbeddingKernel);

}  // namespace lse::backend::hrx_kernels
