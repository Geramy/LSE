#include "lse/graph/kernel_primitive.hpp"

#include <string>

#include "lse/kernels/vec_mem.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

using namespace lse::graph;

namespace {

template <class E, class S = kir::f32>
struct RowGatherArgs {
  env::In<S, E> src;
  env::In<kir::f32, E> rows;
  // Per-element kernel: the value leaves through k.ret, never this slot, but
  // bind requires the output declared.
  env::Out<kir::f32, E> out;
};

// out[s, c] = src[row[s], c]. rows are f32 indices, matching the rest of the
// graph. Width is the last axis of the output. The table is read in whatever
// format it is stored in and widens in register.
std::string emit_row_gather(const KernelShapes& s) {
  if (s.inputs.size() != 2 || s.types.scalar == nullptr ||
      s.intrinsics == nullptr || s.output.rank() == 0 ||
      s.input_dtypes.size() < 2) {
    return {};
  }
  const auto width =
      static_cast<std::uint32_t>(s.output.dim(s.output.rank() - 1));
  if (width == 0) return {};

  return with_elem(s.input_dtypes[0], [&]<class S>() -> std::string {
    kir::KernelBody k(s.types, *s.intrinsics);
    RowGatherArgs<env::Emit, S> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto slot = i / width;
    const auto col = i % width;
    const auto row = cast<kir::u32>(a.rows[slot]);
    e.ret(lse::math::widen(a.src[row * width + col]));
    return k.str();
  });
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

}  // namespace lse::kernels
