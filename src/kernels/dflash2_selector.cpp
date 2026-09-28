#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <cmath>
#include <string>

namespace lse::kernels {
using namespace lse::graph;
namespace math = lse::math;
namespace {
Result<Shape> selector_shape(std::span<const Shape> in) {
  if (in.size() != 4 || in[0].rank() != 4 || in[1].rank() != 3 ||
      in[2].rank() != 4 || in[3].rank() != 3)
    return LSE_ERROR(kInvalidArgument, "DFlash2 selector needs predecessor, gate, successor, and unary tensors");
  const auto batch = in[0].dim(0), positions = in[0].dim(1);
  const auto predecessors = in[0].dim(2), rank = in[0].dim(3);
  const auto successors = in[2].dim(2);
  if (batch <= 0 || positions <= 0 || predecessors <= 0 || successors <= 0 || rank <= 0 ||
      in[1] != Shape{batch, positions, rank} ||
      in[2] != Shape{batch, positions, successors, rank} ||
      in[3] != Shape{batch, positions, successors})
    return LSE_ERROR(kInvalidArgument, "DFlash2 selector tensor geometry does not match");
  return Shape{batch, positions, predecessors, successors};
}
template <class E>
struct SelectorArgs {
  env::In<kir::f32, E> predecessor;
  env::In<kir::f32, E> gate;
  env::In<kir::f32, E> successor;
  env::In<kir::f32, E> unary;
  env::Out<kir::f32, E> out;
};
}
struct DFlash2SelectorKernel final : KernelPrimitive<DFlash2SelectorKernel> {
  static constexpr std::string_view kName = "dflash2.selector";
  static constexpr std::string_view kEntry = "lse_dflash2_selector";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }
  std::string emit_kernel(const KernelShapes& s) const override {
    auto shape = selector_shape(s.inputs);
    if (!shape.ok() || s.output != *shape || s.input_dtypes.size() != 4 ||
        s.output_dtype != DType::kF32 || !s.types.scalar || !s.intrinsics || !s.store) return {};
    for (auto type : s.input_dtypes) if (type != DType::kF32) return {};
    const auto positions = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto predecessors = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto successors = static_cast<std::uint32_t>(s.inputs[2].dim(2));
    const auto rank = static_cast<std::uint32_t>(s.inputs[0].dim(3));
    if (s.attrs != std::array<float, 4>{static_cast<float>(positions), static_cast<float>(predecessors),
        static_cast<float>(successors), static_cast<float>(rank)}) return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    SelectorArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto at = e.let(e.thread_id());
    if (auto valid = e.when(at < static_cast<std::uint32_t>(s.output.elem_count()))) {
      const auto successor = e.let(at % successors);
      const auto predecessor = e.let((at / successors) % predecessors);
      const auto row = e.let(at / (successors * predecessors));
      const auto pbase = e.let((row * predecessors + predecessor) * rank);
      const auto sbase = e.let((row * successors + successor) * rank);
      const auto gbase = e.let(row * rank);
      auto score = e.var(0.0f);
      for (auto r : e.range(rank)) {
        const auto conditioned = e.let(a.predecessor[pbase + r] * a.gate[gbase + r]);
        score = math::fma(conditioned, a.successor[sbase + r], score.read());
      }
      e.store(at, score.read() + a.unary[row * successors + successor]);
    }
    return body.str();
  }
  void eval_cpu(std::span<const float* const> in, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto predecessors = static_cast<std::size_t>(attrs[1]);
    const auto successors = static_cast<std::size_t>(attrs[2]);
    const auto rank = static_cast<std::size_t>(attrs[3]);
    for (std::size_t at = 0; at < count; ++at) {
      const auto successor = at % successors, predecessor = (at / successors) % predecessors;
      const auto row = at / (successors * predecessors);
      const auto pbase = (row * predecessors + predecessor) * rank;
      const auto sbase = (row * successors + successor) * rank;
      const auto gbase = row * rank;
      float score = 0;
      for (std::size_t r = 0; r < rank; ++r)
        score = std::fma(in[0][pbase + r] * in[1][gbase + r], in[2][sbase + r], score);
      out[at] = score + in[3][row * successors + successor];
    }
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override { return selector_shape(in); }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const std::uint32_t threads = s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    plan.workgroup_size[0] = threads;
    plan.workgroup_count[0] = static_cast<std::uint32_t>((s.output.elem_count() + threads - 1) / threads);
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2SelectorKernel);
}  // namespace lse::kernels
