#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"

#include <array>
#include <string>

namespace lse::kernels {
using namespace lse::graph;
namespace {
Result<Shape> convolution_shape(std::span<const Shape> inputs) {
  if (inputs.size() != 3 || inputs[0].rank() != 3 || inputs[1].rank() != 4 ||
      inputs[2].rank() != 2)
    return LSE_ERROR(kInvalidArgument, "DFlash2 convolution requires hidden, dynamic and base tensors");
  const auto& x = inputs[0];
  if (x.dim(0) <= 0 || x.dim(1) <= 0 || x.dim(2) <= 0 ||
      inputs[1].dim(0) != x.dim(0) || inputs[1].dim(1) != x.dim(1) ||
      inputs[1].dim(2) != 2 || inputs[1].dim(3) <= 0 ||
      x.dim(2) % inputs[1].dim(3) || inputs[2] != Shape{2, x.dim(2)})
    return LSE_ERROR(kInvalidArgument, "DFlash2 convolution geometry does not match");
  for (const auto& shape : inputs) {
    std::uint64_t count = 1;
    for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
      const auto dim = static_cast<std::uint64_t>(shape.dim(axis));
      if (dim > UINT32_MAX / count)
        return LSE_ERROR(kInvalidArgument, "DFlash2 convolution exceeds indexing bounds");
      count *= dim;
    }
  }
  return x;
}
template <class E>
struct ConvolutionArgs {
  env::In<kir::f32, E> hidden, dynamic, base;
  env::Out<kir::f32, E> out;
};
struct DFlash2Convolution final : KernelPrimitive<DFlash2Convolution> {
  static constexpr std::string_view kName = "dflash2.convolve2.v1";
  static constexpr std::string_view kEntry = "lse_dflash2_convolve2_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 3; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_host_impl() const noexcept override { return true; }
  Result<Shape> infer_shape(std::span<const Shape> inputs) const override {
    return convolution_shape(inputs);
  }
  DType infer_dtype(std::span<const DType>) const override { return DType::kF32; }
  std::string emit_kernel(const KernelShapes& s) const override {
    const auto shape = convolution_shape(s.inputs);
    if (!shape.ok() || s.output != *shape || s.input_dtypes.size() != 3 ||
        s.output_dtype != DType::kF32 || !s.types.scalar || !s.intrinsics || !s.store)
      return {};
    for (auto type : s.input_dtypes) if (type != DType::kF32) return {};
    const auto tokens = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto width = static_cast<std::uint32_t>(s.inputs[0].dim(2));
    const auto groups = static_cast<std::uint32_t>(s.inputs[1].dim(3));
    const auto group_size = width / groups;
    if (s.attrs != std::array<float, 4>{static_cast<float>(tokens),
        static_cast<float>(width), static_cast<float>(group_size), 0.0f}) return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    ConvolutionArgs<env::Emit> a;
    if (!env::bind(body, a, s)) return {};
    env::Emit e{&body};
    const auto at = e.let(e.thread_id());
    (void)e.ret_if(at >= static_cast<std::uint32_t>(s.output.elem_count()));
    const auto channel = e.let(at % width);
    const auto row = e.let(at / width);
    const auto token = e.let(row % tokens);
    const auto group = e.let(channel / group_size);
    const auto coefficients = e.let(row * (2u * groups) + group);
    const auto current = e.let((a.base[channel] + a.dynamic[coefficients]) * a.hidden[at]);
    auto previous = e.var(0.0f);
    if (auto live_previous = e.when(token != 0u)) previous = a.hidden[at - width];
    const auto delayed = e.let((a.base[channel + width] + a.dynamic[coefficients + groups]) * previous.read());
    e.store(at, current + delayed);
    return body.str();
  }
  void eval_cpu(std::span<const float* const> inputs, float* out, std::size_t count,
                const std::array<float, 4>& attrs) const override {
    const auto tokens = static_cast<std::size_t>(attrs[0]);
    const auto width = static_cast<std::size_t>(attrs[1]);
    const auto group_size = static_cast<std::size_t>(attrs[2]);
    const auto groups = width / group_size;
    for (std::size_t at = 0; at < count; ++at) {
      const auto channel = at % width, row = at / width;
      const auto coefficients = row * 2 * groups + channel / group_size;
      const float current = (inputs[2][channel] + inputs[1][coefficients]) * inputs[0][at];
      const float previous = row % tokens ? inputs[0][at - width] : 0.0f;
      const float delayed = (inputs[2][width + channel] + inputs[1][coefficients + groups]) * previous;
      out[at] = current + delayed;
    }
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    const auto threads = s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    plan.workgroup_size[0] = threads;
    plan.workgroup_count[0] = static_cast<std::uint32_t>((s.output.elem_count() + threads - 1) / threads);
    return plan;
  }
};
LSE_REGISTER_PRIMITIVE(DFlash2Convolution);
}  // namespace
}  // namespace lse::kernels
