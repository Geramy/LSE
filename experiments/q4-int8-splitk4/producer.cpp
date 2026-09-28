#include <array>
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/quant/group_affine_codec.hpp"
namespace lse::kernels::panel_probe {
using namespace graph;
namespace {
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::Out<kir::u32, env::Emit> out;
};
struct Producer final : KernelPrimitive<Producer> {
  static constexpr std::string_view kName = "experiment.q4.shared_panel.v1";
  static constexpr std::string_view kEntry = "lse_experiment_q4_shared_panel_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  DType infer_dtype(std::span<const DType>) const override { return DType::kU32; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 1 || in[0].rank() != 2 || in[0].dim(0) <= 0 ||
        in[0].dim(1) <= 0 || in[0].dim(1) % 64 != 0)
      return LSE_ERROR(kInvalidArgument, "panel requires matrix with complete group64 rows");
    return Shape{in[0].dim(0), in[0].dim(1) * 25 / 64};
  }
  static bool supported(const KernelShapes& s) {
    return s.inputs.size() == 1 && s.input_dtypes.size() == 1 &&
           s.inputs[0].rank() == 2 && s.input_dtypes[0] == DType::kF32 &&
           s.output_dtype == DType::kU32 && s.inputs[0].dim(0) > 0 &&
           s.inputs[0].dim(1) > 0 && s.inputs[0].dim(1) % 64 == 0 &&
           s.output == Shape{s.inputs[0].dim(0), s.inputs[0].dim(1) * 25 / 64} &&
           s.device && s.device->wavefront_size == 32 && s.intrinsics &&
           !s.intrinsics->find("bits.f32").empty() &&
           !s.intrinsics->find("wave.shfl_xor").empty();
  }
  std::string emit_kernel(const KernelShapes& s) const override {
    if (!supported(s)) return {};
    const auto m = static_cast<std::uint32_t>(s.inputs[0].dim(0));
    const auto k = static_cast<std::uint32_t>(s.inputs[0].dim(1));
    const auto chunks = k / 8u, code_words = k / 4u;
    const auto sum_offset = code_words + chunks, stride = k * 25u / 64u;
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    kb.set_store(s.store);
    Args a;
    if (!env::bind(kb, a, s)) return {};
    env::Emit e{&kb};
    const auto lid = e.let(math::local_id());
    const auto row = e.let(math::workgroup_id_y());
    const auto c = e.let(math::workgroup_id_x() * 256u + lid);
    if (auto live = e.when(row < m && c < chunks)) {
      std::array<kir::Val<kir::f32>, 8> v;
      const auto base = e.let(row * k + c * 8u);
      for (std::uint32_t j = 0; j < 8; j += 4) {
        const auto pack = e.load(a.x, e.let(base + j), 16u);
        for (std::uint32_t t = 0; t < 4; ++t) v[j+t] = e.let(pack[static_cast<int>(t)]);
      }
      auto amax = e.let(math::abs(v[0]));
      auto sum = v[0];
      for (std::uint32_t j = 1; j < 8; ++j) {
        amax = e.let(math::max(amax, math::abs(v[j])));
        sum = e.let(sum + v[j]);
      }
      for (std::uint32_t bit = 1; bit < 8; bit <<= 1)
        sum = e.let(sum + math::shfl_xor(sum, e.u32(bit)));
      const auto step = e.let(amax * (1.0f / 127.0f));
      const auto inv = e.let(127.0f / math::max(amax, e.f32(1e-30f)));
      const auto byte_of = [&](int j) {
        const auto code = e.let(math::rint(v[static_cast<std::size_t>(j)] * inv));
        return e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
      };
      const auto dst = e.let(row * stride);
      a.out[dst + c * 2u] = quant::dot4_activation_word(e, byte_of, 0);
      a.out[dst + c * 2u + 1u] = quant::dot4_activation_word(e, byte_of, 1);
      a.out[dst + code_words + c] = math::bits_of<lse::f32>(step);
      if (auto lead = e.when(c % 8u == 0u))
        a.out[dst + sum_offset + c / 8u] = math::bits_of<lse::f32>(sum);
    }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan plan;
    if (!supported(s)) return plan;
    const auto chunks = static_cast<std::uint32_t>(s.inputs[0].dim(1)) / 8u;
    plan.workgroup_size[0] = 256;
    plan.workgroup_count[0] = (chunks + 255u) / 256u;
    plan.workgroup_count[1] = static_cast<std::uint32_t>(s.inputs[0].dim(0));
    return plan;
  }
};
}
const KernelPrimitiveBase* producer() {
  static const Producer instance;
  return &instance;
}
}
