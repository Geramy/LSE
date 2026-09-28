#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/vec_mem.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
namespace lse::kernels {
using namespace graph;
namespace {
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::Out<kir::u32, env::Emit> out;
};
Result<Shape> panel_shape(std::span<const Shape> in) {
  if (in.size() != 1 || !in[0].rank() || !in[0].elem_count() ||
      in[0].elem_count() > UINT32_MAX)
    return LSE_ERROR(kInvalidArgument,
                     "activation panel requires complete group64 rows");
  const auto k = in[0].dim(in[0].rank() - 1);
  if (k <= 0 || k % 64 != 0)
    return LSE_ERROR(kInvalidArgument, "invalid panel row width");
  for (std::size_t i = 0; i < in[0].rank(); ++i)
    if (in[0].dim(i) <= 0)
      return LSE_ERROR(kInvalidArgument, "invalid panel extent");
  return Shape{static_cast<std::int64_t>(in[0].elem_count() /
                                         static_cast<std::uint64_t>(k)),
               k * 25 / 64};
}
struct Q4ActivationPanelKernel final
    : KernelPrimitive<Q4ActivationPanelKernel> {
  static constexpr std::string_view kName =
      "quant_activation.q4_shared_panel.v1";
  static constexpr std::string_view kEntry =
      "lse_quant_activation_q4_shared_panel_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kU32;
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return panel_shape(in);
  }
  static bool supported(const KernelShapes &s) {
    const auto out = panel_shape(s.inputs);
    return out.ok() && s.input_dtypes.size() == 1 &&
           s.input_dtypes[0] == DType::kF32 && s.output_dtype == DType::kU32 &&
           s.output == *out && s.device && s.device->wavefront_size == 32 &&
           s.device->max_threads_per_workgroup >= 256 && s.intrinsics &&
           !s.intrinsics->find("bits.f32").empty() &&
           !s.intrinsics->find("rint").empty() &&
           !s.intrinsics->find("wave.shfl_xor").empty();
  }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4> &,
                        const std::array<std::int32_t, 4> &) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 ||
        out.dtype != DType::kU32)
      return LSE_ERROR(kInvalidArgument, "invalid activation panel storage");
    const std::array shapes{in[0].shape};
    LSE_ASSIGN_OR(const auto expected, panel_shape(shapes));
    if (out.shape != expected ||
        in[0].bytes.size() != in[0].shape.elem_count() * 4 ||
        out.bytes.size() != out.shape.elem_count() * 4)
      return LSE_ERROR(kInvalidArgument, "invalid activation panel buffer");
    const auto k =
        static_cast<std::size_t>(in[0].shape.dim(in[0].shape.rank() - 1));
    const auto rows = in[0].shape.elem_count() / k;
    const auto stride = k * 25 / 64;
    const auto load = [&](std::size_t i) {
      float value;
      std::memcpy(&value, in[0].bytes.data() + i * 4, 4);
      return value;
    };
    const auto store = [&](std::size_t i, std::uint32_t value) {
      std::memcpy(out.bytes.data() + i * 4, &value, 4);
    };
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t group = 0; group < k / 64; ++group) {
        std::array<float, 8> sums{};
        for (std::size_t c = 0; c < 8; ++c) {
          const auto chunk = group * 8 + c;
          std::array<float, 8> values{};
          for (std::size_t j = 0; j < 8; ++j)
            values[j] = load(row * k + chunk * 8 + j);
          auto maximum = std::abs(values[0]);
          auto sum = values[0];
          for (std::size_t j = 1; j < 8; ++j) {
            maximum = std::max(maximum, std::abs(values[j]));
            sum = sum + values[j];
          }
          sums[c] = sum;
          const float step = maximum * (1.0f / 127.0f);
          const float inverse = 127.0f / std::max(maximum, 1e-30f);
          std::array<std::uint32_t, 2> words{};
          for (int plane = 0; plane < 2; ++plane) {
            for (int byte = 0; byte < 4; ++byte) {
              const float rounded =
                  std::nearbyint(values[static_cast<std::size_t>(
                                     quant::dot4_operand_slot(plane, byte))] *
                                 inverse);
              const auto code = std::isfinite(rounded)
                                    ? static_cast<std::int32_t>(rounded)
                                    : 0;
              words[static_cast<std::size_t>(plane)] |=
                  (static_cast<std::uint32_t>(code) & 255u) << (8 * byte);
            }
          }
          store(row * stride + chunk * 2, words[0]);
          store(row * stride + chunk * 2 + 1, words[1]);
          store(row * stride + k / 4 + chunk,
                std::bit_cast<std::uint32_t>(step));
        }
        for (std::size_t bit = 1; bit < 8; bit <<= 1) {
          const auto previous = sums;
          for (std::size_t c = 0; c < 8; ++c)
            sums[c] = previous[c] + previous[c ^ bit];
        }
        store(row * stride + 3 * k / 8 + group,
              std::bit_cast<std::uint32_t>(sums[0]));
      }
    }
    return OkStatus();
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!supported(s))
      return {};
    const auto m = static_cast<std::uint32_t>(s.output.dim(0));
    const auto k =
        static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto chunks = k / 8u, code_words = k / 4u;
    const auto sum_offset = code_words + chunks, stride = (k / 64u) * 25u;
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    kb.set_store(s.store);
    Args a;
    if (!env::bind(kb, a, s))
      return {};
    env::Emit e{&kb};
    const auto lid = e.let(math::local_id());
    const auto row = e.let(math::workgroup_id_y());
    const auto c = e.let(math::workgroup_id_x() * 256u + lid);
    if (auto live = e.when(row < m && c < chunks)) {
      std::array<kir::Val<kir::f32>, 8> v;
      const auto base = e.let(row * k + c * 8u);
      for (std::uint32_t j = 0; j < 8; j += 4) {
        const auto pack = e.load(a.x, e.let(base + j), 16u);
        for (std::uint32_t t = 0; t < 4; ++t)
          v[j + t] = e.let(pack[static_cast<int>(t)]);
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
        const auto code =
            e.let(math::rint(v[static_cast<std::size_t>(j)] * inv));
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
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan plan;
    if (!supported(s))
      return plan;
    const auto chunks =
        static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1)) /
        8u;
    plan.workgroup_size[0] = 256;
    plan.workgroup_count[0] = (chunks + 255u) / 256u;
    plan.workgroup_count[1] = static_cast<std::uint32_t>(s.output.dim(0));
    return plan;
  }
};
} // namespace
LSE_REGISTER_PRIMITIVE(Q4ActivationPanelKernel);
} // namespace lse::kernels
