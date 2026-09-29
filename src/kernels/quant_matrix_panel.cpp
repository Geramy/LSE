#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

namespace lse::kernels {
using namespace graph;
namespace {
// Four K16 fragments per group, followed by interleaved row step/sum words.
constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kRows = dispatch::kQ4MatrixPanelRows;
constexpr std::uint32_t kGroupWords = dispatch::kQ4MatrixPanelGroupWords;
constexpr std::uint32_t kInputRows = 8;

Result<Shape> matrix_panel_shape(std::span<const Shape> in) {
  if (in.size() != 1 || !in[0].rank())
    return LSE_ERROR(kInvalidArgument,
                     "matrix panel needs complete group64 rows");
  const auto k = in[0].dim(in[0].rank() - 1);
  if (k <= 0 || k % 64 != 0 ||
      static_cast<std::uint64_t>(k) > UINT32_MAX / kInputRows ||
      in[0].elem_count() != kInputRows * static_cast<std::uint64_t>(k) ||
      in[0].elem_count() > UINT32_MAX ||
      static_cast<std::uint64_t>(k / 64) * kGroupWords > UINT32_MAX)
    return LSE_ERROR(kInvalidArgument, "invalid matrix panel extent");
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < in[0].rank(); ++axis) {
    const auto extent = in[0].dim(axis);
    if (extent <= 0 || static_cast<std::uint64_t>(extent) > UINT32_MAX / count)
      return LSE_ERROR(kInvalidArgument, "invalid matrix panel row shape");
    count *= static_cast<std::uint64_t>(extent);
  }
  return Shape{k / 64, kGroupWords};
}
struct PanelArgs {
  env::In<kir::f32, env::Emit> x;
  env::Out<kir::u32, env::Emit> out;
};
struct Q4MatrixPanelKernel final : KernelPrimitive<Q4MatrixPanelKernel> {
  static constexpr std::string_view kName =
      "quant_activation.q4_matrix_panel.v1";
  static constexpr std::string_view kEntry =
      "lse_quant_activation_q4_matrix_panel_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 1; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  bool has_typed_host_impl() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kU32;
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    return matrix_panel_shape(in);
  }
  static bool supported(const KernelShapes &s) {
    const auto shape = matrix_panel_shape(s.inputs);
    if (!shape.ok() || s.output != *shape || s.input_dtypes.size() != 1 ||
        s.input_dtypes[0] != DType::kF32 || s.output_dtype != DType::kU32 ||
        !s.device || s.device->wavefront_size != 32 ||
        s.device->max_threads_per_workgroup < kBlock || !s.intrinsics)
      return false;
    for (const auto symbol :
         {"bits.f32", "rint", "wave.shfl_xor", "max", "abs"})
      if (s.intrinsics->find(symbol).empty())
        return false;
    return true;
  }
  Status eval_cpu_typed(std::span<const HostTensorView> in, HostOutputView out,
                        const std::array<float, 4> &,
                        const std::array<std::int32_t, 4> &) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 ||
        out.dtype != DType::kU32)
      return LSE_ERROR(kInvalidArgument, "invalid matrix panel storage");
    const std::array input_shapes{in[0].shape};
    LSE_ASSIGN_OR(const auto expected, matrix_panel_shape(input_shapes));
    if (out.shape != expected ||
        in[0].bytes.size() != in[0].shape.elem_count() * 4 ||
        out.bytes.size() != out.shape.elem_count() * 4)
      return LSE_ERROR(kInvalidArgument, "invalid matrix panel byte extent");
    const auto k =
        static_cast<std::size_t>(in[0].shape.dim(in[0].shape.rank() - 1));
    const auto store = [&](std::size_t at, std::uint32_t bits) {
      std::memcpy(out.bytes.data() + at * 4, &bits, 4);
    };
    for (std::size_t group = 0; group < k / 64; ++group) {
      for (std::size_t row = 0; row < kRows; ++row) {
        std::array<float, 64> values{};
        if (row < kInputRows)
          std::memcpy(values.data(),
                      in[0].bytes.data() + (row * k + group * 64) * 4,
                      values.size() * 4);
        std::array<float, 4> maxima{}, sums{};
        for (std::size_t slice = 0; slice < 4; ++slice) {
          maxima[slice] = std::abs(values[slice * 16]);
          sums[slice] = values[slice * 16];
          for (std::size_t t = 1; t < 16; ++t) {
            const auto value = values[slice * 16 + t];
            maxima[slice] = std::max(maxima[slice], std::abs(value));
            sums[slice] += value;
          }
        }
        auto maximum = maxima[0], sum = sums[0];
        for (std::size_t slice = 1; slice < 4; ++slice) {
          maximum = std::max(maximum, maxima[slice]);
          sum += sums[slice];
        }
        const float inverse = 127.0f / std::max(maximum, 1e-30f);
        for (std::size_t word = 0; word < 16; ++word) {
          std::uint32_t packed = 0;
          for (std::size_t byte = 0; byte < 4; ++byte) {
            const float rounded =
                std::nearbyint(values[word * 4 + byte] * inverse);
            const auto code =
                std::isfinite(rounded) ? static_cast<std::int32_t>(rounded) : 0;
            packed |= (static_cast<std::uint32_t>(code) & 255u) << (8 * byte);
          }
          const auto at = group * kGroupWords + (word / 4) * 64 +
                          ((word % 4) / 2) * 32 + row * 2 + word % 2;
          store(at, packed);
        }
        store(group * kGroupWords + 256 + row * 2,
              std::bit_cast<std::uint32_t>(maximum * (1.0f / 127.0f)));
        store(group * kGroupWords + 257 + row * 2,
              std::bit_cast<std::uint32_t>(sum));
      }
    }
    return OkStatus();
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!supported(s))
      return {};
    const auto k =
        static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
    const auto groups = k / 64u;
    kir::KernelBody kb(s.types, *s.intrinsics, 0);
    kb.set_store(s.store);
    PanelArgs a;
    if (!env::bind(kb, a, s))
      return {};
    env::Emit e{&kb};
    const auto lid = e.let(math::local_id());
    const auto slice = e.let(lid % 4u);
    const auto row = e.let(math::workgroup_id_y());
    const auto group = e.let(math::workgroup_id_x() * 64u + lid / 4u);
    if (auto valid = e.when(row < kRows && group < groups)) {
      std::vector<kir::LValue<kir::f32>> values;
      for (int t = 0; t < 16; ++t)
        values.push_back(e.var(0.0f));
      if (auto actual = e.when(row < kInputRows)) {
        const auto base = e.let(row * k + group * 64u + slice * 16u);
        for (std::uint32_t j = 0; j < 16u; j += 4u) {
          const auto pack = e.load(a.x, e.let(base + j), 16u);
          for (std::uint32_t t = 0; t < 4u; ++t)
            values[j + t] = e.let(pack[static_cast<int>(t)]);
        }
      }
      auto local_max = e.let(math::abs(values[0].read()));
      auto local_sum = e.let(values[0].read());
      for (std::uint32_t t = 1; t < 16; ++t) {
        local_max = e.let(math::max(local_max, math::abs(values[t].read())));
        local_sum = e.let(local_sum + values[t].read());
      }
      auto maximum =
          e.let(math::max(local_max, math::shfl_xor(local_max, e.u32(1))));
      maximum = e.let(math::max(maximum, math::shfl_xor(maximum, e.u32(2))));
      const auto sum1 = e.let(math::shfl_xor(local_sum, e.u32(1)));
      const auto sum2 = e.let(math::shfl_xor(local_sum, e.u32(2)));
      const auto sum3 = e.let(math::shfl_xor(local_sum, e.u32(3)));
      const auto total = e.let(((local_sum + sum1) + sum2) + sum3);
      const auto step = e.let(maximum * (1.0f / 127.0f));
      const auto inverse = e.let(127.0f / math::max(maximum, e.f32(1e-30f)));
      const auto dst = e.let(group * kGroupWords);
      for (std::uint32_t word_index = 0; word_index < 4; ++word_index) {
        auto word = e.let(e.u32(0));
        for (std::uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
          const auto code = e.let(
              math::rint(values[word_index * 4 + byte_index].read() * inverse));
          const auto byte =
              e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
          word = e.let(word + byte * (1u << (8u * byte_index)));
        }
        a.out[dst + slice * 64u + (word_index / 2u) * 32u + row * 2u +
              word_index % 2u] = word;
      }
      if (auto leader = e.when(slice == 0u)) {
        a.out[dst + 256u + row * 2u] = math::bits_of<lse::f32>(step);
        a.out[dst + 257u + row * 2u] = math::bits_of<lse::f32>(total);
      }
    }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    if (!supported(s))
      return tp;
    const auto groups = static_cast<std::uint32_t>(s.output.dim(0));
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = (groups + 63u) / 64u;
    tp.workgroup_count[1] = kRows;
    return tp;
  }
};

KernelShapes original_shapes(const KernelShapes &s) {
  auto original = s;
  original.inputs = s.inputs.first(4);
  original.input_dtypes = s.input_dtypes.first(4);
  return original;
}
bool valid_matrix_panel(const KernelShapes &s) {
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.input_dtypes[4] != DType::kU32)
    return false;
  const auto original = original_shapes(s);
  if (!dispatch::q4_matrix_panel_shape(original))
    return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  return s.inputs[4] == Shape{k / 64, kGroupWords};
}
const KernelPrimitiveBase *legacy(const KernelShapes &s) {
  const auto *base =
      dynamic_cast<const KernelPrimitiveBase *>(find_primitive("quant_linear"));
  return base ? base->specialize(s) : nullptr;
}
struct MatrixArgs {
  env::In<kir::f32, env::Emit> x;
  env::In<kir::u32, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales;
  env::In<lse::bf16, env::Emit> biases;
  env::In<kir::u32, env::Emit> panel;
  env::Out<kir::f32, env::Emit> out;
};
std::string emit_matrix(const KernelShapes &s) {
  using Mma = math::op::Mma<math::MatrixTarget::kRdna4, math::MatrixElem::kI32,
                            math::MatrixElem::kSU8, 16, 16, 16>;
  constexpr auto row = Mma::kRow;
  constexpr auto geo = geometry_of(row);
  constexpr int kFrag = row.a_len / row.chained;
  constexpr int kSlots = row.c_len;
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k =
      static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  const auto groups = k / 64u;
  const auto lanes = k / 8u;
  constexpr std::uint32_t waves = kBlock / 32u;
  const auto tiles_n = (n + 15u) / 16u;
  const auto nblocks = (tiles_n + waves - 1u) / waves;
  kir::KernelBody kb(s.types, *s.intrinsics, 0);
  kb.set_store(s.store);
  MatrixArgs a;
  if (!env::bind(kb, a, s))
    return {};
  env::Emit e{&kb};
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % 32u);
  const auto lo = e.let(lane % 16u);
  const auto hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto m0 = e.let((wg / nblocks) * 16u);
  const auto ntile = e.let((wg % nblocks) * waves + lid / 32u);
  const auto col = e.let(ntile * 16u + lo);
  const auto live = e.let(ntile < tiles_n && col < n);
  const auto safe_col = e.let(select(col < n, col, e.u32(0)));
  std::vector<kir::LValue<kir::f32>> out;
  for (int z = 0; z < kSlots; ++z)
    out.push_back(e.var(0.0f));
  for (auto g : e.range(0u, groups, 1u)) {
    const auto sa = e.let(safe_col * groups + g);
    const auto raw_scale = e.let(a.scales[sa]);
    const auto raw_bias = e.let(a.biases[sa]);
    std::array<kir::Val<kir::u32>, 4> weight;
    for (std::uint32_t t = 0; t < 4u; ++t)
      weight[t] =
          e.let(a.packed[e.let(safe_col * lanes + g * 8u + t * 2u + hi)]);
    std::vector<kir::Pack<kir::u32>> acts, metadata;
    for (std::uint32_t t = 0; t < 4u; ++t)
      acts.push_back(e.load(
          a.panel, e.let(g * kGroupWords + t * 64u + hi * 32u + lo * 2u), 8u));
    for (std::uint32_t z = 0; z < 8u; z += 2u)
      metadata.push_back(e.load(
          a.panel, e.let(g * kGroupWords + 256u + hi * 16u + z * 2u), 16u));
    const auto scale = e.let(math::widen(raw_scale));
    const auto bias = e.let(math::widen(raw_bias));
    const auto acc = e.local<kir::i32, kSlots>();
    for (int z = 0; z < kSlots; ++z)
      acc[z] = kir::cast<kir::i32>(e.u32(0));
    for (std::uint32_t t = 0; t < 4u; ++t) {
      const auto bf = e.local<kir::u32, kFrag>();
      const auto af = e.local<kir::u32, kFrag>();
      for (int f = 0; f < kFrag; ++f) {
        auto expanded = e.let(e.u32(0));
        for (std::uint32_t b = 0; b < 4u; ++b) {
          const auto code =
              e.let((weight[t] /
                     (1u << (4u * (static_cast<std::uint32_t>(f) * 4u + b)))) %
                    16u);
          expanded = e.let(expanded + code * (1u << (8u * b)));
        }
        bf[f] = expanded;
        af[f] = acts[t][f];
      }
      acc = math::mma<Mma>(af.value(), bf.value(), acc.value());
    }
    for (int z = 0; z < kSlots; ++z) {
      const auto step = e.let(math::from_bits<lse::f32>(
          metadata[static_cast<std::size_t>(z / 2)][(z % 2) * 2]));
      const auto sum = e.let(math::from_bits<lse::f32>(
          metadata[static_cast<std::size_t>(z / 2)][(z % 2) * 2 + 1]));
      const auto term = e.let(scale * step);
      out[static_cast<std::size_t>(z)] =
          math::fma(term, kir::cast<kir::f32>(acc[z].read()),
                    out[static_cast<std::size_t>(z)].read()) +
          bias * sum;
    }
  }
  for (int z = 0; z < kSlots; ++z) {
    const auto output_row =
        e.let(m0 + hi * geo.half_rows +
              static_cast<std::uint32_t>(z) * geo.slot_step);
    if (auto valid = e.when(live && output_row < kInputRows))
      e.store(output_row * n + col, out[static_cast<std::size_t>(z)].read());
  }
  return kb.str();
}
struct Q4MatrixPanelLinear final : KernelPrimitive<Q4MatrixPanelLinear> {
  static constexpr std::string_view kName = "quant_linear.q4_matrix_panel.v1";
  static constexpr std::string_view kEntry =
      "lse_quant_linear_q4_matrix_panel_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || !in[0].rank() || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "invalid matrix-panel contraction");
    Shape out;
    for (std::size_t axis = 0; axis + 1 < in[0].rank(); ++axis)
      out.push_back(in[0].dim(axis));
    out.push_back(in[1].dim(0));
    return out;
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!valid_matrix_panel(s) || !s.types.scalar || !s.store || !s.intrinsics)
      return {};
    const auto original = original_shapes(s);
    if (dispatch::q4_matrix_panel_row(original))
      return emit_matrix(s);
    const auto *kernel = legacy(original);
    return kernel ? kernel->emit_kernel(original) : std::string{};
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    if (!valid_matrix_panel(s))
      return tp;
    const auto original = original_shapes(s);
    if (!dispatch::q4_matrix_panel_row(original)) {
      const auto *kernel = legacy(original);
      return kernel ? kernel->plan(original) : tp;
    }
    const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = (n + 127u) / 128u;
    return tp;
  }
};
} // namespace
LSE_REGISTER_PRIMITIVE(Q4MatrixPanelKernel);
LSE_REGISTER_PRIMITIVE(Q4MatrixPanelLinear);
} // namespace lse::kernels
