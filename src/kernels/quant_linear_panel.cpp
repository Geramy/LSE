#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/lds_linear.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

#include <array>
#include <vector>

namespace lse::kernels {
using namespace graph;
namespace {
constexpr std::uint32_t kBlock = 256;
struct Args {
  env::In<kir::f32, env::Emit> x;
  env::In<kir::u32, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales;
  env::In<lse::bf16, env::Emit> biases;
  env::In<kir::u32, env::Emit> panel;
  env::Out<kir::f32, env::Emit> out;
};
KernelShapes original_shapes(const KernelShapes &s) {
  auto original = s;
  original.inputs = s.inputs.first(4);
  original.input_dtypes = s.input_dtypes.first(4);
  return original;
}
bool valid_panel(const KernelShapes &s) {
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.input_dtypes[4] != DType::kU32)
    return false;
  const auto original = original_shapes(s);
  if (!dispatch::q4_shared_panel_shape(original))
    return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto m = static_cast<std::int64_t>(s.inputs[0].elem_count() /
                                           static_cast<std::uint64_t>(k));
  return s.inputs[4] == Shape{m, (k / 64) * 25};
}
const KernelPrimitiveBase *legacy() {
  return dynamic_cast<const KernelPrimitiveBase *>(
      find_primitive("quant_linear"));
}
void emit_run_dot(env::Emit &e, const Args &a,
                  const kir::Val<kir::u32> &row_base,
                  const kir::Val<kir::u32> &scale_base,
                  const kir::Val<kir::u32> &chunk0, std::uint32_t count,
                  std::span<const kir::LValue<kir::f32>> acc, std::uint32_t cpg,
                  const kir::Val<kir::u32> &row, std::uint32_t m,
                  std::uint32_t k, std::uint32_t load_chunks) {
  std::vector<kir::LValue<kir::f32>> facc;
  facc.reserve(acc.size());
  for (std::size_t r = 0; r < acc.size(); ++r)
    facc.push_back(e.var(e.f32(0.0f)));
  const auto words = e.load(a.packed, row_base + chunk0, count * 4u);
  if (load_chunks == 4 && count == 4 && acc.size() == 8) {
    // Four weight chunks are decoded once; each row keeps the original chunk order.
    std::vector<std::array<kir::Val<kir::u32>, 2>> planes;
    for (std::uint32_t j = 0; j < count; ++j) {
      const auto word = e.let(words[static_cast<int>(j)]);
      std::array<kir::Val<kir::u32>, 2> value;
      for (std::size_t plane = 0; plane < 2; ++plane)
        value[plane] = quant::dot4_code_plane(e, word, static_cast<int>(plane));
      planes.push_back(value);
    }
    for (std::size_t row_pair = 0; row_pair < acc.size(); row_pair += 2) {
      std::vector<kir::Pack<kir::u32>> codes_first, codes_second, steps;
      for (std::size_t at = 0; at < 2; ++at) {
        const auto r = row_pair + at;
        const auto input_row =
            e.let(select(row + static_cast<std::uint32_t>(r) < m,
                         row + static_cast<std::uint32_t>(r), e.u32(0)));
        const auto panel_base = e.let(input_row * ((k / 64u) * 25u));
        codes_first.push_back(e.load(a.panel, e.let(panel_base + chunk0 * 2u), 16u));
        codes_second.push_back(e.load(a.panel, e.let(panel_base + chunk0 * 2u + 4u), 16u));
        steps.push_back(e.load(a.panel, e.let(panel_base + k / 4u + chunk0), 16u));
      }
      for (std::uint32_t j = 0; j < count; ++j) {
        for (std::size_t at = 0; at < 2; ++at) {
          const auto r = row_pair + at;
          auto iacc = e.var(kir::cast<kir::i32>(e.u32(0)));
          for (std::size_t plane = 0; plane < 2; ++plane) {
            const auto x = e.let((j < 2 ? codes_first[at] : codes_second[at])
                [static_cast<int>((j % 2) * 2u + plane)]);
            iacc = math::dot4_iu8(kir::cast<kir::i32>(x),
                                  kir::cast<kir::i32>(planes[j][plane]), iacc.read());
          }
          const auto step = e.let(math::from_bits<lse::f32>(steps[at][static_cast<int>(j)]));
          facc[r] = math::fma(step, kir::cast<kir::f32>(iacc.read()), facc[r].read());
        }
      }
    }
  } else if (load_chunks == 2 && count % 2 == 0) {
    for (std::uint32_t first = 0; first < count; first += 2) {
      std::vector<kir::Pack<kir::u32>> code_pairs;
      std::vector<kir::Pack<kir::u32>> step_pairs;
      code_pairs.reserve(acc.size());
      step_pairs.reserve(acc.size());
      const auto first_chunk = e.let(chunk0 + first);
      for (std::size_t r = 0; r < acc.size(); ++r) {
        const auto input_row =
            e.let(select(row + static_cast<std::uint32_t>(r) < m,
                         row + static_cast<std::uint32_t>(r), e.u32(0)));
        const auto panel_base = e.let(input_row * ((k / 64u) * 25u));
        code_pairs.push_back(
            e.load(a.panel, e.let(panel_base + first_chunk * 2u), 16u));
        step_pairs.push_back(
            e.load(a.panel, e.let(panel_base + k / 4u + first_chunk), 8u));
      }
      for (std::uint32_t j = 0; j < 2; ++j) {
        const auto word = e.let(words[static_cast<int>(first + j)]);
        std::array<kir::Val<kir::u32>, 2> planes;
        for (std::size_t plane = 0; plane < 2; ++plane)
          planes[plane] =
              quant::dot4_code_plane(e, word, static_cast<int>(plane));
        for (std::size_t r = 0; r < acc.size(); ++r) {
          auto iacc = e.var(kir::cast<kir::i32>(e.u32(0)));
          for (std::size_t plane = 0; plane < 2; ++plane) {
            const auto x =
                e.let(code_pairs[r][static_cast<int>(j * 2u + plane)]);
            iacc = math::dot4_iu8(kir::cast<kir::i32>(x),
                                  kir::cast<kir::i32>(planes[plane]),
                                  iacc.read());
          }
          const auto step = e.let(
              math::from_bits<lse::f32>(step_pairs[r][static_cast<int>(j)]));
          facc[r] = math::fma(step, kir::cast<kir::f32>(iacc.read()),
                              facc[r].read());
        }
      }
    }
  } else {
    for (auto uu : e.unroll(count)) {
      const auto chunk = e.let(chunk0 + uu);
      const auto word = e.let(words[uu]);
      std::array<kir::Val<kir::u32>, 2> planes;
      for (std::size_t p = 0; p < 2; ++p) {
        planes[p] = quant::dot4_code_plane(e, word, static_cast<int>(p));
      }
      for (std::size_t r = 0; r < acc.size(); ++r) {
        const auto input_row =
            e.let(select(row + static_cast<std::uint32_t>(r) < m,
                         row + static_cast<std::uint32_t>(r), e.u32(0)));
        const auto panel_base = e.let(input_row * ((k / 64u) * 25u));
        const auto codes = e.load(a.panel, e.let(panel_base + chunk * 2u), 8u);
        auto iacc = e.var(kir::cast<kir::i32>(e.u32(0)));
        for (std::size_t p = 0; p < 2; ++p) {
          const auto x = e.let(codes[static_cast<int>(p)]);
          iacc = math::dot4_iu8(kir::cast<kir::i32>(x),
                                kir::cast<kir::i32>(planes[p]), iacc.read());
        }
        const auto step = e.let(
            math::from_bits<lse::f32>(a.panel[panel_base + k / 4u + chunk]));
        facc[r] =
            math::fma(step, kir::cast<kir::f32>(iacc.read()), facc[r].read());
      }
    }
  }
  const auto group = e.let(chunk0 / cpg);
  const auto wscale = e.let(math::widen(a.scales[scale_base + group]));
  for (std::size_t r = 0; r < acc.size(); ++r) {
    acc[r] = math::fma(wscale, facc[r].read(), acc[r].read());
  }
}
void emit_bias(env::Emit &e, const Args &a,
               const kir::Val<kir::u32> &scale_base,
               const kir::Val<kir::u32> &lane, std::uint32_t g_begin,
               std::uint32_t g_end, std::uint32_t wave,
               std::span<const kir::LValue<kir::f32>> acc,
               const kir::Val<kir::u32> &row, std::uint32_t m,
               std::uint32_t k) {
  for (auto g : e.range(e.u32(g_begin) + lane, e.u32(g_end), wave)) {
    const auto b = e.let(math::widen(a.biases[scale_base + g]));
    for (std::size_t r = 0; r < acc.size(); ++r) {
      const auto input_row =
          e.let(select(row + static_cast<std::uint32_t>(r) < m,
                       row + static_cast<std::uint32_t>(r), e.u32(0)));
      const auto panel_base = e.let(input_row * ((k / 64u) * 25u));
      const auto sum = e.let(
          math::from_bits<lse::f32>(a.panel[panel_base + 3u * k / 8u + g]));
      acc[r] = math::fma(b, sum, acc[r].read());
    }
  }
}

std::string emit_panel(const KernelShapes &s, const Dot4Schedule &schedule,
                       std::uint32_t load_chunks) {
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k =
      static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  const auto m = static_cast<std::uint32_t>(s.output.elem_count() / n);
  const auto lanes = k / 8u, groups = k / 64u, nchunks = k / 8u;
  constexpr std::uint32_t chunks_per_group = 8;
  const auto wave = schedule.wave;
  const auto waves = kBlock / wave;
  const auto cpl = schedule.chunks_per_lane;
  const auto span = wave * cpl;
  const auto ntiles = (n + waves - 1) / waves;
  const auto rows = schedule.rows;
  // Keep the original partition ordering: each split restores its bias before
  // the next.
  const auto ksplits = schedule.k_splits;
  kir::KernelBody kb(s.types, *s.intrinsics, 0);
  kb.set_store(s.store);
  Args a;
  if (!env::bind(kb, a, s))
    return {};
  env::Emit e{&kb};
  const auto lid = e.let(math::local_id());
  const auto wave_id = e.let(lid / wave);
  const auto lane = e.let(lid % wave);
  const auto tile = e.let(math::workgroup_id_x());
  const auto col = e.let(tile * waves + wave_id);
  const auto row = e.let(math::workgroup_id_y() * rows);
  std::vector<kir::LValue<kir::f32>> acc;
  acc.reserve(rows);
  for (std::uint32_t r = 0; r < rows; ++r)
    acc.push_back(e.var(0.0f));
  const std::span<const kir::LValue<kir::f32>> accs{acc};
  if (auto in_grid = e.when(tile < ntiles && row < m)) {
    for (std::uint32_t ks = 0; ks < ksplits; ++ks) {
      const auto cb = ks * (nchunks / ksplits);
      const auto ce = cb + nchunks / ksplits;
      if (auto in_cols = e.when(col < n)) {
        const auto row_base = e.let(col * lanes);
        const auto scale_base = e.let(col * groups);
        const auto ntile = ce - cb;
        const auto aligned = (ntile / span) * span;
        for (auto c0 : e.range(cb, cb + aligned, span)) {
          const auto chunk0 = e.let(c0 + lane * cpl);
          emit_run_dot(e, a, row_base, scale_base, chunk0, cpl, accs,
                       chunks_per_group, row, m, k, load_chunks);
        }
        if (aligned < ntile) {
          for (auto chunk :
               e.range(e.u32(cb + aligned) + lane, e.u32(ce), wave))
            emit_run_dot(e, a, row_base, scale_base, chunk, 1, accs,
                         chunks_per_group, row, m, k, load_chunks);
        }
        emit_bias(e, a, scale_base, lane, cb / chunks_per_group,
                  ce / chunks_per_group, wave, accs, row, m, k);
      }
    }
  }
  for (std::uint32_t r = 0; r < rows; ++r)
    for (std::uint32_t bit = 1; bit < wave; bit <<= 1)
      acc[r] = acc[r].read() + math::shfl_xor(acc[r].read(), e.u32(bit));
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto rr = e.let(row + r);
    if (auto lane0 = e.when(lane == 0 && col < n && rr < m))
      e.store(rr * n + col, acc[r].read());
  }
  return kb.str();
}
} // namespace
template <std::uint32_t Rows>
struct Q4GlobalPanelKernel final : KernelPrimitive<Q4GlobalPanelKernel<Rows>> {
  static constexpr std::string_view kName =
      Rows == 8 ? "quant_linear.q4_global_panel.rows8.v2"
                : "quant_linear.q4_global_panel.v1";
  static constexpr std::string_view kEntry =
      Rows == 8 ? "lse_quant_linear_q4_global_panel_rows8_v2"
                : "lse_quant_linear_q4_global_panel_v1";
  static constexpr std::string_view kSource = {};
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5 || !in[0].rank() || in[1].rank() != 2)
      return LSE_ERROR(kInvalidArgument, "invalid shared-panel contraction");
    Shape out;
    for (std::size_t i = 0; i + 1 < in[0].rank(); ++i)
      out.push_back(in[0].dim(i));
    out.push_back(in[1].dim(0));
    return out;
  }
  const KernelPrimitiveBase *specialize(const KernelShapes &s) const override {
    if constexpr (Rows == 0) {
      if (valid_panel(s)) {
        const auto original = original_shapes(s);
        if (dispatch::quant_plan(original).shared_activation_panel &&
            dispatch::q4_shared_panel_rows(original) == 8) {
          static const Q4GlobalPanelKernel<8> rows8;
          return &rows8;
        }
      }
    }
    return this;
  }
  static Dot4Schedule schedule_for(const KernelShapes &original) {
    auto schedule = dot4_schedule(original);
    if constexpr (Rows != 0)
      schedule.rows = Rows;
    return schedule;
  }
  std::string emit_kernel(const KernelShapes &s) const override {
    if (!valid_panel(s) || !s.types.scalar || !s.store || !s.intrinsics)
      return {};
    const auto original = original_shapes(s);
    if (!dispatch::quant_plan(original).shared_activation_panel) {
      const auto *kernel = legacy();
      return kernel ? kernel->emit_kernel(original) : std::string{};
    }
    const auto schedule = schedule_for(original);
    return schedule.valid()
        ? emit_panel(s, schedule, dispatch::q4_shared_panel_load_chunks(original))
        : std::string{};
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    if (!valid_panel(s))
      return {};
    const auto original = original_shapes(s);
    if (!dispatch::quant_plan(original).shared_activation_panel) {
      const auto *kernel = legacy();
      return kernel ? kernel->plan(original) : ThreadPlan{};
    }
    const auto schedule = schedule_for(original);
    if (!schedule.valid())
      return {};
    ThreadPlan plan;
    const auto columns = kBlock / schedule.wave;
    const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
    plan.workgroup_size[0] = kBlock;
    plan.workgroup_count[0] = (n + columns - 1) / columns;
    const auto m = static_cast<std::uint32_t>(s.output.elem_count() / n);
    plan.workgroup_count[1] = (m + schedule.rows - 1) / schedule.rows;
    return plan;
  }
};
using Q4GlobalPanelDefault = Q4GlobalPanelKernel<0>;
LSE_REGISTER_PRIMITIVE(Q4GlobalPanelDefault);
} // namespace lse::kernels
