#include "lse/dispatch/quant.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/kernels/lds_linear.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

#include <algorithm>
#include <array>
#include <type_traits>
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
  if (load_chunks == 4 && count == 4 && acc.size() >= 2) {
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
      const auto pair_rows = std::min<std::size_t>(2, acc.size() - row_pair);
      std::vector<kir::Pack<kir::u32>> codes_first, codes_second, steps;
      for (std::size_t at = 0; at < pair_rows; ++at) {
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
        for (std::size_t at = 0; at < pair_rows; ++at) {
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

// Several columns per wave. emit_panel gives each wave one column, so every
// wave reads every row's activation codes and steps for its column alone: at
// eight rows a lane reads 24 activation words per weight word. Here a wave
// owns `columns` adjacent columns and each lane reads a run's activations
// once for all of them. It pays where the grid is large enough to keep the
// device full with a quarter of the waves -- the vocabulary projection -- and
// costs the narrower projections occupancy, so the shape table chooses it
// (Q4PanelShape::columns).
//
// Every lane keeps emit_panel's chunks, accumulation order and bias pass for
// each of its columns. The closing reduction halves the values a lane holds
// at each xor step instead of reducing every value on every lane; each step
// adds the same two partial sums the xor tree adds, so each output is the
// value emit_panel's lane 0 stores, and the lanes store one output each.
std::string emit_panel_columns(const KernelShapes &s,
                               const Dot4Schedule &schedule,
                               std::uint32_t columns) {
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k =
      static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  const auto m = static_cast<std::uint32_t>(s.output.elem_count() / n);
  const auto lanes = k / 8u, groups = k / 64u, nchunks = k / 8u;
  const auto stride = groups * 25u;
  const auto wave = schedule.wave;
  const auto waves = kBlock / wave;
  const auto cpl = schedule.chunks_per_lane;
  const auto span = wave * cpl;
  const auto per_group = waves * columns;
  const auto ntiles = (n + per_group - 1) / per_group;
  const auto rows = schedule.rows;
  const auto ksplits = schedule.k_splits;
  const auto values = rows * columns;
  if (wave != 32 || (values & (values - 1)) != 0 || values > wave ||
      (cpl != 1 && cpl != 2 && cpl != 4))
    return {};
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
  const auto col0 = e.let((tile * waves + wave_id) * columns);
  const auto row = e.let(math::workgroup_id_y() * rows);
  // acc[c * rows + r]: column c, row r.
  std::vector<kir::LValue<kir::f32>> acc;
  acc.reserve(values);
  for (std::uint32_t v = 0; v < values; ++v)
    acc.push_back(e.var(0.0f));
  // A pass shorter than its schedule's rows (dispatch::verify_rows) computes
  // its own rows alone; the others hold zero through the reduction and are
  // never stored.
  const auto live = std::min(rows, m);
  std::vector<kir::Val<kir::u32>> panel_base, row_base, scale_base;
  for (std::uint32_t r = 0; r < live; ++r) {
    const auto input_row = e.let(select(row + r < m, row + r, e.u32(0)));
    panel_base.push_back(e.let(input_row * stride));
  }
  for (std::uint32_t c = 0; c < columns; ++c) {
    // A column past the end reads the last column; it is never stored.
    const auto col = e.let(col0 + c);
    const auto safe = e.let(select(col < n, col, e.u32(n - 1u)));
    row_base.push_back(e.let(safe * lanes));
    scale_base.push_back(e.let(safe * groups));
  }
  // One run of `count` chunks at `chunk` for every column: the weight words
  // and scales first, then each row's codes once for all columns.
  const auto run = [&](const kir::Val<kir::u32> &chunk, std::uint32_t count) {
    std::vector<std::vector<std::array<kir::Val<kir::u32>, 2>>> planes(columns);
    std::vector<kir::Val<kir::f32>> wscale;
    {
      std::vector<kir::Pack<kir::u32>> words;
      using Raw = std::remove_cvref_t<decltype(a.scales[e.u32(0)])>;
      std::vector<Raw> raw;
      const auto group = e.let(chunk / 8u);
      for (std::uint32_t c = 0; c < columns; ++c) {
        words.push_back(e.load(a.packed, e.let(row_base[c] + chunk), count * 4u));
        raw.push_back(e.let(a.scales[scale_base[c] + group]));
      }
      for (std::uint32_t c = 0; c < columns; ++c) {
        for (std::uint32_t j = 0; j < count; ++j) {
          const auto word = e.let(words[c][static_cast<int>(j)]);
          std::array<kir::Val<kir::u32>, 2> value;
          for (std::size_t plane = 0; plane < 2; ++plane)
            value[plane] =
                quant::dot4_code_plane(e, word, static_cast<int>(plane));
          planes[c].push_back(value);
        }
        wscale.push_back(e.let(math::widen(raw[c])));
      }
    }
    for (std::uint32_t r = 0; r < live; ++r) {
      std::vector<kir::Val<kir::u32>> codes;
      for (std::uint32_t at = 0; at < count * 2u; at += 4u) {
        const auto width = std::min(4u, count * 2u - at);
        const auto pack = e.load(
            a.panel, e.let(panel_base[r] + chunk * 2u + at), width * 4u);
        for (std::uint32_t w = 0; w < width; ++w)
          codes.push_back(e.let(pack[static_cast<int>(w)]));
      }
      const auto steps =
          e.load(a.panel, e.let(panel_base[r] + k / 4u + chunk), count * 4u);
      std::vector<kir::Val<kir::f32>> step;
      for (std::uint32_t j = 0; j < count; ++j)
        step.push_back(
            e.let(math::from_bits<lse::f32>(steps[static_cast<int>(j)])));
      for (std::uint32_t c = 0; c < columns; ++c) {
        auto facc = e.var(e.f32(0.0f));
        for (std::uint32_t j = 0; j < count; ++j) {
          auto iacc = e.var(kir::cast<kir::i32>(e.u32(0)));
          for (std::size_t plane = 0; plane < 2; ++plane)
            iacc = math::dot4_iu8(kir::cast<kir::i32>(codes[j * 2u + plane]),
                                  kir::cast<kir::i32>(planes[c][j][plane]),
                                  iacc.read());
          facc = math::fma(step[j], kir::cast<kir::f32>(iacc.read()),
                           facc.read());
        }
        auto &target = acc[c * rows + r];
        target = math::fma(wscale[c], facc.read(), target.read());
      }
    }
  };
  if (auto in_grid = e.when(tile < ntiles && row < m)) {
    for (std::uint32_t ks = 0; ks < ksplits; ++ks) {
      const auto cb = ks * (nchunks / ksplits);
      const auto ce = cb + nchunks / ksplits;
      const auto ntile = ce - cb;
      const auto aligned = (ntile / span) * span;
      for (auto c0 : e.range(cb, cb + aligned, span))
        run(e.let(c0 + lane * cpl), cpl);
      if (aligned < ntile)
        for (auto chunk : e.range(e.u32(cb + aligned) + lane, e.u32(ce), wave))
          run(chunk, 1);
      for (auto g : e.range(e.u32(cb / 8u) + lane, e.u32(ce / 8u), wave)) {
        std::vector<kir::Val<kir::f32>> bias;
        for (std::uint32_t c = 0; c < columns; ++c)
          bias.push_back(e.let(math::widen(a.biases[scale_base[c] + g])));
        for (std::uint32_t r = 0; r < live; ++r) {
          const auto sum = e.let(math::from_bits<lse::f32>(
              a.panel[panel_base[r] + 3u * k / 8u + g]));
          for (std::uint32_t c = 0; c < columns; ++c) {
            auto &target = acc[c * rows + r];
            target = math::fma(bias[c], sum, target.read());
          }
        }
      }
    }
  }
  // Outside every guard: the shuffles are wave-cooperative. At the step for
  // bit b a lane keeps the half of its values its bit b selects and receives
  // its partner's copy of that half.
  std::vector<kir::Val<kir::f32>> held;
  for (std::uint32_t v = 0; v < values; ++v)
    held.push_back(acc[v].read());
  for (std::uint32_t bit = 1; bit < wave; bit <<= 1) {
    if (held.size() == 1) {
      held[0] = e.let(held[0] + math::shfl_xor(held[0], e.u32(bit)));
      continue;
    }
    const auto upper = e.let(lane / bit % 2u == 1u);
    const auto half = held.size() / 2;
    std::vector<kir::Val<kir::f32>> next;
    for (std::size_t i = 0; i < half; ++i) {
      const auto keep = e.let(select(upper, held[half + i], held[i]));
      const auto send = e.let(select(upper, held[i], held[half + i]));
      next.push_back(e.let(keep + math::shfl_xor(send, e.u32(bit))));
    }
    held = std::move(next);
  }
  // A lane's value index is its low bits reversed: bit 1 chose the top half
  // of all `values`, bit 2 the top half of what remained, and so on.
  auto index = e.u32(0);
  {
    std::uint32_t weight = values / 2u;
    for (std::uint32_t bit = 1; weight > 0; bit <<= 1, weight /= 2u)
      index = e.let(index + (lane / bit % 2u) * weight);
  }
  const auto c = e.let(index / rows);
  const auto r = e.let(index % rows);
  const auto col = e.let(col0 + c);
  const auto rr = e.let(row + r);
  const auto owner = e.let(lane < values);
  if (auto writer = e.when(owner && tile < ntiles && col < n && rr < m))
    e.store(rr * n + col, held[0]);
  return kb.str();
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
  // A pass shorter than its schedule's rows (dispatch::verify_rows) computes
  // its own rows alone, each exactly as the schedule's pass computes it.
  const auto live = std::min(rows, m);
  std::vector<kir::LValue<kir::f32>> acc;
  acc.reserve(live);
  for (std::uint32_t r = 0; r < live; ++r)
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
  for (std::uint32_t r = 0; r < live; ++r)
    for (std::uint32_t bit = 1; bit < wave; bit <<= 1)
      acc[r] = acc[r].read() + math::shfl_xor(acc[r].read(), e.u32(bit));
  for (std::uint32_t r = 0; r < live; ++r) {
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
    if (!schedule.valid())
      return {};
    if (const auto columns = dispatch::q4_shared_panel_columns(original);
        columns > 1)
      return emit_panel_columns(s, schedule, columns);
    return emit_panel(s, schedule,
                      dispatch::q4_shared_panel_load_chunks(original));
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
    const auto columns =
        kBlock / schedule.wave * dispatch::q4_shared_panel_columns(original);
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
