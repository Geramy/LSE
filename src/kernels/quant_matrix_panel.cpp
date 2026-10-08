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
#include <optional>
#include <vector>

namespace lse::kernels {
using namespace graph;
namespace {
// Four K16 fragments per group, followed by interleaved row step/sum words.
constexpr std::uint32_t kBlock = 256;
constexpr std::uint32_t kRows = dispatch::kQ4MatrixPanelRows;
constexpr std::uint32_t kGroupWords = dispatch::kQ4MatrixPanelGroupWords;

// Which of a group's 64 values byte `byte` of panel word `word` carries.
//
// The 64-row prefill tile packs four consecutive values per word. The 16-row
// decode tile instead puts the even values of each run of eight in its first
// word and the odd values in its second, the order a 4-bit weight word's two
// nibble planes come in: the decode contraction then widens a weight word
// into its matrix operand with one shift and one mask per register. The two
// operands meet in the same order, so the integer products are unchanged.
// A panel made for 8-bit weights (attribute 0 = 8) keeps four consecutive
// values per word at every tile height: its weight words are already bytes.
constexpr std::size_t code_slot(std::size_t word, std::size_t byte,
                                std::size_t tile_rows, bool bytes = false) {
  if (tile_rows != kRows || bytes)
    return word * 4 + byte;
  return (word / 2) * 8 + word % 2 + 2 * byte;
}
Result<Shape> matrix_panel_shape(std::span<const Shape> in) {
  if (in.size() != 1 || !in[0].rank())
    return LSE_ERROR(kInvalidArgument,
                     "matrix panel needs complete group64 rows");
  const auto shape = dispatch::q4_matrix_panel_storage_shape(in[0]);
  if (!shape.rank())
    return LSE_ERROR(kInvalidArgument, "invalid matrix panel extent");
  return shape;
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
                        const std::array<std::int32_t, 4> &iattrs) const override {
    if (in.size() != 1 || in[0].dtype != DType::kF32 ||
        out.dtype != DType::kU32)
      return LSE_ERROR(kInvalidArgument, "invalid matrix panel storage");
    const bool bytes = iattrs[0] == 8;
    const std::array input_shapes{in[0].shape};
    LSE_ASSIGN_OR(const auto expected, matrix_panel_shape(input_shapes));
    if (out.shape != expected ||
        in[0].bytes.size() != in[0].shape.elem_count() * 4 ||
        out.bytes.size() != out.shape.elem_count() * 4)
      return LSE_ERROR(kInvalidArgument, "invalid matrix panel byte extent");
    const auto k =
        static_cast<std::size_t>(in[0].shape.dim(in[0].shape.rank() - 1));
    const auto *layout = dispatch::q4_matrix_panel_layout(in[0].shape);
    const auto rows = in[0].shape.elem_count() / k;
    const auto tile_rows = static_cast<std::size_t>(layout->rows);
    const auto padded_rows = (rows + tile_rows - 1) / tile_rows * tile_rows;
    const auto tile_words = tile_rows / kRows * kGroupWords;
    const auto store = [&](std::size_t at, std::uint32_t bits) {
      std::memcpy(out.bytes.data() + at * 4, &bits, 4);
    };
    for (std::size_t group = 0; group < k / 64; ++group) {
      for (std::size_t row = 0; row < padded_rows; ++row) {
        std::array<float, 64> values{};
        if (row < rows)
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
        const auto row16 = row % kRows;
        const auto dst = ((row / tile_rows) * (k / 64) + group) * tile_words +
                         ((row % tile_rows) / kRows) * kGroupWords;
        const float inverse = 127.0f / std::max(maximum, 1e-30f);
        for (std::size_t word = 0; word < 16; ++word) {
          std::uint32_t packed = 0;
          for (std::size_t byte = 0; byte < 4; ++byte) {
            const float rounded =
                std::nearbyint(values[code_slot(word, byte, tile_rows, bytes)] * inverse);
            const auto code =
                std::isfinite(rounded) ? static_cast<std::int32_t>(rounded) : 0;
            packed |= (static_cast<std::uint32_t>(code) & 255u) << (8 * byte);
          }
          const auto at = dst + (word / 4) * 64 + ((word % 4) / 2) * 32 +
                          row16 * 2 + word % 2;
          store(at, packed);
        }
        store(dst + 256 + row16 * 2,
              std::bit_cast<std::uint32_t>(maximum * (1.0f / 127.0f)));
        store(dst + 257 + row16 * 2, std::bit_cast<std::uint32_t>(sum));
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
    const auto *layout = dispatch::q4_matrix_panel_layout(s.inputs[0]);
    const auto input_rows = static_cast<std::uint32_t>(
        s.inputs[0].elem_count() / static_cast<std::uint64_t>(k));
    const auto padded_rows =
        (input_rows + layout->rows - 1u) / layout->rows * layout->rows;
    const auto tile_words = layout->rows / kRows * kGroupWords;
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
    if (auto valid = e.when(row < padded_rows && group < groups)) {
      std::vector<kir::LValue<kir::f32>> values;
      for (int t = 0; t < 16; ++t)
        values.push_back(e.var(0.0f));
      if (auto actual = e.when(row < input_rows)) {
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
      // 16-row tiles past the first (a pass of more than 16 rows) follow it.
      const bool tiles = layout->rows == kRows && padded_rows > kRows;
      const auto row16 = layout->rows == kRows && !tiles ? row : e.let(row % kRows);
      const auto dst =
          layout->rows == kRows
              ? (tiles ? e.let(((row / kRows) * groups + group) * kGroupWords)
                       : e.let(group * kGroupWords))
              : e.let(((row / layout->rows) * groups + group) * tile_words +
                      ((row % layout->rows) / kRows) * kGroupWords);
      for (std::uint32_t word_index = 0; word_index < 4; ++word_index) {
        auto word = e.let(e.u32(0));
        for (std::uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
          const auto code = e.let(math::rint(
              values[code_slot(word_index, byte_index, layout->rows,
                               s.iattrs[0] == 8)].read() *
              inverse));
          const auto byte =
              e.let(kir::cast<kir::u32>(kir::cast<kir::i32>(code)) % 256u);
          word = e.let(word + byte * (1u << (8u * byte_index)));
        }
        a.out[dst + slice * 64u + (word_index / 2u) * 32u + row16 * 2u +
              word_index % 2u] = word;
      }
      if (auto leader = e.when(slice == 0u)) {
        a.out[dst + 256u + row16 * 2u] = math::bits_of<lse::f32>(step);
        a.out[dst + 257u + row16 * 2u] = math::bits_of<lse::f32>(total);
      }
    }
    return kb.str();
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    if (!supported(s))
      return tp;
    const auto *layout = dispatch::q4_matrix_panel_layout(s.inputs[0]);
    const auto groups = static_cast<std::uint32_t>(
        s.inputs[0].dim(s.inputs[0].rank() - 1) / 64);
    tp.workgroup_size[0] = kBlock;
    tp.workgroup_count[0] = (groups + 63u) / 64u;
    const auto input_rows = static_cast<std::uint32_t>(
        s.inputs[0].elem_count() /
        static_cast<std::uint64_t>(s.inputs[0].dim(s.inputs[0].rank() - 1)));
    tp.workgroup_count[1] =
        (input_rows + layout->rows - 1u) / layout->rows * layout->rows;
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
  return s.inputs[4] == dispatch::q4_matrix_panel_storage_shape(s.inputs[0]);
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
// A decode workgroup owns one 16-column tile per wave over the whole K. Halve
// the waves per workgroup until the tiles cover every compute unit.
const dispatch::MatrixPanelTune *matrix_tune(const KernelShapes &s, std::uint32_t bits,
                                   std::uint32_t n, std::uint32_t k) {
  if (!s.device) return nullptr;
  for (const auto &tune : dispatch::arch::tuning(s.device->arch).matrix_panel_tunes)
    if (tune.arch == s.device->arch && tune.bits == bits && tune.n == n && tune.k == k)
      return &tune;
  return nullptr;
}
std::uint32_t matrix_waves(const KernelShapes &s, std::uint32_t n);
// K slices per tile: the part's tune, at most the workgroup's waves.
std::uint32_t matrix_ksplits(const KernelShapes &s, std::uint32_t bits,
                             std::uint32_t n, std::uint32_t k) {
  const auto *tune = matrix_tune(s, bits, n, k);
  const auto waves = matrix_waves(s, n);
  if (!tune || tune->k_splits <= 1 || waves % tune->k_splits != 0 ||
      (k / 64u) % tune->k_splits != 0)
    return 1;
  return tune->k_splits;
}
std::uint32_t matrix_waves(const KernelShapes &s, std::uint32_t n) {
  const auto tiles = (n + 15u) / 16u;
  const auto units = s.device ? static_cast<std::uint32_t>(s.device->compute_units) : 0u;
  auto waves = kBlock / 32u;
  while (waves > 1u && (tiles + waves - 1u) / waves < units)
    waves /= 2u;
  return waves;
}
// Groups whose loads issue together before their products accumulate in
// order. Four is the most the emitter's register budget takes.
// A generation whose lanes carry the whole K step loads twice the operand
// words per group, so it takes half the groups per round.
constexpr std::uint32_t kMatrixUnroll = 4;
std::uint32_t matrix_unroll(std::uint32_t groups,
                            std::uint32_t most = kMatrixUnroll) {
  std::uint32_t unroll = most;
  while (unroll > 1u && groups % unroll != 0u)
    unroll /= 2u;
  return unroll;
}
// The layouts the decode panel can take, as variants (each changes when
// loads issue and which wave does what, never the order an output's terms
// are summed in):
//  - the groups one wave loads per round: the part's tune or the
//    generation's default first, then every other count the slice divides;
//  - an exact split of K across adjacent waves, when the part's tune does
//    not already split it: the slices load a round's groups in turn, and the
//    tile's first wave applies every group of the round to the outputs in
//    group order, on each group's exact integer product, exactly as one wave
//    would have.
// Variant 0 is the part's tune as it stands (a tuned split sums its slices'
// partial outputs, which the exact splits cannot reproduce, so a tuned split
// offers no exact ones).
struct MatrixChoice {
  std::uint32_t unroll = 1;
  std::uint32_t ks = 1;
  bool exact = false;
};
std::vector<MatrixChoice> matrix_menu(const KernelShapes &s, std::uint32_t bits,
                                      std::uint32_t n, std::uint32_t k) {
  const auto groups = k / 64u;
  const auto tuned_ks = matrix_ksplits(s, bits, n, k);
  const auto *tune = matrix_tune(s, bits, n, k);
  const bool split_k = s.device && matrix_target(*s.device) == math::MatrixTarget::kRdna4;
  std::vector<MatrixChoice> menu{{matrix_unroll(
      groups / tuned_ks, tune && tune->unroll ? tune->unroll
                         : split_k ? kMatrixUnroll : kMatrixUnroll / 2u), tuned_ks, false}};
  const auto add = [&](const MatrixChoice &c) {
    for (const auto &m : menu)
      if (m.unroll == c.unroll && m.ks == c.ks && m.exact == c.exact) return;
    menu.push_back(c);
  };
  for (const std::uint32_t most : {kMatrixUnroll, kMatrixUnroll / 2u, 1u})
    add({matrix_unroll(groups / tuned_ks, most), tuned_ks, false});
  if (tuned_ks == 1 && s.device) {
    const auto waves = matrix_waves(s, n);
    const auto budget = backend::workgroup_lds_bytes(s.device);
    for (const std::uint32_t ks : {2u, 4u}) {
      if (waves % ks != 0u || groups % ks != 0u) continue;
      for (const std::uint32_t most : {kMatrixUnroll, kMatrixUnroll / 2u}) {
        const auto u = matrix_unroll(groups / ks, most);
        if (waves * u * 8u * 32u * 4u <= budget) add({u, ks, true});
      }
    }
    // Splits whose slices each sum their own contiguous run of groups, then
    // add the slices' partial outputs in slice order: cheaper than the exact
    // chain, but a different order of sums (variant_reassociates), so only a
    // draft's kernels may take them.
    for (const std::uint32_t ks : {2u, 4u}) {
      if (waves % ks != 0u || groups % ks != 0u ||
          waves * 8u * 32u * 4u > budget) continue;
      for (const std::uint32_t most : {kMatrixUnroll, kMatrixUnroll / 2u})
        add({matrix_unroll(groups / ks, most), ks, false});
    }
  }
  return menu;
}
// The order an output's terms are summed in: one sequence (no split, or an
// exact one), or the slices' partial sums of a split of `ks`.
std::uint32_t matrix_sum_order(const MatrixChoice &c) {
  return c.ks > 1u && !c.exact ? c.ks : 1u;
}
bool matrix_reassociates(const KernelShapes &s, std::uint32_t bits, std::uint32_t variant) {
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  const auto menu = matrix_menu(s, bits, n, k);
  return variant < menu.size() &&
         matrix_sum_order(menu[variant]) != matrix_sum_order(menu[0]);
}
std::uint32_t matrix_variants(const KernelShapes &s, std::uint32_t bits) {
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  return static_cast<std::uint32_t>(matrix_menu(s, bits, n, k).size());
}
// The variant's choice, or nothing for a variant the shape does not offer.
std::optional<MatrixChoice> matrix_choice(const KernelShapes &s, std::uint32_t bits) {
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k = static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  const auto menu = matrix_menu(s, bits, n, k);
  if (s.variant >= menu.size()) return std::nullopt;
  return menu[s.variant];
}
std::uint32_t matrix_lds_bytes(const MatrixChoice &c, std::uint32_t waves) {
  if (c.ks <= 1u) return 0u;
  return waves * (c.exact ? c.unroll : 1u) * 8u * 32u * 4u;
}
// `Bits` is the weight width: 4-bit words widen to matrix operands by nibble
// plane, 8-bit words already are operands, four codes to a register.
//
// `G` is the device's matrix generation. On RDNA4 a lane holds half of each
// K16 product (the half-waves split it); on RDNA3/3.5 a lane holds all of it
// and the half-waves repeat the same rows, so each product reads both halves'
// weight words and activation words, and accumulator slot z is output row
// 2z + half-wave instead of z + 8 * half-wave (lse/math/matrix_rdna3.hpp).
template <std::uint32_t Bits, math::MatrixTarget G>
std::string emit_matrix(const KernelShapes &s) {
  static_assert(Bits == 4 || Bits == 8);
  using Mma = math::op::Mma<G, math::MatrixElem::kI32,
                            math::MatrixElem::kSU8, 16, 16, 16>;
  constexpr auto row = Mma::kRow;
  constexpr auto geo = geometry_of(row);
  constexpr int kFrag = row.a_len / row.chained;
  constexpr int kSlots = row.c_len;
  static_assert(row.wave == 32 && row.chained == 1 && kSlots == 8 &&
                (geo.split_k ? kFrag == 2 : kFrag == 4),
                "the decode panel is written for the wave32 iu8 rows");
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto k =
      static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1));
  // The pass's rows; the tile's other rows are the panel's zero padding.
  const auto input_rows = static_cast<std::uint32_t>(
      s.inputs[0].elem_count() / static_cast<std::uint64_t>(k));
  const auto groups = k / 64u;
  const auto lanes = k * Bits / 32u;
  const auto waves = matrix_waves(s, n);
  // A part's tune may cut K into slices walked by adjacent waves (summed in
  // slice order at the end) and set the groups loaded together; without one
  // the kernel is exactly the single-slice form.
  const auto choice = matrix_choice(s, Bits);
  if (!choice)
    return {};
  const auto ks = choice->ks;
  const auto unroll = choice->unroll;
  const bool exact = choice->exact;
  const auto tiles_n = (n + 15u) / 16u;
  const auto tile_waves = waves / ks;
  const auto nblocks = (tiles_n + tile_waves - 1u) / tile_waves;
  kir::KernelBody kb(s.types, *s.intrinsics,
                     ks > 1 ? backend::workgroup_lds_bytes(s.device) : 0u);
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
  // A pass of more than 16 rows reads its tile of the panel; one tile is the
  // whole panel.
  const auto panel_tile = [&](const kir::Val<kir::u32>& index) {
    return input_rows > 16u ? e.let((m0 / 16u) * (groups * kGroupWords) + index) : index;
  };
  // The single-slice form emits exactly what it always has; only a split
  // names its waves' slices.
  std::optional<kir::Val<kir::u32>> wave_index, kslice;
  std::optional<kir::Tile<kir::f32>> products;
  if (ks > 1) {
    wave_index = e.let(lid / 32u);
    kslice = e.let(*wave_index % ks);
    if (exact)
      products = e.lds<kir::f32>(waves * unroll * kSlots * 32u);
  }
  const auto ntile = ks == 1 ? e.let((wg % nblocks) * waves + lid / 32u)
                             : e.let((wg % nblocks) * tile_waves + *wave_index / ks);
  const auto col = e.let(ntile * 16u + lo);
  const auto live = e.let(ntile < tiles_n && col < n);
  const auto safe_col = e.let(select(col < n, col, e.u32(0)));
  std::vector<kir::LValue<kir::f32>> out;
  for (int z = 0; z < kSlots; ++z)
    out.push_back(e.var(0.0f));
  // A group's four K16 products sum exactly in i32, so the K order inside a
  // group is free as long as weights and activations agree. Lane half `hi`
  // takes the group's words hi*4..hi*4+3 (one 16-byte load) and product t
  // pairs word hi*4+t, codes K[32hi+8t, +8), with the activation half that
  // holds the same codes: slice 2hi+t/2, half t%2.
  // An 8-bit group is twice the words: each half-wave reads its eight in two
  // 16-byte loads and product t takes words 2t and 2t+1.
  struct Loaded {
    kir::Val<lse::bf16> raw_scale, raw_bias;
    std::vector<kir::Pack<kir::u32>> weight, acts, metadata;
  };
  constexpr std::uint32_t kGroupWeightWords = 64u * Bits / 32u;
  // The half-wave's activation words on a split layout; the row's alone on
  // a contiguous one, which reads both halves.
  const auto act_lane = geo.split_k ? e.let(hi * 128u + lo * 2u) : e.let(lo * 2u);
  const auto groups_body = [&](const kir::Val<kir::u32>& g0) {
    std::vector<Loaded> loaded;
    loaded.reserve(unroll);
    for (std::uint32_t u = 0; u < unroll; ++u) {
      // An exact split's waves take every ks-th group of a round, so the
      // round's groups, in order, are the slices' in turn.
      const auto g = exact ? e.let(g0 + u * ks + *kslice) : e.let(g0 + u);
      const auto sa = e.let(safe_col * groups + g);
      Loaded l{e.let(a.scales[sa]), e.let(a.biases[sa]), {}, {}, {}};
      if constexpr (geo.split_k) {
      const auto wbase = e.let(safe_col * lanes + g * kGroupWeightWords +
                               hi * (kGroupWeightWords / 2u));
      for (std::uint32_t w = 0; w < kGroupWeightWords / 2u; w += 4u)
        l.weight.push_back(e.load(a.packed, e.let(wbase + w), 16u));
      for (std::uint32_t t = 0; t < 4u; ++t)
        l.acts.push_back(e.load(
            a.panel, panel_tile(e.let(g * kGroupWords + act_lane + (t / 2u) * 64u + (t % 2u) * 32u)), 8u));
      for (std::uint32_t z = 0; z < 8u; z += 2u)
        l.metadata.push_back(e.load(
            a.panel, panel_tile(e.let(g * kGroupWords + 256u + hi * 16u + z * 2u)), 16u));
      } else {
        // The whole group's weight words; for product t the activation
        // words of both halves (acts[2t + half]); and the step and sum of
        // each row this lane's slots hold (rows 2z + hi), one pair a load.
        const auto wbase = e.let(safe_col * lanes + g * kGroupWeightWords);
        for (std::uint32_t w = 0; w < kGroupWeightWords; w += 4u)
          l.weight.push_back(e.load(a.packed, e.let(wbase + w), 16u));
        for (std::uint32_t t = 0; t < 4u; ++t)
          for (std::uint32_t h = 0; h < 2u; ++h)
            l.acts.push_back(e.load(
                a.panel, panel_tile(e.let(g * kGroupWords + act_lane + h * 128u +
                                          (t / 2u) * 64u + (t % 2u) * 32u)), 8u));
        for (std::uint32_t z = 0; z < 8u; ++z)
          l.metadata.push_back(e.load(
              a.panel, panel_tile(e.let(g * kGroupWords + 256u + hi * 2u + z * 4u)), 8u));
      }
      loaded.push_back(std::move(l));
    }
    for (std::uint32_t u = 0; u < unroll; ++u) {
      const auto &l = loaded[u];
      const auto scale = e.let(math::widen(l.raw_scale));
      const auto bias = e.let(math::widen(l.raw_bias));
      const auto acc = e.local<kir::i32, kSlots>();
      for (int z = 0; z < kSlots; ++z)
        acc[z] = kir::cast<kir::i32>(e.u32(0));
      for (std::uint32_t t = 0; t < 4u; ++t) {
        const auto bf = e.local<kir::u32, kFrag>();
        const auto af = e.local<kir::u32, kFrag>();
        for (int f = 0; f < kFrag; ++f) {
          // Which half of the product this register carries: the lane's own
          // on a split layout, half f / 2 on a contiguous one, whose
          // registers are the two halves' in turn.
          const std::uint32_t half_words = geo.split_k ? 0u : kGroupWeightWords / 2u;
          const std::uint32_t h = geo.split_k ? 0u : static_cast<std::uint32_t>(f) / 2u;
          const int r = geo.split_k ? f : f % 2;
          if constexpr (Bits == 4) {
            // The panel pairs each weight word with its activations in
            // nibble plane order (code_slot): plane f is the word's nibbles
            // f, f+2, f+4 and f+6, one to a byte.
            const auto wi = h * half_words + t;
            const auto word = e.let(l.weight[wi / 4u][static_cast<int>(wi % 4u)]);
            bf[f] = e.let(math::bit_and(
                word / (1u << (4u * static_cast<std::uint32_t>(r))),
                e.u32(0x0f0f0f0fu)));
          } else {
            const auto at = h * half_words + t * 2u + static_cast<std::uint32_t>(r);
            bf[f] = e.let(l.weight[at / 4u][static_cast<int>(at % 4u)]);
          }
          af[f] = l.acts[geo.split_k ? t : t * 2u + h][r];
        }
        acc = math::mma<Mma>(af.value(), bf.value(), acc.value());
      }
      if (exact) {
        for (int z = 0; z < kSlots; ++z)
          (*products)[e.let(((*wave_index * unroll + u) * kSlots +
                             static_cast<std::uint32_t>(z)) * 32u + lane)] =
              kir::cast<kir::f32>(acc[z].read());
        continue;
      }
      for (int z = 0; z < kSlots; ++z) {
        const auto pair = static_cast<std::size_t>(geo.split_k ? z / 2 : z);
        const int lane_pair = geo.split_k ? (z % 2) * 2 : 0;
        const auto step = e.let(math::from_bits<lse::f32>(
            l.metadata[pair][lane_pair]));
        const auto sum = e.let(math::from_bits<lse::f32>(
            l.metadata[pair][lane_pair + 1]));
        const auto term = e.let(scale * step);
        out[static_cast<std::size_t>(z)] =
            math::fma(term, kir::cast<kir::f32>(acc[z].read()),
                      out[static_cast<std::size_t>(z)].read()) +
            bias * sum;
      }
    }
    if (exact) {
      // The tile's first wave applies the round's groups to its outputs in
      // group order: the same scale, step and sum, the same fma and add, on
      // each group's exact product, as the single-wave loop.
      e.barrier();
      if (auto first = e.when(*kslice == 0u)) {
        for (std::uint32_t i = 0; i < unroll * ks; ++i) {
          const std::uint32_t u = i / ks, j = i % ks;
          const auto g = e.let(g0 + i);
          const auto sa = e.let(safe_col * groups + g);
          const auto scale = e.let(math::widen(e.let(a.scales[sa])));
          const auto bias = e.let(math::widen(e.let(a.biases[sa])));
          std::vector<kir::Pack<kir::u32>> metadata;
          if constexpr (geo.split_k) {
            for (std::uint32_t z = 0; z < 8u; z += 2u)
              metadata.push_back(e.load(
                  a.panel, panel_tile(e.let(g * kGroupWords + 256u + hi * 16u + z * 2u)), 16u));
          } else {
            for (std::uint32_t z = 0; z < 8u; ++z)
              metadata.push_back(e.load(
                  a.panel, panel_tile(e.let(g * kGroupWords + 256u + hi * 2u + z * 4u)), 8u));
          }
          for (int z = 0; z < kSlots; ++z) {
            const auto pair = static_cast<std::size_t>(geo.split_k ? z / 2 : z);
            const int lane_pair = geo.split_k ? (z % 2) * 2 : 0;
            const auto step = e.let(math::from_bits<lse::f32>(metadata[pair][lane_pair]));
            const auto sum = e.let(math::from_bits<lse::f32>(metadata[pair][lane_pair + 1]));
            const auto term = e.let(scale * step);
            const auto product = (*products)[e.let(
                (((*wave_index + j) * unroll + u) * kSlots +
                 static_cast<std::uint32_t>(z)) * 32u + lane)].read();
            out[static_cast<std::size_t>(z)] =
                math::fma(term, product, out[static_cast<std::size_t>(z)].read()) +
                bias * sum;
          }
        }
      }
      // The next round's products overwrite this one's.
      e.barrier();
    }
  };
  if (ks == 1) {
    for (auto g0 : e.range(0u, groups, unroll)) groups_body(g0);
  } else if (exact) {
    for (auto g0 : e.range(0u, groups, unroll * ks)) groups_body(g0);
  } else {
    const auto g_begin = e.let(*kslice * (groups / ks));
    for (auto g0 : e.range(g_begin, e.let(g_begin + groups / ks), unroll))
      groups_body(g0);
  }
  if (ks > 1 && !exact) {
    // Each tile's slices are adjacent waves; the first adds the others in
    // slice order.
    const auto partial = e.lds<kir::f32>(waves * kSlots * 32u);
    for (int z = 0; z < kSlots; ++z)
      partial[e.let((*wave_index * kSlots + static_cast<std::uint32_t>(z)) * 32u + lane)] =
          out[static_cast<std::size_t>(z)].read();
    e.barrier();
    for (std::uint32_t j = 1; j < ks; ++j)
      for (int z = 0; z < kSlots; ++z)
        out[static_cast<std::size_t>(z)] =
            out[static_cast<std::size_t>(z)].read() +
            partial[e.let(((*wave_index + j) * kSlots + static_cast<std::uint32_t>(z)) * 32u + lane)].read();
  }
  for (int z = 0; z < kSlots; ++z) {
    const auto output_row =
        e.let(m0 + hi * geo.half_rows +
              static_cast<std::uint32_t>(z) * geo.slot_step);
    if (ks == 1) {
      if (auto valid = e.when(live && output_row < input_rows))
        e.store(output_row * n + col, out[static_cast<std::size_t>(z)].read());
    } else if (auto valid = e.when(live && *kslice == 0u && output_row < input_rows)) {
      e.store(output_row * n + col, out[static_cast<std::size_t>(z)].read());
    }
  }
  return kb.lds().ok() ? kb.str() : std::string{};
}
// The decode panel for the device's matrix generation; a generation without
// the wave32 iu8 row emits nothing.
template <std::uint32_t Bits>
std::string emit_matrix_for(const KernelShapes &s) {
  const auto target = s.device ? matrix_target(*s.device) : std::nullopt;
  if (!target) return {};
  return with_matrix_target<std::string>(
      *target, [&]<math::MatrixTarget G>() -> std::string {
        if constexpr (!math::has_matrix_core_row(G, math::MatrixElem::kI32,
                                                 math::MatrixElem::kSU8, 16,
                                                 16, 16)) {
          return {};
        } else if constexpr (math::matrix_core_row(G, math::MatrixElem::kI32,
                                                   math::MatrixElem::kSU8, 16,
                                                   16, 16).wave != 32) {
          return {};
        } else {
          return emit_matrix<Bits, G>(s);
        }
      });
}
template <bool Cooperative, bool PairStage = false>
std::string emit_prefill_matrix(const KernelShapes &s) {
  static_assert(!PairStage || Cooperative);
  using Mma = math::op::Mma<math::MatrixTarget::kRdna4, math::MatrixElem::kI32,
                            math::MatrixElem::kSU8, 16, 16, 16>;
  constexpr auto row = Mma::kRow;
  constexpr auto geo = geometry_of(row);
  constexpr int frag = row.a_len / row.chained;
  constexpr int slots = row.c_len;
  const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
  const auto m = static_cast<std::uint32_t>(
      s.output.elem_count() / static_cast<std::uint64_t>(s.inputs[1].dim(0)));
  const auto groups =
      static_cast<std::uint32_t>(s.inputs[0].dim(s.inputs[0].rank() - 1) / 64);
  if constexpr (PairStage)
    if (groups % 2u != 0u)
      return {};
  const auto lanes = static_cast<std::uint32_t>(s.inputs[1].dim(1));
  constexpr std::uint32_t waves = kBlock / 32u;
  const auto tiles_n = (n + 15u) / 16u;
  const auto nblocks = (tiles_n + waves - 1u) / waves;
  kir::KernelBody kb(s.types, *s.intrinsics,
                     Cooperative ? backend::workgroup_lds_bytes(s.device) : 0);
  kb.set_store(s.store);
  MatrixArgs a;
  if (!env::bind(kb, a, s))
    return {};
  env::Emit e{&kb};
  std::optional<kir::Tile<kir::u32>> cached_panel;
  if constexpr (Cooperative)
    cached_panel = e.lds<kir::u32>(PairStage ? 2304u : 1152u);
  const auto lid = e.let(math::local_id());
  const auto lane = e.let(lid % 32u);
  const auto lo = e.let(lane % 16u);
  const auto hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto mblock = e.let(wg / nblocks);
  const auto m0 = e.let(mblock * 64u);
  const auto ntile = e.let((wg % nblocks) * waves + lid / 32u);
  const auto col = e.let(ntile * 16u + lo);
  const auto live = e.let(ntile < tiles_n && col < n);
  const auto safe_col = e.let(select(col < n, col, e.u32(0)));
  std::vector<kir::LValue<kir::f32>> out;
  for (int z = 0; z < 4 * slots; ++z)
    out.push_back(e.var(0.0f));
  for (auto g : e.range(0u, groups, PairStage ? 2u : 1u)) {
    if constexpr (Cooperative) {
      for (std::uint32_t pair = 0; pair < (PairStage ? 2u : 1u); ++pair) {
        const auto stage_base = e.let((mblock * groups + g + pair) * 1152u);
        const auto copy_chunk = [&](std::uint32_t chunk) {
          const auto offset = e.let(lid * 4u + chunk * 1024u);
          const auto packed = e.load(a.panel, e.let(stage_base + offset), 16u);
          const auto local_at = e.let(offset + pair * 1152u);
          kb.store_pack<kir::u32>(cached_panel->id(), local_at, packed, 16u);
        };
        copy_chunk(0u);
        if (auto tail = e.when(lid < 32u))
          copy_chunk(1u);
      }
      e.barrier();
    }
    for (std::uint32_t pair = 0; pair < (PairStage ? 2u : 1u); ++pair) {
      const auto group = e.let(g + pair);
      std::optional<kir::Val<kir::u32>> group_base;
      const auto sa = e.let(safe_col * groups + group);
      const auto scale = e.let(math::widen(a.scales[sa]));
      const auto bias = e.let(math::widen(a.biases[sa]));
      std::vector<kir::Local<kir::u32, frag>> bf;
      for (std::uint32_t t = 0; t < 4u; ++t) {
        const auto weight =
            e.let(a.packed[e.let(safe_col * lanes + group * 8u + t * 2u + hi)]);
        bf.push_back(e.local<kir::u32, frag>());
        for (int f = 0; f < frag; ++f) {
          const auto half = e.let(weight / (1u << (16u * f)));
          const auto paired = e.let(math::bit_and(half, e.u32(0xffu)) +
                                    math::bit_and(half, e.u32(0xff00u)) * 256u);
          bf[t][f] = e.let(math::bit_and(paired, e.u32(0x000f000fu)) +
                           math::bit_and(paired, e.u32(0x00f000f0u)) * 16u);
        }
      }
      if constexpr (!Cooperative)
        group_base = e.let((mblock * groups + group) * 1152u);
      for (std::uint32_t i = 0; i < 4u; ++i) {
        const auto block_base = Cooperative
                                    ? e.let(e.u32(pair * 1152u + i * 288u))
                                    : e.let(*group_base + i * 288u);
        const auto acc = e.local<kir::i32, slots>();
        for (int z = 0; z < slots; ++z)
          acc[z] = kir::cast<kir::i32>(e.u32(0));
        for (std::uint32_t t = 0; t < 4u; ++t) {
          const auto index = e.let(block_base + t * 64u + hi * 32u + lo * 2u);
          const auto av = Cooperative ? cached_panel->load(index, 8u)
                                      : e.load(a.panel, index, 8u);
          const auto af = e.local<kir::u32, frag>();
          for (int f = 0; f < frag; ++f)
            af[f] = av[f];
          acc = math::mma<Mma>(af.value(), bf[t].value(), acc.value());
        }
        std::vector<kir::Pack<kir::u32>> metadata;
        for (std::uint32_t z = 0; z < 8u; z += 2u)
          metadata.push_back(
              Cooperative
                  ? cached_panel->load(
                        e.let(block_base + 256u + hi * 16u + z * 2u), 16u)
                  : e.load(a.panel,
                           e.let(block_base + 256u + hi * 16u + z * 2u), 16u));
        for (int z = 0; z < slots; ++z) {
          const auto step = e.let(math::from_bits<lse::f32>(
              metadata[static_cast<std::size_t>(z / 2)][(z % 2) * 2]));
          const auto sum = e.let(math::from_bits<lse::f32>(
              metadata[static_cast<std::size_t>(z / 2)][(z % 2) * 2 + 1]));
          const auto term = e.let(scale * step);
          const auto at = i * slots + static_cast<std::uint32_t>(z);
          out[at] = math::fma(term, kir::cast<kir::f32>(acc[z].read()),
                              out[at].read()) +
                    bias * sum;
        }
      }
    }
    if constexpr (Cooperative)
      e.barrier();
  }
  for (std::uint32_t i = 0; i < 4u; ++i)
    for (int z = 0; z < slots; ++z) {
      const auto output_row =
          e.let(m0 + i * 16u + hi * geo.half_rows +
                static_cast<std::uint32_t>(z) * geo.slot_step);
      if (auto valid = e.when(live && output_row < m))
        e.store(output_row * n + col,
                out[i * slots + static_cast<std::uint32_t>(z)].read());
    }
  return kb.lds().ok() ? kb.str() : std::string{};
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
    if (dispatch::q4_matrix_panel_row(original)) {
      const auto *rule = dispatch::q4_matrix_panel_rule(original);
      if (rule->rows == kRows)
        return emit_matrix_for<4>(s);
      // The 64-row prefill tiles are written for RDNA4's split-K fragment
      // only; another generation has no rule for them, and says so here.
      if (matrix_target(*s.device) != math::MatrixTarget::kRdna4)
        return {};
      return rule->shared_words
                 ? (rule->n == 17408 ? emit_prefill_matrix<true, true>(s)
                                     : emit_prefill_matrix<true>(s))
                 : emit_prefill_matrix<false>(s);
    }
    const auto *kernel = legacy(original);
    return kernel ? kernel->emit_kernel(original) : std::string{};
  }
  std::uint32_t variants(const KernelShapes &s) const override {
    if (!valid_matrix_panel(s) || !s.device)
      return 1;
    const auto original = original_shapes(s);
    if (!dispatch::q4_matrix_panel_row(original) ||
        dispatch::q4_matrix_panel_rule(original)->rows != kRows)
      return 1;
    return matrix_variants(s, 4);
  }
  bool variant_reassociates(const KernelShapes &s, std::uint32_t variant) const override {
    return variants(s) > 1 && matrix_reassociates(s, 4, variant);
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
    const auto *rule = dispatch::q4_matrix_panel_rule(original);
    const auto m = static_cast<std::uint32_t>(
        s.output.elem_count() / static_cast<std::uint64_t>(n));
    tp.lds_bytes = rule->shared_words * sizeof(std::uint32_t);
    tp.workgroup_count[0] =
        ((m + rule->rows - 1u) / rule->rows) * ((n + 127u) / 128u);
    if (rule->rows == kRows) {
      const auto waves = matrix_waves(s, n);
      const auto choice = matrix_choice(s, 4);
      if (!choice) return ThreadPlan{};
      const auto ks = choice->ks;
      const auto tiles = (n + 15u) / 16u;
      tp.workgroup_size[0] = waves * 32u;
      tp.workgroup_count[0] = ((m + rule->rows - 1u) / rule->rows) *
                              ((tiles + waves / ks - 1u) / (waves / ks));
      if (ks > 1) tp.lds_bytes = matrix_lds_bytes(*choice, waves);
    }
    return tp;
  }
};
// The 8-row contraction over 8-bit weights on the shared activation panel.
// The 8-bit matrix kernel (wmma_q8_linear.cpp) quantizes the activations in
// every workgroup, a round of four groups at a time between two barriers,
// with the weight reads of a round waiting behind them. This one reads the
// panel quant_activation.q4_matrix_panel.v1 made once for all its siblings
// (in byte order, attribute 0 = 8) and streams the weights. The panel holds
// the codes, steps and sums the matrix kernel would compute -- the same
// expressions in the same order -- and each output takes the same per-group
// steps in the same group order, so the results are bit-identical.
bool valid_q8_matrix_panel(const KernelShapes &s) {
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.input_dtypes[4] != DType::kU32)
    return false;
  return dispatch::q8_matrix_panel_shape(original_shapes(s)) &&
         s.inputs[4] == dispatch::q4_matrix_panel_storage_shape(s.inputs[0]);
}
struct Q8MatrixPanelLinear final : KernelPrimitive<Q8MatrixPanelLinear> {
  static constexpr std::string_view kName = "quant_linear.q8_matrix_panel.v1";
  static constexpr std::string_view kEntry =
      "lse_quant_linear_q8_matrix_panel_v1";
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
    if (!valid_q8_matrix_panel(s) || !s.types.scalar || !s.store ||
        !s.intrinsics)
      return {};
    const auto original = original_shapes(s);
    if (dispatch::q8_matrix_panel_row(original))
      return emit_matrix_for<8>(s);
    const auto *kernel = legacy(original);
    return kernel ? kernel->emit_kernel(original) : std::string{};
  }
  std::uint32_t variants(const KernelShapes &s) const override {
    if (!valid_q8_matrix_panel(s) || !s.device ||
        !dispatch::q8_matrix_panel_row(original_shapes(s)))
      return 1;
    return matrix_variants(s, 8);
  }
  bool variant_reassociates(const KernelShapes &s, std::uint32_t variant) const override {
    return variants(s) > 1 && matrix_reassociates(s, 8, variant);
  }
  static ThreadPlan plan_impl(const KernelShapes &s) {
    ThreadPlan tp;
    if (!valid_q8_matrix_panel(s))
      return tp;
    const auto original = original_shapes(s);
    if (!dispatch::q8_matrix_panel_row(original)) {
      const auto *kernel = legacy(original);
      return kernel ? kernel->plan(original) : tp;
    }
    const auto n = static_cast<std::uint32_t>(s.inputs[1].dim(0));
    const auto waves = matrix_waves(s, n);
    const auto choice = matrix_choice(s, 8);
    if (!choice) return ThreadPlan{};
    const auto ks = choice->ks;
    tp.workgroup_size[0] = waves * 32u;
    tp.workgroup_count[0] = ((n + 15u) / 16u + waves / ks - 1u) / (waves / ks);
    if (ks > 1) tp.lds_bytes = matrix_lds_bytes(*choice, waves);
    return tp;
  }
};
} // namespace
LSE_REGISTER_PRIMITIVE(Q4MatrixPanelKernel);
LSE_REGISTER_PRIMITIVE(Q4MatrixPanelLinear);
LSE_REGISTER_PRIMITIVE(Q8MatrixPanelLinear);
} // namespace lse::kernels
