// What one architecture's tuning is made of.
//
// Every dispatch rule that names a device lives in that device's own header
// beside this one (gfx1201.hpp, gfx1151.hpp, ...), and arch::tuning(arch)
// hands a kernel or a dispatch decision exactly the rows of the part it runs
// on. A row for one part never sits in a table another part reads, and no
// kernel branches on an arch name: what differs between parts is data here,
// and the matrix instructions and fragment layouts each generation uses are
// rows of lse::math's matrix table, picked by kernels::matrix_target.
//
// Rows that hold on every part are in generic.hpp and are appended after a
// part's own rows, so a part's measured rule is always seen first.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string_view>

#include "lse/dispatch/attention_tuneconfig.h"
#include "lse/dispatch/q8_tuneconfig.h"
#include "lse/dispatch/quant_tuneconfig.h"

namespace lse::dispatch::arch {

struct Tuning {
  // The processor these rows were measured on; empty for the generic rows.
  std::string_view arch;

  // Group-affine contractions (quant.hpp, quant.cpp).
  std::span<const QuantMatrixShape> quant_matrix_shapes;
  std::span<const QuantRowRange> quant_int8_rows;
  std::span<const QuantMatrixRange> quant_matrix_ranges;
  std::span<const QuantScalarShape> quant_scalar_shapes;
  std::span<const QuantRowLadderShape> quant_row_ladders;
  std::span<const QuantPanelDevice> quant_panel_devices;
  std::span<const Q4SwiGluShape> q4_swiglu_shapes;
  std::span<const Q4MatrixPanelShape> q4_matrix_panel_shapes;
  // Contractions that read the shared int8 activation panel on this part,
  // beyond the kQ4PanelShapes every part with the panel takes.
  std::span<const Q4PanelShape> q4_panel_shapes;

  // 8-bit contractions: the WMMA tiles of wmma_q8_linear and the packed
  // weights (q8_matrix.cpp), and separately the 8-row decode form on the
  // shared activation panel (quant_matrix_panel.cpp), since a part may run
  // one well and the other not.
  std::span<const q8_shapes::MatrixRule> q8_matrix_rules;
  std::span<const q8_shapes::MatrixRule> q8_panel_rules;

  // Attention (attention.cpp).
  std::span<const attention_shapes::FlashWmmaRule> flash_wmma;
  std::span<const attention_shapes::FlashCacheRule> flash_cache;
  std::span<const attention_shapes::DecodeRule> decode;
  std::span<const attention_shapes::SplitShortRule> split_short;
  std::span<const attention_shapes::WaveL2Rule> wave_l2;

  // FlashPrefill's block-sparse prefill attention (pool, selector and
  // mean-corrected matrix kernels) is qualified on this part.
  bool flash_prefill = false;

  // The prefill GEMM's tile for wide passes (q4_gemm_tile, M >= 768); unset
  // takes the generic 256 x 128 tile of 64x64 wave tiles.
  Q4GemmTile q4_gemm_wide{};
};

// One part's rows followed by the generic ones, as one array.
template <class T, std::size_t A, std::size_t B>
[[nodiscard]] constexpr std::array<T, A + B> concat(
    const std::array<T, A>& a, const std::array<T, B>& b) noexcept {
  std::array<T, A + B> out{};
  std::copy(a.begin(), a.end(), out.begin());
  std::copy(b.begin(), b.end(), out.begin() + A);
  return out;
}

// Every row of a part's table names that part, so a row pasted into the wrong
// header fails to compile instead of tuning a device nobody measured.
template <class Rows>
[[nodiscard]] constexpr bool rows_name(const Rows& rows,
                                       std::string_view arch) noexcept {
  for (const auto& row : rows)
    if (row.arch != arch && !row.arch.empty()) return false;
  return true;
}

}  // namespace lse::dispatch::arch
