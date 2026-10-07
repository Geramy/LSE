#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace lse::dispatch {

enum class QuantMatrix : std::uint8_t { kNone, kInt8, kInt8Lds, kBF16 };

// Prefill GEMM over 4-bit group-affine weights (q4_gemm.cpp): the K step it
// stages, and the fewest rows it takes. Up to eight rows the decode-shaped
// panel forms are the better schedule; a ragged prefill pass of nine to
// fifteen rows has no panel form and takes the GEMM.
inline constexpr std::uint32_t kQ4GemmStepK = 64;
inline constexpr std::uint64_t kQ4GemmMinRows = 9;
inline constexpr std::uint32_t kQ4MatrixLdsBytes = 6656;
// A prefill GEMM workgroup tile: BM x BN outputs on WM x WN waves.
struct Q4GemmTile {
  std::uint32_t bm = 0, bn = 0, wm = 0, wn = 0;
};
// The wide-pass tile a part without its own (arch::Tuning::q4_gemm_wide)
// takes: 256 x 128 outputs on 4 x 2 waves of 64 x 64.
inline constexpr Q4GemmTile kQ4GemmWideTile{256, 128, 4, 2};

struct QuantMatrixShape {
  std::string_view arch;
  std::uint32_t wave, bits, group;
  std::uint64_t min_m, max_m, n, k;
  QuantMatrix implementation;
  std::uint32_t lds, threads;
};
// Every device-keyed rule below has its rows in the per-architecture headers
// (lse/dispatch/arch/<gfx>.hpp), selected by dispatch::arch::tuning; rows that
// hold on every architecture are in lse/dispatch/arch/generic.hpp.

struct QuantRowRange {
  std::string_view arch;
  std::uint32_t wave, bits, group;
  std::uint64_t min_m, max_m;
};
inline constexpr std::uint64_t kQuantMaxRows = UINT32_MAX;
// Empty arch and zero wave retain capability-qualified generic admission.

struct QuantMatrixRange : QuantRowRange {
  std::uint32_t lds, threads;
};

struct QuantScalarShape {
  std::string_view arch;
  std::uint32_t wave, bits, group, columns, prefill_rows;
};

struct QuantRowLadderShape {
  std::string_view arch;
  std::uint32_t wave, m, ceiling, lds;
  std::uint64_t n, k;
};

struct QuantPanelDevice {
  std::string_view arch;
  std::uint32_t wave, bits, group, threads;
};

struct Q4PanelShape {
  std::int64_t m, n, k;
  std::uint32_t rows = 0;
  std::uint32_t load_chunks = 1;
  // Adjacent columns one wave owns; rows * columns is at most a wave.
  std::uint32_t columns = 1;
};
inline constexpr std::array kQ4PanelShapes{
    Q4PanelShape{1, 17408, 5120, 0, 4},
    Q4PanelShape{1, 5120, 17408, 0, 2},
    Q4PanelShape{4, 17408, 5120, 0, 2},  Q4PanelShape{4, 5120, 17408, 0, 1, 2},
    Q4PanelShape{4, 10240, 5120, 0, 2},  Q4PanelShape{4, 6144, 5120, 0, 2},
    Q4PanelShape{4, 12288, 5120},  Q4PanelShape{4, 5120, 6144},
    Q4PanelShape{4, 248320, 5120}, Q4PanelShape{7, 248320, 5120, 8, 4, 4},
    Q4PanelShape{6, 17408, 5120, 8}, Q4PanelShape{6, 5120, 17408, 8},
    Q4PanelShape{6, 10240, 5120, 8}, Q4PanelShape{6, 6144, 5120, 8},
    Q4PanelShape{6, 12288, 5120, 8}, Q4PanelShape{6, 5120, 6144, 8},
    Q4PanelShape{6, 248320, 5120, 8, 1, 4},
    Q4PanelShape{8, 17408, 5120, 8, 4}, Q4PanelShape{8, 5120, 17408, 8, 2},
    Q4PanelShape{8, 10240, 5120, 8, 4}, Q4PanelShape{8, 6144, 5120, 8, 2},
    Q4PanelShape{8, 12288, 5120, 8, 4}, Q4PanelShape{8, 5120, 6144, 8, 2},
    Q4PanelShape{8, 248320, 5120, 8, 4, 4}, Q4PanelShape{8, 48, 5120, 8},
    Q4PanelShape{8, 1024, 5120, 8, 4},
    Q4PanelShape{4, 1024, 5120, 0, 2}, Q4PanelShape{4, 48, 5120},
};


struct Q4SwiGluShape {
  std::string_view arch;
  std::uint32_t wave, bits, group, threads, chunks_per_lane, k_splits;
  std::int64_t m, n, k;
};

struct Q4MatrixPanelShape {
  std::string_view arch;
  std::uint32_t wave, bits, group, threads;
  std::int64_t m, n, k;
  std::uint32_t rows = 16;
  std::uint32_t shared_words = 0;
};
struct Q4MatrixPanelLayout {
  std::int64_t m, k;
  std::uint32_t rows;
};
inline constexpr std::array kQ4MatrixPanelLayouts{
    Q4MatrixPanelLayout{8, 0, 16},
    Q4MatrixPanelLayout{1024, 17408, 64},
    Q4MatrixPanelLayout{1024, 5120, 64},
};
inline constexpr std::uint32_t kQ4MatrixPanelRows = 16;
inline constexpr std::uint32_t kQ4MatrixPanelGroupWords = 288;

template <class Rule>
[[nodiscard]] constexpr bool quant_shape_device(const Rule& rule,
    std::string_view arch, std::uint32_t wave) noexcept {
  return (rule.arch.empty() || rule.arch == arch) &&
         (rule.wave == 0 || rule.wave == wave);
}

[[nodiscard]] constexpr bool quant_row_range(const QuantRowRange& rule,
    std::string_view arch, std::uint32_t wave, std::uint32_t bits,
    std::uint32_t group, std::uint64_t m) noexcept {
  return quant_shape_device(rule, arch, wave) && rule.bits == bits &&
         rule.group == group && m >= rule.min_m && m <= rule.max_m;
}

}  // namespace lse::dispatch
