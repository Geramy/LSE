#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace lse::dispatch {

enum class QuantMatrix : std::uint8_t { kNone, kInt8, kInt8Lds, kBF16 };
inline constexpr std::uint32_t kQ4MatrixLdsBytes = 6656;

struct QuantMatrixShape {
  std::string_view arch;
  std::uint32_t wave, bits, group;
  std::uint64_t min_m, max_m, n, k;
  QuantMatrix implementation;
  std::uint32_t lds, threads;
};
inline constexpr std::array kQuantMatrixShapes{
    QuantMatrixShape{"gfx1201", 32, 4, 64, 16, 512, 17408, 5120,
                     QuantMatrix::kInt8Lds, kQ4MatrixLdsBytes, 256},
    QuantMatrixShape{"gfx1201", 32, 4, 64, 16, 512, 5120, 17408,
                     QuantMatrix::kInt8Lds, kQ4MatrixLdsBytes, 256},
    QuantMatrixShape{"gfx1201", 32, 6, 64, 512, 512, 17408, 5120,
                     QuantMatrix::kBF16, 16384, 128},
    QuantMatrixShape{"gfx1201", 32, 6, 64, 512, 512, 5120, 17408,
                     QuantMatrix::kBF16, 16384, 128},
};

struct QuantRowRange {
  std::string_view arch;
  std::uint32_t wave, bits, group;
  std::uint64_t min_m, max_m;
};
inline constexpr std::uint64_t kQuantMaxRows = UINT32_MAX;
// Empty arch and zero wave retain capability-qualified generic admission.
inline constexpr std::array kQuantInt8Rows{
    QuantRowRange{"", 0, 4, 64, 1, 1},
    QuantRowRange{"gfx1201", 32, 4, 64, 2, 8},
    QuantRowRange{"gfx1201", 32, 4, 64, 16, kQuantMaxRows},
    QuantRowRange{"", 0, 4, 64, 512, kQuantMaxRows},
};

struct QuantMatrixRange : QuantRowRange {
  std::uint32_t lds, threads;
};
inline constexpr std::array kQuantMatrixRanges{
    QuantMatrixRange{{"gfx1201", 32, 4, 64, 16, kQuantMaxRows},
                     kQ4MatrixLdsBytes, 256},
    QuantMatrixRange{{"", 0, 4, 64, 512, kQuantMaxRows},
                     kQ4MatrixLdsBytes, 256},
};

struct QuantScalarShape {
  std::string_view arch;
  std::uint32_t wave, bits, group, columns, prefill_rows;
};
inline constexpr std::array kQuantScalarShapes{
    QuantScalarShape{"gfx1201", 32, 6, 64, 4, 8},
};

struct QuantRowLadderShape {
  std::string_view arch;
  std::uint32_t wave, m, ceiling, lds;
  std::uint64_t n, k;
};
inline constexpr std::array kQuantRowLadderShapes{
    QuantRowLadderShape{"gfx1201", 32, 3, 4, 33088, 17408, 5120},
    QuantRowLadderShape{"gfx1201", 32, 3, 4, 56128, 5120, 17408},
    QuantRowLadderShape{"gfx1201", 32, 7, 8, 33152, 17408, 5120},
    QuantRowLadderShape{"gfx1201", 32, 7, 8, 56192, 5120, 17408},
};

struct QuantPanelDevice {
  std::string_view arch;
  std::uint32_t wave, bits, group, threads;
};
inline constexpr std::array kQuantPanelDevices{
    QuantPanelDevice{"gfx1201", 32, 4, 64, 256},
};

struct Q4PanelShape {
  std::int64_t m, n, k;
  std::uint32_t rows = 0;
};
inline constexpr std::array kQ4PanelShapes{
    Q4PanelShape{4, 17408, 5120},  Q4PanelShape{4, 5120, 17408},
    Q4PanelShape{4, 10240, 5120},  Q4PanelShape{4, 6144, 5120},
    Q4PanelShape{4, 12288, 5120},  Q4PanelShape{4, 5120, 6144},
    Q4PanelShape{4, 248320, 5120}, Q4PanelShape{7, 248320, 5120, 8},
    Q4PanelShape{6, 17408, 5120, 8}, Q4PanelShape{6, 5120, 17408, 8},
    Q4PanelShape{6, 10240, 5120, 8}, Q4PanelShape{6, 6144, 5120, 8},
    Q4PanelShape{6, 12288, 5120, 8}, Q4PanelShape{6, 5120, 6144, 8},
    Q4PanelShape{6, 248320, 5120, 8},
};


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
