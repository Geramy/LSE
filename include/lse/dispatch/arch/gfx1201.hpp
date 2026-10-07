// gfx1201 (Radeon AI PRO R9700, RDNA4): 64 CUs, wave32, 64 KiB of
// workgroup scratch, WMMA with K split across the half-waves.
//
// Every row here was measured on the part; the comments at each rule's
// definition (quant_tuneconfig.h, q8_tuneconfig.h, attention_tuneconfig.h,
// quant.hpp) give the numbers.
#pragma once

#include "lse/dispatch/arch/generic.hpp"
#include "lse/dispatch/arch/tuning_types.hpp"

namespace lse::dispatch::arch::gfx1201 {

inline constexpr std::string_view kArch = "gfx1201";

// -- Group-affine contractions -------------------------------------------
inline constexpr std::array kQuantMatrixShapes{
    QuantMatrixShape{kArch, 32, 4, 64, 16, 4096, 17408, 5120,
                     QuantMatrix::kInt8Lds, kQ4MatrixLdsBytes, 256},
    QuantMatrixShape{kArch, 32, 4, 64, 16, 4096, 5120, 17408,
                     QuantMatrix::kInt8Lds, kQ4MatrixLdsBytes, 256},
    QuantMatrixShape{kArch, 32, 6, 64, 512, 512, 17408, 5120,
                     QuantMatrix::kBF16, 16384, 128},
    QuantMatrixShape{kArch, 32, 6, 64, 512, 512, 5120, 17408,
                     QuantMatrix::kBF16, 16384, 128},
};
inline constexpr auto kQuantInt8Rows = concat(
    std::array{
        QuantRowRange{kArch, 32, 4, 64, 2, 8},
        QuantRowRange{kArch, 32, 4, 64, 16, kQuantMaxRows},
    },
    generic::kQuantInt8Rows);
inline constexpr auto kQuantMatrixRanges = concat(
    std::array{
        QuantMatrixRange{{kArch, 32, 4, 64, 16, kQuantMaxRows},
                         kQ4MatrixLdsBytes, 256},
    },
    generic::kQuantMatrixRanges);
inline constexpr std::array kQuantScalarShapes{
    QuantScalarShape{kArch, 32, 6, 64, 4, 8},
};
inline constexpr std::array kQuantRowLadderShapes{
    QuantRowLadderShape{kArch, 32, 3, 4, 33088, 17408, 5120},
    QuantRowLadderShape{kArch, 32, 3, 4, 56128, 5120, 17408},
    QuantRowLadderShape{kArch, 32, 7, 8, 33152, 17408, 5120},
    QuantRowLadderShape{kArch, 32, 7, 8, 56192, 5120, 17408},
};
inline constexpr std::array kQuantPanelDevices{
    QuantPanelDevice{kArch, 32, 4, 64, 256},
};
inline constexpr std::array kQ4SwiGluShapes{
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 1, 1, 17408, 5120},
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 1, 4, 17408, 5120},
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 2, 8, 17408, 5120},
};
inline constexpr std::array kQ4MatrixPanelShapes{
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 8, 5120, 17408},
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 1024, 5120, 17408, 64},
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 1024, 17408, 5120, 64, 2304},
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 1024, 10240, 5120, 64, 1152},
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 1024, 6144, 5120, 64, 1152},
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 1024, 12288, 5120, 64, 1152},
};

// -- 8-bit matrix panels --------------------------------------------------
inline constexpr std::array kQ8MatrixRules{
    q8_shapes::MatrixRule{kArch, 32, 8, 16, 1664, 256, 4},
    q8_shapes::MatrixRule{kArch, 32, 8, 64, 6656, 256, 1},
};

// -- Attention ------------------------------------------------------------
inline constexpr std::array kFlashWmmaRules{
    attention_shapes::FlashWmmaRule{kArch, 32, 256, 2, 512},
};
inline constexpr std::array kFlashCacheRules{
    attention_shapes::FlashCacheRule{kArch, 1024, 65536},
};
inline constexpr std::array kDecodeRules{
    attention_shapes::DecodeRule{kArch, 32, 128, 1, 256, 256},
};
inline constexpr std::array kSplitShortRules{
    attention_shapes::SplitShortRule{kArch, 32, 2, 8, 128, 256},
};
inline constexpr std::array kWaveL2Rules{
    attention_shapes::WaveL2Rule{kArch, 32, 128, 1, 1, 16, 128},
};

inline constexpr Tuning kTuning{
    .arch = kArch,
    .quant_matrix_shapes = kQuantMatrixShapes,
    .quant_int8_rows = kQuantInt8Rows,
    .quant_matrix_ranges = kQuantMatrixRanges,
    .quant_scalar_shapes = kQuantScalarShapes,
    .quant_row_ladders = kQuantRowLadderShapes,
    .quant_panel_devices = kQuantPanelDevices,
    .q4_swiglu_shapes = kQ4SwiGluShapes,
    .q4_matrix_panel_shapes = kQ4MatrixPanelShapes,
    .q8_matrix_rules = kQ8MatrixRules,
    .flash_wmma = kFlashWmmaRules,
    .flash_cache = kFlashCacheRules,
    .decode = kDecodeRules,
    .split_short = kSplitShortRules,
    .wave_l2 = kWaveL2Rules,
    .flash_prefill = true,
    // 256 x 128 on 4 x 2 waves of 64 x 64 (q4_gemm_tile's comment has the
    // numbers): the generic wide tile, named here because it is this part's.
    .q4_gemm_wide = kQ4GemmWideTile,
};

static_assert(rows_name(kQuantMatrixShapes, kArch) &&
              rows_name(kQuantInt8Rows, kArch) &&
              rows_name(kQuantMatrixRanges, kArch) &&
              rows_name(kQuantScalarShapes, kArch) &&
              rows_name(kQuantRowLadderShapes, kArch) &&
              rows_name(kQuantPanelDevices, kArch) &&
              rows_name(kQ4SwiGluShapes, kArch) &&
              rows_name(kQ4MatrixPanelShapes, kArch) &&
              rows_name(kQ8MatrixRules, kArch) &&
              rows_name(kFlashWmmaRules, kArch) &&
              rows_name(kFlashCacheRules, kArch) &&
              rows_name(kDecodeRules, kArch) &&
              rows_name(kSplitShortRules, kArch) &&
              rows_name(kWaveL2Rules, kArch));

}  // namespace lse::dispatch::arch::gfx1201
