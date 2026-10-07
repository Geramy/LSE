// gfx1151 (Radeon 8060S, Strix Halo, RDNA 3.5): 40 CUs, wave32, 64 KiB of
// workgroup scratch per workgroup out of a 128 KiB block per WGP, 2 MiB of
// L2, unified LPDDR5X memory. WMMA is the gfx11 form: each lane carries the
// whole K step and the half-waves repeat the same rows (math.hpp's kRdna3
// rows), so fragments are twice RDNA4's width and tiles are sized for that.
//
// A rule appears here only once its kernel has been run on the part; until
// then the part takes the generic rows and the kernels that need no rule.
#pragma once

#include "lse/dispatch/arch/generic.hpp"
#include "lse/dispatch/arch/tuning_types.hpp"

namespace lse::dispatch::arch::gfx1151 {

inline constexpr std::string_view kArch = "gfx1151";

inline constexpr auto kQuantInt8Rows = generic::kQuantInt8Rows;
inline constexpr auto kQuantMatrixRanges = generic::kQuantMatrixRanges;

// -- Group-affine contractions -------------------------------------------
// The 8-row decode contraction over 4-bit weights on the shared int8
// activation panel (quant_matrix_panel.cpp), at the shape gfx1201 measures it
// on: the down projection. 256 threads; the gfx11 form reads each group's
// whole weight row and both activation halves per product, two groups a round.
inline constexpr std::array kQ4MatrixPanelShapes{
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 8, 5120, 17408},
};

// -- 8-bit matrix panels --------------------------------------------------
// The 8-bit iu8 contractions (wmma_q8_linear.cpp, 16- and 64-row tiles, and
// the decode form on the shared panel) at the scratch and threads they ask
// for on every part; their K16 operands follow the gfx11 row's layout.
inline constexpr std::array kQ8MatrixRules{
    q8_shapes::MatrixRule{kArch, 32, 8, 16, 1664, 256, 4},
    q8_shapes::MatrixRule{kArch, 32, 8, 64, 6656, 256, 1},
};

// -- Attention ------------------------------------------------------------
// The flash WMMA prefill tile at the same geometry as gfx1201's: 256 threads
// (eight waves, one 16-key column tile each per 128-key half-window), query
// tiles of 16 rows, head and value widths up to 512. The workgroup scratch it
// asks for (attention_shapes::flash_wmma_lds_bytes plus value staging) is the
// same on both parts and fits the 64 KiB a gfx1151 workgroup addresses. Its
// matrix instructions are the gfx11 ones, with fragments laid out per
// lse/math/matrix_rdna3.hpp.
inline constexpr std::array kFlashWmmaRules{
    attention_shapes::FlashWmmaRule{kArch, 32, 256, 2, 512},
};

// Decode-time attention (sdpa.cpp, attention.cpp). Without these rules a
// decode step's attention runs as one workgroup per KV-head pair over the
// whole cache -- six workgroups on a 40-CU part, 52 ms a layer at a 1K cache
// on gfx1151. The split forms cut the cache into 128-key partitions, one
// 128-thread workgroup each, and merge them: the same kernels gfx1201 runs,
// at the same wave, threads, head width and block bound, with scratch well
// inside 64 KiB. The wave-per-row L2 normalization rides along at the same
// geometry.
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
    .quant_int8_rows = kQuantInt8Rows,
    .quant_matrix_ranges = kQuantMatrixRanges,
    .q4_matrix_panel_shapes = kQ4MatrixPanelShapes,
    .q8_matrix_rules = kQ8MatrixRules,
    .flash_wmma = kFlashWmmaRules,
    .decode = kDecodeRules,
    .split_short = kSplitShortRules,
    .wave_l2 = kWaveL2Rules,
};

static_assert(rows_name(kQuantInt8Rows, kArch) &&
              rows_name(kQuantMatrixRanges, kArch) &&
              rows_name(kQ4MatrixPanelShapes, kArch) &&
              rows_name(kQ8MatrixRules, kArch) &&
              rows_name(kFlashWmmaRules, kArch) &&
              rows_name(kDecodeRules, kArch) &&
              rows_name(kSplitShortRules, kArch) &&
              rows_name(kWaveL2Rules, kArch));

}  // namespace lse::dispatch::arch::gfx1151
