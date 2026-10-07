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

// -- Verify-pass contractions (2 to 8 rows) ------------------------------
// MTP and DFlash2 verify passes run 4-bit contractions of 2 to 8 rows. Without
// these rules gfx1151 ran them as one f32 row at a time (the weights read once
// per row). With them it takes gfx1201's int8 path: the activation rows
// quantized once to a shared panel, products by v_dot4_i32_iu8 (RDNA3.5 has
// the dot8 instructions), and the row ladder and fused SwiGLU forms at the
// shapes gfx1201 measured them on. 256 threads, scratch within 64 KiB.
inline constexpr auto kQuantInt8Rows = concat(
    std::array{QuantRowRange{kArch, 32, 4, 64, 2, 8}},
    generic::kQuantInt8Rows);
inline constexpr std::array kQuantPanelDevices{
    QuantPanelDevice{kArch, 32, 4, 64, 256},
};
inline constexpr std::array kQuantRowLadderShapes{
    QuantRowLadderShape{kArch, 32, 3, 4, 33088, 17408, 5120},
    QuantRowLadderShape{kArch, 32, 3, 4, 56128, 5120, 17408},
    QuantRowLadderShape{kArch, 32, 7, 8, 33152, 17408, 5120},
    QuantRowLadderShape{kArch, 32, 7, 8, 56192, 5120, 17408},
};
inline constexpr std::array kQ4SwiGluShapes{
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 1, 1, 17408, 5120},
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 1, 4, 17408, 5120},
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 2, 8, 17408, 5120},
};
inline constexpr auto kQuantMatrixRanges = generic::kQuantMatrixRanges;

// -- Group-affine contractions -------------------------------------------
// The 8-row decode contraction over 4-bit weights on the shared int8
// activation panel (quant_matrix_panel.cpp), at the shape gfx1201 measures it
// on: the down projection. 256 threads; the gfx11 form reads each group's
// whole weight row and both activation halves per product, two groups a round.
inline constexpr std::array kQ4MatrixPanelShapes{
    Q4MatrixPanelShape{kArch, 32, 4, 64, 256, 8, 5120, 17408},
};

// The prefill GEMM's wide-pass tile (M >= 768): 256 x 128 outputs on 4 x 4
// waves of 64 x 32. gfx1201's 64 x 64 wave tile holds 16 accumulators and
// 8 operand fragments, and a gfx11 fragment is twice as wide, so on gfx1151
// it compiles to 256 VGPRs with a spill and its fused-epilogue forms exceed
// loomc's spill-materialization limit (the engine cannot start). 64 x 32 wave
// tiles compile to 184 VGPRs with no scratch. Measured on the 8060S,
// test_q4_gemm --gpu, ms per launch, N17408 K5120 / N5120 K17408:
//   256x128 on 4x4 waves (this):  M1024 9.6 / 9.6,   M4096 37.8 / 39.0
//   256x128 on 8x2 waves:         M1024 9.7 / 9.6,   M4096 37.6 / 38.9
//   128x128 on 2x4 waves:         M1024 9.7 / 11.2,  M4096 38.6 / 44.8
//   48x32 wave tiles (generic):   M1024 10.5 / 10.9, M4096 41.0 / 43.5
//   256x128 on 4x2 waves (spills, plain GEMM only): M1024 8.6 / 8.8
inline constexpr Q4GemmTile kQ4GemmWide{256, 128, 4, 4};

// -- 8-bit matrix panels --------------------------------------------------
// The 16-row tile of the 8-bit iu8 contractions: the decode form on the
// shared panel (quant_matrix_panel.cpp) and wmma_q8_linear's 16-row form, at
// the scratch and threads they ask for on every part. The 64-row tile has no
// gfx1151 rule: 8-bit prefill takes the f16 GEMM.
inline constexpr std::array kQ8MatrixRules{
    q8_shapes::MatrixRule{kArch, 32, 8, 16, 1664, 256, 4},
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
    .quant_row_ladders = kQuantRowLadderShapes,
    .quant_panel_devices = kQuantPanelDevices,
    .q4_swiglu_shapes = kQ4SwiGluShapes,
    .q4_matrix_panel_shapes = kQ4MatrixPanelShapes,
    .q8_matrix_rules = kQ8MatrixRules,
    .flash_wmma = kFlashWmmaRules,
    .decode = kDecodeRules,
    .split_short = kSplitShortRules,
    .wave_l2 = kWaveL2Rules,
    .q4_gemm_wide = kQ4GemmWide,
};

static_assert(rows_name(kQuantInt8Rows, kArch) &&
              rows_name(kQuantPanelDevices, kArch) &&
              rows_name(kQuantRowLadderShapes, kArch) &&
              rows_name(kQ4SwiGluShapes, kArch) &&
              rows_name(kQuantMatrixRanges, kArch) &&
              rows_name(kQ4MatrixPanelShapes, kArch) &&
              rows_name(kQ8MatrixRules, kArch) &&
              rows_name(kFlashWmmaRules, kArch) &&
              rows_name(kDecodeRules, kArch) &&
              rows_name(kSplitShortRules, kArch) &&
              rows_name(kWaveL2Rules, kArch));

}  // namespace lse::dispatch::arch::gfx1151
