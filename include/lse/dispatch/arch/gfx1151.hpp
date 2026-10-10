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
// Verify passes of two, three, five, six and seven rows run on the four- and
// eight-row kernels with the extra rows masked (dispatch::verify_rows).
// Without them each such width took the one-row-at-a-time contraction.
// Verify ms per pass on the 8060S at fixed widths 1..8 (DFlash2, 256-token
// prompt): 80 103 208 98 323 125 239 140 -> 80 89 93 98 119 120 133 140;
// MTP=3 widths 1..4: 76 98 203 90 -> 75 84 87 91.
inline constexpr std::array kQuantVerifyRows{
    QuantVerifyRows{kArch, 2, 4}, QuantVerifyRows{kArch, 3, 4},
    QuantVerifyRows{kArch, 5, 8}, QuantVerifyRows{kArch, 6, 8},
    QuantVerifyRows{kArch, 7, 8},
};
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
    // Eight rows: the panel staged in scratch per 128-chunk block.
    Q4SwiGluShape{kArch, 32, 4, 64, 256, 4, 2, 8, 17408, 5120, true},
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

// Single-row decode contractions over the shared int8 activation panel. The
// panel is quantized once per activation (quant_activation_panel.cpp) and
// every contraction over it reads it, where without a row each workgroup
// requantizes the whole activation in its own scratch. kQ4PanelShapes gives
// every part the MLP's two; these are the attention projections that share
// the layer's normed input. Measured on the 8060S in the pinned model (GPU
// kernel time per call, rocprofv3, 1024-token prompt, 64 decoded tokens):
//   GDN in_proj_qkv 10240 x 5120   136.0 -> 127.5 us
//   GDN in_proj_z    6144 x 5120    86.2 ->  78.6 us
//   attn q+gate     12288 x 5120   164.3 -> 152.6 us
//   attn k, v        1024 x 5120    18.9 ->  14.8 us
//   GDN in_proj_a, b   48 x 5120   9.2, 7.9 -> 6.0, 5.0 us
// and a decode step 75.4 -> 73.6 ms (13.17 -> 13.48 tok/s) with the panel
// launches they add. Two-chunk activation loads; one chunk for the 48-row
// projections, whose single wave column per row has no second to pair. Not
// here: the 5120 x 6144 output projection, which reads the panel slower
// (85.5 -> 108.4 us), and two columns per wave, slower on every shape (down
// projection 222 -> 251 us, vocabulary projection 3066 -> 3225 us).
inline constexpr std::array kQ4DecodePanelShapes{
    Q4PanelShape{1, 10240, 5120, 0, 2, 1, kArch},
    Q4PanelShape{1, 6144, 5120, 0, 2, 1, kArch},
    Q4PanelShape{1, 12288, 5120, 0, 2, 1, kArch},
    Q4PanelShape{1, 1024, 5120, 0, 2, 1, kArch},
    Q4PanelShape{1, 48, 5120, 0, 1, 1, kArch},
    // MTP=3's four-row verify passes, every row in one workgroup row: the
    // shared rows leave the row count to the scratch-priced ladder, which on
    // this part picked two and read every weight twice, and kept the MLP
    // from fusing into the SwiGLU pair. rocprofv3, us per call:
    //   gate + up        456 + 441 (two passes each) -> 448 fused
    //   vocabulary      6153 -> 3411    attn q+gate  288 -> 177
    //   GDN qkv          186 -> 140     GDN z        118 ->  91
    //   output proj      164 -> 118
    // MTP=3 decode 17.56 -> 23.34 tok/s (256-token prompt, 128 out).
    Q4PanelShape{4, 17408, 5120, 4, 2, 1, kArch},
    Q4PanelShape{4, 10240, 5120, 4, 2, 1, kArch},
    Q4PanelShape{4, 6144, 5120, 4, 2, 1, kArch},
    Q4PanelShape{4, 12288, 5120, 4, 1, 1, kArch},
    Q4PanelShape{4, 5120, 6144, 4, 1, 1, kArch},
    Q4PanelShape{4, 248320, 5120, 4, 1, 1, kArch},
    Q4PanelShape{4, 1024, 5120, 4, 2, 1, kArch},
    Q4PanelShape{4, 48, 5120, 4, 1, 1, kArch},
    // Eight-row verify passes (DFlash2) with two adjacent columns per wave,
    // which read each row's panel codes once for both. The eight-row form
    // reads 24 panel words per weight word, and at one column a wave that
    // is what binds it. rocprofv3, us per call at width 8:
    //   GDN qkv 224 -> 177, GDN z 158 -> 115, output proj 154 -> 140
    // (four columns: 192, 131, 176). The MLP's eight rows stay in the
    // SwiGLU pair: as two two-column contractions they cost the same.
    Q4PanelShape{8, 10240, 5120, 8, 4, 2, kArch},
    Q4PanelShape{8, 6144, 5120, 8, 2, 2, kArch},
    Q4PanelShape{8, 5120, 6144, 8, 2, 2, kArch},
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

// The prefill GEMM's tile by pass rows (q4_gemm_kernel_tile), measured as on
// gfx1201 (gfx1201.hpp) and on this part's own: every wave tile here is at
// most the wide tile's 64 x 32. Weighted GEMM time per pass on the 8060S, ms,
// shape-only -> band; every candidate wrote the same bytes:
//   256 386.7 -> 339.0                         (256 rows on 4 x 4 waves)
//   304 520.4 -> 420.6, 320 526.9 -> 428.3     (320 rows on 5 x 4 waves)
//   368 542.7 -> 496.6, 384 543.6 -> 501.7     (192 rows on 4 x 4 waves)
//   448 687.7 -> 590.7                         (224 rows on 7 x 4 waves)
//   496 728.5 -> 626.7, 512 729.2 -> 636.2     (256 rows on 4 x 4 waves)
//   608 948.4 -> 835.5, 640 955.4 -> 826.2     (320 rows on 5 x 4 waves)
//   736 1077.2 -> 937.4, 752 1090.6 -> 939.2   (256 rows on 4 x 4 waves)
//   784 1208.8 -> 1041.6, 800 1214.1 -> 1053.4 (160 rows on 5 x 4 waves)
//   832 1219.6 -> 1081.0, 864 1218.6 -> 1101.5 (144 rows on 3 x 4 waves)
inline constexpr std::array kQ4GemmRowTiles{
    Q4GemmRowTile{241, 256, {256, 128, 4, 4}},
    Q4GemmRowTile{289, 320, {320, 128, 5, 4}},
    Q4GemmRowTile{337, 384, {192, 128, 4, 4}},
    Q4GemmRowTile{433, 448, {224, 128, 7, 4}},
    Q4GemmRowTile{449, 464, {160, 128, 5, 4}},
    Q4GemmRowTile{465, 512, {256, 128, 4, 4}},
    Q4GemmRowTile{593, 640, {320, 128, 5, 4}},
    Q4GemmRowTile{641, 672, {224, 128, 7, 4}},
    Q4GemmRowTile{721, 767, {256, 128, 4, 4}},
    Q4GemmRowTile{769, 800, {160, 128, 5, 4}},
    Q4GemmRowTile{817, 864, {144, 128, 3, 4}},
    Q4GemmRowTile{865, 896, {224, 128, 7, 4}},
    Q4GemmRowTile{1073, 1104, {224, 128, 7, 4}},
};

// -- 8-bit contractions ---------------------------------------------------
// The 8-row decode form on the shared panel (quant_matrix_panel.cpp, the
// DFlash2 draft's 8-row passes) at the scratch and threads it takes on every
// part. wmma_q8_linear's tiles get no rule here: on the 8060S the generic
// contraction is faster for the rows they would take. Measured per call,
// 17408 x 5120 8-bit, MTP=3 smoke:
//   generic:             M1 0.76 ms, M2 1.55, M3 2.27, M4 2.95
//   16-row WMMA tile:    11.2 ms at every M (2.4-2.6 ms with whole-group
//                        16-byte weight loads)
//   8-row shared panel:  1.38 ms at M8, against 5.78 ms generic
// so MTP=3 decodes at 17.0 tok/s without the tile against 8.9 with it, and
// DFlash2 at 16.0 with the panel against 14.1 without.
inline constexpr std::array kQ8PanelRules{
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
    .quant_verify_rows = kQuantVerifyRows,
    .q4_swiglu_shapes = kQ4SwiGluShapes,
    .q4_matrix_panel_shapes = kQ4MatrixPanelShapes,
    .q4_panel_shapes = kQ4DecodePanelShapes,
    .q8_panel_rules = kQ8PanelRules,
    .flash_wmma = kFlashWmmaRules,
    .decode = kDecodeRules,
    .split_short = kSplitShortRules,
    .wave_l2 = kWaveL2Rules,
    // FlashPrefill V2 (block-sparse prefill attention, alpha 0.1): its pool
    // and selector kernels and the mean-corrected flash kernel on the gfx11
    // fragments pass test_sparse_attention --gpu on the 8060S. Pinned model,
    // prefill tok/s, dense -> FlashPrefill V2: 4096 tokens 488 -> 501,
    // 8192 442 -> 485, 16384 378 -> 467, 32768 533 -> 696; 1024 and 2048
    // unchanged (attention is a small share of a short prefill here).
    .flash_prefill = true,
    .q4_gemm_wide = kQ4GemmWide,
    .q4_gemm_rows = kQ4GemmRowTiles,
};

static_assert(rows_name(kQuantInt8Rows, kArch) &&
              rows_name(kQuantVerifyRows, kArch) &&
              rows_name(kQuantPanelDevices, kArch) &&
              rows_name(kQuantRowLadderShapes, kArch) &&
              rows_name(kQ4SwiGluShapes, kArch) &&
              rows_name(kQuantMatrixRanges, kArch) &&
              rows_name(kQ4MatrixPanelShapes, kArch) &&
              rows_name(kQ4DecodePanelShapes, kArch) &&
              rows_name(kQ8PanelRules, kArch) &&
              rows_name(kFlashWmmaRules, kArch) &&
              rows_name(kDecodeRules, kArch) &&
              rows_name(kSplitShortRules, kArch) &&
              rows_name(kWaveL2Rules, kArch));

}  // namespace lse::dispatch::arch::gfx1151
