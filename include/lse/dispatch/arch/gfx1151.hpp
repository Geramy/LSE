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

inline constexpr Tuning kTuning{
    .arch = kArch,
    .quant_int8_rows = kQuantInt8Rows,
    .quant_matrix_ranges = kQuantMatrixRanges,
    .flash_wmma = kFlashWmmaRules,
};

static_assert(rows_name(kQuantInt8Rows, kArch) &&
              rows_name(kQuantMatrixRanges, kArch) &&
              rows_name(kFlashWmmaRules, kArch));

}  // namespace lse::dispatch::arch::gfx1151
