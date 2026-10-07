// RDNA4 (gfx12: gfx1200, gfx1201) matrix rows.
//
// WMMA, wave32. Half RDNA3's operand width per lane: K splits across the
// half-waves, A[i][k] in lane i + 16*(k/(K/2)), value k%(K/2), and nothing
// repeats (OperandLayout::kLaneRowSplitK). The accumulator gives each half-wave
// a block of rows: D[i][j] is lane j + 16*(i/8), register i%8
// (AccLayout::kRowBlockHalfWave).
//
// References: AMD "RDNA4 Instruction Set Architecture" reference guide, WMMA
// section; AMD's matrix instruction calculator (--architecture rdna4, e.g.
// v_wmma_i32_16x16x16_iu8: D[m][n] = v{m%8}{n + 16*(m/8)}); loomc's
// rdna4_wmma_* layouts at the pinned HRX revision. Verified on gfx1201 by the
// device-vs-oracle suite.
//
// Included by lse/math.hpp after MatrixCoreRow and its layout enums, never on
// its own: these are rows of math::matrix_core_table(), the one table every
// kernel asks for its matrix instruction, operand widths and lane layouts.
// One generation per header, so a part's instruction set and the layouts
// measured for it sit together and apart from every other generation's.
#pragma once

namespace lse::math::detail {

inline constexpr std::array<MatrixCoreRow, 11> kRdna4Rows{{
    // -- RDNA4, wave32. Half the RDNA3 operand width: K splits across the
    // half-waves, which is a different fill, not just a narrower one. gfx1201
    // is offline here, so nothing below has a measured layout.
    // Layouts hypothesized from the su8 row's ISA-calculator derivation and
    // VERIFIED ON DEVICE (gfx1201): the split-K operand fill and block
    // accumulator mapping are what the generalized tile emits, and the
    // device-vs-oracle suite is the measurement.
    {"wmma12.f32.16x16x16.f16", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 8, Scalar::kF16, 8, Scalar::kF32, 8, 1, 16,
     16, 16, 32, MatrixCap::kWmma12F16, 1, 16, 200,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    {"wmma12.f32.16x16x16.bf16", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 8, Scalar::kBF16, 8, Scalar::kF32, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmma12Bf16, 1, 16, 200,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    {"wmma12.f16.16x16x16.f16", MatrixTarget::kRdna4, MatrixElem::kF16,
     MatrixElem::kF16, Scalar::kF16, 8, Scalar::kF16, 8, Scalar::kF16, 8, 1, 16,
     16, 16, 32, MatrixCap::kWmma12F16, 1, 16, 200, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"wmma12.bf16.16x16x16.bf16", MatrixTarget::kRdna4, MatrixElem::kBF16,
     MatrixElem::kBF16, Scalar::kBF16, 8, Scalar::kBF16, 8, Scalar::kBF16, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmma12Bf16, 1, 16, 200,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.i32.16x16x16.iu8", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Int8, 1, 16, 400,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    // The double-K form of the same instruction: 8 int8 per lane is a 64-bit
    // read on a 128-bit path, so two chained steps fill the load.
    {"wmma12.i32.16x16x16.iu8", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Int8, 2, 32, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    // The mixed form, as on RDNA3: a group-affine code is unsigned and the
    // activation signed. gfx12 spells the same two signedness immediates, so
    // this is one row, not a different instruction. Its lane mapping is not
    // measured, so it does not emit -- see the note above.
    {"wmma12.i32.16x16x16.su8", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kSU8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Int8, 1, 16, 400,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    {"wmma12.i32.16x16x32.iu4", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI4, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 8, 16,
     16, 32, 32, MatrixCap::kWmma12Int4, 1, 32, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.f32.16x16x16.fp8_fp8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 1, 16, 400,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    {"wmma12.f32.16x16x16.bf8_bf8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kBf8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 1, 16, 400,
     OperandLayout::kLaneRowSplitK, AccLayout::kRowBlockHalfWave},
    {"wmma12.f32.16x16x16.fp8_fp8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 2, 32, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
}};

}  // namespace lse::math::detail
