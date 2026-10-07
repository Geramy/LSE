// RDNA3 / RDNA3.5 (gfx11: gfx1100-gfx1103, gfx1150-gfx1153) matrix rows.
//
// WMMA, wave32. Each lane holds the whole K step of its operand row, and the
// two half-waves must hold the same rows: A[i][k] sits in lane i and lane i+16,
// value k, and B[k][j] likewise in lanes j and j+16 (OperandLayout::
// kLaneRowContiguousK). The f32/i32 accumulator interleaves the half-waves:
// D[i][j] is lane j + 16*(i%2), register i/2 (AccLayout::kPairRowHalfWave).
//
// References: AMD "RDNA3 Instruction Set Architecture" reference guide, section
// "Wave Matrix Multiply Accumulate (WMMA)"; AMD's matrix instruction calculator
// (amd_matrix_instruction_calculator, --architecture rdna3, e.g.
// v_wmma_f32_16x16x16_f16 --detail-instruction); GPUOpen, "How to accelerate AI
// applications on RDNA 3 using WMMA" (the A/B replication and the
// c[16*(2*ele + lane/16) + lane%16] store); rocWMMA's gfx11 path, which
// duplicates the inputs across the half-waves; and loomc's rdna3_wmmar3_*
// fragment layouts at the pinned HRX revision
// (loom/py/loom/target/arch/amdgpu/matrix_fragment_layouts.py and its
// test_rdna3_integer_wmma_layout_matches_instruction_coordinates). The layouts
// were also measured on gfx1151 (256/256 slots, max err 5e-7) and are checked
// against a CPU model of the documented instruction in test_loom_matrix.
//
// Included by lse/math.hpp after MatrixCoreRow and its layout enums, never on
// its own: these are rows of math::matrix_core_table(), the one table every
// kernel asks for its matrix instruction, operand widths and lane layouts.
// One generation per header, so a part's instruction set and the layouts
// measured for it sit together and apart from every other generation's.
#pragma once

namespace lse::math::detail {

inline constexpr std::array<MatrixCoreRow, 7> kRdna3Rows{{
    // -- RDNA3 / 3.5, wave32. A/B 16 elements per lane, whole K in each lane.
    {"wmma.f32.16x16x16.f16", MatrixTarget::kRdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 16, Scalar::kF16, 16, Scalar::kF32, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmmaF16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    {"wmma.f32.16x16x16.bf16", MatrixTarget::kRdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 16, Scalar::kBF16, 16, Scalar::kF32, 8,
     1, 16, 16, 16, 32, MatrixCap::kWmmaBf16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    // The narrow-accumulate forms take an opsel that selects which half of a
    // 16-wide accumulator the 8 results land in, so their D mapping is not the
    // f32 one and has never been measured here.
    {"wmma.f16.16x16x16.f16", MatrixTarget::kRdna3, MatrixElem::kF16,
     MatrixElem::kF16, Scalar::kF16, 16, Scalar::kF16, 16, Scalar::kF16, 16, 1,
     16, 16, 16, 32, MatrixCap::kWmmaF16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kUnmeasured},
    {"wmma.bf16.16x16x16.bf16", MatrixTarget::kRdna3, MatrixElem::kBF16,
     MatrixElem::kBF16, Scalar::kBF16, 16, Scalar::kBF16, 16, Scalar::kBF16, 16,
     1, 16, 16, 16, 32, MatrixCap::kWmmaBf16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kUnmeasured},
    // Four int8 to an i32 lane, eight int4. Same D layout as the f32 form —
    // measured, see tests/test_jit.cpp matrix_core_int8_*.
    {"wmma.i32.16x16x16.iu8", MatrixTarget::kRdna3, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmmaInt8, 1, 16, 200,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    {"wmma.i32.16x16x16.su8", MatrixTarget::kRdna3, MatrixElem::kI32,
     MatrixElem::kSU8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmmaInt8, 1, 16, 200,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    {"wmma.i32.16x16x16.iu4", MatrixTarget::kRdna3, MatrixElem::kI32,
     MatrixElem::kI4, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 8, 16,
     16, 16, 32, MatrixCap::kWmmaInt4, 1, 16, 400,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
}};

}  // namespace lse::math::detail
