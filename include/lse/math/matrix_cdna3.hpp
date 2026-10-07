// CDNA3 (gfx942) matrix rows.
//
// MFMA, wave64: a different instruction family, not a wider WMMA. 64 lanes
// cooperate and a lane holds a slice of K, not all of it. No layout has been
// measured here, so none of these rows emits.
//
// Included by lse/math.hpp after MatrixCoreRow and its layout enums, never on
// its own: these are rows of math::matrix_core_table(), the one table every
// kernel asks for its matrix instruction, operand widths and lane layouts.
// One generation per header, so a part's instruction set and the layouts
// measured for it sit together and apart from every other generation's.
#pragma once

namespace lse::math::detail {

inline constexpr std::array<MatrixCoreRow, 7> kCdna3Rows{{
    // -- CDNA3, wave64, MFMA. A different instruction family, not a wider
    // WMMA: 64 lanes cooperate and a lane holds a slice of K, not all of it.
    // No MI300X here.
    {"mfma.f32.16x16x16.f16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 4, Scalar::kF16, 4, Scalar::kF32, 4, 1, 16,
     16, 16, 64, MatrixCap::kMfmaF16, 1, 16, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.32x32x8.f16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 4, Scalar::kF16, 4, Scalar::kF32, 16, 1,
     32, 32, 8, 64, MatrixCap::kMfmaF16, 1, 8, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.16x16x16.bf16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 4, Scalar::kBF16, 4, Scalar::kF32, 4, 1,
     16, 16, 16, 64, MatrixCap::kMfmaBf16, 1, 16, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"mfma.f32.32x32x8.bf16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 4, Scalar::kBF16, 4, Scalar::kF32, 16, 1,
     32, 32, 8, 64, MatrixCap::kMfmaBf16, 1, 8, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    // MFMA takes its packed operands as one i64 per lane, not a vector.
    {"mfma.i32.16x16x32.i8", MatrixTarget::kCdna3, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kI32, 4, 8, 16,
     16, 32, 64, MatrixCap::kMfmaInt8, 1, 32, 800, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.16x16x32.fp8_fp8", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kF32, 4, 8, 16,
     16, 32, 64, MatrixCap::kMfmaFp8, 1, 32, 800, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.32x32x16.fp8_fp8", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kF32, 16, 8,
     32, 32, 16, 64, MatrixCap::kMfmaFp8, 1, 16, 800,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
}};

}  // namespace lse::math::detail
