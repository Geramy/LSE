// Rows that hold on every architecture: capability-qualified admissions with
// an empty arch and a zero wave, which match any part that has the matrix
// row, the instructions and the scratch they ask for. A part's own header
// appends these after its measured rows.
#pragma once

#include "lse/dispatch/arch/tuning_types.hpp"

namespace lse::dispatch::arch::generic {

inline constexpr std::array kQuantInt8Rows{
    QuantRowRange{"", 0, 4, 64, 1, 1},
    QuantRowRange{"", 0, 4, 64, 512, kQuantMaxRows},
};
inline constexpr std::array kQuantMatrixRanges{
    QuantMatrixRange{{"", 0, 4, 64, 512, kQuantMaxRows},
                     kQ4MatrixLdsBytes, 256},
};

inline constexpr Tuning kTuning{
    .arch = {},
    .quant_int8_rows = kQuantInt8Rows,
    .quant_matrix_ranges = kQuantMatrixRanges,
};

}  // namespace lse::dispatch::arch::generic
