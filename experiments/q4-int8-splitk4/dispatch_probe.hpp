#pragma once
#include "lse/dispatch/q8_matrix.hpp"
namespace lse::dispatch::i8_splitk4_probe {
[[nodiscard]] AffineMatrixPlan q4_small_matrix_plan(const graph::KernelShapes&);
}
