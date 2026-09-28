#pragma once
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"
namespace lse::dispatch {
struct AffineMatrixPlan {
  const math::MatrixCoreRow* matrix = nullptr;
  std::uint32_t m = 0, n = 0, k = 0, lanes = 0, groups = 0, lds_bytes = 0, round_groups = 0;
};
[[nodiscard]] std::uint32_t q8_matrix_rows(const graph::KernelShapes&);
[[nodiscard]] AffineMatrixPlan q8_matrix_plan(const graph::KernelShapes&,
                                        std::uint32_t rows);
[[nodiscard]] AffineMatrixPlan q4_small_matrix_plan(const graph::KernelShapes&);
}  // namespace lse::dispatch
