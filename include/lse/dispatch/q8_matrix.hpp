#pragma once
#include "lse/graph/kernel_primitive.hpp"
#include "lse/dispatch/q8_tuneconfig.h"
#include "lse/math.hpp"
namespace lse::dispatch {
struct AffineMatrixPlan {
  const math::MatrixCoreRow* matrix = nullptr;
  std::uint32_t m = 0, n = 0, k = 0, lanes = 0, groups = 0, lds_bytes = 0, round_groups = 0;
};
[[nodiscard]] std::uint32_t q8_matrix_rows(const graph::KernelShapes&);
[[nodiscard]] AffineMatrixPlan q8_matrix_plan(const graph::KernelShapes&,
                                        std::uint32_t rows);
[[nodiscard]] inline bool q8_packed_weight_shape(std::uint32_t columns, std::uint32_t features) {
  for (const auto& rule : q8_shapes::kPackedShapes)
    if (columns == rule.n && features == rule.k) return true;
  return false;
}

[[nodiscard]] inline bool q8_packed_matrix_shape(const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || s.iattrs[0] != 8 || s.iattrs[1] != 64 ||
      !s.inputs[0].rank() || s.inputs[1].rank() != 2) return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  if (n <= 0 || k <= 0 || n > UINT32_MAX || k > UINT32_MAX ||
      !q8_packed_weight_shape(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k)) ||
      s.inputs[0].elem_count() != static_cast<std::uint64_t>(k) ||
      s.inputs[1] != Shape{n, k / 4} || s.inputs[2] != Shape{n, k / 64} ||
      s.inputs[3] != s.inputs[2]) return false;
  Shape expected;
  for (std::size_t i = 0; i + 1 < s.inputs[0].rank(); ++i) {
    if (s.inputs[0].dim(i) <= 0) return false;
    expected.push_back(s.inputs[0].dim(i));
  }
  expected.push_back(n);
  return s.output == expected;
}

[[nodiscard]] bool q8_packed_weight_device(const backend::DeviceInfo&);
// The architectures q8_packed_weight_device can admit, by name alone. For
// planning without a device; the device check also asks for the matrix-core
// capabilities and LDS a real part reports.
[[nodiscard]] inline bool q8_packed_weight_arch(std::string_view arch) noexcept {
  for (const auto& rule : q8_shapes::kMatrixRules)
    if (rule.bits == 8 && rule.rows == 16 && rule.arch == arch) return true;
  return false;
}
[[nodiscard]] AffineMatrixPlan q8_packed_matrix_plan(const graph::KernelShapes&);
}  // namespace lse::dispatch
