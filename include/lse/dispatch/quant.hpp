#pragma once

#include <cstdint>

#include "lse/dispatch/quant_shapes.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::dispatch {

struct QuantPlan {
  const math::MatrixCoreRow* matrix = nullptr;
  QuantMatrix implementation = QuantMatrix::kNone;
  bool int8_activations = false;
  bool shared_activation_panel = false;
  bool rotate_decode_panel = false;
  std::uint32_t decode_columns = 1;
  std::uint32_t prefill_rows = 1;
  std::uint32_t row_ladder_ceiling = 0;
};

// Shape eligibility is shared by graph construction and device dispatch.
[[nodiscard]] inline const Q4PanelShape* q4_shared_panel_rule(
    const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || s.iattrs[0] != 4 || s.iattrs[1] != 64 ||
      !s.inputs[0].rank() || s.inputs[1].rank() != 2)
    return nullptr;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  const Q4PanelShape* measured = nullptr;
  for (const auto& shape : kQ4PanelShapes)
    if (n == shape.n && k == shape.k &&
        s.inputs[0].elem_count() == static_cast<std::uint64_t>(shape.m * k))
      measured = &shape;
  if (!measured || s.inputs[1] != Shape{n, k / 8} ||
      s.inputs[2] != Shape{n, k / 64} || s.inputs[3] != s.inputs[2])
    return nullptr;
  Shape output;
  for (std::size_t i = 0; i + 1 < s.inputs[0].rank(); ++i)
    output.push_back(s.inputs[0].dim(i));
  output.push_back(n);
  return s.output == output ? measured : nullptr;
}
[[nodiscard]] inline bool q4_shared_panel_shape(const graph::KernelShapes& s) {
  return q4_shared_panel_rule(s) != nullptr;
}
[[nodiscard]] inline std::uint32_t q4_shared_panel_rows(
    const graph::KernelShapes& s) {
  const auto* rule = q4_shared_panel_rule(s);
  return rule ? rule->rows : 0;
}

[[nodiscard]] QuantPlan quant_plan(const graph::KernelShapes&, bool indexed = false);
[[nodiscard]] const math::MatrixCoreRow* linear_matrix_row(const graph::KernelShapes&);

inline constexpr std::uint32_t kTableRevision = 1;
[[nodiscard]] constexpr std::uint64_t implementation_id(std::string_view name) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (char c : name) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace lse::dispatch
