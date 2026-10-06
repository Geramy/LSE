#pragma once

#include <cstdint>

#include "lse/dispatch/quant_tuneconfig.h"
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
//
// The tiled prefill GEMM: 4-bit codes, groups a whole number of K steps, f32
// activations and output, float affine planes, and at least kQ4GemmMinRows
// rows. Whether the device can run it is asked again at dispatch.
[[nodiscard]] inline bool q4_gemm_shape(const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      (s.input_dtypes[2] != DType::kBF16 && s.input_dtypes[2] != DType::kF16) ||
      s.input_dtypes[3] != s.input_dtypes[2] || s.output_dtype != DType::kF32 ||
      s.iattrs[0] != 4 || s.iattrs[1] <= 0 ||
      s.iattrs[1] % static_cast<std::int32_t>(kQ4GemmStepK) != 0 ||
      !s.inputs[0].rank() || s.inputs[1].rank() != 2)
    return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  if (k <= 0 || n <= 0 || k % s.iattrs[1] != 0 ||
      s.inputs[1] != Shape{n, k / 8} ||
      s.inputs[2] != Shape{n, k / s.iattrs[1]} || s.inputs[3] != s.inputs[2])
    return false;
  Shape output;
  for (std::size_t i = 0; i + 1 < s.inputs[0].rank(); ++i)
    output.push_back(s.inputs[0].dim(i));
  output.push_back(n);
  if (s.output != output) return false;
  const auto m = s.inputs[0].elem_count() / static_cast<std::uint64_t>(k);
  return m >= kQ4GemmMinRows && m * static_cast<std::uint64_t>(k) <= UINT32_MAX &&
         m * static_cast<std::uint64_t>(n) <= UINT32_MAX &&
         static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k / 8) <= UINT32_MAX;
}
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

[[nodiscard]] inline const Q4MatrixPanelShape* q4_matrix_panel_rule(
    const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || !s.inputs[0].rank())
    return nullptr;
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < s.inputs[0].rank(); ++axis) {
    const auto extent = s.inputs[0].dim(axis);
    if (extent <= 0 || static_cast<std::uint64_t>(extent) > UINT32_MAX / count)
      return nullptr;
    count *= static_cast<std::uint64_t>(extent);
  }
  for (const auto& rule : kQ4MatrixPanelShapes) {
    if (s.iattrs[0] != static_cast<std::int32_t>(rule.bits) ||
        s.iattrs[1] != static_cast<std::int32_t>(rule.group) ||
        s.inputs[0].dim(s.inputs[0].rank() - 1) != rule.k ||
        s.inputs[0].elem_count() != static_cast<std::uint64_t>(rule.m * rule.k) ||
        s.inputs[1] != Shape{rule.n, rule.k / 8} ||
        s.inputs[2] != Shape{rule.n, rule.k / rule.group} ||
        s.inputs[3] != s.inputs[2])
      continue;
    Shape expected;
    for (std::size_t axis = 0; axis + 1 < s.inputs[0].rank(); ++axis)
      expected.push_back(s.inputs[0].dim(axis));
    expected.push_back(rule.n);
    return s.output == expected ? &rule : nullptr;
  }
  return nullptr;
}
[[nodiscard]] inline bool q4_matrix_panel_shape(const graph::KernelShapes& s) {
  return q4_matrix_panel_rule(s) != nullptr;
}

[[nodiscard]] inline const Q4MatrixPanelLayout* q4_matrix_panel_layout(
    const Shape& input) {
  if (!input.rank()) return nullptr;
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < input.rank(); ++axis) {
    const auto extent = input.dim(axis);
    if (extent <= 0 || static_cast<std::uint64_t>(extent) > UINT32_MAX / count)
      return nullptr;
    count *= static_cast<std::uint64_t>(extent);
  }
  const auto k = input.dim(input.rank() - 1);
  if (k % 64 != 0) return nullptr;
  const auto m = count / static_cast<std::uint64_t>(k);
  for (const auto& rule : kQ4MatrixPanelLayouts) {
    if (m != static_cast<std::uint64_t>(rule.m) || (rule.k && k != rule.k))
      continue;
    const auto blocks = (m + rule.rows - 1) / rule.rows;
    const auto words = blocks * static_cast<std::uint64_t>(k / 64) *
                       (rule.rows / kQ4MatrixPanelRows) * kQ4MatrixPanelGroupWords;
    if (words > UINT32_MAX) return nullptr;
    return &rule;
  }
  return nullptr;
}

[[nodiscard]] inline Shape q4_matrix_panel_storage_shape(const Shape& input) {
  const auto* rule = q4_matrix_panel_layout(input);
  if (!rule) return {};
  const auto groups = input.dim(input.rank() - 1) / 64;
  if (rule->rows == kQ4MatrixPanelRows)
    return Shape{groups, kQ4MatrixPanelGroupWords};
  return Shape{(rule->m + rule->rows - 1) / rule->rows, groups,
               (rule->rows / kQ4MatrixPanelRows) * kQ4MatrixPanelGroupWords};
}
[[nodiscard]] const math::MatrixCoreRow* q4_matrix_panel_row(
    const graph::KernelShapes&);

[[nodiscard]] QuantPlan quant_plan(const graph::KernelShapes&, bool indexed = false);
[[nodiscard]] inline std::uint32_t q4_shared_panel_load_chunks(
    const graph::KernelShapes& s) {
  const auto* rule = q4_shared_panel_rule(s);
  return rule && quant_plan(s).shared_activation_panel ? rule->load_chunks : 1;
}

[[nodiscard]] inline const Q4SwiGluShape *
q4_swiglu_shape(const graph::KernelShapes &s) {
  if (s.inputs.size() != 8 || s.input_dtypes.size() != 8 ||
      s.output_dtype != DType::kF32 || s.input_dtypes[0] != DType::kF32 ||
      s.input_dtypes[1] != DType::kU32 || s.input_dtypes[4] != DType::kU32 ||
      s.input_dtypes[7] != DType::kU32)
    return nullptr;
  for (const auto slot : {2u, 3u, 5u, 6u})
    if (s.input_dtypes[slot] != DType::kBF16)
      return nullptr;
  for (const auto &rule : kQ4SwiGluShapes) {
    if (s.iattrs[0] != static_cast<std::int32_t>(rule.bits) ||
        s.iattrs[1] != static_cast<std::int32_t>(rule.group) ||
        s.inputs[0] != Shape{1, rule.m, rule.k} ||
        s.output != Shape{1, rule.m, rule.n} ||
        s.inputs[1] != Shape{rule.n, rule.k / 8} ||
        s.inputs[4] != s.inputs[1] ||
        s.inputs[7] != Shape{rule.m, (rule.k / rule.group) * 25})
      continue;
    bool affines = true;
    for (const auto slot : {2u, 3u, 5u, 6u})
      affines &= s.inputs[slot] == Shape{rule.n, rule.k / rule.group};
    if (affines)
      return &rule;
  }
  return nullptr;
}
[[nodiscard]] inline const Q4SwiGluShape *
q4_swiglu_rule(const graph::KernelShapes &s) {
  const auto *rule = q4_swiglu_shape(s);
  if (!rule || !s.device || !s.intrinsics ||
      !quant_shape_device(*rule, s.device->arch, s.device->wavefront_size) ||
      s.device->max_threads_per_workgroup < rule->threads ||
      s.intrinsics->find("silu").empty())
    return nullptr;
  auto original = s;
  original.inputs = s.inputs.first(4);
  original.input_dtypes = s.input_dtypes.first(4);
  return quant_plan(original).shared_activation_panel ? rule : nullptr;
}

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
