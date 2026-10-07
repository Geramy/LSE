#include "lse/dispatch/quant.hpp"

#include <algorithm>

#include "lse/dispatch/arch/tuning.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/quant/group_affine_codec.hpp"

namespace lse::dispatch {
namespace {
using graph::KernelShapes;

const math::MatrixCoreRow* matrix_row(const KernelShapes& s,
                                    math::MatrixElem acc,
                                    math::MatrixElem operand) {
  if (!s.device || !s.intrinsics) return nullptr;
  const auto target = kernels::matrix_target(*s.device);
  if (!target) return nullptr;
  const auto caps = kernels::device_matrix_caps(*s.device);
  for (const auto& row : math::matrix_core_table()) {
    if (row.target == *target && row.wave == s.device->wavefront_size &&
        row.acc == acc && row.operand == operand &&
        row.m == 16 && row.n == 16 && row.k_step == 16 && row.emittable() &&
        math::has_cap(caps, row.cap) && !s.intrinsics->find(row.key).empty())
      return &row;
  }
  return nullptr;
}

bool addressable_shape(const Shape& shape) {
  if (!shape.rank()) return false;
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
    const auto extent = shape.dim(axis);
    if (extent <= 0 || static_cast<std::uint64_t>(extent) > UINT32_MAX / count)
      return false;
    count *= static_cast<std::uint64_t>(extent);
  }
  return true;
}

bool dense_quant_shape(const KernelShapes& s, std::uint64_t& m,
                       std::uint64_t& n, std::uint64_t& k) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      !dtype_info(s.input_dtypes[2]).is_float || s.input_dtypes[3] != s.input_dtypes[2] ||
      s.output_dtype != DType::kF32 || !s.inputs[0].rank() ||
      s.inputs[1].rank() != 2 || s.inputs[2].rank() != 2 ||
      s.inputs[3].rank() != 2 || s.iattrs[0] <= 0 || s.iattrs[1] <= 0)
    return false;
  for (const auto& input : s.inputs)
    if (!addressable_shape(input)) return false;
  if (!addressable_shape(s.output)) return false;
  auto spec = quant::GroupAffine::make(s.iattrs[0], s.iattrs[1]);
  if (!spec.ok()) return false;
  const auto width = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto outputs = s.inputs[1].dim(0);
  if (width <= 0 || outputs <= 0 || width % s.iattrs[1] ||
      s.inputs[1].dim(1) * 32 != width * s.iattrs[0] ||
      s.inputs[2] != Shape{outputs, width / s.iattrs[1]} ||
      s.inputs[3] != s.inputs[2] || !s.output.elem_count() ||
      s.output.elem_count() % static_cast<std::uint64_t>(outputs)) return false;
  n = static_cast<std::uint64_t>(outputs);
  k = static_cast<std::uint64_t>(width);
  m = s.output.elem_count() / n;
  return m * k == s.inputs[0].elem_count() && m * n <= UINT32_MAX &&
         m * k <= UINT32_MAX && n * static_cast<std::uint64_t>(s.inputs[1].dim(1)) <= UINT32_MAX;
}
}  // namespace

QuantPlan quant_plan(const KernelShapes& s, bool indexed) {
  QuantPlan plan;
  std::uint64_t m = 0, n = 0, k = 0;
  if (indexed || !s.device || !s.intrinsics || !dense_quant_shape(s, m, n, k))
    return plan;
  const auto budget = backend::workgroup_lds_bytes(s.device);
  const bool staged = !s.staged.name.empty() || !s.staged_quant.codes.empty();
  const auto* amd = backend::device_extension<backend::AmdDeviceInfo>(*s.device);
  const arch::Tuning& tune = arch::tuning(s.device->arch);
  bool int8_shape = false;
  for (const auto& rule : tune.quant_int8_rows)
    int8_shape |= quant_row_range(rule, s.device->arch, s.device->wavefront_size,
                                 static_cast<std::uint32_t>(s.iattrs[0]),
                                 static_cast<std::uint32_t>(s.iattrs[1]), m);
  if (int8_shape && amd && amd->has_dot4_iu8) {
    plan.int8_activations = true;
    for (const auto symbol : quant::kGroupAffineDotSymbols)
      if (s.intrinsics->find(symbol).empty()) plan.int8_activations = false;
  }
  bool panel_device = false;
  for (const auto& rule : tune.quant_panel_devices)
    panel_device |= quant_shape_device(rule, s.device->arch, s.device->wavefront_size) &&
        static_cast<std::uint32_t>(s.iattrs[0]) == rule.bits &&
        static_cast<std::uint32_t>(s.iattrs[1]) == rule.group &&
        s.device->max_threads_per_workgroup >= rule.threads;
  plan.shared_activation_panel = !staged && plan.int8_activations &&
      q4_shared_panel_shape(s) && panel_device;
  if (!staged && plan.int8_activations && s.input_dtypes[2] == DType::kBF16 &&
      s.device->max_threads_per_workgroup >= 256) {
    for (const auto& rule : tune.quant_row_ladders) {
      if (quant_shape_device(rule, s.device->arch, s.device->wavefront_size) &&
          m == rule.m && n == rule.n && k == rule.k && budget >= rule.lds) {
        plan.row_ladder_ceiling = rule.ceiling;
        break;
      }
    }
  }
  if (!staged) {
    for (const auto& rule : tune.quant_matrix_shapes) {
      if (!quant_shape_device(rule, s.device->arch, s.device->wavefront_size) ||
          static_cast<std::uint32_t>(s.iattrs[0]) != rule.bits || static_cast<std::uint32_t>(s.iattrs[1]) != rule.group ||
          m < rule.min_m || m > rule.max_m || n != rule.n || k != rule.k ||
          budget < rule.lds || s.device->max_threads_per_workgroup < rule.threads ||
          (rule.bits == 6 && s.input_dtypes[2] != DType::kBF16)) continue;
      const auto* row = matrix_row(s, rule.bits == 4 ? math::MatrixElem::kI32
                                                   : math::MatrixElem::kF32,
                                  rule.bits == 4 ? math::MatrixElem::kSU8
                                                : math::MatrixElem::kBF16);
      if (!row || row->chained != 1) continue;
      plan.matrix = row;
      plan.implementation = rule.implementation;
      return plan;
    }
    if (plan.int8_activations) {
      for (const auto& rule : tune.quant_matrix_ranges) {
        if (!quant_row_range(rule, s.device->arch, s.device->wavefront_size,
                             static_cast<std::uint32_t>(s.iattrs[0]),
                             static_cast<std::uint32_t>(s.iattrs[1]), m) ||
            budget < rule.lds || s.device->max_threads_per_workgroup < rule.threads)
          continue;
        if (const auto* row = matrix_row(s, math::MatrixElem::kI32,
                                         math::MatrixElem::kSU8);
            row && row->chained == 1) {
          plan.matrix = row;
          plan.implementation = QuantMatrix::kInt8;
          return plan;
        }
      }
    }
  }
  if (!staged && s.input_dtypes[2] == DType::kBF16 &&
      s.device->max_threads_per_workgroup >= 256) {
    for (const auto& rule : tune.quant_scalar_shapes) {
      if (!quant_shape_device(rule, s.device->arch, s.device->wavefront_size) ||
          static_cast<std::uint32_t>(s.iattrs[0]) != rule.bits || static_cast<std::uint32_t>(s.iattrs[1]) != rule.group) continue;
      if (m == 1) {
        plan.rotate_decode_panel = true;
        if (n % rule.columns == 0) plan.decode_columns = rule.columns;
      }
      for (const auto rows : {rule.prefill_rows, 4u, 2u}) {
        if (m < rows || (rows == rule.prefill_rows && m < 32)) continue;
        const auto tile = std::min<std::uint64_t>(k, rows >= 8 ? 512u : 1024u);
        if (budget >= rows * tile * 4u) {
          plan.prefill_rows = rows;
          break;
        }
      }
    }
  }
  return plan;
}

const math::MatrixCoreRow* q4_matrix_panel_row(const KernelShapes& s) {
  const auto* rule = q4_matrix_panel_rule(s);
  if (!rule || !s.device || !s.intrinsics ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty())
    return nullptr;
  for (const auto symbol :
       {"bits.f32", "value.f32", "wave.shfl_xor", "rint", "max", "abs"})
    if (s.intrinsics->find(symbol).empty()) return nullptr;
  if (!quant_shape_device(*rule, s.device->arch, s.device->wavefront_size) ||
      s.device->max_threads_per_workgroup < rule->threads ||
      backend::workgroup_lds_bytes(s.device) <
          rule->shared_words * sizeof(std::uint32_t))
    return nullptr;
  if (rule->shared_words && s.intrinsics->find("barrier").empty())
    return nullptr;
  const auto* row = matrix_row(s, math::MatrixElem::kI32, math::MatrixElem::kSU8);
  return row && row->chained == 1 ? row : nullptr;
}

const math::MatrixCoreRow* q8_matrix_panel_row(const KernelShapes& s) {
  if (!q8_matrix_panel_shape(s) || !s.device || !s.intrinsics ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty() ||
      s.device->wavefront_size != 32 || s.device->max_threads_per_workgroup < 256)
    return nullptr;
  for (const auto symbol :
       {"bits.f32", "value.f32", "wave.shfl_xor", "rint", "max", "abs"})
    if (s.intrinsics->find(symbol).empty()) return nullptr;
  bool measured = false;
  for (const auto& rule : arch::tuning(s.device->arch).q8_matrix_rules)
    measured |= rule.arch == s.device->arch && rule.wave == s.device->wavefront_size &&
                rule.bits == 8 && rule.rows == kQ4MatrixPanelRows;
  if (!measured) return nullptr;
  const auto* row = matrix_row(s, math::MatrixElem::kI32, math::MatrixElem::kSU8);
  return row && row->chained == 1 ? row : nullptr;
}

const math::MatrixCoreRow* linear_matrix_row(const KernelShapes& s) {
  if (s.inputs.size() != 2 || s.input_dtypes.size() != 2 || !s.device ||
      s.device->max_threads_per_workgroup < 64 || s.output_dtype != DType::kF32 ||
      s.inputs[1].rank() != 2 || !addressable_shape(s.inputs[0]) ||
      !addressable_shape(s.inputs[1]) || !addressable_shape(s.output)) return nullptr;
  const auto input = s.input_dtypes[1] == DType::kI32 ? DType::kI32 : DType::kF32;
  if (s.input_dtypes[0] != input) return nullptr;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  Shape expected;
  for (std::size_t axis = 0; axis + 1 < s.inputs[0].rank(); ++axis)
    expected.push_back(s.inputs[0].dim(axis));
  expected.push_back(n);
  if (s.output != expected || s.inputs[1].dim(1) != k) return nullptr;
  struct Key { bool valid = false; math::MatrixElem acc{}, operand{}; };
  const auto key = kernels::with_matrix_operand<Key>(s.input_dtypes[1],
      []<class, class, math::MatrixElem A, math::MatrixElem T>() { return Key{true, A, T}; });
  if (!key.valid) return nullptr;
  const auto* row = matrix_row(s, key.acc, key.operand);
  return row && k >= row->k_step / row->pack && n >= row->n ? row : nullptr;
}
}  // namespace lse::dispatch
