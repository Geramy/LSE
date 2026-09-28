#include "lse/dispatch/q8_matrix.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include "lse/kernels/wmma.hpp"
namespace lse::dispatch {
namespace {
struct Rule { std::string_view arch; std::uint32_t wave, bits, rows, lds, threads, max_round_groups; };
constexpr std::array kRules{
    Rule{"gfx1201", 32, 8, 16, 1664, 256, 4},
    Rule{"gfx1201", 32, 8, 64, 6656, 256, 1},
    Rule{"gfx1201", 32, 4, 16, 1664, 256, 4},
};
bool addressable(const Shape& shape) {
  if (!shape.rank()) return false;
  std::uint64_t count = 1;
  for (std::size_t i = 0; i < shape.rank(); ++i) {
    if (shape.dim(i) <= 0 || static_cast<std::uint64_t>(shape.dim(i)) > UINT32_MAX / count)
      return false;
    count *= static_cast<std::uint64_t>(shape.dim(i));
  }
  return true;
}
AffineMatrixPlan matrix_plan(const graph::KernelShapes& s, std::uint32_t rows, std::uint32_t bits) {
  AffineMatrixPlan plan;
  if (!s.device || !s.intrinsics || s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || s.iattrs[0] != static_cast<std::int32_t>(bits) || s.iattrs[1] != 64 ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty()) return plan;
  for (const auto& in : s.inputs) if (!addressable(in)) return plan;
  if (!addressable(s.output) || s.inputs[1].rank() != 2) return plan;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  if (k % 64 || s.inputs[1].dim(1) != k / (32 / bits) ||
      s.inputs[2] != Shape{n, k / 64} || s.inputs[3] != s.inputs[2]) return plan;
  Shape expected;
  for (std::size_t i = 0; i + 1 < s.inputs[0].rank(); ++i) expected.push_back(s.inputs[0].dim(i));
  expected.push_back(n);
  if (s.output != expected) return plan;
  const auto m = s.inputs[0].elem_count() / static_cast<std::uint64_t>(k);
  const auto groups = static_cast<std::uint32_t>(k / 64);
  const Rule* rule = nullptr;
  std::uint32_t round_groups = 0, lds_bytes = 0;
  for (const auto& r : kRules) {
    const auto rounds = std::min(groups, r.max_round_groups);
    const auto lds = r.lds * rounds;
    if (s.device->arch == r.arch && s.device->wavefront_size == r.wave && bits == r.bits && rows == r.rows &&
        backend::workgroup_lds_bytes(s.device) >= lds &&
        s.device->max_threads_per_workgroup >= r.threads) {
      rule = &r; round_groups = rounds; lds_bytes = lds;
    }
  }
  if (!rule) return plan;
  const auto target = kernels::matrix_target(*s.device);
  if (!target) return plan;
  const auto caps = kernels::device_matrix_caps(*s.device);
  for (const auto& row : math::matrix_core_table()) {
    if (row.target != *target || static_cast<std::uint32_t>(row.wave) != rule->wave || row.acc != math::MatrixElem::kI32 ||
        row.operand != math::MatrixElem::kSU8 || row.m != 16 || row.n != 16 || row.k != 16 ||
        row.k_step != 16 || row.chained != 1 || !row.emittable() ||
        !math::has_cap(caps, row.cap) || s.intrinsics->find(row.key).empty()) continue;
    plan = {&row, static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
            static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(k / (32 / bits)),
            groups, lds_bytes, round_groups};
    return plan;
  }
  return plan;
}
}  // namespace
std::uint32_t q8_matrix_rows(const graph::KernelShapes& s) {
  struct ShapeRule { std::uint32_t min_m, max_m, n, k, rows; };
  constexpr std::array rules{
      ShapeRule{64, UINT32_MAX, 0, 0, 64},
      ShapeRule{1, 8, 5120, 10240, 16},
      ShapeRule{1, 8, 17408, 5120, 16},
      ShapeRule{3, 8, 5120, 17408, 16},
      ShapeRule{8, 8, 4096, 5120, 16},
      ShapeRule{8, 8, 5120, 4096, 16},
      ShapeRule{8, 8, 1024, 5120, 16},
      ShapeRule{8, 8, 1280, 5120, 16},
      ShapeRule{3, 7, 5120, 25600, 16},
  };
  for (const auto& rule : rules) {
    const auto plan = matrix_plan(s, rule.rows, 8);
    if (!plan.matrix || plan.m < rule.min_m || plan.m > rule.max_m ||
        plan.n < 128 || (rule.n != 0 && plan.n != rule.n) ||
        (rule.k != 0 && plan.k != rule.k)) continue;
    return rule.rows;
  }
  return 0;
}
AffineMatrixPlan q8_matrix_plan(const graph::KernelShapes& s, std::uint32_t rows) {
  return matrix_plan(s, rows, 8);
}
AffineMatrixPlan q4_small_matrix_plan(const graph::KernelShapes& s) {
  return matrix_plan(s, 16, 4);
}
}  // namespace lse::dispatch
