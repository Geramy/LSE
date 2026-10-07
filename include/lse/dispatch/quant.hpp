#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/dispatch/arch/tuning.hpp"
#include "lse/dispatch/q8_tuneconfig.h"
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
// The tiled prefill GEMM: 4- or 8-bit codes, groups a whole number of K steps, f32
// activations and output, float affine planes, and at least kQ4GemmMinRows
// rows. Whether the device can run it is asked again at dispatch.
// Whether a device can run the tiled 4-bit GEMM at all: 32-lane waves, a
// 256-thread workgroup and the RDNA f16 matrix instructions. Asked when the
// graph is built, because slicing K changes the graph; the kernel asks the
// precise question (row, load width, scratch) again at dispatch.
[[nodiscard]] inline bool q4_gemm_device(const backend::DeviceInfo& d) noexcept {
  const auto f = backend::arch_family(d.arch);
  return d.wavefront_size == 32 && d.max_threads_per_workgroup >= 256 &&
         (f == backend::ArchFamily::kRdna3 || f == backend::ArchFamily::kRdna35 ||
          f == backend::ArchFamily::kRdna4);
}

// The workgroup tile for an M x N contraction. Tall when M fills it; short
// otherwise, so a short prompt still spreads over the device.
//
// From 65 to 767 rows the tile is built from 48 x 32 wave tiles, four wave
// columns over 128 output columns and two to four wave rows. Each step's
// fixed work (the weight decode, the scratch stores, two barriers) is then
// shared by up to 192 rows instead of 64, and the waves of one workgroup
// hide each other's latencies. Up to 192 rows one row tile holds the whole
// pass; past that, 96- or 144-row tiles, whichever pads less (gfx1201, N
// 17408 x K 5120: M 80 0.43 -> 0.31 ms, M 144 0.58 -> 0.44 ms, M 272 1.00
// -> 0.80 ms, M 656 1.84 -> 1.68 ms; N 5120 x K 17408: M 144 0.65 -> 0.43
// ms, M 272 1.13 -> 0.84 ms).
// The wide-pass tile is the part's (arch::Tuning::q4_gemm_wide): a
// generation whose operand fragments are wider than RDNA4's spends its
// registers differently, so the tile is measured per part.
[[nodiscard]] constexpr Q4GemmTile q4_gemm_tile(
    std::uint64_t m, const arch::Tuning& tune = arch::generic::kTuning, std::uint64_t exp_n = 0) noexcept {
  if (m <= 32) return {32, 128, 1, 8};
  if (m <= 64) return {64, 128, 2, 4};
  if (m <= 192) {
    const auto wm = static_cast<std::uint32_t>((m + 47) / 48);
    return {48 * wm, 128, wm, 4};
  }
  // Wide passes take 64x64 wave tiles, which read each operand fragment
  // from scratch for four matrix instructions instead of two: 256 rows on
  // eight waves from 768 rows up (gfx1201, M 1024: gate 1.91 -> 1.66 ms,
  // down 2.01 -> 1.64 ms). They need sixteen accumulators per wave, which
  // the compiler takes only with its arrays growing per overflow, not per
  // append (patches/hrx/loom-grow-arrays-only-when-full.patch).
  if (m >= 768) {
    if (!std::is_constant_evaluated())
      if (const char* env = std::getenv("LSE_EXP_Q4_WIDE"); env &&
          exp_n >= (std::getenv("LSE_EXP_Q4_WIDE_MIN_N") ? std::strtoull(std::getenv("LSE_EXP_Q4_WIDE_MIN_N"), nullptr, 10) : 0)) {
        unsigned v[4] = {0, 0, 0, 0};
        if (std::sscanf(env, "%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3]) == 4)
          return {v[0], v[1], v[2], v[3]};
      }
    return tune.q4_gemm_wide.bm ? tune.q4_gemm_wide : kQ4GemmWideTile;
  }
  const std::uint64_t rows96 = (m + 95) / 96 * 96, rows144 = (m + 143) / 144 * 144;
  return rows144 <= rows96 ? Q4GemmTile{144, 128, 3, 4} : Q4GemmTile{96, 128, 2, 4};
}
// Whether a tile is one of the 48-row-wave tiles above.
[[nodiscard]] constexpr bool q4_gemm_wave48(const Q4GemmTile& t) noexcept {
  return t.wm != 0 && t.bm == 48 * t.wm && t.wn == 4 && t.bn == 128;
}
// How many slices of K one contraction is cut into. A workgroup walks its K
// range serially, so a grid with fewer workgroups than the device can seat
// is bound by that walk, not by math or memory; slicing K multiplies the
// grid until it fills `compute_units` twice over (three times for the
// 48-row-wave tiles, whose workgroups are larger: M 144 gate 0.48 -> 0.44 ms
// at two slices). Each slice keeps at least kQ4GemmMinSliceSteps steps, and
// slices divide the steps evenly.
inline constexpr std::uint32_t kQ4GemmMinSliceSteps = 16;
inline constexpr std::uint32_t kQ4GemmMaxSlices = 8;
[[nodiscard]] constexpr std::uint32_t q4_gemm_slices(
    std::uint64_t m, std::uint64_t n, std::uint64_t k,
    std::uint32_t compute_units,
    const arch::Tuning& tune = arch::generic::kTuning) noexcept {
  if (compute_units == 0 || k % kQ4GemmStepK != 0) return 1;
  const auto t = q4_gemm_tile(m, tune, n);
  const std::uint64_t tiles = ((m + t.bm - 1) / t.bm) * ((n + t.bn - 1) / t.bn);
  const std::uint64_t steps = k / kQ4GemmStepK;
  const std::uint64_t fill = (q4_gemm_wave48(t) ? 3ull : 2ull) * compute_units;
  std::uint32_t s = 1;
  while (s < kQ4GemmMaxSlices && tiles * s < fill &&
         steps % (2ull * s) == 0 && steps / (2ull * s) >= kQ4GemmMinSliceSteps)
    s *= 2;
  return s;
}

[[nodiscard]] inline bool q4_gemm_shape(const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      (s.input_dtypes[2] != DType::kBF16 && s.input_dtypes[2] != DType::kF16) ||
      s.input_dtypes[3] != s.input_dtypes[2] || s.output_dtype != DType::kF32 ||
      (s.iattrs[0] != 4 && s.iattrs[0] != 8) || s.iattrs[1] <= 0 ||
      s.iattrs[1] % static_cast<std::int32_t>(kQ4GemmStepK) != 0 ||
      !s.inputs[0].rank() || s.inputs[1].rank() != 2)
    return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  const auto lanes = k * s.iattrs[0] / 32;
  if (k <= 0 || n <= 0 || k % s.iattrs[1] != 0 ||
      s.inputs[1] != Shape{n, lanes} ||
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
         static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(lanes) <= UINT32_MAX;
}
// `arch` adds that part's own rows (arch::Tuning::q4_panel_shapes) after the
// shared table; the graph names the device it builds for, a kernel reads it
// from its shapes.
[[nodiscard]] inline const Q4PanelShape* q4_shared_panel_rule(
    const graph::KernelShapes& s, std::string_view arch) {
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
  if (!measured && !arch.empty())
    for (const auto& shape : arch::tuning(arch).q4_panel_shapes)
      if (shape.arch == arch && n == shape.n && k == shape.k &&
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
[[nodiscard]] inline const Q4PanelShape* q4_shared_panel_rule(
    const graph::KernelShapes& s) {
  return q4_shared_panel_rule(
      s, s.device ? std::string_view(s.device->arch) : std::string_view{});
}
[[nodiscard]] inline bool q4_shared_panel_shape(const graph::KernelShapes& s,
                                                std::string_view arch) {
  return q4_shared_panel_rule(s, arch) != nullptr;
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
  const Shape& x = s.inputs[0];
  const auto* rule = arch::first_rule<&arch::Tuning::q4_matrix_panel_shapes>(
      s.device ? std::string_view(s.device->arch) : std::string_view{},
      [&](const Q4MatrixPanelShape& r) {
        return s.iattrs[0] == static_cast<std::int32_t>(r.bits) &&
               s.iattrs[1] == static_cast<std::int32_t>(r.group) &&
               x.dim(x.rank() - 1) == r.k &&
               x.elem_count() == static_cast<std::uint64_t>(r.m * r.k) &&
               s.inputs[1] == Shape{r.n, r.k / 8} &&
               s.inputs[2] == Shape{r.n, r.k / r.group} &&
               s.inputs[3] == s.inputs[2];
      });
  if (rule == nullptr) return nullptr;
  Shape expected;
  for (std::size_t axis = 0; axis + 1 < x.rank(); ++axis)
    expected.push_back(x.dim(axis));
  expected.push_back(rule->n);
  return s.output == expected ? rule : nullptr;
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

// An 8-row contraction over 8-bit group-affine weights that reads the shared
// 16-row activation panel (quant_activation.q4_matrix_panel.v1, attribute 0 =
// 8) instead of quantizing its activations in every workgroup: the 8-row
// shapes the 8-bit matrix table measures (q8_shapes::kShapes, 16-row tile).
[[nodiscard]] inline bool q8_matrix_panel_shape(const graph::KernelShapes& s) {
  if (s.inputs.size() != 4 || s.input_dtypes.size() != 4 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || s.iattrs[0] != 8 || s.iattrs[1] != 64 ||
      !s.inputs[0].rank() || s.inputs[1].rank() != 2)
    return false;
  const auto* layout = q4_matrix_panel_layout(s.inputs[0]);
  if (!layout || layout->rows != kQ4MatrixPanelRows) return false;
  const auto k = s.inputs[0].dim(s.inputs[0].rank() - 1);
  const auto n = s.inputs[1].dim(0);
  if (n < 128 || s.inputs[1] != Shape{n, k / 4} ||
      s.inputs[2] != Shape{n, k / 64} || s.inputs[3] != s.inputs[2])
    return false;
  Shape expected;
  for (std::size_t axis = 0; axis + 1 < s.inputs[0].rank(); ++axis)
    expected.push_back(s.inputs[0].dim(axis));
  expected.push_back(n);
  if (s.output != expected) return false;
  const auto m = static_cast<std::uint64_t>(layout->m);
  for (const auto& rule : q8_shapes::kShapes)
    if (rule.rows == kQ4MatrixPanelRows && m >= rule.min_m && m <= rule.max_m &&
        (rule.n == 0 || static_cast<std::int64_t>(rule.n) == n) &&
        (rule.k == 0 || static_cast<std::int64_t>(rule.k) == k))
      return true;
  return false;
}
[[nodiscard]] const math::MatrixCoreRow* q8_matrix_panel_row(
    const graph::KernelShapes&);

[[nodiscard]] QuantPlan quant_plan(const graph::KernelShapes&, bool indexed = false);
[[nodiscard]] inline std::uint32_t q4_shared_panel_load_chunks(
    const graph::KernelShapes& s) {
  const auto* rule = q4_shared_panel_rule(s);
  return rule && quant_plan(s).shared_activation_panel ? rule->load_chunks : 1;
}

[[nodiscard]] inline std::uint32_t q4_shared_panel_columns(
    const graph::KernelShapes& s) {
  const auto* rule = q4_shared_panel_rule(s);
  return rule && quant_plan(s).shared_activation_panel ? rule->columns : 1;
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
  return arch::first_rule<&arch::Tuning::q4_swiglu_shapes>(
      s.device ? std::string_view(s.device->arch) : std::string_view{},
      [&](const Q4SwiGluShape &rule) {
        if (s.iattrs[0] != static_cast<std::int32_t>(rule.bits) ||
            s.iattrs[1] != static_cast<std::int32_t>(rule.group) ||
            s.inputs[0] != Shape{1, rule.m, rule.k} ||
            s.output != Shape{1, rule.m, rule.n} ||
            s.inputs[1] != Shape{rule.n, rule.k / 8} ||
            s.inputs[4] != s.inputs[1] ||
            s.inputs[7] != Shape{rule.m, (rule.k / rule.group) * 25})
          return false;
        bool affines = true;
        for (const auto slot : {2u, 3u, 5u, 6u})
          affines &= s.inputs[slot] == Shape{rule.n, rule.k / rule.group};
        return affines;
      });
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
