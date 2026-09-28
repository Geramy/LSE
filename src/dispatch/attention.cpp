#include "lse/dispatch/attention.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kv/block.hpp"

namespace lse::dispatch {
namespace {
using graph::KernelShapes;

constexpr std::uint32_t kThreads = 256, kKeyWindow = 256;
constexpr std::uint32_t kSplitRecord = 258;

struct FlashRule {
  std::string_view arch;
  std::uint32_t min_rows, tile_rows;
  AttentionPlan plan;
};
constexpr std::array kFlashRules{
    FlashRule{"gfx1201", 12, 12, AttentionPlan::kFlash12},
    FlashRule{{}, 2, 8, AttentionPlan::kFlash8},
};

struct SplitShortRule {
  std::string_view arch;
  std::uint32_t wave, min_rows, max_rows, threads, max_keys;
};
constexpr std::array kSplitShortRules{
    SplitShortRule{"gfx1201", 32, 2, 8, 128, 8192},
};

struct SplitShortDefaultRule {
  std::uint32_t queries, capacity;
};
constexpr std::array kSplitShortDefaultRules{
    SplitShortDefaultRule{3, 1024}, SplitShortDefaultRule{3, 2048},
    SplitShortDefaultRule{4, 1024}, SplitShortDefaultRule{4, 2048},
    SplitShortDefaultRule{7, 1024}, SplitShortDefaultRule{7, 2048},
    SplitShortDefaultRule{8, 1024}, SplitShortDefaultRule{8, 2048},
};

bool short_default_shape(const Shape& query, std::int64_t capacity) {
  if (query.rank() != 4 || query.dim(0) != 1 || query.dim(1) != 24 ||
      query.dim(3) != 256) return false;
  for (const auto& rule : kSplitShortDefaultRules)
    if (query.dim(2) == rule.queries && capacity == rule.capacity) return true;
  return false;
}

bool power_of_two(std::int64_t n) { return n >= 2 && (n & (n - 1)) == 0; }

bool positive_shape(const Shape& shape) {
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

bool f32_inputs(const KernelShapes& s, std::size_t count) {
  if (s.input_dtypes.size() != count || s.output_dtype != DType::kF32) return false;
  for (auto dtype : s.input_dtypes)
    if (dtype != DType::kF32) return false;
  return true;
}

// Shape-only graph construction has no dialect table yet.
bool has_ops(const KernelShapes& s, std::initializer_list<std::string_view> ops) {
  if (!s.intrinsics) return true;
  for (auto op : ops)
    if (s.intrinsics->find(op).empty()) return false;
  return true;
}

bool attention_ops(const KernelShapes& s) {
  return has_ops(s, {"thread.local_id", "thread.workgroup_id.x", "barrier",
                     "fma", "max", "exp", "neg_inf"});
}

bool paged_inputs(const KernelShapes& s) {
  if (s.inputs.size() != 5 || !f32_inputs(s, 5) ||
      s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
      s.inputs[2].rank() != 4 || s.inputs[4].rank() != 2 ||
      (s.iattrs[0] != 0 && s.iattrs[0] != 1 && s.iattrs[1] < 0)) return false;
  for (const auto& shape : s.inputs)
    if (!positive_shape(shape)) return false;
  const auto& q = s.inputs[0];
  const auto& k = s.inputs[1];
  const auto& v = s.inputs[2];
  const auto& table = s.inputs[4];
  if (q.dim(0) > (INT32_MAX - kv::kStepMetaHeader) / kv::kStepMetaPerRow ||
      q.dim(1) % k.dim(1) || q.dim(3) != k.dim(3) ||
      k.dim(0) != v.dim(0) || k.dim(1) != v.dim(1) || k.dim(2) != v.dim(2) ||
      table.dim(0) < q.dim(0) || !power_of_two(k.dim(2)) ||
      s.iattrs[3] != k.dim(2) || table.dim(1) > UINT32_MAX / k.dim(2) ||
      s.inputs[3].elem_count() < static_cast<std::size_t>(
          kv::step_meta_elems(static_cast<std::int32_t>(q.dim(0))))) return false;
  return true;
}

std::uint64_t flash_lds_bytes(const FlashDims& d, std::uint32_t rows) {
  return static_cast<std::uint64_t>(rows) * (d.dh + 2ull * kKeyWindow + 4) * sizeof(float);
}
}  // namespace

FlashDims flash_dimensions(const KernelShapes& s) {
  FlashDims d;
  if (!paged_inputs(s) || s.inputs[1].dim(2) > kKeyWindow ||
      kKeyWindow % s.inputs[1].dim(2)) return d;
  const auto& q = s.inputs[0];
  const auto& k = s.inputs[1];
  d.bsz = static_cast<std::uint32_t>(q.dim(0));
  d.qh = static_cast<std::uint32_t>(q.dim(1));
  d.tq = static_cast<std::uint32_t>(q.dim(2));
  d.dh = static_cast<std::uint32_t>(q.dim(3));
  d.kvh = static_cast<std::uint32_t>(k.dim(1));
  d.ts = static_cast<std::uint32_t>(k.dim(2));
  d.dv = static_cast<std::uint32_t>(s.inputs[2].dim(3));
  d.group = d.qh / d.kvh;
  d.stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
  d.scale = s.attrs[0];
  d.mask = s.iattrs[0];
  d.window = static_cast<std::uint32_t>(s.iattrs[1]);
  d.valid = true;
  return d;
}

bool flash_supported(const KernelShapes& s, std::uint32_t query_rows) {
  const auto d = flash_dimensions(s);
  return (query_rows == 8 || query_rows == 12) && d.valid && s.device &&
         s.device->max_threads_per_workgroup >= kThreads && attention_ops(s) &&
         flash_lds_bytes(d, query_rows) <= backend::workgroup_lds_bytes(s.device);
}

bool split_decode_scope(const Shape& query, std::int64_t offset, std::int64_t capacity) {
  return query.rank() == 4 && query.dim(2) == 1 && offset >= 511 &&
         capacity > 0 && capacity <= 4096;
}

bool shared_decode_supported(const KernelShapes& s) {
  if (!paged_inputs(s) || !s.device || s.device->arch != "gfx1201" ||
      s.device->max_threads_per_workgroup < kThreads || !attention_ops(s)) return false;
  const auto& q = s.inputs[0];
  const auto& keys = s.inputs[1];
  const auto stride = s.inputs[4].dim(1), block = keys.dim(2);
  return q.dim(2) == 1 && q.dim(3) == 256 && s.inputs[2] == keys &&
         block <= 256 && stride <= 8192 / block &&
         static_cast<std::uint64_t>(stride * block) * sizeof(float) <=
             backend::workgroup_lds_bytes(s.device);
}

bool split_decode_supported(const KernelShapes& s) {
  return shared_decode_supported(s) && s.device->wavefront_size == 32;
}

bool split_decode_merge_supported(const KernelShapes& s) {
  if (s.inputs.size() != 1 || s.inputs[0].rank() != 4 ||
      !positive_shape(s.inputs[0]) || s.inputs[0].dim(2) > 64 ||
      s.inputs[0].dim(3) != kSplitRecord || !f32_inputs(s, 1) ||
      !s.device || s.device->arch != "gfx1201" ||
      s.device->max_threads_per_workgroup < kThreads || !attention_ops(s)) return false;
  const auto bytes = static_cast<std::uint32_t>(s.inputs[0].dim(2)) * sizeof(float);
  return ((bytes + 15u) / 16u) * 16u <= backend::workgroup_lds_bytes(s.device);
}

bool split_short_scope(const Shape& query, std::int64_t offset, std::int64_t capacity) {
  return offset >= 512 && short_default_shape(query, capacity);
}

bool split_short_default_supported(const KernelShapes& s) {
  return split_short_supported(s) && s.inputs[1].dim(1) == 4 &&
         s.inputs[1].dim(2) == 16 && s.iattrs[0] == 1 && s.iattrs[1] == 0 &&
         short_default_shape(s.inputs[0], s.inputs[4].dim(1) * s.inputs[1].dim(2));
}

bool split_short_supported(const KernelShapes& s) {
  if (!paged_inputs(s) || !s.device || !attention_ops(s) ||
      !has_ops(s, {"wave.shfl_xor"}) || s.inputs[0].dim(3) != 256 || s.inputs[1] != s.inputs[2] ||
      !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f) return false;
  const auto capacity = s.inputs[4].dim(1) * s.inputs[1].dim(2);
  for (const auto& rule : kSplitShortRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.inputs[0].dim(2) >= rule.min_rows && s.inputs[0].dim(2) <= rule.max_rows &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        capacity <= rule.max_keys &&
        backend::workgroup_lds_bytes(s.device) >= 128u * sizeof(float)) {
      const Shape partial{s.inputs[0].dim(0), s.inputs[0].dim(1),
                          s.inputs[0].dim(2), (capacity + 127) / 128, kSplitRecord};
      return positive_shape(partial);
    }
  return false;
}

bool split_short_merge_supported(const KernelShapes& s) {
  if (s.inputs.size() != 1 || s.inputs[0].rank() != 5 ||
      !positive_shape(s.inputs[0]) || s.inputs[0].dim(3) > 64 ||
      s.inputs[0].dim(4) != kSplitRecord || !f32_inputs(s, 1) ||
      !s.device || !attention_ops(s)) return false;
  for (const auto& rule : kSplitShortRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.inputs[0].dim(2) >= rule.min_rows && s.inputs[0].dim(2) <= rule.max_rows &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        ((static_cast<std::uint32_t>(s.inputs[0].dim(3)) * sizeof(float) + 15u) / 16u) * 16u <=
            backend::workgroup_lds_bytes(s.device)) return true;
  return false;
}

AttentionPlan attention_plan(const KernelShapes& s) {
  if (shared_decode_supported(s)) return AttentionPlan::kSharedExp;
  const auto d = flash_dimensions(s);
  if (!d.valid || !s.device || s.device->max_threads_per_workgroup < kThreads ||
      !attention_ops(s)) return AttentionPlan::kScalar;
  const auto budget = backend::workgroup_lds_bytes(s.device);
  for (const auto& rule : kFlashRules)
    if ((rule.arch.empty() || s.device->arch == rule.arch) &&
        d.tq >= rule.min_rows && flash_lds_bytes(d, rule.tile_rows) <= budget)
      return rule.plan;
  return AttentionPlan::kScalar;
}

bool reduction_row_supported(const KernelShapes& s) {
  return !s.inputs.empty() && s.inputs[0].rank() && s.types.scalar && s.intrinsics &&
         s.inputs[0].dim(s.inputs[0].rank() - 1) > 0;
}

bool cooperative_rms_supported(const KernelShapes& s) {
  if (!reduction_row_supported(s) || s.inputs.size() != 2 ||
      s.input_dtypes.size() != 2 || s.inputs[1].rank() != 1 ||
      s.output != s.inputs[0] || s.input_dtypes[0] != DType::kF32 ||
      s.output_dtype != DType::kF32 || !s.device ||
      s.device->wavefront_size < 32 ||
      s.device->max_threads_per_workgroup < kThreads ||
      backend::workgroup_lds_bytes(s.device) < kThreads * sizeof(float) ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty()) return false;
  const auto dtype = s.input_dtypes[1];
  if (dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16) return false;
  const auto width = s.inputs[0].dim(s.inputs[0].rank() - 1);
  return width >= 32 && s.inputs[1].elem_count() == static_cast<std::size_t>(width) &&
         s.output.elem_count() && s.output.elem_count() <= UINT32_MAX / sizeof(float) &&
         has_ops(s, {"thread.local_id", "thread.workgroup_id.x", "barrier", "fma", "rsqrt", "wave.shfl_xor"});
}

bool phase_cooperative_rms_supported(const KernelShapes& s) {
  if (!cooperative_rms_supported(s)) return false;
  const auto width = s.inputs[0].dim(s.inputs[0].rank() - 1);
  return width >= kThreads && width % kThreads == 0 && s.output.elem_count() % kThreads == 0;
}

bool wave32_l2_supported(const KernelShapes& s) {
  return reduction_row_supported(s) && s.device && s.device->arch == "gfx1201" &&
         s.device->wavefront_size == 32 && s.device->max_threads_per_workgroup >= 128 &&
         s.inputs.size() == 1 && f32_inputs(s, 1) && s.inputs[0] == Shape{1, 1, 16, 128} &&
         s.output == s.inputs[0] && std::isfinite(s.attrs[0]) && s.attrs[0] > 0.0f &&
         s.staged.name.empty() && s.staged_quant.codes.empty() &&
         has_ops(s, {"thread.local_id", "thread.workgroup_id.x", "wave.shfl_xor", "fma", "sqrt", "max"});
}

std::uint32_t scalar_threads(const KernelShapes& s) {
  return s.device && s.device->max_threads_per_workgroup >= kThreads ? kThreads : 64u;
}

}  // namespace lse::dispatch
