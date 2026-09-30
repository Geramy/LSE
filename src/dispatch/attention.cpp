#include "lse/dispatch/attention.hpp"
#include "lse/dispatch/attention_shapes.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>

#include "lse/backends/hrx/device_info.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/kernels/wmma.hpp"

namespace lse::dispatch {
namespace {
using graph::KernelShapes;

namespace shapes = attention_shapes;
constexpr std::uint32_t kThreads = 256;

bool short_default_shape(const Shape& query, std::int64_t capacity,
                         const shapes::ShortDefaultRule& geometry) {
  if (query.rank() != 4) return false;
  for (const auto& rule : shapes::kSplitShortRules)
    if (query.dim(0) == geometry.batch && query.dim(1) == geometry.query_heads &&
        query.dim(2) >= rule.min_rows && query.dim(2) <= rule.max_rows &&
        query.dim(3) == rule.head_dim && capacity >= geometry.min_keys &&
        capacity <= UINT32_MAX) return true;
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
  if (s.inputs.size() != 5 || s.input_dtypes.size() != 5 ||
      s.output_dtype != DType::kF32 || s.input_dtypes[0] != DType::kF32 ||
      s.input_dtypes[3] != DType::kF32 || s.input_dtypes[4] != DType::kF32 ||
      s.input_dtypes[1] != s.input_dtypes[2] ||
      !kv::valid_storage(s.input_dtypes[1], s.attrs[1]) ||
      s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
      s.inputs[2].rank() != 4 || s.inputs[4].rank() != 2 ||
      (s.iattrs[0] != 0 && s.iattrs[0] != 1 && s.iattrs[1] < 0)) return false;
  for (const auto& shape : s.inputs)
    if (!positive_shape(shape)) return false;
  const auto& q = s.inputs[0];
  const auto& k = s.inputs[1];
  const auto& v = s.inputs[2];
  const auto& table = s.inputs[4];
  const auto storage = kv::cache_dtype(s.input_dtypes[1], s.attrs[1]);
  if (q.dim(0) > (INT32_MAX - kv::kStepMetaHeader) / kv::kStepMetaPerRow ||
      q.dim(1) % k.dim(1) || q.dim(3) != kv::logical_width(storage, k.dim(3)) ||
      k.dim(0) != v.dim(0) || k.dim(1) != v.dim(1) || k.dim(2) != v.dim(2) ||
      table.dim(0) < q.dim(0) || !power_of_two(k.dim(2)) ||
      s.iattrs[3] != k.dim(2) || table.dim(1) > UINT32_MAX / k.dim(2) ||
      s.inputs[3].elem_count() < static_cast<std::size_t>(
          kv::step_meta_elems(static_cast<std::int32_t>(q.dim(0))))) return false;
  if (kv::packed_cache(storage)) {
    for (const auto* pool : {&k, &v}) {
      const auto width = kv::logical_width(storage, pool->dim(3));
      const auto vectors = pool->elem_count() / static_cast<std::size_t>(pool->dim(3));
      if (width <= 0 || vectors > UINT32_MAX / static_cast<std::uint64_t>(width))
        return false;
    }
  }
  return true;
}

}  // namespace

bool paged_attention_inputs_valid(const KernelShapes& s) { return paged_inputs(s); }

FlashDims flash_dimensions(const KernelShapes& s) {
  FlashDims d;
  if (!paged_inputs(s) || s.inputs[1].dim(2) > shapes::kFlashKeyWindow ||
      shapes::kFlashKeyWindow % s.inputs[1].dim(2)) return d;
  const auto& q = s.inputs[0];
  const auto& k = s.inputs[1];
  d.bsz = static_cast<std::uint32_t>(q.dim(0));
  d.qh = static_cast<std::uint32_t>(q.dim(1));
  d.tq = static_cast<std::uint32_t>(q.dim(2));
  d.dh = static_cast<std::uint32_t>(q.dim(3));
  d.kvh = static_cast<std::uint32_t>(k.dim(1));
  d.ts = static_cast<std::uint32_t>(k.dim(2));
  d.dv = static_cast<std::uint32_t>(kv::logical_width(
      kv::cache_dtype(s.input_dtypes[1], s.attrs[1]), s.inputs[2].dim(3)));
  d.group = d.qh / d.kvh;
  d.stride = static_cast<std::uint32_t>(s.inputs[4].dim(1));
  d.scale = s.attrs[0];
  d.mask = s.iattrs[0];
  d.window = static_cast<std::uint32_t>(s.iattrs[1]);
  d.valid = true;
  return d;
}

bool flash_wmma_supported(const KernelShapes& s) {
  const auto d = flash_dimensions(s);
  if (!d.valid || !s.device || !s.intrinsics || !s.types.scalar ||
      !attention_ops(s) || !has_ops(s, {"wave.shfl_xor"}) ||
      s.output != Shape{d.bsz, d.qh, d.tq, d.dv} ||
      !std::isfinite(d.scale) || d.scale <= 0.0f ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty()) return false;
  bool geometry = false;
  for (const auto& rule : shapes::kFlashWmmaRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.device->max_threads_per_workgroup >= rule.threads && d.tq >= rule.min_rows &&
        d.dh <= rule.max_head_dim && d.dv <= rule.max_head_dim &&
        backend::workgroup_lds_bytes(s.device) >= shapes::flash_wmma_lds_bytes(d.dh))
      geometry = true;
  if (!geometry) return false;
  const auto target = kernels::matrix_target(*s.device);
  if (!target) return false;
  const auto operand = kv::cache_dtype(s.input_dtypes[1], s.attrs[1]) == kv::CacheDType::kF16
      ? math::MatrixElem::kF16 : math::MatrixElem::kBF16;
  const auto caps = kernels::device_matrix_caps(*s.device);
  for (const auto& row : math::matrix_core_table())
    if (row.target == *target && row.wave == s.device->wavefront_size &&
        row.acc == math::MatrixElem::kF32 && row.operand == operand &&
        row.m == 16 && row.n == 16 && row.k_step == 16 && row.chained == 1 &&
        row.a_len == 8 && row.b_len == 8 && row.c_len == 8 && row.emittable() &&
        math::has_cap(caps, row.cap) && !s.intrinsics->find(row.key).empty()) return true;
  return false;
}

bool split_decode_scope(const Shape& query, std::int64_t offset, std::int64_t capacity) {
  if (query.rank() != 4 || offset < 0 || capacity <= 0) return false;
  for (const auto& rule : shapes::kDecodeRules)
    if (query.dim(2) == rule.query_rows) return true;
  return false;
}

bool split_decode_supported(const KernelShapes& s) {
  if (!paged_inputs(s) || !s.device || !attention_ops(s) ||
      s.inputs[1] != s.inputs[2] || !std::isfinite(s.attrs[0]) ||
      s.attrs[0] <= 0.0f) return false;
  const auto& q = s.inputs[0];
  const auto block = s.inputs[1].dim(2);
  const auto capacity = s.inputs[4].dim(1) * block;
  for (const auto& rule : shapes::kDecodeRules) {
    if (s.device->arch != rule.arch || s.device->wavefront_size != rule.wave ||
        s.device->max_threads_per_workgroup < rule.threads ||
        q.dim(2) != rule.query_rows || q.dim(3) != rule.head_dim ||
        block > rule.max_block ||
        backend::workgroup_lds_bytes(s.device) < shapes::kShortKeyWindow * sizeof(float) *
            shapes::decode_head_tile(static_cast<std::uint32_t>(q.dim(1)),
                                     static_cast<std::uint32_t>(s.inputs[1].dim(1))))
      continue;
    const auto parts = (capacity + shapes::kShortKeyWindow - 1) / shapes::kShortKeyWindow;
    const auto merge_bytes = (static_cast<std::uint64_t>(parts) * sizeof(float) + 15u) / 16u * 16u;
    const Shape partial{q.dim(0), q.dim(1), parts, shapes::kSplitRecord};
    return merge_bytes <= backend::workgroup_lds_bytes(s.device) &&
        positive_shape(partial) && (s.output == q || s.output == partial);
  }
  return false;
}

bool split_decode_merge_supported(const KernelShapes& s) {
  if (s.inputs.size() != 1 || s.inputs[0].rank() != 4 ||
      !positive_shape(s.inputs[0]) || s.inputs[0].dim(3) != shapes::kSplitRecord ||
      !f32_inputs(s, 1) || !s.device || !attention_ops(s)) return false;
  const auto bytes = static_cast<std::uint64_t>(s.inputs[0].dim(2)) * sizeof(float);
  for (const auto& rule : shapes::kDecodeRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        s.output == Shape{s.inputs[0].dim(0), s.inputs[0].dim(1), 1, rule.head_dim} &&
        ((bytes + 15u) / 16u) * 16u <= backend::workgroup_lds_bytes(s.device)) return true;
  return false;
}

bool split_short_scope(const Shape& query, std::int64_t offset, std::int64_t capacity) {
  for (const auto& rule : shapes::kShortDefaults)
    if (offset >= rule.min_offset && short_default_shape(query, capacity, rule)) return true;
  return false;
}

bool split_short_default_supported(const KernelShapes& s) {
  if (!split_short_supported(s)) return false;
  const auto capacity = s.inputs[4].dim(1) * s.inputs[1].dim(2);
  for (const auto& rule : shapes::kShortDefaults)
    if (s.inputs[1].dim(1) == rule.key_heads && s.inputs[1].dim(2) == rule.block &&
        s.iattrs[0] == rule.mask && s.iattrs[1] == rule.window &&
        short_default_shape(s.inputs[0], capacity, rule)) return true;
  return false;
}

bool split_short_supported(const KernelShapes& s) {
  if (!paged_inputs(s) || !s.device || !attention_ops(s) ||
      !has_ops(s, {"wave.shfl_xor"}) || s.inputs[1] != s.inputs[2] ||
      !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f) return false;
  const auto capacity = s.inputs[4].dim(1) * s.inputs[1].dim(2);
  for (const auto& rule : shapes::kSplitShortRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.inputs[0].dim(2) >= rule.min_rows && s.inputs[0].dim(2) <= rule.max_rows &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        s.inputs[0].dim(3) == rule.head_dim &&
        shapes::split_merge_lds_bytes(shapes::split_partitions(capacity)) <=
            backend::workgroup_lds_bytes(s.device) &&
        backend::workgroup_lds_bytes(s.device) >= rule.threads * sizeof(float) *
            shapes::short_query_tile(static_cast<std::uint32_t>(s.inputs[0].dim(2)),
                                     static_cast<std::uint32_t>(capacity)) *
            shapes::short_head_tile(static_cast<std::uint32_t>(s.inputs[0].dim(2)),
                                    static_cast<std::uint32_t>(s.inputs[0].dim(1)),
                                    static_cast<std::uint32_t>(s.inputs[1].dim(1)),
                                    static_cast<std::uint32_t>(capacity))) {
      const Shape partial{s.inputs[0].dim(0), s.inputs[0].dim(1),
                          s.inputs[0].dim(2),
                          static_cast<std::int64_t>(shapes::split_partitions(capacity)),
                          shapes::kSplitRecord};
      return positive_shape(partial);
    }
  return false;
}

bool split_short_merge_supported(const KernelShapes& s) {
  if (s.inputs.size() != 1 || s.inputs[0].rank() != 5 ||
      !positive_shape(s.inputs[0]) || s.inputs[0].dim(4) != shapes::kSplitRecord || !f32_inputs(s, 1) ||
      !s.device || !attention_ops(s)) return false;
  for (const auto& rule : shapes::kSplitShortRules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.inputs[0].dim(2) >= rule.min_rows && s.inputs[0].dim(2) <= rule.max_rows &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        s.output == Shape{s.inputs[0].dim(0), s.inputs[0].dim(1),
                          s.inputs[0].dim(2), rule.head_dim} &&
        shapes::split_merge_lds_bytes(s.inputs[0].dim(3)) <=
            backend::workgroup_lds_bytes(s.device)) return true;
  return false;
}

AttentionPlan attention_plan(const KernelShapes& s) {
  return flash_wmma_supported(s) ? AttentionPlan::kFlashWmma : AttentionPlan::kScalar;
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
  if (!reduction_row_supported(s) || !s.device || s.inputs.size() != 1 ||
      !f32_inputs(s, 1) || s.output != s.inputs[0] ||
      !std::isfinite(s.attrs[0]) || s.attrs[0] <= 0.0f ||
      !s.staged.name.empty() || !s.staged_quant.codes.empty() ||
      !has_ops(s, {"thread.local_id", "thread.workgroup_id.x", "wave.shfl_xor",
                   "fma", "sqrt", "max"})) return false;
  for (const auto& rule : shapes::kWaveL2Rules)
    if (s.device->arch == rule.arch && s.device->wavefront_size == rule.wave &&
        s.device->max_threads_per_workgroup >= rule.threads &&
        s.inputs[0] == Shape{rule.batch, rule.heads, rule.rows, rule.width}) return true;
  return false;
}

std::uint32_t scalar_threads(const KernelShapes& s) {
  return s.device && s.device->max_threads_per_workgroup >= kThreads ? kThreads : 64u;
}

}  // namespace lse::dispatch
