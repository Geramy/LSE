#include "lse/ops/attention.hpp"

#include <cmath>

#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"

namespace lse::ops {

Array split_heads(const Array& x, std::int64_t heads, std::int64_t head_dim) {
  const Shape& s = x.shape();
  Array r = graph::reshape(x, Shape{s.dim(0), s.dim(1), heads, head_dim});
  return graph::transpose(r, {0, 2, 1, 3});
}

Array merge_heads(const Array& x) {
  const Shape& s = x.shape();  // [B, H, T, D]
  Array t = graph::transpose(x, {0, 2, 1, 3});
  return graph::reshape(t, Shape{s.dim(0), s.dim(2), s.dim(1) * s.dim(3)});
}

namespace {

Result<Array> alloc_kv(const Shape& shape, DType dtype) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to allocate the KV cache");
  }
  const std::size_t bytes =
      dtype_storage_bytes(dtype, static_cast<std::size_t>(shape.elem_count()));
  if (bytes == 0) {
    return LSE_ERROR(kInvalidArgument, "empty KV allocation");
  }
  auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
  if (!buf.ok()) return buf.status();
  return Array::from_buffer(buf.release(), shape, dtype);
}

Status ensure_padded(AttentionCache* cache, const Array& chunk,
                     std::int64_t kvh, std::int64_t hd) {
  const Shape& cs = chunk.shape();
  const auto b = cs.dim(0);
  const auto t = cs.dim(2);
  if (cache->used < 0 || t < 0) {
    return LSE_ERROR(kInvalidArgument, "KV write has a negative extent");
  }
  if (static_cast<std::int64_t>(cache->used) + t > cache->capacity) {
    return LSE_ERROR(kOutOfRange, "KV write at ", std::to_string(cache->used),
                     " + ", std::to_string(t), " exceeds capacity ",
                     std::to_string(cache->capacity));
  }
  if (cache->keys.valid() && cache->values.valid()) return OkStatus();
  const Shape cap{b, kvh, cache->capacity, hd};
  LSE_ASSIGN_OR(cache->keys, alloc_kv(cap, chunk.dtype()));
  LSE_ASSIGN_OR(cache->values, alloc_kv(cap, chunk.dtype()));
  return OkStatus();
}

}  // namespace

Result<Array> gated_attention(const Array& x, const GatedAttentionWeights& w,
                              const GatedAttentionSpec& spec,
                              const RopeTables& rope, std::int32_t offset,
                              AttentionCache* cache) {
  if (spec.q_heads <= 0 || spec.kv_heads <= 0 ||
      spec.q_heads % spec.kv_heads != 0 || spec.head_dim <= 0) {
    return LSE_ERROR(kInvalidArgument,
                     "q_heads must be a positive multiple of kv_heads");
  }
  const auto qh = static_cast<std::int64_t>(spec.q_heads);
  const auto kvh = static_cast<std::int64_t>(spec.kv_heads);
  const auto hd = static_cast<std::int64_t>(spec.head_dim);

  Array q_lin = graph::linear(x, w.q_proj);
  Array gate_lin;
  if (spec.gate == GateSource::kFusedInQProj) {
    gate_lin = graph::slice(q_lin, -1, qh * hd, 2 * qh * hd);
    q_lin = graph::slice(q_lin, -1, 0, qh * hd);
  } else {
    gate_lin = graph::linear(x, w.g_proj);
  }

  Array q = split_heads(q_lin, qh, hd);
  Array k = split_heads(graph::linear(x, w.k_proj), kvh, hd);
  Array v = split_heads(graph::linear(x, w.v_proj), kvh, hd);

  // Per-head RMSNorm over head_dim, before the rotation.
  q = graph::rms_norm(q, w.q_norm, spec.norm_eps, spec.zero_centered_norm);
  k = graph::rms_norm(k, w.k_norm, spec.norm_eps, spec.zero_centered_norm);

  const bool padded = cache != nullptr &&
                      (cache->capacity > 0 || spec.kv_length > 0);
  if (padded && cache->capacity <= 0) {
    cache->capacity = spec.kv_length;
  }
  if (padded && cache->used == 0 && offset > 0) cache->used = offset;

  if (padded && cache->pos.valid()) {
    LSE_ASSIGN_OR(q, apply_rope(q, rope, cache->pos));
    LSE_ASSIGN_OR(k, apply_rope(k, rope, cache->pos));
  } else {
    LSE_ASSIGN_OR(q, apply_rope(q, rope, offset));
    LSE_ASSIGN_OR(k, apply_rope(k, rope, offset));
  }

  Array k_attn = k;
  Array v_attn = v;
  if (cache != nullptr) {
    if (padded) {
      LSE_RETURN_IF_ERROR(ensure_padded(cache, k, kvh, hd));
      if (!cache->pos.valid()) {
        return LSE_ERROR(kInternal, "padded KV needs a position slot");
      }
      k_attn = graph::overwrite_slice(cache->keys, k, 2, cache->pos);
      v_attn = graph::overwrite_slice(cache->values, v, 2, cache->pos);
    } else {
      if (cache->keys.valid()) {
        k_attn = graph::concat({cache->keys, k}, 2);
        v_attn = graph::concat({cache->values, v}, 2);
      }
      cache->keys = k_attn;
      cache->values = v_attn;
    }
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Array o = padded && cache->pos.valid()
                ? graph::sdpa(q, k_attn, v_attn, scale, spec.mask, spec.window,
                              cache->pos)
                : graph::sdpa(q, k_attn, v_attn, scale, spec.mask, spec.window,
                              offset);

  return graph::linear(merge_heads(o) * graph::sigmoid(gate_lin), w.o_proj);
}

}  // namespace lse::ops
