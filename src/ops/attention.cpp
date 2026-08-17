#include "lse/ops/attention.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
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

std::size_t PagedKvLayer::pool_bytes() const noexcept {
  std::size_t total = 0;
  if (keys.valid()) {
    total += dtype_storage_bytes(keys.dtype(), keys.shape().elem_count());
  }
  if (values.valid()) {
    total += dtype_storage_bytes(values.dtype(), values.shape().elem_count());
  }
  return total;
}

std::int32_t paged_pool_blocks(std::int32_t tokens, std::int32_t rows,
                               std::int32_t capacity) noexcept {
  if (rows <= 0) return 0;
  const std::int32_t per_row = kv::blocks_for(tokens, kv::kBlockSize);
  const std::int32_t ceiling =
      kv::blocks_for(capacity, kv::kBlockSize) * rows;
  return kv::pool_rung(per_row * rows, ceiling);
}

namespace {

Result<Array> alloc_zeroed(const Shape& shape, DType dtype) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to allocate the KV pool");
  }
  const std::size_t bytes =
      dtype_storage_bytes(dtype, static_cast<std::size_t>(shape.elem_count()));
  if (bytes == 0) {
    return LSE_ERROR(kInvalidArgument, "empty KV allocation");
  }
  auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
  if (!buf.ok()) return buf.status();
  backend::DeviceBuffer owned = buf.release();
  // Zeroed, not just allocated. A partially filled block is read by the attention
  // kernel's last iteration and multiplied by a zero weight; with garbage bytes
  // a NaN would survive `fma(0, NaN, acc)` and poison the whole row.
  const std::vector<std::byte> zeros(bytes, std::byte{0});
  LSE_RETURN_IF_ERROR(sched->backend().copy_h2d(zeros.data(), owned, bytes, 0));
  return Array::from_buffer(std::move(owned), shape, dtype);
}

// Moves a pool to a bigger rung. Block ids keep their meaning — the allocator
// only appends — so the used prefix is copied verbatim and nothing re-prefills.
// It goes through the host because backend::IBackend has no device-to-device
// copy; the crossing happens once per rung, i.e. at 128, 256, 512 ... tokens.
Result<Array> regrow_pool(const Array& old, const Shape& want, DType dtype) {
  LSE_ASSIGN_OR(Array grown, alloc_zeroed(want, dtype));
  if (!old.valid()) return grown;
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to grow the KV pool");
  }
  const std::size_t bytes =
      dtype_storage_bytes(dtype, static_cast<std::size_t>(old.shape().elem_count()));
  if (bytes == 0) return grown;
  graph::Node& src = *old.node();
  if (!src.buffer.valid()) return grown;
  if (src.buffer.size_bytes < bytes) {
    return LSE_ERROR(kInternal, "KV pool ", old.shape().to_string(),
                     " wants ", std::to_string(bytes), " bytes but its buffer holds ",
                     std::to_string(src.buffer.size_bytes));
  }
  std::vector<std::byte> staging(bytes);
  LSE_RETURN_IF_ERROR(
      sched->backend().copy_d2h(src.buffer, staging.data(), bytes, 0));
  LSE_RETURN_IF_ERROR(sched->backend().copy_h2d(
      staging.data(), grown.node()->buffer, bytes, 0));
  return grown;
}

Status upload_table(PagedKvLayer& layer) {
  if (!layer.table.valid()) {
    return LSE_ERROR(kInternal, "paged KV has no block table to upload");
  }
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to upload the block table");
  }
  graph::Node& n = *layer.table.node();
  if (!n.buffer.valid()) {
    LSE_RETURN_IF_ERROR(
        graph::interpreter::ensure_output_buffer(n, sched->backend()));
  }
  const std::size_t count = n.element_count();
  std::vector<float> image(count, 0.0f);
  // Block 0 is the pad: a padded batch row, and a table slot the sequence has
  // not reached, must still name a real block so every row runs the identical
  // address arithmetic. Pad rows never write and their output is discarded.
  LSE_RETURN_IF_ERROR(kv::write_table_rows(layer.tables, layer.stride(),
                                           /*pad=*/0, image));
  const std::size_t bytes = dtype_storage_bytes(n.dtype, count);
  if (n.host_mirror.size() < bytes) n.host_mirror.resize(bytes);
  for (std::size_t i = 0; i < count; ++i) {
    graph::interpreter::store_element(n, i, image[i]);
  }
  n.host_dirty = true;
  n.device_dirty = false;
  n.materialized = true;
  LSE_RETURN_IF_ERROR(
      graph::interpreter::sync_to_device(n, sched->backend()));
  layer.table_dirty = false;
  return OkStatus();
}

// Allocates or grows the pools, the block table and the block lists so `rows`
// rows can each hold `tokens` positions.
Status ensure_paged(PagedKvLayer& layer, std::int32_t rows, std::int32_t tokens,
                    std::int32_t capacity, std::int64_t kvh, std::int64_t hd,
                    DType dtype) {
  if (rows <= 0) {
    return LSE_ERROR(kInvalidArgument, "paged KV needs at least one row");
  }
  if (tokens > capacity) {
    return LSE_ERROR(kOutOfRange, "KV write reaching ", std::to_string(tokens),
                     " tokens exceeds capacity ", std::to_string(capacity));
  }
  const std::int32_t stride = kv::blocks_for(capacity, kv::kBlockSize);
  const std::int32_t want_blocks = paged_pool_blocks(tokens, rows, capacity);

  if (layer.tables.size() != static_cast<std::size_t>(rows)) {
    layer.tables.assign(static_cast<std::size_t>(rows),
                        kv::BlockTable(kv::kBlockSize));
    layer.table_dirty = true;
  }

  const Shape pool{want_blocks, kvh, kv::kBlockSize, hd};
  const bool resize = !layer.keys.valid() ||
                      layer.keys.shape().dim(0) != want_blocks;
  if (resize && layer.keys.valid()) {
    // Two reasons, both fatal without it. Reading the old pool back needs the
    // previous pass's writes to have landed, and the old buffer is freed when the
    // program that referenced it is replaced — which happens while the previous
    // pass's dispatches may still be reading it. Growth is one event per rung, so
    // the drain costs nothing per token.
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) {
      return LSE_ERROR(kInternal, "no backend to grow the KV pool");
    }
    LSE_RETURN_IF_ERROR(sched->backend().synchronize());
  }
  if (resize) {
    LSE_ASSIGN_OR(layer.keys, regrow_pool(layer.keys, pool, dtype));
    LSE_ASSIGN_OR(layer.values, regrow_pool(layer.values, pool, dtype));
    LSE_RETURN_IF_ERROR(layer.alloc.grow(want_blocks));
  }
  if (!layer.table.valid() || layer.stride() != stride ||
      layer.table.shape().dim(0) != rows) {
    LSE_ASSIGN_OR(layer.table,
                  alloc_zeroed(Shape{rows, stride}, DType::kF32));
    layer.table_dirty = true;
  }
  for (kv::BlockTable& t : layer.tables) {
    const std::int32_t before = t.size();
    LSE_RETURN_IF_ERROR(layer.alloc.cover(t, tokens));
    if (t.size() != before) layer.table_dirty = true;
  }
  if (layer.table_dirty) LSE_RETURN_IF_ERROR(upload_table(layer));
  return OkStatus();
}

}  // namespace

Result<bool> extend_paged(PagedKvLayer& layer, std::int32_t tokens) {
  if (!layer.valid()) return true;
  const std::int32_t want = kv::blocks_for(tokens, kv::kBlockSize);
  for (kv::BlockTable& t : layer.tables) {
    if (want <= t.size()) continue;
    // Out of blocks: the pool has to move to a bigger rung, and that is a new
    // buffer. Say so rather than failing, so the caller rebuilds instead of
    // replaying a program that points at the old pool.
    if (want - t.size() > layer.alloc.free_count()) return true;
    LSE_RETURN_IF_ERROR(layer.alloc.cover(t, tokens));
    layer.table_dirty = true;
  }
  if (layer.table_dirty) LSE_RETURN_IF_ERROR(upload_table(layer));
  return false;
}

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

  const bool paged = cache != nullptr && cache->paged != nullptr &&
                     (cache->capacity > 0 || spec.kv_length > 0);
  if (paged && cache->capacity <= 0) cache->capacity = spec.kv_length;
  if (paged && cache->used == 0 && offset > 0) cache->used = offset;

  if (paged && cache->meta.valid()) {
    LSE_ASSIGN_OR(q, apply_rope(q, rope, cache->meta));
    LSE_ASSIGN_OR(k, apply_rope(k, rope, cache->meta));
  } else {
    LSE_ASSIGN_OR(q, apply_rope(q, rope, offset));
    LSE_ASSIGN_OR(k, apply_rope(k, rope, offset));
  }

  Array k_attn = k;
  Array v_attn = v;
  if (cache != nullptr) {
    if (paged) {
      if (!cache->meta.valid()) {
        return LSE_ERROR(kInternal, "paged KV needs a step descriptor");
      }
      const Shape& ks = k.shape();
      const auto rows = static_cast<std::int32_t>(ks.dim(0));
      const auto t = static_cast<std::int32_t>(ks.dim(2));
      LSE_RETURN_IF_ERROR(ensure_paged(
          *cache->paged, rows, cache->used + t,
          static_cast<std::int32_t>(cache->capacity), kvh, hd, k.dtype()));
      cache->keys = cache->paged->keys;
      cache->values = cache->paged->values;
      cache->table = cache->paged->table;
      k_attn = graph::kv_page_write(cache->keys, k, cache->meta, cache->table,
                                    kv::kBlockSize);
      v_attn = graph::kv_page_write(cache->values, v, cache->meta, cache->table,
                                    kv::kBlockSize);
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
  Array o = paged ? graph::sdpa_paged(q, k_attn, v_attn, scale, spec.mask,
                                      spec.window, cache->meta, cache->table,
                                      kv::kBlockSize)
                  : graph::sdpa(q, k_attn, v_attn, scale, spec.mask, spec.window,
                                offset);

  return graph::linear(merge_heads(o) * graph::sigmoid(gate_lin), w.o_proj);
}

}  // namespace lse::ops
