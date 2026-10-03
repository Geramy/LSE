#include "lse/ops/attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lse/dispatch/attention.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/sizing.hpp"

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
    total += keys.node()->kv_fragments
        ? keys.node()->kv_fragments->fragment_count() * kv::kFragmentBytes
        : dtype_storage_bytes(keys.dtype(), keys.shape().elem_count());
  }
  if (values.valid()) {
    total += values.node()->kv_fragments
        ? values.node()->kv_fragments->fragment_count() * kv::kFragmentBytes
        : dtype_storage_bytes(values.dtype(), values.shape().elem_count());
  }
  return total;
}

std::int32_t paged_pool_blocks(std::span<const std::int32_t> row_tokens,
                               std::int32_t ceiling) noexcept {
  // A resident session that wants program replay across requests cannot
  // tolerate a regrow: regrow_pool swaps the pool arrays, which orphans the
  // KV leaves every retained program was bound against. Pre-sizing at the
  // ceiling keeps the arrays (and so the leaves) stable for the process
  // lifetime, at the cost of committing the full pool up front.
  if (std::getenv("LSE_KV_PREALLOC") != nullptr) return ceiling;
  std::int32_t want = 0;
  for (std::int32_t t : row_tokens) want += kv::blocks_for(t, kv::kBlockSize);
  return kv::pool_rung(want, ceiling);
}

std::int32_t paged_pool_blocks(std::int32_t tokens, std::int32_t rows,
                               std::int32_t capacity) noexcept {
  if (rows <= 0) return 0;
  const std::int32_t per_row = kv::blocks_for(tokens, kv::kBlockSize);
  const std::int32_t ceiling =
      kv::blocks_for(capacity, kv::kBlockSize) * rows;
  if (std::getenv("LSE_KV_PREALLOC") != nullptr) return ceiling;
  return kv::pool_rung(per_row * rows, ceiling);
}

namespace {

Result<Array> alloc_pool(const Shape& shape, DType dtype,
                         std::size_t initialized_prefix_bytes) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to allocate the KV pool");
  }
  const std::size_t bytes =
      dtype_storage_bytes(dtype, static_cast<std::size_t>(shape.elem_count()));
  if (bytes == 0) {
    return LSE_ERROR(kInvalidArgument, "empty KV allocation");
  }
  if (initialized_prefix_bytes > bytes) {
    return LSE_ERROR(kInvalidArgument, "KV prefix exceeds the new pool");
  }
  // The pool belongs to the layer that reads it. Left on whoever is primary,
  // every attention group on every other device fetches the whole pool across
  // the link once per layer per token -- which is what it cost before the
  // layer's own member was asked here.
  backend::IDeviceSet& set = sched->devices();
  const std::size_t member = graph::preferred_member();
  backend::IBackend& be =
      member < set.size() ? set.device(member) : sched->backend();
  const backend::Stream at =
      member < set.size()
          ? set.stream_for(member).value_or(backend::kDefaultStream)
          : backend::kDefaultStream;
  const backend::ScopedAllocationSite site(backend::AllocationSite::kKvCache);
  auto buf = be.allocate(bytes, backend::MemoryClass::kDevice, at);
  if (!buf.ok()) {
    return Status(buf.status().code(), detail::concat(
        "KV pool allocation (", shape.to_string(), ", ", std::to_string(bytes),
        " bytes): ", buf.status().message()));
  }
  backend::DeviceBuffer owned = buf.release();
  // A partially filled block can be read under a zero attention weight. Zero
  // the new tail so NaN bytes cannot survive fma(0, NaN, acc). On growth the
  // old prefix is copied below; zeroing that prefix first wastes a full pool
  // transfer and a pool-sized host allocation at every context rung.
  const std::size_t tail_bytes = bytes - initialized_prefix_bytes;
  if (tail_bytes != 0) {
    const std::vector<std::byte> zeros(tail_bytes, std::byte{0});
    LSE_RETURN_IF_ERROR(be.copy_h2d(
        zeros.data(), owned, tail_bytes, initialized_prefix_bytes));
  }
  return Array::from_buffer(std::move(owned), shape, dtype);
}

Result<Array> alloc_zeroed(const Shape& shape, DType dtype) {
  return alloc_pool(shape, dtype, 0);
}

Result<Array> grow_fragments(PagedKvLayer& layer, const Array& old,
                             const Shape& shape, DType dtype, std::size_t capacity_bytes) {
  auto* sched = graph::default_scheduler();
  if (!sched) return LSE_ERROR(kInternal, "no backend for K/V fragments");
  auto& set = sched->devices();
  const auto member = graph::preferred_member();
  auto& be = member < set.size() ? set.device(member) : sched->backend();
  const auto stream = member < set.size()
      ? set.stream_for(member).value_or(backend::kDefaultStream)
      : backend::kDefaultStream;
  auto storage = old.valid() ? old.node()->kv_fragments : nullptr;
  if (old.valid() && !storage)
    return LSE_ERROR(kInvalidArgument, "cannot change a live K/V storage layout");
  const backend::ScopedAllocationSite site(backend::AllocationSite::kKvCache);
  if (!layer.memory) layer.memory = sched->kv_memory();
  if (!storage)
    storage = std::make_shared<kv::FragmentStorage>(layer.memory, be, stream);
  LSE_RETURN_IF_ERROR(storage->reserve(capacity_bytes));
  const auto block_bytes = dtype_storage_bytes(dtype, shape.elem_count() / shape.dim(0));
  LSE_RETURN_IF_ERROR(storage->grow(layer.resident_blocks * block_bytes));
  LSE_ASSIGN_OR(auto buffer, storage->binding());
  auto result = Array::from_buffer(std::move(buffer), shape, dtype);
  result.node()->kv_fragments = std::move(storage);
  return result;
}

Status grow_resident_fragments(PagedKvLayer& layer) {
  for (const auto* pool : {&layer.keys, &layer.values}) {
    if (!pool->valid() || !pool->node()->kv_fragments) continue;
    const auto block_bytes = dtype_storage_bytes(pool->dtype(),
        pool->shape().elem_count() / pool->shape().dim(0));
    LSE_RETURN_IF_ERROR(pool->node()->kv_fragments->grow(layer.resident_blocks * block_bytes));
  }
  return OkStatus();
}

// Moves a pool to a bigger rung. Block ids keep their meaning — the allocator
// only appends — so the used prefix is copied verbatim and nothing re-prefills.
// The move is one engine call and stays on the device; it happens once per
// rung, i.e. at 128, 256, 512 ... tokens.
Result<Array> regrow_pool(const Array& old, const Shape& want, DType dtype) {
  if (!old.valid()) return alloc_zeroed(want, dtype);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to grow the KV pool");
  }
  const std::size_t bytes =
      dtype_storage_bytes(dtype, static_cast<std::size_t>(old.shape().elem_count()));
  if (bytes == 0) return alloc_zeroed(want, dtype);
  graph::Node& src = *old.node();
  if (!src.buffer.valid()) return alloc_zeroed(want, dtype);
  if (src.buffer.size_bytes < bytes) {
    return LSE_ERROR(kInternal, "KV pool ", old.shape().to_string(),
                     " wants ", std::to_string(bytes), " bytes but its buffer holds ",
                     std::to_string(src.buffer.size_bytes));
  }
  LSE_ASSIGN_OR(Array grown, alloc_pool(want, dtype, bytes));
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(src, sched->backend()));
  // One statement: move the used prefix into the bigger pool. Both ends are
  // device memory, so this is the copy engine and the bytes never touch the
  // host -- it used to stage them down and back up, which is two transfers and
  // a buffer the size of the pool for a move that never needed either.
  LSE_RETURN_IF_ERROR(
      sched->backend().copy({grown.node()->buffer}, {src.buffer}, bytes));
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

// What each row must cover once the pending pass lands. `row_tokens` is the
// batch driver's answer; without one, every row reaches `tokens`, which is what
// a single sequence needs.
Result<std::vector<std::int32_t>> row_demand(const PagedKvLayer& layer,
                                             std::int32_t rows,
                                             std::int32_t tokens) {
  if (layer.row_tokens.empty()) {
    return std::vector<std::int32_t>(static_cast<std::size_t>(rows), tokens);
  }
  if (layer.row_tokens.size() != static_cast<std::size_t>(rows)) {
    return LSE_ERROR(kInvalidArgument, "paged KV has ",
                     std::to_string(layer.row_tokens.size()),
                     " per-row demands for a pass of ", std::to_string(rows),
                     " rows");
  }
  return layer.row_tokens;
}

std::int32_t pool_ceiling(const PagedKvLayer& layer, std::int32_t rows,
                          std::int32_t capacity) noexcept {
  if (layer.block_ceiling > 0) return layer.block_ceiling;
  return kv::blocks_for(capacity, kv::kBlockSize) * rows;
}

}  // namespace

Status ensure_paged_kv(PagedKvLayer& layer, std::int32_t rows, std::int32_t tokens,
                       std::int32_t capacity, std::int64_t kvh, std::int64_t hd,
                       kv::CacheDType format) {
  const auto dtype = kv::storage_dtype(format);
  const auto pitch = kv::storage_width(format, hd);
  if (pitch <= 0) return LSE_ERROR(kInvalidArgument, "FP8 KV head width must be a positive multiple of four");
  if (layer.valid() && layer.storage != format)
    return LSE_ERROR(kInvalidArgument, "cannot change a live KV pool format");
  if (rows <= 0) {
    return LSE_ERROR(kInvalidArgument, "paged KV needs at least one row");
  }
  LSE_ASSIGN_OR(const std::vector<std::int32_t> want_tokens,
                row_demand(layer, rows, tokens));
  for (std::int32_t t : want_tokens) {
    if (t > capacity) {
      return LSE_ERROR(kOutOfRange, "KV write reaching ", std::to_string(t),
                       " tokens exceeds capacity ", std::to_string(capacity));
    }
  }
  const std::int32_t table_ceiling = kv::blocks_for(capacity, kv::kBlockSize);
  std::int32_t want_blocks =
      paged_pool_blocks(want_tokens, pool_ceiling(layer, rows, capacity));
  // The pool never shrinks. Its block count is an input dimension the JIT keys
  // on, so giving it back when a sequence leaves would recompile the attention
  // groups on every retirement — and regrow_pool copies the used prefix, which
  // has nowhere to go in a smaller buffer.
  if (layer.keys.valid()) {
    want_blocks = std::max(
        want_blocks, static_cast<std::int32_t>(layer.keys.shape().dim(0)));
  }
  const std::int32_t stride =
      std::max(layer.stride(), std::min(table_ceiling, want_blocks));

  if (layer.tables.size() != static_cast<std::size_t>(rows)) {
    // Rows that survive keep their blocks: widening or narrowing the batch must
    // not cost the sequences already in it their KV. Only the rows that go away
    // hand theirs back — dropping a table on the floor would leave its
    // refcounts at 1 forever, and the free list would shrink by a whole batch
    // every time the width changed.
    for (std::size_t r = static_cast<std::size_t>(rows); r < layer.tables.size();
         ++r) {
      LSE_RETURN_IF_ERROR(layer.alloc.release_all(layer.tables[r]));
    }
    layer.tables.resize(static_cast<std::size_t>(rows),
                        kv::BlockTable(kv::kBlockSize));
    layer.table_dirty = true;
  }

  const Shape pool{want_blocks, kvh, kv::kBlockSize, pitch};
  const bool resize = !layer.keys.valid() ||
                      layer.keys.shape().dim(0) != want_blocks;
  if (resize && layer.keys.valid()) {
    // Complete users of the old graph before changing its pool shape. On the
    // contiguous backends this also orders the old-to-new pool copy.
    graph::Scheduler* sched = graph::default_scheduler();
    if (sched == nullptr) {
      return LSE_ERROR(kInternal, "no backend to grow the KV pool");
    }
    LSE_RETURN_IF_ERROR(sched->backend().synchronize());
  }
  if (resize) {
    auto* sched = graph::default_scheduler();
    const auto member = graph::preferred_member();
    auto* owner = sched ? (member < sched->devices().size()
        ? &sched->devices().device(member) : &sched->backend()) : nullptr;
    const auto* toolchain = owner ? owner->toolchain(sched->dialect()) : nullptr;
    const bool fragmented = layer.keys.valid() ? bool(layer.keys.node()->kv_fragments)
        : toolchain && toolchain->dialect == graph::Dialect::kLoom;
    if (fragmented) {
      const auto capacity_bytes =
          static_cast<std::size_t>(pool_ceiling(layer, rows, capacity)) *
          kv::block_plane_bytes(format, kvh, hd);
      LSE_ASSIGN_OR(layer.keys, grow_fragments(layer, layer.keys, pool, dtype, capacity_bytes));
      LSE_ASSIGN_OR(layer.values, grow_fragments(layer, layer.values, pool, dtype, capacity_bytes));
    } else {
      LSE_ASSIGN_OR(layer.keys, regrow_pool(layer.keys, pool, dtype));
      LSE_ASSIGN_OR(layer.values, regrow_pool(layer.values, pool, dtype));
    }
    LSE_RETURN_IF_ERROR(layer.alloc.grow(want_blocks));
    layer.storage = format;
  }
  if (!layer.table.valid() || layer.stride() != stride ||
      layer.table.shape().dim(0) != rows) {
    LSE_ASSIGN_OR(layer.table,
                  alloc_zeroed(Shape{rows, stride}, DType::kF32));
    layer.table_dirty = true;
  }
  for (std::size_t r = 0; r < layer.tables.size(); ++r) {
    kv::BlockTable& t = layer.tables[r];
    const std::int32_t before = t.size();
    LSE_RETURN_IF_ERROR(layer.alloc.cover(t, want_tokens[r]));
    for (std::int32_t i = before; i < t.size(); ++i)
      layer.resident_blocks = std::max(layer.resident_blocks,
          static_cast<std::int32_t>(t.blocks()[i]) + 1);
    if (t.size() != before) layer.table_dirty = true;
  }
  LSE_RETURN_IF_ERROR(grow_resident_fragments(layer));
  if (layer.table_dirty) LSE_RETURN_IF_ERROR(upload_table(layer));
  return OkStatus();
}

Result<bool> extend_paged(PagedKvLayer& layer, std::int32_t tokens) {
  if (!layer.valid()) return true;
  const auto rows = static_cast<std::int32_t>(layer.tables.size());
  LSE_ASSIGN_OR(const std::vector<std::int32_t> want_tokens,
                row_demand(layer, rows, tokens));
  for (std::int32_t t : want_tokens)
    if (kv::blocks_for(t, kv::kBlockSize) > layer.stride()) return true;
  for (std::size_t r = 0; r < layer.tables.size(); ++r) {
    kv::BlockTable& t = layer.tables[r];
    const std::int32_t want = kv::blocks_for(want_tokens[r], kv::kBlockSize);
    if (want <= t.size()) continue;
    // A larger logical pool changes the compiled shape. Rebuild the graph;
    // fragmented storage keeps its existing bytes and addresses in place.
    if (want - t.size() > layer.alloc.free_count()) return true;
    const auto before = t.size();
    LSE_RETURN_IF_ERROR(layer.alloc.cover(t, want_tokens[r]));
    for (std::int32_t i = before; i < t.size(); ++i)
      layer.resident_blocks = std::max(layer.resident_blocks,
          static_cast<std::int32_t>(t.blocks()[i]) + 1);
    layer.table_dirty = true;
  }
  LSE_RETURN_IF_ERROR(grow_resident_fragments(layer));
  if (layer.table_dirty) LSE_RETURN_IF_ERROR(upload_table(layer));
  return false;
}

Status release_row(PagedKvLayer& layer, std::int32_t row) {
  if (row < 0 || static_cast<std::size_t>(row) >= layer.tables.size()) {
    return LSE_ERROR(kOutOfRange, "no paged row ", std::to_string(row),
                     " in a layer of ", std::to_string(layer.tables.size()));
  }
  kv::BlockTable& t = layer.tables[static_cast<std::size_t>(row)];
  if (t.empty()) return OkStatus();
  LSE_RETURN_IF_ERROR(layer.alloc.release_all(t));
  // The device image still names the blocks this row just gave up. It is
  // re-uploaded by the next extend_paged, which every pass runs, and until then
  // the row's live length is zero so no kernel reads through it.
  layer.table_dirty = true;
  return OkStatus();
}

Result<Array> gated_attention(const Array& x, const GatedAttentionWeights& w,
                              const GatedAttentionSpec& spec,
                              const RopeTables& rope, std::int32_t offset,
                              AttentionCache* cache, AttentionExecutionPhase phase) {
  if (spec.q_heads <= 0 || spec.kv_heads <= 0 ||
      spec.q_heads % spec.kv_heads != 0 || spec.head_dim <= 0) {
    return LSE_ERROR(kInvalidArgument,
                     "q_heads must be a positive multiple of kv_heads");
  }
  const auto sparse = attention_for_phase(spec.sparse_attention, phase, x.shape().dim(1));
  if (sparse.enabled() &&
      (!cache || !cache->paged))
    return LSE_ERROR(kInvalidArgument, "Sparse attention requires paged attention storage");
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
      LSE_RETURN_IF_ERROR(ensure_paged_kv(
          *cache->paged, rows, cache->used + t,
          static_cast<std::int32_t>(cache->capacity), kvh, hd, spec.kv_cache_dtype));
      cache->keys = cache->paged->keys;
      cache->values = cache->paged->values;
      cache->table = cache->paged->table;
      k_attn = graph::kv_page_write(cache->keys, k, cache->meta, cache->table,
                                    kv::kBlockSize, spec.kv_cache_dtype);
      v_attn = graph::kv_page_write(cache->values, v, cache->meta, cache->table,
                                    kv::kBlockSize, spec.kv_cache_dtype);
    } else {
      if (cache->keys.valid()) {
        k_attn = graph::concat({cache->keys, k}, 2);
        v_attn = graph::concat({cache->values, v}, 2);
      }
      cache->keys = k_attn;
      cache->values = v_attn;
    }
  }

  const backend::DeviceInfo* split_device = nullptr;
  const auto table_capacity = paged ? cache->table.shape().dim(1) * kv::kBlockSize : 0;
  if (paged && (dispatch::split_decode_scope(q.shape(), offset, table_capacity) ||
                dispatch::split_short_scope(q.shape(), offset, table_capacity))) {
    if (auto* scheduler = graph::default_scheduler()) {
      const auto member = graph::preferred_member();
      if (member < scheduler->devices().size())
        split_device = &scheduler->devices().device(member).device_info();
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Array o = paged ? graph::sdpa_paged(q, k_attn, v_attn, scale, spec.mask,
                                      spec.window, cache->meta, cache->table,
                                      kv::kBlockSize, split_device, spec.kv_cache_dtype,
                                      sparse)
                  : graph::sdpa(q, k_attn, v_attn, scale, spec.mask, spec.window,
                                offset);

  return graph::linear(merge_heads(o) * graph::sigmoid(gate_lin), w.o_proj);
}

}  // namespace lse::ops
