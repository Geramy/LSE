// How many bytes the paged K/V cache holds, answered without a device.
//
// These are the allocator's own rules, not a second opinion about them: the
// paged attention path (ops::ensure_paged_kv) sizes its pools and fragment
// tables through the same functions, so a model-info or memory estimate that
// reads these agrees with what a running engine allocates.
//
// One attention layer keeps two planes, K and V, each of
// [blocks, kv_heads, kBlockSize, pitch] in the cache format's storage dtype.
// The 8-bit formats pack four codes per word and add one FP32 scale word per
// token and head, which is what storage_width accounts for.
#pragma once

#include <cstddef>
#include <cstdint>

#include "lse/core/dtype.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/kv/memory.hpp"

namespace lse::kv {

// Bytes of one block of ONE plane (K or V) of one layer. Zero when the format
// cannot store this head width (FP8 needs a multiple of four).
[[nodiscard]] inline std::size_t block_plane_bytes(CacheDType format,
                                                   std::int64_t kv_heads,
                                                   std::int64_t head_dim) noexcept {
  const std::int64_t pitch = storage_width(format, head_dim);
  if (pitch <= 0 || kv_heads <= 0) return 0;
  return dtype_storage_bytes(storage_dtype(format),
                             static_cast<std::size_t>(kv_heads * kBlockSize * pitch));
}

// K and V of one token in one layer, before any paging granularity.
[[nodiscard]] inline std::size_t token_bytes(CacheDType format,
                                             std::int64_t kv_heads,
                                             std::int64_t head_dim) noexcept {
  return 2 * block_plane_bytes(format, kv_heads, head_dim) /
         static_cast<std::size_t>(kBlockSize);
}

// The pool's block count once every row holds `tokens`, for a pool that may
// reach `capacity` tokens per row. The rung rule ops::paged_pool_blocks uses,
// minus its LSE_KV_PREALLOC diagnostic, which pins the pool at the ceiling.
[[nodiscard]] inline std::int32_t pool_blocks(std::int32_t tokens,
                                              std::int32_t capacity,
                                              std::int32_t rows) noexcept {
  if (rows <= 0) return 0;
  const std::int32_t ceiling = blocks_for(capacity, kBlockSize) * rows;
  return pool_rung(blocks_for(tokens, kBlockSize) * rows, ceiling);
}

// Contiguous storage (the HIP path): both planes are whole pools at the rung.
[[nodiscard]] inline std::size_t contiguous_layer_bytes(
    CacheDType format, std::int64_t kv_heads, std::int64_t head_dim,
    std::int32_t tokens, std::int32_t capacity, std::int32_t rows) noexcept {
  return 2 * static_cast<std::size_t>(pool_blocks(tokens, capacity, rows)) *
         block_plane_bytes(format, kv_heads, head_dim);
}

// While a contiguous pool moves to its next rung, the old and the new pool of
// that layer are both live. Layers grow one at a time, so the transient is one
// layer's previous pool. Zero when no growth is pending at `tokens`.
[[nodiscard]] inline std::size_t contiguous_growth_bytes(
    CacheDType format, std::int64_t kv_heads, std::int64_t head_dim,
    std::int32_t tokens, std::int32_t capacity, std::int32_t rows) noexcept {
  if (tokens <= kBlockSize) return 0;
  const std::int32_t now = pool_blocks(tokens, capacity, rows);
  const std::int32_t before = pool_blocks(tokens - kBlockSize, capacity, rows);
  return now == before ? 0
                       : 2 * static_cast<std::size_t>(before) *
                             block_plane_bytes(format, kv_heads, head_dim);
}

// Fragmented storage (the Loom path): each plane is backed by whole 256 KiB
// fragments covering the blocks actually handed out. A fresh pool hands blocks
// out in ascending order, so the resident count is the blocks in use; the
// allocator starts it at one block.
[[nodiscard]] inline std::size_t plane_fragments(CacheDType format,
                                                 std::int64_t kv_heads,
                                                 std::int64_t head_dim,
                                                 std::int32_t resident_blocks) noexcept {
  const std::size_t bytes = static_cast<std::size_t>(resident_blocks < 1 ? 1 : resident_blocks) *
                            block_plane_bytes(format, kv_heads, head_dim);
  return (bytes + kFragmentBytes - 1) / kFragmentBytes;
}

[[nodiscard]] inline std::size_t fragmented_layer_fragments(
    CacheDType format, std::int64_t kv_heads, std::int64_t head_dim,
    std::int32_t tokens, std::int32_t rows) noexcept {
  const std::int32_t resident = blocks_for(tokens, kBlockSize) * (rows > 0 ? rows : 1);
  return 2 * plane_fragments(format, kv_heads, head_dim, resident);
}

// Fragments come out of 256 MiB arenas shared by every layer on a device, so
// the device-side reservation is the fragment total rounded up to arenas.
[[nodiscard]] inline std::size_t arena_bytes(std::size_t fragments) noexcept {
  constexpr std::size_t per_arena = kArenaBytes / kFragmentBytes;
  return (fragments + per_arena - 1) / per_arena * kArenaBytes;
}

// The per-plane device table of fragment addresses, sized for the whole
// capacity when the plane is first made (FragmentStorage::reserve).
[[nodiscard]] inline std::size_t fragment_table_bytes(CacheDType format,
                                                      std::int64_t kv_heads,
                                                      std::int64_t head_dim,
                                                      std::int32_t capacity,
                                                      std::int32_t rows) noexcept {
  const std::size_t ceiling =
      static_cast<std::size_t>(blocks_for(capacity, kBlockSize)) *
      static_cast<std::size_t>(rows > 0 ? rows : 1);
  const std::size_t bytes = ceiling * block_plane_bytes(format, kv_heads, head_dim);
  return (bytes + kFragmentBytes - 1) / kFragmentBytes * sizeof(std::uint64_t);
}

// The [rows, stride] f32 block-id table one layer uploads. Its stride follows
// the pool rung, bounded by the per-row block limit.
[[nodiscard]] inline std::size_t block_table_bytes(std::int32_t tokens,
                                                   std::int32_t capacity,
                                                   std::int32_t rows) noexcept {
  const std::int32_t limit = blocks_for(capacity, kBlockSize);
  const std::int32_t pool = pool_blocks(tokens, capacity, rows);
  const std::int32_t stride = pool < limit ? pool : limit;
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(stride) * sizeof(float);
}

}  // namespace lse::kv
