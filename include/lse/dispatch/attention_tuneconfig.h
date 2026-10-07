#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "lse/dispatch/attention.hpp"

namespace lse::dispatch::attention_shapes {

inline constexpr std::uint32_t kFlashThreads = 256, kFlashKeyWindow = 256;
inline constexpr std::uint32_t kFlashQueryTile = 16;
inline constexpr std::uint32_t kFlashPrefillSelectorThreads = 128;
static_assert(kFlashPrefillSelectorThreads % 32 == 0 &&
              kFlashPrefillSelectorThreads >= 32 && kFlashPrefillSelectorThreads <= 1024);
inline constexpr std::uint32_t kSplitRecord = 258, kShortKeyWindow = 128;

struct FlashWmmaRule {
  std::string_view arch;
  std::uint32_t wave, threads, min_rows, max_head_dim;
};
// The device rows of these rules live in the per-architecture headers
// (lse/dispatch/arch/<gfx>.hpp) and are selected by dispatch::arch::tuning.

struct FlashCacheRule {
  std::string_view arch;
  std::uint32_t min_rows, min_capacity;
};

// Query tile, scores and softmax state, plus the smallest value-staging block
// (16 keys of value_dim + 8 halves) when value_dim is given.
[[nodiscard]] constexpr std::uint64_t flash_wmma_lds_bytes(std::uint32_t head_dim,
                                                           std::uint32_t value_dim) {
  const auto padded = (static_cast<std::uint64_t>(head_dim) + 15u) / 16u * 16u;
  const std::uint64_t staging =
      value_dim == 0 ? 0 : 16u * (static_cast<std::uint64_t>(value_dim) + 8u) * 2u;
  return kFlashQueryTile * (padded * 2u + kFlashKeyWindow * 4u + 3u * 4u) + staging;
}

struct DecodeRule {
  std::string_view arch;
  std::uint32_t wave, threads, query_rows, head_dim;
  std::uint32_t max_block;
};

struct DecodeHeadRule {
  std::uint32_t queries_per_kv, head_tile;
};
inline constexpr std::array kDecodeHeadRules{
    DecodeHeadRule{6, 6},
};

[[nodiscard]] constexpr std::uint32_t decode_head_tile(std::uint32_t heads,
                                                      std::uint32_t kvheads) noexcept {
  if (kvheads == 0 || heads % kvheads != 0) return 1;
  for (const auto& rule : kDecodeHeadRules)
    if (heads / kvheads == rule.queries_per_kv) return rule.head_tile;
  return 1;
}

struct SplitShortRule {
  std::string_view arch;
  std::uint32_t wave, min_rows, max_rows, threads, head_dim;
};

struct ShortTileRule {
  std::uint32_t min_rows, max_rows, query_tile;
};
inline constexpr std::array kShortTileRules{
    ShortTileRule{4, 8, 4},
};

[[nodiscard]] constexpr std::uint64_t split_partitions(std::uint64_t capacity) noexcept {
  return capacity / kShortKeyWindow + (capacity % kShortKeyWindow != 0);
}

[[nodiscard]] constexpr std::uint64_t split_merge_lds_bytes(std::uint64_t parts) noexcept {
  return (parts * sizeof(float) + 15u) / 16u * 16u;
}

struct ShortHeadRule {
  std::uint32_t queries_per_kv, query_tile, head_tile;
};
inline constexpr std::array kShortHeadRules{
    ShortHeadRule{6, 4, 2},
};

[[nodiscard]] constexpr std::uint32_t short_rule_query_tile(std::uint32_t rows) noexcept {
  for (const auto& rule : kShortTileRules)
    if (rows >= rule.min_rows && rows <= rule.max_rows) return rule.query_tile;
  return 1;
}

[[nodiscard]] constexpr std::uint32_t short_rule_head_tile(std::uint32_t heads,
    std::uint32_t kvheads, std::uint32_t query_tile) noexcept {
  if (kvheads == 0 || heads % kvheads != 0 || query_tile == 1) return 1;
  for (const auto& rule : kShortHeadRules)
    if (heads / kvheads == rule.queries_per_kv && query_tile == rule.query_tile)
      return rule.head_tile;
  return 1;
}

// A tile of query rows and heads reads each key window once instead of once
// per row, which is most of the untiled kernel's memory traffic, but leaves
// query_tile * head_tile times fewer workgroups. Tile while the workgroups that
// are certain to have keys still cover every compute unit: a KV rung doubles
// as it grows, so a pass at `capacity` holds more than half of it.
[[nodiscard]] constexpr std::uint32_t short_query_tile(std::uint32_t rows,
    std::uint32_t heads, std::uint32_t kvheads, std::uint32_t capacity,
    std::uint32_t compute_units) noexcept {
  const auto tile = short_rule_query_tile(rows);
  if (tile == 1 || compute_units == 0 || heads == 0) return 1;
  const auto head_tile = short_rule_head_tile(heads, kvheads, tile);
  const auto live_parts = static_cast<std::uint64_t>(capacity / 2u) / kShortKeyWindow;
  const auto tiled = live_parts * (heads / head_tile) * ((rows + tile - 1u) / tile);
  return tiled >= compute_units ? tile : 1u;
}

[[nodiscard]] constexpr std::uint32_t short_head_tile(std::uint32_t rows,
    std::uint32_t heads, std::uint32_t kvheads, std::uint32_t capacity,
    std::uint32_t compute_units) noexcept {
  return short_rule_head_tile(heads, kvheads,
      short_query_tile(rows, heads, kvheads, capacity, compute_units));
}

[[nodiscard]] constexpr bool short_skips_empty_partitions(std::uint32_t rows,
    std::uint32_t heads, std::uint32_t kvheads, std::uint32_t capacity,
    std::uint32_t compute_units) noexcept {
  return short_query_tile(rows, heads, kvheads, capacity, compute_units) > 1;
}

struct ShortDefaultRule {
  std::uint32_t batch, query_heads, key_heads, block;
  std::int64_t min_keys, min_offset;
  std::int32_t mask, window;
};
inline constexpr std::array kShortDefaults{
    ShortDefaultRule{1, 24, 4, 16, 1024, 512, 1, 0},
};

struct WaveL2Rule {
  std::string_view arch;
  std::uint32_t wave, threads, batch, heads, rows, width;
};

}  // namespace lse::dispatch::attention_shapes
