#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "lse/dispatch/attention.hpp"

namespace lse::dispatch::attention_shapes {

inline constexpr std::uint32_t kFlashThreads = 256, kFlashKeyWindow = 256;
inline constexpr std::uint32_t kFlashQueryTile = 16;
inline constexpr std::uint32_t kSplitRecord = 258, kShortKeyWindow = 128;

struct FlashWmmaRule {
  std::string_view arch;
  std::uint32_t wave, threads, min_rows, max_head_dim;
};
inline constexpr std::array kFlashWmmaRules{
    FlashWmmaRule{"gfx1201", 32, 256, 2, 512},
};

struct FlashCacheRule {
  std::string_view arch;
  std::uint32_t min_rows, min_capacity;
};
inline constexpr std::array kFlashCacheRules{
    FlashCacheRule{"gfx1201", 1024, 65536},
};

[[nodiscard]] constexpr bool flash_retain_far_cache(std::string_view arch,
    std::uint32_t rows, std::uint64_t capacity) noexcept {
  for (const auto& rule : kFlashCacheRules)
    if (arch == rule.arch && rows >= rule.min_rows && capacity >= rule.min_capacity)
      return true;
  return false;
}

[[nodiscard]] constexpr std::uint64_t flash_wmma_lds_bytes(std::uint32_t head_dim) {
  const auto padded = (static_cast<std::uint64_t>(head_dim) + 15u) / 16u * 16u;
  return kFlashQueryTile * (padded * 2u + kFlashKeyWindow * 4u + 3u * 4u);
}

struct DecodeRule {
  std::string_view arch;
  std::uint32_t wave, threads, query_rows, head_dim;
  std::uint32_t max_block;
};
inline constexpr std::array kDecodeRules{
    DecodeRule{"gfx1201", 32, 128, 1, 256, 256},
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
inline constexpr std::array kSplitShortRules{
    SplitShortRule{"gfx1201", 32, 2, 8, 128, 256},
};

struct ShortTileRule {
  std::uint32_t min_rows, max_rows, min_capacity, query_tile;
};
inline constexpr std::array kShortTileRules{
    ShortTileRule{4, 8, 8192, 4},
};

[[nodiscard]] constexpr std::uint64_t split_partitions(std::uint64_t capacity) noexcept {
  return capacity / kShortKeyWindow + (capacity % kShortKeyWindow != 0);
}

[[nodiscard]] constexpr std::uint64_t split_merge_lds_bytes(std::uint64_t parts) noexcept {
  return (parts * sizeof(float) + 15u) / 16u * 16u;
}

[[nodiscard]] constexpr std::uint32_t short_query_tile(std::uint32_t rows,
                                                      std::uint32_t capacity) noexcept {
  for (const auto& rule : kShortTileRules)
    if (rows >= rule.min_rows && rows <= rule.max_rows && capacity >= rule.min_capacity)
      return rule.query_tile;
  return 1;
}

[[nodiscard]] constexpr bool short_skips_empty_partitions(std::uint32_t rows,
                                                         std::uint32_t capacity) noexcept {
  return short_query_tile(rows, capacity) > 1;
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
inline constexpr std::array kWaveL2Rules{
    WaveL2Rule{"gfx1201", 32, 128, 1, 1, 16, 128},
};

}  // namespace lse::dispatch::attention_shapes
