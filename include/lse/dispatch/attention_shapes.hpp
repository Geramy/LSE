#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "lse/dispatch/attention.hpp"

namespace lse::dispatch::attention_shapes {

inline constexpr std::uint32_t kFlashThreads = 256, kFlashKeyWindow = 256;
inline constexpr std::uint32_t kFlashDefaultQueryTile = 8, kFlashPrefillQueryTile = 12;
inline constexpr std::uint32_t kSplitRecord = 258, kShortKeyWindow = 128;

struct FlashRule {
  std::string_view arch;
  std::uint32_t min_rows, tile_rows;
  AttentionPlan plan;
  bool reuse_keys;
};
inline constexpr std::array kFlashRules{
    FlashRule{"gfx1201", kFlashPrefillQueryTile, kFlashPrefillQueryTile, AttentionPlan::kFlash12, true},
    FlashRule{{}, 2, kFlashDefaultQueryTile, AttentionPlan::kFlash8, false},
};

[[nodiscard]] constexpr bool flash_reuses_keys(std::uint32_t tile_rows) noexcept {
  for (const auto& rule : kFlashRules)
    if (rule.tile_rows == tile_rows) return rule.reuse_keys;
  return false;
}

struct DecodeRule {
  std::string_view arch;
  std::uint32_t wave, threads, query_rows, head_dim;
  std::uint32_t max_block, max_keys, merge_parts;
  std::int64_t split_min_offset, split_max_keys;
};
inline constexpr std::array kDecodeRules{
    DecodeRule{"gfx1201", 32, 256, 1, 256, 256, 8192, 64, 511, 4096},
};

struct SplitShortRule {
  std::string_view arch;
  std::uint32_t wave, min_rows, max_rows, threads, head_dim;

  // Each merge lane initializes one partition weight.
  [[nodiscard]] constexpr std::uint32_t max_keys() const {
    return kShortKeyWindow * threads;
  }
};
inline constexpr std::array kSplitShortRules{
    SplitShortRule{"gfx1201", 32, 2, 8, 128, 256},
};

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
