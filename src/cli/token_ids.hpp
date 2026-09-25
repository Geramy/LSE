#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <span>

namespace lse::cli {

// Generator::generate returns emitted IDs; a sampled stop token is excluded.
// Serialize existing vectors only after generation, leaving callbacks untouched.
inline bool print_token_ids(FILE* file, std::span<const std::uint32_t> prompt,
                            std::span<const std::uint32_t> generated) {
  if (std::fputs("[token-ids] {\"prompt_ids\":[", file) < 0) return false;
  const auto write = [file](std::span<const std::uint32_t> ids) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (std::fprintf(file, "%s%" PRIu32, i == 0 ? "" : ",", ids[i]) < 0)
        return false;
    }
    return true;
  };
  return write(prompt) &&
         std::fputs("],\"generated_ids\":[", file) >= 0 && write(generated) &&
         std::fputs("],\"stop_tokens_excluded\":true}\n", file) >= 0 &&
         std::fflush(file) == 0;
}

}  // namespace lse::cli
