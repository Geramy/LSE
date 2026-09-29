#pragma once

#include <cstdint>

namespace lse::runtime {

struct PrefillBatch {
  std::uint32_t batch_size = 1024;
  std::uint32_t ubatch_size = 1024;

  [[nodiscard]] static constexpr bool valid_size(std::uint32_t size) noexcept {
    return size >= 128 && size <= 4096 && (size & (size - 1)) == 0;
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return valid_size(batch_size) && valid_size(ubatch_size) &&
           ubatch_size <= batch_size;
  }
};

}  // namespace lse::runtime
