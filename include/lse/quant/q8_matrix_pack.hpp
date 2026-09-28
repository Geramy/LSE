#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::quant {

struct Q8MatrixPack {
  std::uint32_t columns = 0;
  std::uint32_t features = 0;
  std::uint32_t padded_columns = 0;
  std::vector<std::uint32_t> words;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint16_t> biases;
};

[[nodiscard]] Result<Q8MatrixPack>
pack_q8_matrix(std::uint32_t columns, std::uint32_t features,
               std::span<const std::byte> words,
               std::span<const std::byte> scales,
               std::span<const std::byte> biases);

// Reserves future original-weight allocations, not the runtime/KV working set.
[[nodiscard]] bool
q8_packed_memory_admitted(std::optional<std::size_t> free_bytes,
                          std::span<const std::size_t> remaining_tensor_bytes,
                          std::size_t slab_bytes, std::size_t packed_bytes);

} // namespace lse::quant
