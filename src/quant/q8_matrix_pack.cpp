#include "lse/quant/q8_matrix_pack.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace lse::quant {

Result<Q8MatrixPack> pack_q8_matrix(std::uint32_t columns,
                                    std::uint32_t features,
                                    std::span<const std::byte> words,
                                    std::span<const std::byte> scales,
                                    std::span<const std::byte> biases) {
  if (columns == 0 || columns > UINT32_MAX - 15u || features == 0 ||
      features % 64u != 0)
    return LSE_ERROR(kInvalidArgument, "invalid Q8 matrix extent");
  const auto padded = (columns + 15u) / 16u * 16u;
  const auto lanes = features / 4u;
  const auto groups = features / 64u;
  const auto weight_count = static_cast<std::uint64_t>(padded) * lanes;
  const auto affine_count = static_cast<std::uint64_t>(padded) * groups;
  if (weight_count > UINT32_MAX || affine_count > UINT32_MAX ||
      weight_count > std::numeric_limits<std::size_t>::max() / 4u ||
      affine_count > std::numeric_limits<std::size_t>::max() / 2u ||
      words.size() != static_cast<std::uint64_t>(columns) * lanes * 4u ||
      scales.size() != static_cast<std::uint64_t>(columns) * groups * 2u ||
      biases.size() != scales.size())
    return LSE_ERROR(kInvalidArgument, "invalid Q8 matrix storage");

  Q8MatrixPack out;
  out.columns = columns;
  out.features = features;
  out.padded_columns = padded;
  out.words.resize(static_cast<std::size_t>(weight_count));
  out.scales.resize(static_cast<std::size_t>(affine_count));
  out.biases.resize(out.scales.size());
  for (std::uint32_t column = 0; column < padded; ++column) {
    const auto source = column < columns ? column : 0u;
    const auto tile = column / 16u;
    const auto lane = column % 16u;
    for (std::uint32_t word = 0; word < lanes; ++word) {
      // N16 / K16 / fragment / half-wave / output lane.
      const auto dst = static_cast<std::size_t>(tile) * lanes * 16u +
                       (word / 4u) * 64u + (word % 2u) * 32u +
                       ((word / 2u) % 2u) * 16u + lane;
      const auto src = (static_cast<std::size_t>(source) * lanes + word) * 4u;
      std::memcpy(&out.words[dst], words.data() + src, 4u);
    }
    for (std::uint32_t group = 0; group < groups; ++group) {
      const auto dst =
          static_cast<std::size_t>(tile) * groups * 16u + group * 16u + lane;
      const auto src = (static_cast<std::size_t>(source) * groups + group) * 2u;
      std::memcpy(&out.scales[dst], scales.data() + src, 2u);
      std::memcpy(&out.biases[dst], biases.data() + src, 2u);
    }
  }
  return out;
}

bool q8_packed_memory_admitted(
    std::optional<std::size_t> free_bytes,
    std::span<const std::size_t> remaining_tensor_bytes, std::size_t slab_bytes,
    std::size_t packed_bytes) {
  constexpr std::size_t alignment = 4096;
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  if (!free_bytes || slab_bytes < alignment || slab_bytes % alignment != 0)
    return false;
  std::size_t regular = 0, oversized = 0, largest = 0;
  for (const auto bytes : remaining_tensor_bytes) {
    if (bytes > limit - (alignment - 1))
      return false;
    const auto rounded = (bytes + alignment - 1) & ~(alignment - 1);
    auto &sum = rounded > slab_bytes ? oversized : regular;
    if (rounded > limit - sum)
      return false;
    sum += rounded;
    if (rounded <= slab_bytes)
      largest = std::max(largest, rounded);
  }
  // Every nonfinal slab must consume at least this much before first-fit
  // requests another. Existing slab holes are deliberately not credited.
  std::size_t slabs = 0;
  if (regular != 0) {
    const auto usable = slab_bytes - largest + alignment;
    slabs = regular / usable + (regular % usable != 0);
  }
  if (slabs > (limit - oversized) / slab_bytes)
    return false;
  const auto reserve = oversized + slabs * slab_bytes;
  return reserve <= *free_bytes && packed_bytes <= *free_bytes - reserve;
}

} // namespace lse::quant
