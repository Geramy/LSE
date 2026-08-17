#include "lse/kv/block.hpp"

namespace lse::kv {

std::int32_t pool_rung(std::int32_t blocks, std::int32_t ceiling) noexcept {
  if (ceiling <= 0) return 0;
  if (blocks >= ceiling) return ceiling;
  std::int32_t rung = kMinPoolBlocks;
  while (rung < blocks && rung < ceiling) rung *= 2;
  return rung < ceiling ? rung : ceiling;
}

Result<BlockTable::Slot> BlockTable::locate(std::int32_t token) const {
  if (token < 0) {
    return LSE_ERROR(kInvalidArgument, "negative KV position ",
                     std::to_string(token));
  }
  const std::int32_t index = token / block_size_;
  if (index >= size()) {
    return LSE_ERROR(kOutOfRange, "KV position ", std::to_string(token),
                     " needs block ", std::to_string(index),
                     " but the sequence holds ", std::to_string(size()));
  }
  return Slot{blocks_[static_cast<std::size_t>(index)], token % block_size_};
}

Status write_table_rows(std::span<const BlockTable> tables, std::int32_t stride,
                        BlockId pad, std::span<float> out) {
  if (stride <= 0) {
    return LSE_ERROR(kInvalidArgument, "block table stride must be positive");
  }
  if (pad == kNoBlock) {
    return LSE_ERROR(kInvalidArgument,
                     "block table padding must be a real block");
  }
  if (out.size() % static_cast<std::size_t>(stride) != 0) {
    return LSE_ERROR(kInvalidArgument, "block table image of ",
                     std::to_string(out.size()),
                     " floats is not a whole number of rows of ",
                     std::to_string(stride));
  }
  const std::size_t rows = out.size() / static_cast<std::size_t>(stride);
  if (tables.size() > rows) {
    return LSE_ERROR(kOutOfRange, "block table image holds ",
                     std::to_string(rows), " rows, got ",
                     std::to_string(tables.size()), " sequences");
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const std::span<const BlockId> ids =
        r < tables.size() ? tables[r].blocks() : std::span<const BlockId>{};
    if (ids.size() > static_cast<std::size_t>(stride)) {
      return LSE_ERROR(kOutOfRange, "sequence ", std::to_string(r), " holds ",
                       std::to_string(ids.size()),
                       " blocks, the table stride is ", std::to_string(stride));
    }
    float* row = out.data() + r * static_cast<std::size_t>(stride);
    for (std::size_t i = 0; i < static_cast<std::size_t>(stride); ++i) {
      const BlockId id = i < ids.size() ? ids[i] : pad;
      row[i] = static_cast<float>(id);
    }
  }
  return OkStatus();
}

}  // namespace lse::kv
