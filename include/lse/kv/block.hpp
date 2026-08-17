// Paged attention state: the geometry of a KV block and the per-sequence list
// of blocks that holds one sequence's keys and values.
//
// A paged cache is a buffer whose bytes are a LIST OF BLOCKS reached through a
// table the kernel reads, not one contiguous span per sequence. That is the
// difference between Orca and vLLM: contiguous max-capacity buffers reserve
// `capacity` tokens for every sequence and throw away whatever the sequence
// does not reach, which on this APU comes out of the same system RAM the
// weights need. It is also what lets residency be per-block later — a
// contiguous "one buffer, one device" cache would have to be rewritten for it.
//
// Depends on core only, on purpose: nothing here knows about a device, a graph
// or a session, so the allocation policy is unit-testable with no backend.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::kv {

using BlockId = std::uint32_t;

// Not a valid index into any pool. Distinct from block 0, which is a real
// block, so an unfilled table row reads as "nothing here" rather than as the
// first sequence's first block.
inline constexpr BlockId kNoBlock = 0xffffffffu;

// Tokens per block.
//
// Must stay a power of two: the SDPA and KV-write kernels turn `pos / kBlockSize`
// and `pos % kBlockSize` into a shift and a mask, and a non-power-of-two would
// put an integer division in the innermost address chain of the one loop that
// cannot afford it.
//
// 16 is vLLM's default and it is the right trade here for the same reason: the
// block is the quantum of waste (a sequence wastes at most 15 tokens of KV) and
// also the reuse distance of the block-table load, which the kernel hoists out
// of 16 key dots.
inline constexpr std::int32_t kBlockSize = 16;
inline constexpr std::int32_t kBlockShift = 4;

static_assert(kBlockSize == (1 << kBlockShift), "kBlockSize must be 1<<kBlockShift");

// A device pool's block count is a dimension of the tensors the JIT keys on, so
// sizing it to exactly what a sequence needs would make every context length
// its own set of kernels. Pools therefore come in rungs — powers of two — and a
// pool grows by moving to the next rung, which bounds the attention shapes the
// engine can ever compile at log2(capacity / block_size) instead of one per
// length. Same shape of answer as the prefill chunk ladder.
inline constexpr std::int32_t kMinPoolBlocks = 8;

// Smallest rung that holds `blocks`, never below kMinPoolBlocks and never above
// `ceiling` (which is the whole engine capacity in blocks and is itself a valid
// pool size).
[[nodiscard]] std::int32_t pool_rung(std::int32_t blocks,
                                     std::int32_t ceiling) noexcept;

// Blocks needed to hold `tokens`, rounded up.
[[nodiscard]] constexpr std::int32_t blocks_for(std::int32_t tokens,
                                                std::int32_t block_size) noexcept {
  if (tokens <= 0 || block_size <= 0) return 0;
  return (tokens + block_size - 1) / block_size;
}

// What one block of one attention layer holds. `head_dim` is the K width; a
// model whose V width differs would need two geometries, and none here does.
struct BlockGeometry {
  std::int32_t block_size = kBlockSize;
  std::int32_t kv_heads = 0;
  std::int32_t head_dim = 0;

  [[nodiscard]] bool valid() const noexcept {
    return block_size > 0 && kv_heads > 0 && head_dim > 0;
  }
  // One block is [kv_heads, block_size, head_dim] of one plane (K or V).
  [[nodiscard]] std::int64_t elems_per_block() const noexcept {
    return static_cast<std::int64_t>(kv_heads) * block_size * head_dim;
  }
  [[nodiscard]] std::size_t bytes_per_block(DType dt) const noexcept {
    return dtype_storage_bytes(dt,
                               static_cast<std::size_t>(elems_per_block()));
  }
};

// The ordered blocks holding one sequence, oldest first. Position p lives in
// block `blocks()[p / block_size]` at offset `p % block_size`.
//
// The table does not own its blocks: BlockAllocator does the refcounting, and
// two tables sharing a prefix is the whole point of that being refcounted.
class BlockTable {
 public:
  BlockTable() = default;
  explicit BlockTable(std::int32_t block_size) : block_size_(block_size) {}

  [[nodiscard]] std::int32_t block_size() const noexcept { return block_size_; }
  [[nodiscard]] std::int32_t size() const noexcept {
    return static_cast<std::int32_t>(blocks_.size());
  }
  [[nodiscard]] std::int32_t capacity_tokens() const noexcept {
    return size() * block_size_;
  }
  [[nodiscard]] std::span<const BlockId> blocks() const noexcept {
    return blocks_;
  }
  [[nodiscard]] bool empty() const noexcept { return blocks_.empty(); }

  void push(BlockId b) { blocks_.push_back(b); }
  void clear() noexcept { blocks_.clear(); }

  struct Slot {
    BlockId block = kNoBlock;
    std::int32_t offset = 0;
  };
  [[nodiscard]] Result<Slot> locate(std::int32_t token) const;

 private:
  std::int32_t block_size_ = kBlockSize;
  std::vector<BlockId> blocks_;
};

// Flattens block tables into the row-major [rows, stride] image the kernels
// read, as f32 — which is how every index travels in this engine's graph
// (graph::embedding, rope offsets, MoE routing all do the same).
//
// `stride` is baked into the generated address arithmetic, so it is fixed at
// the engine's maximum blocks-per-sequence and never varies with how many
// blocks a sequence currently holds.
//
// Padded rows and slots past a table's length take `pad`, which must be a real
// block: a padded batch row runs the identical code path, and a row that
// addressed kNoBlock would read out of the pool. Making pad rows
// indistinguishable to the kernel is the width-invariance rule, not an
// optimization — a `if (row < real_rows)` in the address arithmetic is exactly
// the defect shape that made every token read row 0.
Status write_table_rows(std::span<const BlockTable> tables, std::int32_t stride,
                        BlockId pad, std::span<float> out);

}  // namespace lse::kv
