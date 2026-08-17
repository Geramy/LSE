// The block pool: a free list with a refcount per block.
//
// Refcounts are not speculative generality. Prefix sharing — two sequences that
// begin with the same system prompt pointing at the same blocks — is the single
// largest KV win available after paging itself, and it is nearly free once a
// block can be held by more than one table. Building the free list without them
// means rewriting it later, which is the same mistake as building residency
// around one contiguous buffer.
#pragma once

#include <cstdint>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/kv/block.hpp"

namespace lse::kv {

class BlockAllocator {
 public:
  BlockAllocator() = default;
  explicit BlockAllocator(std::int32_t blocks) { reset(blocks); }

  // Drops every holder and re-fills the free list at `blocks`.
  void reset(std::int32_t blocks);

  [[nodiscard]] std::int32_t total() const noexcept {
    return static_cast<std::int32_t>(refs_.size());
  }
  [[nodiscard]] std::int32_t free_count() const noexcept {
    return static_cast<std::int32_t>(free_.size());
  }
  [[nodiscard]] std::int32_t used() const noexcept {
    return total() - free_count();
  }

  // Adds blocks at the end. Existing ids and refcounts keep their meaning, so
  // the bytes a caller already wrote stay addressable — which is what makes
  // growing the device pool a copy of the used prefix rather than a re-prefill.
  Status grow(std::int32_t blocks);

  // A block with refcount 1. kOutOfMemory when the pool is dry, naming the
  // pool size: an exhausted pool is a scheduling decision to make (preempt,
  // swap), not a condition to paper over.
  Result<BlockId> acquire();

  Status retain(BlockId b);
  // True when the block went back to the free list, i.e. this was the last
  // holder.
  Result<bool> release(BlockId b);

  [[nodiscard]] Result<std::int32_t> refcount(BlockId b) const;

  // Tops `table` up until it covers `tokens`, acquiring only what is missing.
  // Leaves the table untouched and returns the allocator unchanged when it
  // cannot cover the request, so a failed admission does not strand blocks.
  Status cover(BlockTable& table, std::int32_t tokens);

  // Releases every block in `table` and empties it.
  Status release_all(BlockTable& table);

 private:
  std::vector<std::uint32_t> refs_;
  // Freed blocks come back off the end: a block just released is the warmest
  // one in cache, and nothing here depends on ids being handed out in order.
  std::vector<BlockId> free_;
};

}  // namespace lse::kv
