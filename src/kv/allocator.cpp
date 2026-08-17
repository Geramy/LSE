#include "lse/kv/allocator.hpp"

namespace lse::kv {

namespace {

Status check_id(const std::vector<std::uint32_t>& refs, BlockId b) {
  if (b == kNoBlock || b >= refs.size()) {
    return LSE_ERROR(kOutOfRange, "block ", std::to_string(b),
                     " is outside a pool of ", std::to_string(refs.size()));
  }
  return OkStatus();
}

}  // namespace

void BlockAllocator::reset(std::int32_t blocks) {
  const auto n = blocks > 0 ? static_cast<std::size_t>(blocks) : 0;
  refs_.assign(n, 0);
  free_.clear();
  free_.reserve(n);
  // Descending, so the first acquire hands out block 0 and a fresh pool lays a
  // sequence down in ascending order. Nothing depends on it; it makes a dumped
  // block table readable.
  for (std::size_t i = n; i-- > 0;) free_.push_back(static_cast<BlockId>(i));
}

Status BlockAllocator::grow(std::int32_t blocks) {
  if (blocks < total()) {
    return LSE_ERROR(kInvalidArgument, "cannot shrink a block pool from ",
                     std::to_string(total()), " to ", std::to_string(blocks));
  }
  const auto want = static_cast<std::size_t>(blocks);
  const std::size_t was = refs_.size();
  refs_.resize(want, 0);
  free_.reserve(free_.size() + (want - was));
  for (std::size_t i = want; i-- > was;) free_.push_back(static_cast<BlockId>(i));
  return OkStatus();
}

Result<BlockId> BlockAllocator::acquire() {
  if (free_.empty()) {
    return LSE_ERROR(kOutOfMemory, "KV block pool exhausted: all ",
                     std::to_string(total()), " blocks are held");
  }
  const BlockId b = free_.back();
  free_.pop_back();
  refs_[b] = 1;
  return b;
}

Status BlockAllocator::retain(BlockId b) {
  LSE_RETURN_IF_ERROR(check_id(refs_, b));
  if (refs_[b] == 0) {
    return LSE_ERROR(kInvalidArgument, "cannot retain free block ",
                     std::to_string(b));
  }
  ++refs_[b];
  return OkStatus();
}

Result<bool> BlockAllocator::release(BlockId b) {
  LSE_RETURN_IF_ERROR(check_id(refs_, b));
  if (refs_[b] == 0) {
    return LSE_ERROR(kInvalidArgument, "double release of block ",
                     std::to_string(b));
  }
  if (--refs_[b] != 0) return false;
  free_.push_back(b);
  return true;
}

Result<std::int32_t> BlockAllocator::refcount(BlockId b) const {
  LSE_RETURN_IF_ERROR(check_id(refs_, b));
  return static_cast<std::int32_t>(refs_[b]);
}

Status BlockAllocator::cover(BlockTable& table, std::int32_t tokens) {
  const std::int32_t want = blocks_for(tokens, table.block_size());
  const std::int32_t have = table.size();
  if (want <= have) return OkStatus();
  const std::int32_t need = want - have;
  if (need > free_count()) {
    return LSE_ERROR(kOutOfMemory, "covering ", std::to_string(tokens),
                     " tokens needs ", std::to_string(need),
                     " more block(s); ", std::to_string(free_count()),
                     " of ", std::to_string(total()), " are free");
  }
  for (std::int32_t i = 0; i < need; ++i) {
    // Checked against free_count() above, so this cannot fail; the assignment
    // still goes through Result so a future change cannot make it silent.
    LSE_ASSIGN_OR(const BlockId b, acquire());
    table.push(b);
  }
  return OkStatus();
}

Status BlockAllocator::release_all(BlockTable& table) {
  for (BlockId b : table.blocks()) {
    LSE_ASSIGN_OR(const bool freed, release(b));
    (void)freed;
  }
  table.clear();
  return OkStatus();
}

}  // namespace lse::kv
