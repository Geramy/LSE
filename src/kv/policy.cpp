#include "lse/kv/policy.hpp"

#include <algorithm>
#include <string>

namespace lse::kv {

std::int32_t BlockPolicy::shortfall(const SequenceDemand& want) const noexcept {
  const std::int32_t need = blocks_for(want.tokens_after, block_size_);
  return need > want.held ? need - want.held : 0;
}

Admission BlockPolicy::admit(const SequenceDemand& want,
                             const BlockAllocator& pool,
                             std::span<const SequenceDemand> resident) const {
  Admission out;
  out.blocks_needed = shortfall(want);
  if (out.blocks_needed == 0) {
    out.reason = "already covered";
    return out;
  }

  // A sequence appending its next block is not subject to the reserve: the
  // reserve exists to keep exactly that append possible. Anything that has not
  // started yet must leave it intact.
  const bool resuming = want.held > 0;
  // Negative when the reserve is already larger than what is free. Kept signed
  // rather than clamped: the comparisons below must see that the pool is over
  // its own reserve, not that it has nothing spare.
  const std::int32_t usable =
      resuming ? pool.free_count() : pool.free_count() - reserve_;
  const std::string spare =
      std::to_string(pool.free_count()) + " free" +
      (resuming ? "" : ", " + std::to_string(reserve_) + " reserved");
  if (out.blocks_needed <= usable) {
    out.reason = "fits in " + std::to_string(pool.free_count()) + " free block(s)";
    return out;
  }

  // Oldest first, and never the sequence being admitted.
  std::vector<const SequenceDemand*> victims;
  victims.reserve(resident.size());
  for (const SequenceDemand& s : resident) {
    if (s.id == want.id || s.held == 0) continue;
    victims.push_back(&s);
  }
  std::sort(victims.begin(), victims.end(),
            [](const SequenceDemand* a, const SequenceDemand* b) {
              if (a->last_used != b->last_used) return a->last_used < b->last_used;
              return a->id < b->id;
            });

  std::int32_t reclaimable = 0;
  for (const SequenceDemand* s : victims) reclaimable += s->held;
  if (out.blocks_needed > usable + reclaimable) {
    out.verdict = Verdict::kRefuse;
    out.reason = "needs " + std::to_string(out.blocks_needed) +
                 " block(s); " + spare + " and " +
                 std::to_string(reclaimable) + " reclaimable across " +
                 std::to_string(victims.size()) + " sequence(s)";
    return out;
  }

  out.verdict = Verdict::kPreempt;
  std::int32_t got = usable > 0 ? usable : 0;
  std::int32_t host_left = host_blocks_;
  for (const SequenceDemand* s : victims) {
    if (got >= out.blocks_needed) break;
    Preemption p;
    p.id = s->id;
    p.blocks = s->held;
    // Swap while there is host room. A dropped sequence pays a full re-prefill
    // of tokens_after tokens when it returns; a swapped one pays the copy out
    // and back. With no host room the copy has nowhere to land, so it drops.
    if (host_left >= s->held) {
      p.how = Eviction::kSwapOut;
      host_left -= s->held;
    } else {
      p.how = Eviction::kDrop;
    }
    got += s->held;
    out.preempt.push_back(std::move(p));
  }
  out.reason = "needs " + std::to_string(out.blocks_needed) + " block(s); " +
               spare + ", preempting " +
               std::to_string(out.preempt.size()) + " sequence(s)";
  return out;
}

}  // namespace lse::kv
