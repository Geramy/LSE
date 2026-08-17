// What to do when the block pool cannot serve everyone: admit, preempt, swap.
//
// Decisions only. The policy sees block counts and ages and returns a plan; the
// caller owns the pool, the device buffers and the copies, because a decision
// that depends on nothing but numbers is the half that can be tested without a
// device — and the half that is wrong in every engine that gets this wrong.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lse/kv/allocator.hpp"
#include "lse/kv/block.hpp"

namespace lse::kv {

// One live sequence as the policy sees it.
struct SequenceDemand {
  std::string id;
  // Blocks the sequence already holds.
  std::int32_t held = 0;
  // Absolute tokens the sequence will have written once the pending step lands.
  std::int32_t tokens_after = 0;
  // Step counter at its last use. Smaller is older; ties break on id so the
  // plan is deterministic.
  std::uint64_t last_used = 0;
};

enum class Verdict : std::uint8_t {
  // The pool can serve it now.
  kAdmit,
  // It fits only after the named sequences give their blocks back.
  kPreempt,
  // It does not fit even with the pool empty. Not a transient condition.
  kRefuse,
};

// A preempted sequence either loses its blocks (and re-prefills when it comes
// back) or has them copied to host memory first. Dropping is free now and costs
// a re-prefill later; swapping costs the copy both ways. The rule is the one
// vLLM converged on: swap while there is host room, drop when there is not.
enum class Eviction : std::uint8_t { kDrop, kSwapOut };

struct Preemption {
  std::string id;
  Eviction how = Eviction::kDrop;
  std::int32_t blocks = 0;
};

struct Admission {
  Verdict verdict = Verdict::kAdmit;
  // Blocks the sequence must acquire for the pending step.
  std::int32_t blocks_needed = 0;
  // Oldest first. Empty unless the verdict is kPreempt.
  std::vector<Preemption> preempt;
  // Why, in the words the error or the log should use.
  std::string reason;
};

class BlockPolicy {
 public:
  // `reserve` blocks are held back from admission so a sequence already
  // decoding can always append its next block: a pool run to exactly zero
  // deadlocks on the sequence that is mid-token.
  //
  // `host_blocks` is the swap space, in blocks. Zero means no swap space, and
  // then every preemption is a drop — said once here rather than discovered as
  // a silent behaviour change.
  BlockPolicy(std::int32_t block_size, std::int32_t reserve,
              std::int32_t host_blocks)
      : block_size_(block_size), reserve_(reserve), host_blocks_(host_blocks) {}

  [[nodiscard]] std::int32_t block_size() const noexcept { return block_size_; }
  [[nodiscard]] std::int32_t reserve() const noexcept { return reserve_; }
  [[nodiscard]] std::int32_t host_blocks() const noexcept { return host_blocks_; }

  // Blocks `want` still has to acquire to cover tokens_after.
  [[nodiscard]] std::int32_t shortfall(const SequenceDemand& want) const noexcept;

  // `resident` is every live sequence including `want` itself, which is skipped
  // as a preemption candidate: a sequence never evicts itself to make room for
  // itself.
  [[nodiscard]] Admission admit(const SequenceDemand& want,
                                const BlockAllocator& pool,
                                std::span<const SequenceDemand> resident) const;

 private:
  std::int32_t block_size_ = kBlockSize;
  std::int32_t reserve_ = 0;
  std::int32_t host_blocks_ = 0;
};

}  // namespace lse::kv
