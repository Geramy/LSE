// A conversation's live state, owned separately from whoever is decoding it.
//
// A session is the unit that holds cache: per-layer MixerState (a paged K/V
// block pool plus this sequence's block lists on attention layers, a fixed
// recurrent matrix on GDN layers), the token history the repetition penalty
// reads, and the absolute position. Keeping it out of the generator is what lets
// a server interleave conversations and resume one without replaying its prompt.
//
// The KV is paged: a sequence holds a list of kv::kBlockSize-token blocks in a
// pool sized to the rung above what it uses, not a contiguous span at the engine
// capacity. Sequence length and the batch row count are dispatch values the
// kernels read, so one code object serves every length and every batch up to a
// bucket.
//
// Still not batched, and that is the remaining half: sessions decode one at a
// time. What is missing is the driver — admitting and preempting sequences
// between steps and running one ragged attention over them — not the cache.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/kv/block.hpp"
#include "lse/model/hybrid_lm.hpp"

namespace lse::runtime {

class Session {
 public:
  Session(std::string id, std::size_t layers)
      : id_(std::move(id)), states_(layers) {}

  [[nodiscard]] const std::string& id() const noexcept { return id_; }
  [[nodiscard]] std::vector<model::MixerState>& states() noexcept { return states_; }
  [[nodiscard]] std::vector<std::uint32_t>& history() noexcept { return history_; }
  [[nodiscard]] const std::vector<std::uint32_t>& history() const noexcept {
    return history_;
  }

  // Tokens the cache already covers. The next forward pass starts here.
  [[nodiscard]] std::int32_t position() const noexcept { return position_; }
  void advance(std::int32_t tokens) noexcept { position_ += tokens; }

  // Drops the cache but keeps the id, so the next turn is a fresh prefill.
  void clear();

  // A fresh sequence on the SAME arrays: the state tensors keep their nodes
  // and buffers — zeroed on device — the paged rows are released, and every
  // position returns to zero. clear() drops the arrays and forces the next
  // pass to rebuild the graph; this keeps the graph replayable, which is what
  // lets a server reuse one retained program across requests instead of
  // paying partition and emit per request.
  Status restart();

  // What the cache currently costs: the block pools actually allocated plus the
  // GDN state and its conv tails. Attention pools sit at a rung above the blocks
  // in use, not at the engine capacity.
  [[nodiscard]] std::size_t cache_bytes() const noexcept;

  // Blocks this session holds, summed over layers. The unit a budget and a
  // preemption decision are counted in; see kv::BlockPolicy.
  [[nodiscard]] std::size_t kv_blocks() const noexcept;

 private:
  std::string id_;
  std::vector<model::MixerState> states_;
  std::vector<std::uint32_t> history_;
  std::int32_t position_ = 0;
};

// Sessions by id, evicted least-recently-used once the total cache exceeds the
// budget. A server holds one of these; a single-shot CLI run does not need it.
//
// Deliberately not a batching scheduler: sessions here are independent and are
// decoded one at a time. The paged pool and the runtime batch extent are landed
// (see kv/ and model::batch_bucket); what continuous batching still needs is the
// policy layer on top — admitting and preempting sequences between steps with
// kv::BlockPolicy, and stepping the admitted set together.
class SessionStore {
 public:
  SessionStore(std::size_t layers, std::size_t budget_bytes)
      : layers_(layers), budget_bytes_(budget_bytes) {}

  // Creates the session if `id` is new. The pointer stays valid until the
  // session is evicted or erased.
  Session& get_or_create(const std::string& id);

  [[nodiscard]] Session* find(const std::string& id);
  void erase(const std::string& id);

  [[nodiscard]] std::size_t size() const noexcept { return sessions_.size(); }
  [[nodiscard]] std::size_t cache_bytes() const noexcept;

  // Evicts least-recently-used sessions until the budget is met. Called after
  // every decode step; safe to call directly in a test.
  void enforce_budget();

 private:
  void touch(const std::string& id);

  std::size_t layers_;
  std::size_t budget_bytes_;
  std::unordered_map<std::string, std::unique_ptr<Session>> sessions_;
  // Most recently used at the back.
  std::deque<std::string> lru_;
};

}  // namespace lse::runtime
