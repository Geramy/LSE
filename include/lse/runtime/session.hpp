// A conversation's live state, owned separately from whoever is decoding it.
//
// A session is the unit that holds cache: per-layer MixerState (a KV pair
// allocated at the engine length on attention layers, a fixed recurrent
// matrix on GDN layers), the token history the repetition penalty reads, and
// the absolute position. Keeping it out of the generator is what lets a
// server interleave conversations and resume one without replaying its prompt.
//
// Not batched. Every session decodes on its own; there is no shared step, no
// paged KV, no cross-sequence scheduler. See SessionStore's note on what that
// would take.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/core/status.hpp"
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

  // What the cache currently costs. Attention layers sit at the engine KV
  // length; GDN layers stay flat.
  [[nodiscard]] std::size_t cache_bytes() const noexcept;

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
// decoded one at a time. Continuous batching in the vLLM sense would replace
// the per-session contiguous KV tensors with a paged pool, admit and preempt
// sequences between steps, and run one ragged attention over all of them.
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
