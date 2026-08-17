#include "lse/runtime/session.hpp"

#include <algorithm>

namespace lse::runtime {

namespace {

std::size_t array_bytes(const graph::Array& a) noexcept {
  if (!a.valid()) return 0;
  return dtype_storage_bytes(a.dtype(), a.shape().elem_count());
}

}  // namespace

void Session::clear() {
  for (model::MixerState& s : states_) s = model::MixerState{};
  history_.clear();
  position_ = 0;
}

std::size_t Session::cache_bytes() const noexcept {
  std::size_t total = 0;
  for (const model::MixerState& s : states_) {
    total += array_bytes(s.gdn_state);
    // The conv tails are live device tensors too. Leaving them out under-reported
    // the cache by 0.26-5.6 MiB per session depending on the model, which is the
    // accounting a block budget would have inherited.
    total += array_bytes(s.gdn_conv_q);
    total += array_bytes(s.gdn_conv_k);
    total += array_bytes(s.gdn_conv_v);
    total += array_bytes(s.gdn_conv_qkv);
    // Paged: what the pools actually hold, which is a rung above the blocks in
    // use, not the engine capacity. This is the number paging moves.
    total += s.paged.pool_bytes();
    total += array_bytes(s.paged.table);
  }
  return total;
}

std::size_t Session::kv_blocks() const noexcept {
  std::size_t total = 0;
  for (const model::MixerState& s : states_) {
    for (const kv::BlockTable& t : s.paged.tables) {
      total += static_cast<std::size_t>(t.size());
    }
  }
  return total;
}

Session& SessionStore::get_or_create(const std::string& id) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    it = sessions_.emplace(id, std::make_unique<Session>(id, layers_)).first;
  }
  touch(id);
  return *it->second;
}

Session* SessionStore::find(const std::string& id) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) return nullptr;
  touch(id);
  return it->second.get();
}

void SessionStore::erase(const std::string& id) {
  sessions_.erase(id);
  lru_.erase(std::remove(lru_.begin(), lru_.end(), id), lru_.end());
}

std::size_t SessionStore::cache_bytes() const noexcept {
  std::size_t total = 0;
  for (const auto& [id, s] : sessions_) total += s->cache_bytes();
  return total;
}

void SessionStore::touch(const std::string& id) {
  lru_.erase(std::remove(lru_.begin(), lru_.end(), id), lru_.end());
  lru_.push_back(id);
}

void SessionStore::enforce_budget() {
  if (budget_bytes_ == 0) return;
  // Always keeps the most recent session, even alone over budget: evicting the
  // one being served would turn every step into a full re-prefill.
  while (cache_bytes() > budget_bytes_ && lru_.size() > 1) {
    const std::string victim = lru_.front();
    lru_.pop_front();
    sessions_.erase(victim);
  }
}

}  // namespace lse::runtime
