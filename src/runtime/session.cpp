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
    total += array_bytes(s.key_cache);
    total += array_bytes(s.value_cache);
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
