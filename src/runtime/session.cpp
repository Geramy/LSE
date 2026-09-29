#include "lse/runtime/session.hpp"

#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/model/mtp.hpp"
#include "lse/ops/attention.hpp"

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
  mtp_tail_ = {};
  mtp_owner_ = nullptr;
  mtp_revision_ = 0;
  mtp_position_ = 0;
  position_ = 0;
}

namespace {

// Whole-array device zero: the sibling of BatchScheduler's zero_device_row,
// for a state that is being restarted rather than one row of a batch.
Status zero_device_array(graph::Array& a) {
  if (!a.valid() || !a.node()) return OkStatus();
  graph::Node& n = *a.node();
  if (!n.buffer.valid()) return OkStatus();  // never written: still zeros
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no backend to clear a state");
  }
  const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
  if (bytes == 0) return OkStatus();
  const std::vector<std::byte> zeros(bytes, std::byte{0});
  LSE_RETURN_IF_ERROR(
      sched->backend().copy({n.buffer}, zeros.data(), bytes));
  n.materialized = true;
  n.device_dirty = true;
  n.host_dirty = false;
  return OkStatus();
}

}  // namespace

Status Session::restart() {
  // Zeroing the recurrence is a host write into buffers the previous pass
  // may still be reading on a host-coherent device; see Scheduler::drain.
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    LSE_RETURN_IF_ERROR(sched->drain());
  }
  for (model::MixerState& s : states_) {
    LSE_RETURN_IF_ERROR(zero_device_array(s.gdn_state));
    LSE_RETURN_IF_ERROR(zero_device_array(s.gdn_conv_q));
    LSE_RETURN_IF_ERROR(zero_device_array(s.gdn_conv_k));
    LSE_RETURN_IF_ERROR(zero_device_array(s.gdn_conv_v));
    LSE_RETURN_IF_ERROR(zero_device_array(s.gdn_conv_qkv));
    // Releasing every row makes the stale KV unreachable — the bytes stay,
    // the table stops naming them — and position zero is what makes the next
    // pass a prefill.
    for (std::size_t r = 0; r < s.paged.tables.size(); ++r) {
      LSE_RETURN_IF_ERROR(
          ops::release_row(s.paged, static_cast<std::int32_t>(r)));
    }
    s.position = 0;
  }
  history_.clear();
  mtp_tail_ = {};
  mtp_owner_ = nullptr;
  mtp_revision_ = 0;
  mtp_position_ = 0;
  position_ = 0;
  return OkStatus();
}

Status Session::retain_mtp_tail(const graph::Array& hidden,
                                std::int32_t position,
                                const model::MtpModule& module) {
  if (!hidden.valid() || hidden.dtype() != DType::kF32 ||
      hidden.shape().rank() != 3 || hidden.shape().dim(0) != 1 ||
      hidden.shape().dim(1) != 1 || hidden.shape().dim(2) != module.config().hidden_size ||
      position <= 0 || module.position() != position) {
    return LSE_ERROR(kInvalidArgument, "invalid resident MTP target tail");
  }
  graph::Array held = hidden;
  LSE_RETURN_IF_ERROR(held.materialize());
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend for MTP target tail");
  graph::Node& src = *hidden.node();
  const auto member = sched->devices().member_of(src.buffer.residency);
  if (!src.buffer.valid() || member >= sched->devices().size())
    return LSE_ERROR(kInvalidArgument, "MTP target tail has no resident storage");
  auto& owner = sched->devices().device(member);
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(src, owner));
  const auto bytes = dtype_storage_bytes(hidden.dtype(), hidden.shape().elem_count());
  if (!mtp_tail_.valid() || mtp_tail_.shape() != hidden.shape() ||
      mtp_tail_.node()->buffer.residency != src.buffer.residency) {
    LSE_ASSIGN_OR(auto buffer, owner.allocate(bytes, backend::MemoryClass::kDevice));
    mtp_tail_ = graph::Array::from_buffer(std::move(buffer), hidden.shape(), hidden.dtype());
  }
  LSE_RETURN_IF_ERROR(owner.copy({mtp_tail_.node()->buffer}, {src.buffer}, bytes));
  mtp_tail_.node()->host_dirty = false;
  mtp_tail_.node()->device_dirty = mtp_tail_.node()->buffer.ptr == nullptr;
  mtp_tail_.node()->materialized = true;
  mtp_owner_ = &module;
  mtp_revision_ = module.revision();
  mtp_position_ = position;
  return OkStatus();
}

bool Session::mtp_context_matches(const model::MtpModule& module) const noexcept {
  return mtp_tail_.valid() && mtp_owner_ == &module &&
         mtp_revision_ == module.revision() && mtp_position_ == position_ &&
         module.position() == position_;
}

std::size_t Session::cache_bytes() const noexcept {
  std::size_t total = array_bytes(mtp_tail_);
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
