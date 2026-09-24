#pragma once

#include <array>
#include <cstddef>
#include <mutex>

namespace lse::backend::hrx_kernels {

// Reserve quarantine capacity before allocating or issuing work. A failed
// drain does not prove that the GPU stopped referencing a buffer: retain its
// owner until process exit and disable subsequent probes. The registry itself
// intentionally has process lifetime, so static destruction cannot free an
// allocation still in use by a failed device. Concurrent reservations are
// bounded; callers must decline before allocating when no slot is available.
template <class Buffer, std::size_t Capacity = 16>
class ProbeRetirement {
  struct Slot {
    bool used = false;
    std::array<Buffer, 2> buffers{};
    std::size_t count = 0;
  };
  struct Registry {
    std::mutex mutex;
    bool failed = false;
    std::array<Slot, Capacity> slots{};
  };
  static Registry& registry() {
    static Registry* instance = new Registry;
    return *instance;
  }
 public:
  ProbeRetirement() {
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    if (r.failed) return;
    for (auto& slot : r.slots) {
      if (!slot.used) {
        slot.used = true;
        slot_ = &slot;
        break;
      }
    }
  }
  ProbeRetirement(const ProbeRetirement&) = delete;
  ProbeRetirement& operator=(const ProbeRetirement&) = delete;
  ~ProbeRetirement() {
    if (slot_ == nullptr) return;
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    if (slot_->count == 0) slot_->used = false;
    else r.failed = true;  // Unwinding without a confirmed drain retains owners.
  }
  bool available() const { return slot_ != nullptr; }
  bool track(const Buffer& buffer) {
    if (slot_ == nullptr || slot_->count == slot_->buffers.size()) return false;
    slot_->buffers[slot_->count++] = buffer;
    return true;
  }
  template <class Drain, class Release>
  bool finish(Drain&& drain, Release&& release) {
    if (slot_ == nullptr) return false;
    // Always drain, including after a failed later launch/copy. Its failure
    // may have left earlier successful submissions in flight.
    if (!drain()) {
      auto& r = registry();
      std::lock_guard lock(r.mutex);
      r.failed = true;
      slot_ = nullptr;  // Registry retains this occupied slot permanently.
      return false;
    }
    for (std::size_t i = 0; i < slot_->count; ++i) release(slot_->buffers[i]);
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    *slot_ = {};
    slot_ = nullptr;
    return true;
  }
 private:
  Slot* slot_ = nullptr;
};

}  // namespace lse::backend::hrx_kernels
