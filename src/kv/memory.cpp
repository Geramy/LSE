#include "lse/kv/memory.hpp"

#include <algorithm>

namespace lse::kv {

struct MemoryManager::Lease {
  std::shared_ptr<MemoryManager> manager;
  std::size_t arena;
  std::uint32_t slot;
  ~Lease() { manager->release(arena, slot); }
};

Result<backend::DeviceBuffer> MemoryManager::acquire(
    backend::IBackend& backend, backend::Stream stream) {
  const std::lock_guard lock(mutex_);
  std::size_t selected = arenas_.size();
  std::size_t empty = arenas_.size();
  for (std::size_t i = 0; i < arenas_.size(); ++i) {
    const auto& arena = arenas_[i];
    if (!arena.buffer.valid() && empty == arenas_.size()) empty = i;
    if (arena.backend == &backend && arena.stream == stream &&
        arena.buffer.valid() && !arena.available.empty()) {
      selected = i;
      break;
    }
  }
  if (selected == arenas_.size()) {
    LSE_ASSIGN_OR(auto buffer, backend.allocate(kArenaBytes,
        backend::MemoryClass::kDevice, stream));
    Arena arena;
    arena.buffer = std::move(buffer);
    arena.backend = &backend;
    arena.stream = stream;
    for (std::uint32_t i = kArenaBytes / kFragmentBytes; i != 0; --i)
      arena.available.push_back(i - 1);
    if (empty < arenas_.size()) {
      arenas_[empty] = std::move(arena);
      selected = empty;
    } else {
      arenas_.push_back(std::move(arena));
      selected = arenas_.size() - 1;
    }
  }
  auto& arena = arenas_[selected];
  const auto slot = arena.available.back();
  arena.available.pop_back();
  ++arena.assigned;
  auto view = arena.buffer;
  view.offset += slot * kFragmentBytes;
  view.size_bytes = kFragmentBytes;
  view.storage = std::shared_ptr<Lease>(
      new Lease{shared_from_this(), selected, slot});
  return view;
}

void MemoryManager::release(std::size_t index, std::uint32_t slot) {
  const std::lock_guard lock(mutex_);
  auto& arena = arenas_[index];
  arena.available.push_back(slot);
  if (--arena.assigned == 0) {
    arena.backend->deallocate(arena.buffer);
    arena.available.clear();
  }
}

MemoryManager::Stats MemoryManager::stats() const {
  const std::lock_guard lock(mutex_);
  Stats result;
  for (const auto& arena : arenas_) {
    if (!arena.buffer.valid()) continue;
    result.reserved_bytes += kArenaBytes;
    result.assigned_bytes += arena.assigned * kFragmentBytes;
    ++result.arenas;
  }
  return result;
}

Status FragmentStorage::reserve(std::size_t bytes) {
  const auto count = bytes / kFragmentBytes + (bytes % kFragmentBytes != 0);
  if (count <= table_.size_bytes / sizeof(std::uint64_t)) return OkStatus();
  std::vector<std::uint64_t> addresses(count, 0);
  for (std::size_t i = 0; i < fragments_.size(); ++i) {
    LSE_ASSIGN_OR(auto pointer, backend_.device_pointer(fragments_[i]));
    addresses[i] = reinterpret_cast<std::uintptr_t>(pointer);
  }
  LSE_ASSIGN_OR(auto table, backend_.allocate(addresses.size() * sizeof(std::uint64_t),
      backend::MemoryClass::kDevice, stream_));
  // Ordered behind the launches already issued rather than after a wait for
  // them: nothing has read this table yet, and every reader is queued later.
  LSE_RETURN_IF_ERROR(backend_.write_ordered(table, addresses.data(), table.size_bytes, 0));
  table_ = std::move(table);
  return OkStatus();
}

Status FragmentStorage::grow(std::size_t bytes) {
  if (bytes <= size_bytes_) return OkStatus();
  const auto count = bytes / kFragmentBytes + (bytes % kFragmentBytes != 0);
  if (count == fragments_.size()) {
    size_bytes_ = bytes;
    return OkStatus();
  }
  LSE_RETURN_IF_ERROR(reserve(bytes));
  auto fragments = fragments_;
  while (fragments.size() < count) {
    LSE_ASSIGN_OR(auto fragment, manager_->acquire(backend_, stream_));
    fragments.push_back(std::move(fragment));
  }
  for (std::size_t i = fragments_.size(); i < fragments.size();) {
    auto span = fragments[i++];
    while (i < fragments.size() && fragments[i].handle == span.handle &&
           fragments[i].ptr == span.ptr &&
           fragments[i].offset == span.offset + span.size_bytes) {
      span.size_bytes += fragments[i++].size_bytes;
    }
    LSE_RETURN_IF_ERROR(backend_.zero_ordered(span, span.size_bytes, 0));
  }
  std::vector<std::uint64_t> addresses;
  addresses.reserve(count - fragments_.size());
  for (std::size_t i = fragments_.size(); i < fragments.size(); ++i) {
    LSE_ASSIGN_OR(auto pointer, backend_.device_pointer(fragments[i]));
    addresses.push_back(reinterpret_cast<std::uintptr_t>(pointer));
  }
  LSE_RETURN_IF_ERROR(backend_.write_ordered(table_, addresses.data(),
      addresses.size() * sizeof(std::uint64_t), fragments_.size() * sizeof(std::uint64_t)));
  fragments_ = std::move(fragments);
  size_bytes_ = bytes;
  return OkStatus();
}

Result<backend::DeviceBuffer> FragmentStorage::binding() {
  if (!table_.valid()) return LSE_ERROR(kInvalidArgument, "empty K/V fragment table");
  struct Lease {
    std::shared_ptr<FragmentStorage> storage;
    backend::DeviceBuffer table;
  };
  auto view = table_;
  view.storage = std::make_shared<Lease>(Lease{shared_from_this(), table_});
  return view;
}

Status FragmentStorage::read(void* dst, std::size_t bytes, std::size_t offset) const {
  if (offset > size_bytes_ || bytes > size_bytes_ - offset)
    return LSE_ERROR(kOutOfRange, "read exceeds K/V fragment storage");
  auto* out = static_cast<std::byte*>(dst);
  while (bytes) {
    const auto in_fragment = offset % kFragmentBytes;
    const auto n = std::min(bytes, kFragmentBytes - in_fragment);
    LSE_RETURN_IF_ERROR(backend_.copy_d2h(fragments_[offset / kFragmentBytes], out, n, in_fragment));
    out += n;
    offset += n;
    bytes -= n;
  }
  return OkStatus();
}

Status FragmentStorage::write(const void* src, std::size_t bytes, std::size_t offset) {
  if (offset > size_bytes_ || bytes > size_bytes_ - offset)
    return LSE_ERROR(kOutOfRange, "write exceeds K/V fragment storage");
  const auto* in = static_cast<const std::byte*>(src);
  while (bytes) {
    const auto in_fragment = offset % kFragmentBytes;
    const auto n = std::min(bytes, kFragmentBytes - in_fragment);
    LSE_RETURN_IF_ERROR(backend_.copy_h2d(in, fragments_[offset / kFragmentBytes], n, in_fragment));
    in += n;
    offset += n;
    bytes -= n;
  }
  return OkStatus();
}

}  // namespace lse::kv
