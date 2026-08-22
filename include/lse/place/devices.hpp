// The devices this process holds: their lifetime, their identity, and which of
// them holds which bytes.
//
// probe::qualify_pool MEASURES a pool; this OWNS one. That is the whole
// difference and it is why this layer exists: a probe::PoolMember is a
// description that borrows a backend, a Member here *is* the backend, and every
// buffer stamped with its DeviceIndex is released by it. Nothing in probe can
// hold a device open, route an allocation to one, or say which of them a
// buffer belongs to.
//
// This is the topmost layer in the build for the same reason: it depends on
// probe (to measure), on graph (to hand the set to a scheduler) and on the
// backend registry (to bind), and nothing depends on it but the entry point.
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/place/residency.hpp"
#include "lse/probe/pool.hpp"
#include "lse/probe/profile.hpp"

namespace lse::place {

// One device this process has bound.
struct Member {
  probe::DeviceId id;             // "hrx:0" — how everything above addresses it
  backend::DeviceIndex index{};   // what its buffers carry
  backend::IBackend* backend = nullptr;
  // What enumeration said about this device before it was bound, including its
  // peer row. Default-constructed when the backend has no enumerator, in which
  // case every fact in it reads kUnknown — which is the truth.
  backend::DeviceDescriptor descriptor;
};

// Devices a run may use, backend-qualified and in preference order:
// "hrx:0,cpu:0". Whitespace around an entry is ignored. Empty is not an error
// and not a set — see Devices::open.
[[nodiscard]] Result<std::vector<probe::DeviceId>> parse_selector(
    std::string_view text);

class Devices final : public backend::IDeviceSet {
 public:
  // Binds every device the selector names, in the order named; the first is the
  // primary. A named device that will not come up is an error and never a
  // silent omission: a set that quietly lost a member would place work by a
  // plan nobody agreed to.
  //
  // An EMPTY selector means "whatever comes up", which is the one-device answer
  // and the default: it walks default_backend_order() and takes the first
  // backend that initializes, at LSE_DEVICE's ordinal. Widening the default to
  // every device present would measure hardware the run will not use, and
  // PLAN.md's rule that `auto` must not mean "use everything" is the same rule
  // one layer down.
  [[nodiscard]] static Result<std::unique_ptr<Devices>> open(
      std::string_view selector);
  ~Devices() override;

  Devices(const Devices&) = delete;
  Devices& operator=(const Devices&) = delete;

  std::size_t size() const noexcept override;
  std::optional<backend::Stream> stream_for(
      std::size_t member) const noexcept override;
  backend::IBackend& device(std::size_t i) const override;
  std::size_t primary() const noexcept override;
  std::size_t member_of(backend::DeviceIndex d) const noexcept override;
  Status may_read(backend::DeviceIndex held,
                  std::size_t target) const override;

  [[nodiscard]] std::span<const Member> members() const noexcept;
  [[nodiscard]] const Member* find(const probe::DeviceId& id) const noexcept;

  // Backends that were tried and refused while an empty selector looked for
  // one, joined by "; ". Empty when nothing was declined. A caller reports it;
  // falling back from a code-generating backend to a host one is a
  // two-order-of-magnitude cliff and it must not be silent.
  [[nodiscard]] std::string_view declined() const noexcept;

  // How work on `target` reaches bytes resident on `held`. Derived only from
  // what was queried (the backend's peer row) or measured (the pool's link
  // matrix); a pair nothing answered for is kUnknown and stays that way.
  [[nodiscard]] Reach reach(backend::DeviceIndex held,
                            std::size_t target) const noexcept;

  // --- buffer ownership ---------------------------------------------------
  // Routing an allocation to a member, and a release back to whoever owns it.
  // The part probe does not have: a buffer knows its device, so freeing it is
  // no longer the caller's job to remember.

  [[nodiscard]] Result<backend::DeviceBuffer> allocate(
      std::size_t member, std::size_t bytes,
      backend::MemoryClass cls = backend::MemoryClass::kDevice);
  // Releases through the member that allocated it. Refuses rather than leaking
  // silently when the buffer names a device this set does not hold — the caller
  // asked the wrong set, and guessing which member owns it is how a live
  // allocation gets freed under someone else.
  [[nodiscard]] Status deallocate(backend::DeviceBuffer& buf);

  // --- qualification ------------------------------------------------------

  // The members as probe addresses them, ranks in member order.
  [[nodiscard]] std::vector<probe::PoolMember> pool_members() const;
  // Measures every member and the paths between them, once. The profile is
  // persisted under the pool's fingerprint, so a normal start reads it back.
  [[nodiscard]] Status qualify(const probe::PoolOptions& options = {});
  // Empty until qualify() succeeded.
  [[nodiscard]] const probe::PoolProfile& profile() const noexcept;

 private:
  Devices();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Opens the process's set on an explicit selector — what a --pool flag passes.
//
// Must be called before anything asks for a device. Once the set is open it is
// open: re-selecting would leave live buffers stamped with devices the new set
// does not hold, so a late call refuses instead of quietly running on the other
// one. An empty selector here still defers to LSE_POOL.
Status open_default_devices(std::string_view selector);

// The set the engine runs on, opened on first use from LSE_POOL when nothing
// opened it explicitly. Null when nothing came up at all.
//
// This is also what graph::default_scheduler() binds: this layer registers
// itself as the factory at static-initialization time, so a binary that links it
// gets the whole set and one that does not still gets a single device.
[[nodiscard]] Devices* default_devices();

}  // namespace lse::place
