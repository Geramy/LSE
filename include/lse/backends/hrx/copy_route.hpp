#pragma once
#include "lse/backend/backend.hpp"
namespace lse::backend::detail {
// Residency is the non-recycled token Backend::init assigns to this instance,
// stamped by Backend::allocate. A shared backend family or equal foreign tokens
// do not authorize use of this instance's queues. On a single physical device,
// valid member indices name local streams; a spanning device needs peer policy.
inline bool own_single_device_copy(DeviceIndex owner, unsigned physical_count,
                                   std::size_t stream_count,
                                   const DeviceBuffer &src,
                                   const DeviceBuffer &dst) noexcept {
  return owner.bound() && physical_count == 1 && src.residency == owner &&
         dst.residency == owner && src.member < stream_count &&
         dst.member < stream_count;
}
} // namespace lse::backend::detail
