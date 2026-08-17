// Which device holds which bytes, and whether work on one device may read them.
//
// probe measures what moving bytes between two devices COSTS; this settles what
// is LEGAL. The two are not the same question and must not be traded against
// each other: an illegal read is not a slow read, it is the wrong answer. A
// kernel launched on device A against a buffer resident on B is accepted by
// every runtime here, runs, and produces plausible numbers — which is why the
// engine has to refuse it rather than price it.
//
// Not to be confused with graph/sharding.hpp's Placement (kReplicated /
// kSharded / kPartial). That is how a TENSOR is distributed; residency is which
// device holds a given buffer's bytes. A sharded tensor has one residency per
// shard.
#pragma once

#include <cstdint>

#include "lse/backend/backend.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/probe/profile.hpp"

namespace lse::place {

// How work on one device reaches bytes held by another.
//
// Five answers rather than a bool because they are five different situations
// and only one of them is "go ahead": a path that exists but has to be walked
// (kStaged) is not the same as one a kernel can dereference (kPeer), and
// "nothing here can tell" is not the same as "no".
#define LSE_REACH_LIST(X)                                                    \
  X(kUnknown, "unknown")     /* both are devices and nothing answers */      \
  X(kUnclaimed, "unclaimed") /* the bytes name no device at all */           \
  X(kSame, "same")           /* one device's own memory */                   \
  X(kPeer, "peer")           /* addressable without a host bounce */         \
  X(kStaged, "staged")       /* reachable only by copying through host */    \
  X(kNo, "no")               /* the runtime says never */

LSE_DECLARE_ENUM(Reach, std::uint8_t, LSE_REACH_LIST)

// Whether a kernel may load from bytes it reaches this way.
//
// kStaged is false: the bytes have to be moved first, and treating a copy path
// as a read is exactly how a kernel comes to dereference another device's
// memory through a pointer that happens to be mapped. kUnknown is false for the
// same reason every unmeasured term in this engine refuses — an unanswerable
// question about two devices must not pass by default.
[[nodiscard]] constexpr bool readable(Reach r) noexcept {
  return r == Reach::kUnclaimed || r == Reach::kSame || r == Reach::kPeer;
}

// What a backend's own peer query says, in these terms. kOnRequest is a real
// path that is not switched on yet, so it reads as kPeer for legality and the
// caller that acts on it is the one that has to grant access.
[[nodiscard]] Reach reach_of(backend::PeerAccess access) noexcept;

// What a measured link says. A link is about moving bytes, so nothing here
// answers kSame except the same-device entry, and every fabric path is staged
// as far as a kernel is concerned — a NIC that can DMA out of device memory
// still does not let a shader load from the other machine.
[[nodiscard]] Reach reach_of(probe::PathKind path) noexcept;

}  // namespace lse::place
