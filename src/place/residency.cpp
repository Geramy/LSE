#include "lse/place/residency.hpp"

namespace lse::place {

Reach reach_of(backend::PeerAccess access) noexcept {
  switch (access) {
    case backend::PeerAccess::kSelf:
      return Reach::kSame;
    case backend::PeerAccess::kYes:
    case backend::PeerAccess::kOnRequest:
      return Reach::kPeer;
    case backend::PeerAccess::kNo:
      return Reach::kNo;
    case backend::PeerAccess::kUnknown:
      break;
  }
  return Reach::kUnknown;
}

Reach reach_of(probe::PathKind path) noexcept {
  switch (path) {
    case probe::PathKind::kSameDevice:
      return Reach::kSame;
    case probe::PathKind::kPeerDirect:
      return Reach::kPeer;
    case probe::PathKind::kHostStaged:
    case probe::PathKind::kRdmaDirect:
    case probe::PathKind::kRdmaStaged:
      return Reach::kStaged;
    case probe::PathKind::kUnknown:
      break;
  }
  return Reach::kUnknown;
}

}  // namespace lse::place
