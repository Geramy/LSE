#include "lse/dist/peer.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>


namespace lse::dist {
namespace {

struct MailboxKey {
  Rank dst = 0;
  Rank src = 0;
  int tag = 0;

  friend bool operator<(const MailboxKey& a, const MailboxKey& b) noexcept {
    return std::tie(a.dst, a.src, a.tag) < std::tie(b.dst, b.src, b.tag);
  }
};


// What this rank's link actually does, timed on the copy the transport will
// make. Measured rather than declared: the same code runs on a pool wired for
// peer-direct and on one that quietly falls back to host staging, and the cost
// model has to be able to tell those apart -- an order of magnitude separates
// them, and it decides whether a compressed collective is worth its codec.
struct LinkRate {
  std::uint64_t bytes_per_s = 1;
  std::uint64_t latency_ns = 0;
};

LinkRate measure_link(backend::IDeviceSet& set, std::size_t member) {
  constexpr std::size_t kBytes = 4u << 20;
  LinkRate out;
  if (set.size() < 2) return out;
  const std::size_t peer = member == 0 ? 1 : 0;
  auto src = set.device(member).allocate(kBytes, backend::MemoryClass::kDevice);
  auto dst = set.device(peer).allocate(kBytes, backend::MemoryClass::kDevice);
  if (!src.ok() || !dst.ok()) return out;
  backend::DeviceBuffer a = src.release();
  backend::DeviceBuffer b = dst.release();
  backend::IBackend& from = set.device(member);
  // One untimed pass: the first touch of a fresh mapping faults pages in and
  // reads several times slower than the steady state, and a cold number here
  // would send the cost model to the wrong collective for the whole run.
  if (!from.copy_peer(a, b, kBytes, 0, 0).ok()) return out;
  const auto t0 = std::chrono::steady_clock::now();
  int reps = 0;
  for (; reps < 4; ++reps) {
    if (!from.copy_peer(a, b, kBytes, 0, 0).ok()) return out;
  }
  const auto dt = std::chrono::steady_clock::now() - t0;
  const double ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
  if (ns > 0.0) {
    out.bytes_per_s = static_cast<std::uint64_t>(
        static_cast<double>(kBytes) * static_cast<double>(reps) * 1e9 / ns);
  }
  // The intercept, from the smallest transfer the link will carry. Warmed
  // first: the bandwidth loop above leaves multi-megabyte transfers in flight,
  // and timing the next small copy against that backlog measured 905 us for a
  // 64 B move -- two orders above the truth, and enough to talk the collective
  // selector out of every algorithm whose cost is latency-bound.
  for (int i = 0; i < 4; ++i) {
    if (!from.copy_peer(a, b, 64, 0, 0).ok()) return out;
  }
  constexpr int kPings = 64;
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < kPings; ++i) {
    if (!from.copy_peer(a, b, 64, 0, 0).ok()) return out;
  }
  const auto dt1 = std::chrono::steady_clock::now() - t1;
  out.latency_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(dt1).count() /
      kPings);
  return out;
}

}  // namespace

// A posted send, waiting for the matching recv to pull it. The sender's buffer
// is borrowed, not copied: the whole point of this transport is that the bytes
// move once, straight across the link, so the source has to stay put until the
// receiver reports the transfer done.
struct PeerTransport::Posted {
  const backend::DeviceBuffer* device = nullptr;
  const void* host = nullptr;
  std::size_t offset = 0;
  std::size_t bytes = 0;
  std::size_t member = 0;
  bool done = false;
  Status result = OkStatus();
};

struct PeerTransport::Group {
  std::mutex mu;
  std::condition_variable cv;
  int joined = 0;
  std::map<MailboxKey, std::deque<std::shared_ptr<PeerTransport::Posted>>> mailboxes;
};

namespace {

struct GroupRegistry {
  std::mutex mu;
  std::map<std::string, std::weak_ptr<PeerTransport::Group>> groups;
};

GroupRegistry& registry() {
  static GroupRegistry r;
  return r;
}

std::shared_ptr<PeerTransport::Group> join(const std::string& id) {
  GroupRegistry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.groups.find(id);
  if (it != r.groups.end()) {
    if (auto live = it->second.lock()) return live;
  }
  auto fresh = std::make_shared<PeerTransport::Group>();
  r.groups[id] = fresh;
  return fresh;
}

}  // namespace

PeerTransport::PeerTransport() = default;
PeerTransport::~PeerTransport() { disconnect_impl(); }

void PeerTransport::bind(backend::IDeviceSet& set, std::size_t member) noexcept {
  set_ = &set;
  member_ = member;
}

Status PeerTransport::connect_impl(const TransportConfig& cfg) {
  if (set_ == nullptr) {
    return LSE_ERROR(kInvalidArgument,
                     "peer transport was not bound to a device set");
  }
  if (member_ >= set_->size()) {
    return LSE_ERROR(kOutOfRange, "peer rank is bound to member ",
                     std::to_string(member_), " of a ",
                     std::to_string(set_->size()), "-device set");
  }
  rank_ = cfg.rank;
  world_ = cfg.world_size;
  timeout_ms_ = cfg.timeout_ms;
  group_ = join(cfg.group_id.empty() ? std::string("peer") : cfg.group_id);
  {
    std::lock_guard lock(group_->mu);
    ++group_->joined;
  }
  group_->cv.notify_all();

  caps_.native_collectives = false;
  // Measured on the path this transport actually takes, so a pool that ends up
  // host-staging reports what it got and the cost model plans for that instead.
  caps_.device_memory_direct = true;
  caps_.reliable = true;
  caps_.ordered = true;
  caps_.full_duplex = true;
  caps_.asynchronous = true;
  const LinkRate rate = measure_link(*set_, member_);
  caps_.bandwidth_bytes_per_s = rate.bytes_per_s;
  caps_.latency_ns = rate.latency_ns;
  caps_.max_message_bytes = 0;
  return OkStatus();
}

void PeerTransport::disconnect_impl() noexcept {
  if (!group_) return;
  {
    std::lock_guard lock(group_->mu);
    --group_->joined;
  }
  group_->cv.notify_all();
  group_.reset();
}

Result<CommHandle> PeerTransport::send_impl(const CommBuffer& buf, Rank dst,
                                            int tag) {
  if (!group_) return LSE_ERROR(kInternal, "peer transport not connected");
  if (dst < 0 || dst >= world_) {
    return LSE_ERROR(kOutOfRange, "peer send to rank ", std::to_string(dst));
  }
  if (!buf.on_device() && buf.host == nullptr) {
    return LSE_ERROR(kInvalidArgument, "peer send from an empty buffer");
  }
  auto posted = std::make_shared<PeerTransport::Posted>();
  posted->device = buf.device;
  posted->host = buf.host;
  posted->offset = buf.offset;
  posted->bytes = buf.bytes;
  posted->member = member_;
  {
    std::lock_guard lock(group_->mu);
    group_->mailboxes[MailboxKey{dst, rank_, tag}].push_back(posted);
    pending_.push_back(posted);
  }
  group_->cv.notify_all();
  return CommHandle{static_cast<std::uint64_t>(pending_.size())};
}

Result<CommHandle> PeerTransport::recv_impl(CommBuffer& buf, Rank src,
                                            int tag) {
  if (!group_) return LSE_ERROR(kInternal, "peer transport not connected");
  if (src < 0 || src >= world_) {
    return LSE_ERROR(kOutOfRange, "peer recv from rank ", std::to_string(src));
  }
  const MailboxKey key{rank_, src, tag};
  std::shared_ptr<PeerTransport::Posted> msg;
  {
    std::unique_lock lock(group_->mu);
    const bool arrived = group_->cv.wait_for(
        lock, std::chrono::milliseconds(timeout_ms_), [&] {
          auto it = group_->mailboxes.find(key);
          return it != group_->mailboxes.end() && !it->second.empty();
        });
    if (!arrived) {
      return LSE_ERROR(kCancelled, "peer recv from rank ", std::to_string(src),
                       " tag ", std::to_string(tag), " timed out");
    }
    auto& q = group_->mailboxes[key];
    msg = std::move(q.front());
    q.pop_front();
  }
  if (msg->bytes != buf.bytes) {
    return LSE_ERROR(kInvalidArgument, "peer message is ",
                     std::to_string(msg->bytes), " B, receiver expects ",
                     std::to_string(buf.bytes));
  }

  // The receiver drives the transfer because it is the only side holding both
  // ends. Device-to-device goes straight across the link; the mixed and
  // host-only cases fall back to the path that exists for them.
  Status moved = OkStatus();
  if (msg->device != nullptr && buf.on_device()) {
    moved = set_->device(msg->member)
                .copy_peer(*msg->device, *const_cast<backend::DeviceBuffer*>(
                                             buf.device),
                           buf.bytes, msg->offset, buf.offset);
  } else if (msg->device != nullptr) {
    moved = set_->device(msg->member)
                .copy_d2h(*msg->device,
                          static_cast<std::byte*>(buf.host) + buf.offset,
                          buf.bytes, msg->offset);
  } else if (buf.on_device()) {
    moved = set_->device(member_).copy_h2d(
        static_cast<const std::byte*>(msg->host) + msg->offset,
        *const_cast<backend::DeviceBuffer*>(buf.device), buf.bytes, buf.offset);
  } else {
    std::memcpy(static_cast<std::byte*>(buf.host) + buf.offset,
                static_cast<const std::byte*>(msg->host) + msg->offset,
                buf.bytes);
  }

  {
    std::lock_guard lock(group_->mu);
    msg->result = moved;
    msg->done = true;
  }
  group_->cv.notify_all();
  LSE_RETURN_IF_ERROR(moved);
  return CommHandle{1};
}

Status PeerTransport::wait_impl(CommHandle h) {
  if (!h.valid()) return LSE_ERROR(kInvalidArgument, "invalid peer handle");
  if (!group_) return OkStatus();
  const auto i = static_cast<std::size_t>(h.id) - 1;
  if (i >= pending_.size()) return OkStatus();
  std::shared_ptr<PeerTransport::Posted> posted = pending_[i];
  std::unique_lock lock(group_->mu);
  const bool landed = group_->cv.wait_for(
      lock, std::chrono::milliseconds(timeout_ms_), [&] { return posted->done; });
  if (!landed) {
    return LSE_ERROR(kCancelled, "peer send was never received");
  }
  return posted->result;
}

Status PeerTransport::fence_impl() {
  if (!group_) return OkStatus();
  for (const std::shared_ptr<PeerTransport::Posted>& p : pending_) {
    std::unique_lock lock(group_->mu);
    const bool landed = group_->cv.wait_for(
        lock, std::chrono::milliseconds(timeout_ms_), [&] { return p->done; });
    if (!landed) return LSE_ERROR(kCancelled, "peer fence timed out");
    LSE_RETURN_IF_ERROR(p->result);
  }
  pending_.clear();
  return OkStatus();
}

}  // namespace lse::dist
