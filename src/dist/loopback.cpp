#include "lse/dist/loopback.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "lse/dist/adapter.hpp"

namespace lse::dist {

namespace {

// Measured once per process: an in-process transport's "link" is a memcpy, so
// its bandwidth is the machine's, not a constant someone typed. The cost model
// reads this the same way it would read an RDMA NIC's rate.
std::uint64_t measured_copy_bandwidth() {
  static const std::uint64_t rate = [] {
    constexpr std::size_t kBytes = 8u << 20;
    std::vector<std::byte> a(kBytes), b(kBytes);
    std::memset(a.data(), 1, kBytes);
    const auto t0 = std::chrono::steady_clock::now();
    int reps = 0;
    for (; reps < 8; ++reps) std::memcpy(b.data(), a.data(), kBytes);
    const auto dt = std::chrono::steady_clock::now() - t0;
    const double ns =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
    if (ns <= 0.0) return std::uint64_t{1};
    return static_cast<std::uint64_t>(static_cast<double>(kBytes) *
                                      static_cast<double>(reps) * 1e9 / ns);
  }();
  return rate;
}

struct MailboxKey {
  Rank dst;
  Rank src;
  int tag;
  friend auto operator<=>(const MailboxKey&, const MailboxKey&) = default;
};

}  // namespace

struct LoopbackTransport::Group {
  std::mutex mu;
  std::condition_variable cv;
  Rank world = 1;
  int joined = 0;
  std::map<MailboxKey, std::deque<std::vector<std::byte>>> mailboxes;
};

namespace {

struct GroupRegistry {
  std::mutex mu;
  std::map<std::string, std::weak_ptr<LoopbackTransport::Group>> groups;
};

GroupRegistry& registry() {
  static GroupRegistry r;
  return r;
}

std::shared_ptr<LoopbackTransport::Group> join_group(const std::string& id,
                                                     Rank world) {
  GroupRegistry& r = registry();
  std::lock_guard lock(r.mu);
  auto it = r.groups.find(id);
  if (it != r.groups.end()) {
    if (auto live = it->second.lock()) return live;
  }
  auto fresh = std::make_shared<LoopbackTransport::Group>();
  fresh->world = world;
  r.groups[id] = fresh;
  return fresh;
}

// A CommBuffer's payload as bytes. A device-resident buffer on this transport
// is a pointer handoff only when the backend maps it host-visible; there is no
// residency model yet to say when that is true, so it is refused rather than
// read through a pointer nothing promised.
Result<std::span<const std::byte>> payload(const CommBuffer& b) {
  if (b.on_device()) {
    return LSE_ERROR(kUnimplemented,
                     "loopback moves host payloads; a device-resident buffer "
                     "needs the placement seam to say who may read it");
  }
  if (b.host == nullptr) {
    return LSE_ERROR(kInvalidArgument, "loopback got a null buffer");
  }
  return std::span<const std::byte>(
      static_cast<const std::byte*>(b.host) + b.offset, b.bytes);
}

}  // namespace

LoopbackTransport::LoopbackTransport() = default;
LoopbackTransport::~LoopbackTransport() { disconnect_impl(); }

Status LoopbackTransport::connect_impl(const TransportConfig& cfg) {
  if (cfg.world_size <= 0 || cfg.rank < 0 || cfg.rank >= cfg.world_size) {
    return LSE_ERROR(kInvalidArgument, "loopback rank ",
                     std::to_string(cfg.rank), " is not in [0, ",
                     std::to_string(cfg.world_size), ")");
  }
  rank_ = cfg.rank;
  world_ = cfg.world_size;
  timeout_ms_ = cfg.timeout_ms;
  group_ = join_group(cfg.group_id.empty() ? std::string("default")
                                           : cfg.group_id,
                      world_);
  {
    std::lock_guard lock(group_->mu);
    if (group_->world != world_) {
      return LSE_ERROR(kInvalidArgument, "loopback group '", cfg.group_id,
                       "' was formed with world size ",
                       std::to_string(group_->world));
    }
    ++group_->joined;
  }

  caps_ = Capabilities{};
  caps_.native_collectives = false;
  // Same process, same device: a buffer handed to another rank crosses no
  // fabric and needs no bounce.
  caps_.device_memory_direct = true;
  caps_.reliable = true;
  caps_.ordered = true;
  caps_.full_duplex = true;
  caps_.asynchronous = true;
  caps_.bandwidth_bytes_per_s = measured_copy_bandwidth();
  // A condition-variable handoff, measured on this class of machine. Small
  // enough that the latency-bound threshold lands in the low tens of KB.
  caps_.latency_ns = 2000;
  caps_.max_message_bytes = 0;
  return OkStatus();
}

void LoopbackTransport::disconnect_impl() noexcept {
  if (!group_) return;
  {
    std::lock_guard lock(group_->mu);
    --group_->joined;
  }
  group_->cv.notify_all();
  group_.reset();
}

Result<CommHandle> LoopbackTransport::send_impl(const CommBuffer& buf, Rank dst,
                                               int tag) {
  if (!group_) return LSE_ERROR(kInternal, "loopback transport not connected");
  if (dst < 0 || dst >= world_) {
    return LSE_ERROR(kOutOfRange, "loopback send to rank ",
                     std::to_string(dst));
  }
  LSE_ASSIGN_OR(auto bytes, payload(buf));
  {
    std::lock_guard lock(group_->mu);
    group_->mailboxes[MailboxKey{dst, rank_, tag}].emplace_back(bytes.begin(),
                                                                bytes.end());
  }
  group_->cv.notify_all();
  // Copy-on-send: the message is already delivered, so the handle is only a
  // receipt. A transport with real asynchrony returns a pending one here.
  return CommHandle{1};
}

Result<CommHandle> LoopbackTransport::recv_impl(CommBuffer& buf, Rank src,
                                                int tag) {
  if (!group_) return LSE_ERROR(kInternal, "loopback transport not connected");
  if (src < 0 || src >= world_) {
    return LSE_ERROR(kOutOfRange, "loopback recv from rank ",
                     std::to_string(src));
  }
  if (buf.on_device() || buf.host == nullptr) {
    return LSE_ERROR(kUnimplemented,
                     "loopback delivers into host memory only");
  }
  const MailboxKey key{rank_, src, tag};
  std::vector<std::byte> msg;
  {
    std::unique_lock lock(group_->mu);
    const bool arrived = group_->cv.wait_for(
        lock, std::chrono::milliseconds(timeout_ms_), [&] {
          auto it = group_->mailboxes.find(key);
          return it != group_->mailboxes.end() && !it->second.empty();
        });
    if (!arrived) {
      return LSE_ERROR(kCancelled, "loopback recv from rank ",
                       std::to_string(src), " tag ", std::to_string(tag),
                       " timed out");
    }
    auto& q = group_->mailboxes[key];
    msg = std::move(q.front());
    q.pop_front();
  }
  if (msg.size() != buf.bytes) {
    return LSE_ERROR(kInvalidArgument, "loopback message is ",
                     std::to_string(msg.size()), " B, receiver expects ",
                     std::to_string(buf.bytes));
  }
  std::memcpy(static_cast<std::byte*>(buf.host) + buf.offset, msg.data(),
              msg.size());
  return CommHandle{1};
}

Status LoopbackTransport::wait_impl(CommHandle h) {
  return h.valid() ? OkStatus()
                   : LSE_ERROR(kInvalidArgument, "invalid loopback handle");
}

Status LoopbackTransport::fence_impl() { return OkStatus(); }

LSE_REGISTER_TRANSPORT("loopback", LoopbackTransport)

}  // namespace lse::dist
