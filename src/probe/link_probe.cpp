#include "lse/probe/link_probe.hpp"

#include <unistd.h>

#include <cstdio>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

#include "lse/probe/wire.hpp"

namespace lse::probe {

namespace {

using Clock = std::chrono::steady_clock;
using dist::CommBuffer;
using dist::Rank;
using wire::bytes_of;
using wire::exchange_bytes;
using wire::recv_bytes;
using wire::send_bytes;

double ns_since(Clock::time_point t0) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
          .count());
}

constexpr int kTagRtt = 0x51;
constexpr int kTagBurst = 0x52;
constexpr int kTagAck = 0x53;
constexpr int kTagRow = 0x54;

// A row of the matrix as it crosses the wire. Fixed width so a gather is one
// exchange of plain bytes and needs no length negotiation.
constexpr std::size_t kWirePoints = 8;

struct WireLink {
  std::uint8_t path = 0;
  std::uint8_t latency_provenance = 0;
  std::uint8_t bandwidth_provenance = 0;
  std::uint8_t point_count = 0;
  double latency_ns = 0.0;
  double bandwidth_bytes_per_s = 0.0;
  double fit_error = 0.0;
  std::uint64_t bytes[kWirePoints] = {};
  double ns[kWirePoints] = {};
};

int reps_for(std::size_t bytes, std::size_t budget) noexcept {
  if (bytes == 0) return 2;
  const std::size_t n = budget / bytes;
  return static_cast<int>(std::clamp<std::size_t>(n, 2, 16));
}

}  // namespace

std::string host_identity() {
  std::array<char, 256> buf{};
  if (::gethostname(buf.data(), buf.size() - 1) != 0) return "localhost";
  return std::string(buf.data());
}

PathKind classify_path(const LinkMember& src, const LinkMember& dst,
                       const dist::Capabilities& caps) noexcept {
  if (src.id == dst.id) return PathKind::kSameDevice;
  const bool same_host = src.host == dst.host && !src.host.empty();
  if (same_host) {
    return caps.device_memory_direct ? PathKind::kPeerDirect
                                     : PathKind::kHostStaged;
  }
  // `device_memory_direct` is exactly the GPUDirect question: whether the NIC
  // can DMA out of device memory, or the bytes have to land in host memory
  // first and be sent from there.
  return caps.device_memory_direct ? PathKind::kRdmaDirect
                                   : PathKind::kRdmaStaged;
}

Result<std::vector<LinkProfile>> probe_links(dist::ITransport& transport,
                                             std::span<const LinkMember> members,
                                             const LinkProbeConfig& cfg) {
  const auto n = static_cast<Rank>(members.size());
  if (n <= 0) return LSE_ERROR(kInvalidArgument, "an empty pool has no links");
  if (transport.world_size() != n) {
    return LSE_ERROR(kInvalidArgument, "pool has ", std::to_string(n),
                     " members but the transport has a world of ",
                     std::to_string(transport.world_size()));
  }
  // One slot is spent on the round-trip anchor below.
  if (cfg.sizes.size() + 1 > kWirePoints) {
    return LSE_ERROR(kInvalidArgument, "link probe takes at most ",
                     std::to_string(kWirePoints - 1), " transfer sizes");
  }

  // Members are indexed by rank so the matrix and the schedule agree.
  std::vector<std::size_t> by_rank(members.size(), members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    const Rank r = members[i].rank;
    if (r < 0 || r >= n || by_rank[static_cast<std::size_t>(r)] != members.size()) {
      return LSE_ERROR(kInvalidArgument,
                       "pool member ranks must be a permutation of [0, ",
                       std::to_string(n), ")");
    }
    by_rank[static_cast<std::size_t>(r)] = i;
  }
  const Rank me = transport.rank();
  if (me < 0 || me >= n) {
    return LSE_ERROR(kOutOfRange, "transport rank ", std::to_string(me),
                     " is not a pool member");
  }
  const std::size_t self = by_rank[static_cast<std::size_t>(me)];
  const dist::Capabilities& caps = transport.capabilities();

  // This rank's outgoing row.
  std::vector<LinkProfile> row(members.size());
  for (std::size_t j = 0; j < members.size(); ++j) {
    row[j].src = members[self].id;
    row[j].dst = members[j].id;
    row[j].path = classify_path(members[self], members[j], caps);
    if (row[j].path == PathKind::kSameDevice) {
      // Not a transfer at all. Zero is the measurement, not a placeholder.
      row[j].latency_ns = Measured::measured(0.0);
      row[j].bandwidth_bytes_per_s = Measured::unknown();
    }
  }

  std::vector<std::byte> payload(
      cfg.sizes.empty() ? 0 : *std::max_element(cfg.sizes.begin(), cfg.sizes.end()),
      std::byte{0x5A});
  std::array<std::byte, 8> token{};
  std::array<std::byte, 8> ack{};

  Rank width = 1;
  while (width < n) width <<= 1;

  for (Rank step = 1; step < width; ++step) {
    const Rank peer = me ^ step;
    if (peer >= n) continue;
    const std::size_t pi = by_rank[static_cast<std::size_t>(peer)];
    const bool low = me < peer;
    LinkProfile& out = row[pi];

    // Round trip, timed by each side in turn. A one-way time needs two clocks
    // that agree, which nothing here has, so the intercept is half a round trip
    // and is the same number in both directions by construction.
    double rtt_ns = 0.0;
    for (int turn = 0; turn < 2; ++turn) {
      const bool i_time = (turn == 0) == low;
      if (i_time) {
        // The minimum, not the mean. Every source of noise here — a scheduler
        // slice, a page fault, another rank on the same core — only ever adds
        // time, so the fastest round trip is the one closest to what the link
        // costs and the average is a measurement of the machine's load.
        double best = 0.0;
        for (int r = 0; r < cfg.latency_reps; ++r) {
          CommBuffer send = bytes_of(token.data(), token.size());
          CommBuffer back = bytes_of(ack.data(), ack.size());
          const auto t0 = Clock::now();
          LSE_RETURN_IF_ERROR(send_bytes(transport, send, peer, kTagRtt));
          LSE_RETURN_IF_ERROR(recv_bytes(transport, back, peer, kTagRtt));
          const double one = ns_since(t0);
          if (best == 0.0 || one < best) best = one;
        }
        rtt_ns = best;
      } else {
        for (int r = 0; r < cfg.latency_reps; ++r) {
          CommBuffer back = bytes_of(ack.data(), ack.size());
          LSE_RETURN_IF_ERROR(recv_bytes(transport, back, peer, kTagRtt));
          CommBuffer send = bytes_of(token.data(), token.size());
          LSE_RETURN_IF_ERROR(send_bytes(transport, send, peer, kTagRtt));
        }
      }
    }

    // Half a round trip of an 8-byte token is a direct measurement of what a
    // transfer costs before any bytes matter, and it anchors the intercept. A
    // one-way time would need two clocks that agree; nothing here has them, so
    // the two directions necessarily share this number.
    if (rtt_ns > 0.0) {
      out.points.push_back(TransferPoint{token.size(), rtt_ns / 2.0});
    }

    for (std::size_t s = 0; s < cfg.sizes.size(); ++s) {
      const std::size_t bytes = cfg.sizes[s];
      const int reps = reps_for(bytes, cfg.byte_budget);
      for (int dir = 0; dir < 2; ++dir) {
        const bool i_send = (dir == 0) == low;
        if (i_send) {
          const auto t0 = Clock::now();
          for (int r = 0; r < reps; ++r) {
            CommBuffer msg = bytes_of(payload.data(), bytes);
            LSE_RETURN_IF_ERROR(send_bytes(transport, msg, peer, kTagBurst));
          }
          CommBuffer back = bytes_of(ack.data(), ack.size());
          LSE_RETURN_IF_ERROR(recv_bytes(transport, back, peer, kTagAck));
          const double elapsed = ns_since(t0);
          // The burst pays for one acknowledgement round trip whatever its
          // size; charging it to the messages would put a size-independent
          // constant into the slope, which is the term it does not belong to.
          const double net = elapsed > rtt_ns ? elapsed - rtt_ns : elapsed;
          out.points.push_back(
              TransferPoint{bytes, net / static_cast<double>(reps)});
        } else {
          for (int r = 0; r < reps; ++r) {
            CommBuffer msg = bytes_of(payload.data(), bytes);
            LSE_RETURN_IF_ERROR(recv_bytes(transport, msg, peer, kTagBurst));
          }
          CommBuffer send = bytes_of(token.data(), token.size());
          LSE_RETURN_IF_ERROR(send_bytes(transport, send, peer, kTagAck));
        }
      }
    }

    fit_link(out);
  }

  if (n == 1) return row;

  // Every rank has its own row; the matrix is the rows put together.
  std::vector<WireLink> mine(members.size());
  for (std::size_t j = 0; j < members.size(); ++j) {
    const LinkProfile& l = row[j];
    WireLink& w = mine[j];
    w.path = static_cast<std::uint8_t>(l.path);
    w.latency_provenance = static_cast<std::uint8_t>(l.latency_ns.provenance);
    w.bandwidth_provenance =
        static_cast<std::uint8_t>(l.bandwidth_bytes_per_s.provenance);
    w.latency_ns = l.latency_ns.value;
    w.bandwidth_bytes_per_s = l.bandwidth_bytes_per_s.value;
    w.fit_error = l.fit_error;
    w.point_count = static_cast<std::uint8_t>(
        std::min(l.points.size(), kWirePoints));
    for (std::size_t p = 0; p < w.point_count; ++p) {
      w.bytes[p] = l.points[p].bytes;
      w.ns[p] = l.points[p].ns;
    }
  }

  std::vector<WireLink> matrix(members.size() * members.size());
  std::copy(mine.begin(), mine.end(), matrix.begin() + static_cast<std::ptrdiff_t>(self * members.size()));

  const std::size_t row_bytes = mine.size() * sizeof(WireLink);
  std::vector<WireLink> inbox(members.size());
  for (Rank step = 1; step < n; ++step) {
    const Rank dst = static_cast<Rank>((me + step) % n);
    const Rank src = static_cast<Rank>((me - step + n) % n);
    CommBuffer out = bytes_of(mine.data(), row_bytes);
    CommBuffer in = bytes_of(inbox.data(), row_bytes);
    LSE_RETURN_IF_ERROR(
        exchange_bytes(transport, out, dst, in, src, kTagRow + step));
    const std::size_t si = by_rank[static_cast<std::size_t>(src)];
    std::copy(inbox.begin(), inbox.end(),
              matrix.begin() + static_cast<std::ptrdiff_t>(si * members.size()));
  }

  std::vector<LinkProfile> out(members.size() * members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    for (std::size_t j = 0; j < members.size(); ++j) {
      const WireLink& w = matrix[i * members.size() + j];
      LinkProfile& l = out[i * members.size() + j];
      l.src = members[i].id;
      l.dst = members[j].id;
      l.path = static_cast<PathKind>(w.path);
      l.latency_ns = {w.latency_ns,
                      static_cast<Provenance>(w.latency_provenance)};
      l.bandwidth_bytes_per_s = {
          w.bandwidth_bytes_per_s,
          static_cast<Provenance>(w.bandwidth_provenance)};
      l.fit_error = w.fit_error;
      for (std::size_t p = 0; p < w.point_count && p < kWirePoints; ++p) {
        l.points.push_back(TransferPoint{static_cast<std::size_t>(w.bytes[p]),
                                         w.ns[p]});
      }
    }
  }
  return out;
}

Result<std::vector<LinkProfile>> probe_local_links(
    std::span<const LocalMember> members, const LinkProbeConfig& cfg) {
  const std::size_t n = members.size();
  if (n == 0) return LSE_ERROR(kInvalidArgument, "an empty pool has no links");

  std::vector<LinkProfile> out(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      LinkProfile& l = out[i * n + j];
      l.src = members[i].id;
      l.dst = members[j].id;
      if (i == j) {
        // Not a transfer at all. Zero is the measurement, not a placeholder.
        l.path = PathKind::kSameDevice;
        l.latency_ns = Measured::measured(0.0);
        continue;
      }
      if (members[i].backend == nullptr || members[j].backend == nullptr) {
        l.path = PathKind::kUnknown;
        continue;
      }
    }
  }
  if (n == 1 || cfg.sizes.empty()) return out;

  const std::size_t widest =
      *std::max_element(cfg.sizes.begin(), cfg.sizes.end());
  // The bounce goes through memory the device can reach, not through whatever
  // the allocator hands back. A plain vector is pageable, and a transfer out
  // of pageable memory is copied into a driver-owned staging buffer first --
  // which is what made a bounce across PCIe 5.0 x16 measure 0.71 GB/s, two
  // orders under the bus, and made the link look like the problem when the
  // host buffer was.
  std::vector<std::byte> paged;
  void* host_ptr = nullptr;
  backend::DeviceBuffer pinned;
  bool have_pinned = false;
  if (members[0].backend != nullptr) {
    auto staged =
        members[0].backend->allocate(widest, backend::MemoryClass::kStaging);
    if (staged.ok()) {
      pinned = staged.release();
      if (pinned.ptr != nullptr) {
        host_ptr = pinned.ptr;
        have_pinned = true;
      }
    }
  }
  if (!have_pinned) {
    paged.assign(widest, std::byte{0x5A});
    host_ptr = paged.data();
  }
  struct PinnedGuard {
    backend::IBackend* be;
    backend::DeviceBuffer* buf;
    bool live;
    ~PinnedGuard() { if (live && be != nullptr) be->deallocate(*buf); }
  } pinned_guard{members[0].backend, &pinned, have_pinned};

  for (std::size_t i = 0; i < n; ++i) {
    if (members[i].backend == nullptr) continue;
    backend::IBackend& src_be = *members[i].backend;
    auto src = src_be.allocate(widest, backend::MemoryClass::kDevice);
    if (!src.ok()) return src.status();
    backend::DeviceBuffer src_buf = src.release();

    for (std::size_t j = 0; j < n; ++j) {
      if (i == j || members[j].backend == nullptr) continue;
      backend::IBackend& dst_be = *members[j].backend;
      auto dst = dst_be.allocate(widest, backend::MemoryClass::kDevice);
      if (!dst.ok()) {
        src_be.deallocate(src_buf);
        return dst.status();
      }
      backend::DeviceBuffer dst_buf = dst.release();

      LinkProfile& l = out[i * n + j];
      l.path = PathKind::kHostStaged;
      Status failed;
      for (const std::size_t bytes : cfg.sizes) {
        const int reps = reps_for(bytes, cfg.byte_budget);
        // The minimum, not the mean, for the same reason the transport probe
        // takes it: every source of noise here only ever adds time, so the
        // fastest of the repeats is the one closest to what the move costs.
        double best = 0.0;
        for (int r = 0; r < reps && failed.ok(); ++r) {
          const auto t0 = Clock::now();
          failed = src_be.copy_d2h(src_buf, host_ptr, bytes, 0);
          if (failed.ok()) {
            failed = dst_be.copy_h2d(host_ptr, dst_buf, bytes, 0);
          }
          const double one = ns_since(t0);
          if (best == 0.0 || one < best) best = one;
        }
        if (!failed.ok()) break;
        if (best > 0.0) l.points.push_back(TransferPoint{bytes, best});
      }
      dst_be.deallocate(dst_buf);
      if (!failed.ok()) {
        src_be.deallocate(src_buf);
        return failed;
      }
      fit_link(l);
    }
    src_be.deallocate(src_buf);
  }
  return out;
}

}  // namespace lse::probe
