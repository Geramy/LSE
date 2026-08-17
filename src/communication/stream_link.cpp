#include "lse/communication/stream_link.hpp"

#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include "lse/core/debug.hpp"

namespace lse::comm {

namespace {

// A frame's payload length is 32 bits on the wire, and a gigabyte is already
// far past the point where segmenting costs anything measurable.
constexpr std::size_t kMaxFrameBytes = 1u << 30;
constexpr std::size_t kDiscardChunk = 64u << 10;

std::size_t effective_frame_max(std::size_t requested) noexcept {
  if (requested == 0 || requested > kMaxFrameBytes) return kMaxFrameBytes;
  return requested;
}

Interest interest_of(bool read, bool write) noexcept {
  return static_cast<Interest>(static_cast<std::uint8_t>(read ? 1 : 0) |
                               static_cast<std::uint8_t>(write ? 2 : 0));
}

bool would_block() noexcept { return errno == EAGAIN || errno == EWOULDBLOCK; }

}  // namespace

Result<LinkOptions> link_options_from(const Endpoint& ep) {
  LinkOptions opt;
  LSE_ASSIGN_OR(const std::uint64_t ring,
                ep.option_u64("ctrl_ring", kDefaultControlRingBytes));
  if (ring < kFrameHeaderBytes + 1 || ring > kMaxControlRingBytes) {
    return LSE_ERROR(kOutOfRange, "ctrl_ring=", std::to_string(ring),
                     " is outside [", std::to_string(kFrameHeaderBytes + 1),
                     ", ", std::to_string(kMaxControlRingBytes), "]");
  }
  opt.ctrl_ring = static_cast<std::uint32_t>(ring);

  LSE_ASSIGN_OR(const std::uint64_t inflight, ep.option_u64("max_inflight", 64));
  if (inflight == 0 || inflight > (1u << 20)) {
    return LSE_ERROR(kOutOfRange, "max_inflight=", std::to_string(inflight),
                     " is outside [1, 1048576]");
  }
  opt.max_inflight = static_cast<std::uint32_t>(inflight);

  LSE_ASSIGN_OR(const std::uint64_t deadline_ms,
                ep.option_u64("deadline_ms", 30000));
  opt.deadline_ns = deadline_ms * 1'000'000ull;
  LSE_ASSIGN_OR(const std::uint64_t offer_ms,
                ep.option_u64("offer_ms", deadline_ms));
  opt.offer_ns = offer_ms * 1'000'000ull;

  LSE_ASSIGN_OR(const std::uint64_t frame_max, ep.option_u64("frame_max", 0));
  if (frame_max != 0 && frame_max < 64) {
    return LSE_ERROR(kOutOfRange, "frame_max=", std::to_string(frame_max),
                     " is below the 64-byte floor a header would dominate");
  }
  opt.frame_max = static_cast<std::size_t>(frame_max);
  return opt;
}

StreamLink::StreamLink(Poller& poller, EventSink& sink, std::uint64_t channel,
                       Endpoint peer, Capabilities caps, const LinkOptions& opt,
                       std::array<int, 2> fds, bool client, std::uint64_t id_hi,
                       std::uint64_t id_lo)
    : poller_(&poller),
      sink_(&sink),
      channel_(channel),
      peer_(std::move(peer)),
      caps_(caps),
      opt_(opt),
      id_hi_(id_hi),
      id_lo_(id_lo),
      client_(client),
      announced_(!client) {
  for (std::size_t i = 0; i < 2; ++i) {
    LaneIo& io = lanes_[i];
    io.fd = fds[i];
    io.lane = i == 0 ? Lane::kControl : Lane::kData;
    io.connecting = client;
    // The acceptor has already read the peer's hello — that is how it knew
    // which two sockets belong together — so only its own reply is outstanding.
    io.hello_read = !client;
    build_hello(io);
    if (io.lane == Lane::kControl) {
      io.ctrl_tx.resize(static_cast<std::size_t>(opt_.ctrl_ring) +
                        kFrameHeaderBytes);
      io.rx.resize(static_cast<std::size_t>(opt_.ctrl_ring) +
                   kFrameHeaderBytes);
    }
    io.interest = interest_of(false, true);
    const Status added = poller_->add(io.fd, io.interest, this);
    if (!added.ok() && error_.ok()) error_ = added;
  }
  if (!error_.ok()) retire(StatusCode::kIoError);
}

StreamLink::~StreamLink() {
  for (LaneIo& io : lanes_) {
    if (io.fd >= 0) {
      poller_->remove(io.fd);
      ::close(io.fd);
      io.fd = -1;
    }
  }
}

void StreamLink::build_hello(LaneIo& io) {
  Hello h{};
  h.magic = kFrameMagic;
  h.version = kWireVersion;
  h.lane = static_cast<std::uint8_t>(io.lane == Lane::kControl ? 0 : 1);
  h.flags = host_is_little_endian() ? kHelloLittleEndian : std::uint8_t{0};
  h.channel_hi = id_hi_;
  h.channel_lo = id_lo_;
  h.control_ring_bytes = opt_.ctrl_ring;
  h.reserved = 0;
  std::memcpy(io.hello_out.data(), &h, sizeof(h));
}

// A lane that wants neither read nor write is handed BACK to the poller rather
// than left registered with an empty mask. epoll reports EPOLLHUP and EPOLLERR
// whatever the mask says and both are level state, so a data lane holding an
// unanswered offer whose peer has hung up would be reported ready for ever:
// poll() would return zero events immediately, on every call, until the offer
// deadline. Registration is therefore driven from the wanted interest, and this
// is the one place that adds, modifies or removes a lane's descriptor.
void StreamLink::arm(LaneIo& io) {
  if (io.fd < 0 || retired_) return;
  bool want_write = false;
  bool want_read = false;
  if (io.connecting) {
    want_write = true;
  } else {
    want_write = !io.write_closed && (!io.hello_sent || !io.tx.empty());
    // A data lane holding an unanswered offer stops reading, which closes the
    // window and stops the peer rather than buffering behind the caller.
    want_read = !io.eof && !io.offer_pending;
  }
  const Interest want = interest_of(want_read, want_write);
  if (want == Interest::kNone) {
    if (io.polled) {
      poller_->remove(io.fd);
      io.polled = false;
      io.interest = Interest::kNone;
    }
    return;
  }
  if (!io.polled) {
    const Status added = poller_->add(io.fd, want, this);
    if (!added.ok()) {
      fault(added, StatusCode::kIoError);
      return;
    }
    io.polled = true;
    io.interest = want;
    return;
  }
  if (want == io.interest) return;
  const Status s = poller_->modify(io.fd, want);
  if (!s.ok()) {
    fault(s, StatusCode::kIoError);
    return;
  }
  io.interest = want;
}

void StreamLink::on_ready(int fd, Readiness r) {
  for (LaneIo& io : lanes_) {
    if (io.fd == fd) {
      service(io, r);
      return;
    }
  }
}

void StreamLink::finish_connect(LaneIo& io) {
  int err = 0;
  socklen_t len = sizeof(err);
  if (::getsockopt(io.fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0) {
    fault(LSE_ERROR(kIoError, "getsockopt(SO_ERROR) on ", peer_.str(), ": ",
                    std::strerror(errno)),
          StatusCode::kIoError);
    return;
  }
  if (err != 0) {
    fault(LSE_ERROR(kIoError, "connect to ", peer_.str(), ": ",
                    std::strerror(err)),
          StatusCode::kIoError);
    return;
  }
  io.connecting = false;
}

void StreamLink::drive_hello(LaneIo& io) {
  while (!io.hello_sent) {
    const std::size_t left = kHelloBytes - io.hello_out_sent;
    const ssize_t n = ::send(io.fd, io.hello_out.data() + io.hello_out_sent,
                             left, MSG_NOSIGNAL);
    if (n < 0) {
      if (would_block()) return;
      fault(LSE_ERROR(kIoError, "hello send to ", peer_.str(), ": ",
                      std::strerror(errno)),
            StatusCode::kIoError);
      return;
    }
    io.hello_out_sent += static_cast<std::size_t>(n);
    if (io.hello_out_sent == kHelloBytes) io.hello_sent = true;
  }
}

void StreamLink::announce_open_if_ready() {
  if (announced_ || retired_) return;
  for (const LaneIo& io : lanes_) {
    if (!io.hello_sent || !io.hello_read) return;
  }
  announced_ = true;
  Event ev;
  ev.kind = EventKind::kConnected;
  ev.channel = channel_;
  sink_->emit(ev);
}

void StreamLink::service(LaneIo& io, Readiness r) {
  if (retired_) return;

  if (io.connecting) {
    if (!r.writable && !r.hangup) return;
    finish_connect(io);
    if (retired_ || io.connecting) return;
  }

  if (!io.hello_sent) {
    drive_hello(io);
    if (retired_) return;
  }

  if (!io.hello_read && (r.readable || r.hangup)) {
    while (io.hello_in_read < kHelloBytes) {
      const ssize_t n = ::recv(io.fd, io.hello_in.data() + io.hello_in_read,
                               kHelloBytes - io.hello_in_read, 0);
      if (n == 0) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " closed before its hello arrived"),
              StatusCode::kIoError);
        return;
      }
      if (n < 0) {
        if (would_block()) break;
        fault(LSE_ERROR(kIoError, "hello recv from ", peer_.str(), ": ",
                        std::strerror(errno)),
              StatusCode::kIoError);
        return;
      }
      io.hello_in_read += static_cast<std::size_t>(n);
    }
    if (io.hello_in_read == kHelloBytes) {
      Hello h{};
      std::memcpy(&h, io.hello_in.data(), sizeof(h));
      if (h.magic != kFrameMagic) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " is not speaking this protocol"),
              StatusCode::kIoError);
        return;
      }
      if (h.version != kWireVersion) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(), " speaks wire version ",
                        std::to_string(h.version), ", this build speaks ",
                        std::to_string(kWireVersion)),
              StatusCode::kIoError);
        return;
      }
      if (((h.flags & kHelloLittleEndian) != 0) != host_is_little_endian()) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " has the opposite byte order; this wire is written "
                        "as-is and has no swap path"),
              StatusCode::kIoError);
        return;
      }
      if (io.lane == Lane::kControl) {
        peer_ctrl_ring_ = std::min(h.control_ring_bytes, kMaxControlRingBytes);
      }
      io.hello_read = true;
    }
  }

  announce_open_if_ready();
  if (retired_) return;

  if (io.hello_sent && io.hello_read) {
    if (r.writable) {
      drain_tx(io);
      if (retired_) return;
    }
    if (r.readable || r.hangup) {
      if (io.lane == Lane::kControl) {
        drain_rx_control(io);
      } else {
        drain_rx_data(io);
      }
      if (retired_) return;
    }
  }
  arm(io);
}

// --- transmit --------------------------------------------------------------

void StreamLink::drain_tx(LaneIo& io) {
  while (!io.tx.empty()) {
    TxOp& op = io.tx.front();

    if (io.lane == Lane::kControl) {
      const std::size_t left = op.ctrl_len - op.frame_sent;
      const ssize_t n =
          ::send(io.fd, io.ctrl_tx.data() + io.ctrl_head + op.frame_sent, left,
                 MSG_NOSIGNAL);
      if (n < 0) {
        if (would_block()) break;
        fault(LSE_ERROR(kIoError, "control send to ", peer_.str(), ": ",
                        std::strerror(errno)),
              StatusCode::kIoError);
        return;
      }
      op.started = true;
      op.frame_sent += static_cast<std::size_t>(n);
      if (op.frame_sent < op.ctrl_len) break;
      io.ctrl_head += op.ctrl_len;
      const std::uint64_t done_op = op.op;
      const std::uint32_t done_tag = op.tag;
      const std::size_t done_bytes = op.total;
      io.tx.pop_front();
      if (io.ctrl_head == io.ctrl_tail) {
        io.ctrl_head = 0;
        io.ctrl_tail = 0;
      }
      complete(io.lane, Direction::kSend, done_op, done_tag, done_bytes,
               StatusCode::kOk);
      continue;
    }

    if (!op.framed) {
      const std::size_t cap = effective_frame_max(opt_.frame_max);
      op.frame_payload = std::min(cap, op.total - op.sent);
      op.frame_sent = 0;
      op.hdr_sent = 0;
      op.hdr = FrameHeader{};
      op.hdr.magic = kFrameMagic;
      op.hdr.lane = 1;
      op.hdr.flags =
          (op.sent + op.frame_payload == op.total) ? kFrameLast : std::uint8_t{0};
      op.hdr.reserved = 0;
      op.hdr.tag = op.tag;
      op.hdr.bytes = static_cast<std::uint32_t>(op.frame_payload);
      op.hdr.total_bytes = op.total;
      op.framed = true;
    }

    std::array<iovec, 2> iov{};
    int count = 0;
    if (op.hdr_sent < kFrameHeaderBytes) {
      iov[static_cast<std::size_t>(count)].iov_base =
          reinterpret_cast<char*>(&op.hdr) + op.hdr_sent;
      iov[static_cast<std::size_t>(count)].iov_len =
          kFrameHeaderBytes - op.hdr_sent;
      ++count;
    }
    if (op.frame_sent < op.frame_payload) {
      iov[static_cast<std::size_t>(count)].iov_base = const_cast<std::byte*>(
          op.src + op.sent + op.frame_sent);
      iov[static_cast<std::size_t>(count)].iov_len =
          op.frame_payload - op.frame_sent;
      ++count;
    }
    if (count == 0) {
      // A zero-byte frame whose header is already out: nothing left to write.
      op.sent += op.frame_payload;
      op.framed = false;
      if (op.sent >= op.total) {
        const std::uint64_t done_op = op.op;
        const std::uint32_t done_tag = op.tag;
        const std::size_t done_bytes = op.sent;
        io.tx.pop_front();
        complete(io.lane, Direction::kSend, done_op, done_tag, done_bytes,
                 StatusCode::kOk);
      }
      continue;
    }

    msghdr msg{};
    msg.msg_iov = iov.data();
    msg.msg_iovlen = static_cast<std::size_t>(count);
    const ssize_t n = ::sendmsg(io.fd, &msg, MSG_NOSIGNAL);
    if (n < 0) {
      if (would_block()) break;
      fault(LSE_ERROR(kIoError, "data send to ", peer_.str(), ": ",
                      std::strerror(errno)),
            StatusCode::kIoError);
      return;
    }
    op.started = true;
    auto left = static_cast<std::size_t>(n);
    if (op.hdr_sent < kFrameHeaderBytes) {
      const std::size_t take =
          std::min(left, kFrameHeaderBytes - op.hdr_sent);
      op.hdr_sent += take;
      left -= take;
    }
    op.frame_sent += left;

    if (op.hdr_sent == kFrameHeaderBytes && op.frame_sent == op.frame_payload) {
      op.sent += op.frame_payload;
      op.framed = false;
      if (op.sent >= op.total) {
        const std::uint64_t done_op = op.op;
        const std::uint32_t done_tag = op.tag;
        const std::size_t done_bytes = op.sent;
        io.tx.pop_front();
        complete(io.lane, Direction::kSend, done_op, done_tag, done_bytes,
                 StatusCode::kOk);
        continue;
      }
      continue;
    }
    break;  // partial write: the socket is full
  }

  if (io.lane == Lane::kControl && io.ctrl_refused) {
    const std::size_t used = io.ctrl_tail - io.ctrl_head;
    if (used * 2 <= io.ctrl_tx.size()) {
      io.ctrl_refused = false;
      Event ev;
      ev.kind = EventKind::kWritable;
      ev.channel = channel_;
      ev.lane = Lane::kControl;
      ev.bytes = credit(Lane::kControl);
      sink_->emit(ev);
    }
  }

  half_close_if_drained();
}

// Graceful means the peer gets everything that was queued: the write half is
// shut down so it sees a FIN rather than a reset, and the channel stays up
// until the peer closes back or the channel's own deadline runs out. Dropping
// the descriptor the moment our queue empties would let a reset discard bytes
// the peer had not read yet.
// A lane at EOF still has a sibling that may be mid-transfer, so the channel
// ends only when both have run out. Until then the spent descriptor is handed
// back to the poller rather than waited on, because a hangup is level state and
// would be reported for ever.
void StreamLink::note_eof(LaneIo& io) {
  if (io.eof && !io.polled) return;
  io.eof = true;
  if (io.tx.empty() && io.fd >= 0) {
    poller_->remove(io.fd);
    io.polled = false;
    io.interest = Interest::kNone;
  } else {
    arm(io);
  }
  if (lanes_[0].eof && lanes_[1].eof) {
    retire(closing_ ? StatusCode::kCancelled : StatusCode::kIoError);
  }
}

void StreamLink::half_close_if_drained() {
  if (retired_ || !closing_) return;
  for (LaneIo& io : lanes_) {
    if (io.fd < 0 || io.write_closed || !io.tx.empty()) continue;
    ::shutdown(io.fd, SHUT_WR);
    io.write_closed = true;
    arm(io);
  }
}

// --- receive ---------------------------------------------------------------

void StreamLink::drain_rx_control(LaneIo& io) {
  for (;;) {
    const std::size_t room = io.rx.size() - io.rx_len;
    if (room == 0) {
      // A frame larger than the ring this side advertised: the peer ignored
      // the hello, and there is no honest way to deliver it.
      fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                      " sent a control frame larger than the ",
                      std::to_string(opt_.ctrl_ring),
                      " B ring it was told about"),
            StatusCode::kIoError);
      return;
    }
    const ssize_t n = ::recv(io.fd, io.rx.data() + io.rx_len, room, 0);
    if (n == 0) {
      io.eof = true;
    } else if (n < 0) {
      if (!would_block()) {
        fault(LSE_ERROR(kIoError, "control recv from ", peer_.str(), ": ",
                        std::strerror(errno)),
              StatusCode::kIoError);
        return;
      }
    } else {
      io.rx_len += static_cast<std::size_t>(n);
    }

    std::size_t cursor = 0;
    while (io.rx_len - cursor >= kFrameHeaderBytes) {
      FrameHeader h{};
      std::memcpy(&h, io.rx.data() + cursor, sizeof(h));
      if (h.magic != kFrameMagic || h.lane != 0) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " sent an unrecognised control frame"),
              StatusCode::kIoError);
        return;
      }
      if (h.bytes > opt_.ctrl_ring) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(), " sent a ",
                        std::to_string(h.bytes),
                        " B control message; this side advertised ",
                        std::to_string(opt_.ctrl_ring)),
              StatusCode::kIoError);
        return;
      }
      if (io.rx_len - cursor < kFrameHeaderBytes + h.bytes) break;
      const std::byte* payload = io.rx.data() + cursor + kFrameHeaderBytes;
      Event ev;
      ev.kind = EventKind::kControl;
      ev.channel = channel_;
      ev.lane = Lane::kControl;
      ev.dir = Direction::kRecv;
      ev.tag = h.tag;
      ev.bytes = h.bytes;
      ev.data = sink_->retain({payload, h.bytes});
      sink_->emit(ev);
      cursor += kFrameHeaderBytes + h.bytes;
    }
    if (cursor > 0) {
      io.rx_len -= cursor;
      if (io.rx_len > 0) {
        std::memmove(io.rx.data(), io.rx.data() + cursor, io.rx_len);
      }
    }

    if (io.eof) {
      if (io.rx_len > 0) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " closed the control lane mid-frame"),
              StatusCode::kIoError);
        return;
      }
      note_eof(io);
      return;
    }
    if (n < 0 || static_cast<std::size_t>(n) < room) return;  // socket drained
  }
}

bool StreamLink::bind_inbound(LaneIo& io) {
  const auto it =
      std::find_if(io.posted.begin(), io.posted.end(),
                   [&](const RxOp& r) { return r.tag == io.inbound_tag; });
  if (it == io.posted.end()) return false;
  const RxOp claimed = *it;
  io.posted.erase(it);

  if (io.inbound_total > claimed.capacity) {
    complete(Lane::kData, Direction::kRecv, claimed.op, claimed.tag, 0,
             StatusCode::kOutOfRange);
    io.inbound_op = 0;
    io.inbound_dst = nullptr;
    io.inbound_discard = true;
  } else {
    io.inbound_op = claimed.op;
    io.inbound_dst = claimed.dst;
    io.inbound_discard = false;
  }
  io.inbound_done = 0;
  io.inbound_active = true;
  return true;
}

void StreamLink::drain_rx_data(LaneIo& io) {
  for (;;) {
    if (!io.in_frame) {
      while (io.hdr_in_len < kFrameHeaderBytes) {
        const ssize_t n = ::recv(io.fd, io.hdr_in.data() + io.hdr_in_len,
                                 kFrameHeaderBytes - io.hdr_in_len, 0);
        if (n == 0) {
          io.eof = true;
          break;
        }
        if (n < 0) {
          if (would_block()) return;
          fault(LSE_ERROR(kIoError, "data recv from ", peer_.str(), ": ",
                          std::strerror(errno)),
                StatusCode::kIoError);
          return;
        }
        io.hdr_in_len += static_cast<std::size_t>(n);
      }
      if (io.eof) {
        if (io.hdr_in_len > 0 || io.inbound_active) {
          fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                          " closed the data lane mid-transfer after ",
                          std::to_string(io.inbound_done), " B"),
                StatusCode::kIoError);
        } else {
          note_eof(io);
        }
        return;
      }
      std::memcpy(&io.cur, io.hdr_in.data(), sizeof(io.cur));
      io.hdr_in_len = 0;
      if (io.cur.magic != kFrameMagic || io.cur.lane != 1) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " sent an unrecognised data frame"),
              StatusCode::kIoError);
        return;
      }
      io.in_frame = true;
      io.frame_left = io.cur.bytes;
      if (!io.inbound_active) {
        io.inbound_tag = io.cur.tag;
        io.inbound_total = io.cur.total_bytes;
      }
      // The landing region was accepted against total_bytes, so a frame that
      // would carry this transfer past the total the peer declared is a peer
      // contradicting itself — and taking its word for it is a write past the
      // end of the caller's region, not a protocol nicety.
      if (io.inbound_done > io.inbound_total ||
          io.frame_left > io.inbound_total - io.inbound_done) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(), " sent a ",
                        std::to_string(io.frame_left), " B frame that takes tag ",
                        std::to_string(io.inbound_tag), " past the ",
                        std::to_string(io.inbound_total),
                        " B transfer it declared"),
              StatusCode::kIoError);
        return;
      }
    }

    if (!io.inbound_active) {
      if (io.offer_pending) return;
      if (!bind_inbound(io)) {
        io.offer_pending = true;
        io.offer_tag = io.inbound_tag;
        io.offer_since_ns = steady_now_ns();
        Event ev;
        ev.kind = EventKind::kDataOffered;
        ev.channel = channel_;
        ev.lane = Lane::kData;
        ev.dir = Direction::kRecv;
        ev.tag = io.inbound_tag;
        ev.bytes = static_cast<std::size_t>(io.inbound_total);
        sink_->emit(ev);
        arm(io);
        return;
      }
    }

    while (io.frame_left > 0) {
      std::byte* dst = nullptr;
      std::size_t want = io.frame_left;
      if (io.inbound_discard) {
        if (discard_.size() < kDiscardChunk) discard_.resize(kDiscardChunk);
        dst = discard_.data();
        want = std::min(want, discard_.size());
      } else {
        dst = io.inbound_dst + io.inbound_done;
      }
      const ssize_t n = ::recv(io.fd, dst, want, 0);
      if (n == 0) {
        fault(LSE_ERROR(kIoError, "peer ", peer_.str(),
                        " closed the data lane after ",
                        std::to_string(io.inbound_done), " B of ",
                        std::to_string(io.inbound_total)),
              StatusCode::kIoError);
        return;
      }
      if (n < 0) {
        if (would_block()) return;
        fault(LSE_ERROR(kIoError, "data recv from ", peer_.str(), ": ",
                        std::strerror(errno)),
              StatusCode::kIoError);
        return;
      }
      const auto got = static_cast<std::size_t>(n);
      io.frame_left -= got;
      io.inbound_done += got;
    }

    io.in_frame = false;
    if ((io.cur.flags & kFrameLast) != 0) {
      const std::uint64_t op = io.inbound_op;
      const std::uint32_t tag = io.inbound_tag;
      const auto moved = static_cast<std::size_t>(io.inbound_done);
      io.inbound_active = false;
      io.inbound_op = 0;
      io.inbound_dst = nullptr;
      io.inbound_discard = false;
      io.inbound_done = 0;
      if (op != 0) {
        complete(Lane::kData, Direction::kRecv, op, tag, moved,
                 StatusCode::kOk);
      }
    }
  }
}

// --- caller surface --------------------------------------------------------

Status StreamLink::post(Lane lane, Direction dir, const Transfer& t,
                        std::uint64_t op) {
  if (retired_) {
    return LSE_ERROR(kNotFound, "channel to ", peer_.str(), " is closed");
  }
  if (closing_) {
    return LSE_ERROR(kOutOfRange, "channel to ", peer_.str(),
                     " is closing; no further posts are accepted");
  }
  if (!announced_) {
    return LSE_ERROR(kOutOfRange, "channel to ", peer_.str(),
                     " is not connected yet; wait for its kConnected event");
  }

  LaneIo& io = lane_of(lane);

  if (lane == Lane::kControl) {
    if (dir != Direction::kSend) {
      return LSE_ERROR(kInvalidArgument,
                       "control messages arrive unsolicited; there is nothing "
                       "to post a receive against");
    }
    const std::size_t limit =
        static_cast<std::size_t>(peer_ctrl_ring_) - kFrameHeaderBytes;
    if (t.bytes > limit) {
      return LSE_ERROR(kInvalidArgument, "control message of ",
                       std::to_string(t.bytes), " B exceeds the ",
                       std::to_string(limit), " B the peer advertised");
    }
    const std::size_t need = kFrameHeaderBytes + t.bytes;
    if (io.ctrl_head > 0 && io.ctrl_tx.size() - io.ctrl_tail < need) {
      const std::size_t used = io.ctrl_tail - io.ctrl_head;
      if (used > 0) {
        std::memmove(io.ctrl_tx.data(), io.ctrl_tx.data() + io.ctrl_head, used);
      }
      io.ctrl_head = 0;
      io.ctrl_tail = used;
    }
    if (io.ctrl_tx.size() - io.ctrl_tail < need) {
      io.ctrl_refused = true;
      return LSE_ERROR(kOutOfMemory, "the ", std::to_string(opt_.ctrl_ring),
                       " B control ring to ", peer_.str(),
                       " is full; wait for kWritable");
    }

    FrameHeader h{};
    h.magic = kFrameMagic;
    h.lane = 0;
    h.flags = kFrameLast;
    h.reserved = 0;
    h.tag = t.tag;
    h.bytes = static_cast<std::uint32_t>(t.bytes);
    h.total_bytes = t.bytes;
    std::memcpy(io.ctrl_tx.data() + io.ctrl_tail, &h, sizeof(h));
    if (t.bytes > 0) {
      std::memcpy(io.ctrl_tx.data() + io.ctrl_tail + kFrameHeaderBytes,
                  static_cast<const std::byte*>(t.region.host) + t.offset,
                  t.bytes);
    }
    io.ctrl_tail += need;

    TxOp queued;
    queued.op = op;
    queued.tag = t.tag;
    queued.total = t.bytes;
    queued.ctrl_len = need;
    io.tx.push_back(queued);
    drain_tx(io);
    if (!retired_) arm(io);
    return OkStatus();
  }

  if (t.bytes > 0 && t.region.host == nullptr) {
    return LSE_ERROR(kUnimplemented, std::string(peer_.scheme()),
                     " needs a host address for the bytes; a device-resident "
                     "region has no path here — register it or stage it first");
  }
  if (t.region.bytes != 0 && t.offset + t.bytes > t.region.bytes) {
    return LSE_ERROR(kOutOfRange, "transfer [", std::to_string(t.offset), ", ",
                     std::to_string(t.offset + t.bytes),
                     ") is outside its ", std::to_string(t.region.bytes),
                     " B region");
  }

  if (dir == Direction::kSend) {
    TxOp queued;
    queued.op = op;
    queued.tag = t.tag;
    queued.src = static_cast<const std::byte*>(t.region.host) + t.offset;
    queued.total = t.bytes;
    io.tx.push_back(queued);
    drain_tx(io);
    if (!retired_) arm(io);
    return OkStatus();
  }

  RxOp landing;
  landing.op = op;
  landing.tag = t.tag;
  landing.dst = static_cast<std::byte*>(t.region.host) + t.offset;
  landing.capacity = t.bytes;
  io.posted.push_back(landing);
  if (io.offer_pending && io.offer_tag == t.tag) {
    io.offer_pending = false;
    drain_rx_data(io);
  }
  if (!retired_) arm(io);
  return OkStatus();
}

Status StreamLink::cancel(std::uint64_t op) {
  for (LaneIo& io : lanes_) {
    for (auto it = io.tx.begin(); it != io.tx.end(); ++it) {
      if (it->op != op) continue;
      if (it->started) {
        return LSE_ERROR(kOutOfRange,
                         "that transfer is already on the wire; it cannot be "
                         "un-sent, only abandoned with abort()");
      }
      const std::uint32_t tag = it->tag;
      if (io.lane == Lane::kControl) {
        // The bytes were copied into the ring at post time. Nothing of this op
        // has been written and nothing behind it has either — only the head can
        // be partly sent — so the hole closes with one memmove.
        std::size_t at = io.ctrl_head;
        for (auto scan = io.tx.begin(); scan != it; ++scan) at += scan->ctrl_len;
        const std::size_t hole = it->ctrl_len;
        const std::size_t tail_bytes = io.ctrl_tail - (at + hole);
        if (tail_bytes > 0) {
          std::memmove(io.ctrl_tx.data() + at,
                       io.ctrl_tx.data() + at + hole, tail_bytes);
        }
        io.ctrl_tail -= hole;
        if (io.ctrl_head == io.ctrl_tail) {
          io.ctrl_head = 0;
          io.ctrl_tail = 0;
        }
      }
      io.tx.erase(it);
      complete(io.lane, Direction::kSend, op, tag, 0, StatusCode::kCancelled);
      arm(io);
      return OkStatus();
    }
    for (auto it = io.posted.begin(); it != io.posted.end(); ++it) {
      if (it->op != op) continue;
      const std::uint32_t tag = it->tag;
      io.posted.erase(it);
      complete(io.lane, Direction::kRecv, op, tag, 0, StatusCode::kCancelled);
      return OkStatus();
    }
    if (io.inbound_active && io.inbound_op == op) {
      return LSE_ERROR(kOutOfRange,
                       "that receive is already taking bytes off the wire");
    }
  }
  return LSE_ERROR(kNotFound, "no such outstanding operation on the channel to ",
                   peer_.str());
}

std::size_t StreamLink::credit(Lane lane) const noexcept {
  if (lane != Lane::kControl) return static_cast<std::size_t>(-1);
  const LaneIo& io = lane_of(Lane::kControl);
  const std::size_t used = io.ctrl_tail - io.ctrl_head;
  const std::size_t free_bytes = io.ctrl_tx.size() - used;
  return free_bytes > kFrameHeaderBytes ? free_bytes - kFrameHeaderBytes : 0;
}

Status StreamLink::reject(std::uint32_t tag) {
  LaneIo& io = lane_of(Lane::kData);
  if (!io.offer_pending || io.offer_tag != tag) {
    return LSE_ERROR(kNotFound, "no offer for tag ", std::to_string(tag),
                     " is outstanding on the channel to ", peer_.str());
  }
  io.offer_pending = false;
  io.inbound_op = 0;
  io.inbound_dst = nullptr;
  io.inbound_discard = true;
  io.inbound_done = 0;
  io.inbound_active = true;
  drain_rx_data(io);
  if (!retired_) arm(io);
  return OkStatus();
}

Status StreamLink::close() {
  if (retired_) return OkStatus();
  closing_ = true;
  close_by_ns_ = steady_now_ns() + opt_.deadline_ns;
  for (LaneIo& io : lanes_) {
    if (!io.tx.empty()) drain_tx(io);
    if (retired_) return OkStatus();
  }
  half_close_if_drained();
  return OkStatus();
}

void StreamLink::abort() noexcept {
  if (retired_) return;
  if (error_.ok()) {
    error_ = Status(StatusCode::kCancelled, "channel aborted locally");
  }
  retire(StatusCode::kCancelled);
}

void StreamLink::tick(std::uint64_t now_ns) {
  if (retired_) return;
  for (LaneIo& io : lanes_) {
    if (!io.offer_pending) continue;
    if (now_ns < io.offer_since_ns + opt_.offer_ns) continue;
    fault(LSE_ERROR(kCancelled, "no receive was posted for tag ",
                    std::to_string(io.offer_tag), " within ",
                    std::to_string(opt_.offer_ns / 1'000'000ull),
                    " ms; the data lane to ", peer_.str(),
                    " cannot make progress"),
          StatusCode::kCancelled);
    return;
  }
  half_close_if_drained();
  // A peer that never answers a FIN would otherwise hold the channel open for
  // ever; the channel's own deadline is what bounds the wait.
  if (closing_ && close_by_ns_ != 0 && now_ns >= close_by_ns_) {
    retire(StatusCode::kCancelled);
  }
}

std::uint64_t StreamLink::next_timer_ns() const noexcept {
  std::uint64_t soonest = closing_ ? close_by_ns_ : 0;
  for (const LaneIo& io : lanes_) {
    if (!io.offer_pending) continue;
    const std::uint64_t at = io.offer_since_ns + opt_.offer_ns;
    if (soonest == 0 || at < soonest) soonest = at;
  }
  return soonest;
}

// --- completion and teardown ----------------------------------------------

void StreamLink::complete(Lane lane, Direction dir, std::uint64_t op,
                          std::uint32_t tag, std::size_t bytes,
                          StatusCode code) {
  Event ev;
  ev.kind = dir == Direction::kSend ? EventKind::kSendComplete
                                    : EventKind::kRecvComplete;
  ev.code = code;
  ev.lane = lane;
  ev.dir = dir;
  ev.tag = tag;
  ev.channel = channel_;
  ev.op = op;
  ev.bytes = bytes;
  sink_->emit(ev);
}

void StreamLink::fault(Status why, StatusCode outstanding) {
  if (error_.ok()) error_ = std::move(why);
  retire(outstanding);
}

void StreamLink::retire(StatusCode outstanding) {
  if (retired_) return;
  retired_ = true;

  // Exactly one completion per accepted post, in post order, control lane
  // first: the plan a peer already sent is finished with before the bulk it
  // describes.
  for (LaneIo& io : lanes_) {
    for (const TxOp& op : io.tx) {
      // How far it actually got, including the frame that was in flight: that
      // offset is what a caller replays from. On the control lane frame_sent
      // counts the pre-formatted header too, which is not payload.
      const std::size_t moved =
          io.lane == Lane::kControl
              ? (op.frame_sent > kFrameHeaderBytes
                     ? op.frame_sent - kFrameHeaderBytes
                     : 0)
              : op.sent + op.frame_sent;
      complete(io.lane, Direction::kSend, op.op, op.tag, moved, outstanding);
    }
    io.tx.clear();
    if (io.inbound_active && io.inbound_op != 0) {
      complete(io.lane, Direction::kRecv, io.inbound_op, io.inbound_tag,
               static_cast<std::size_t>(io.inbound_done), outstanding);
      io.inbound_op = 0;
    }
    io.inbound_active = false;
    for (const RxOp& op : io.posted) {
      complete(io.lane, Direction::kRecv, op.op, op.tag, 0, outstanding);
    }
    io.posted.clear();
    if (io.fd >= 0) {
      poller_->remove(io.fd);
      ::close(io.fd);
      io.fd = -1;
    }
    if (debug()) {
      std::fill(io.rx.begin(), io.rx.end(), std::byte{0xDD});
    }
  }
  emit_closed(error_.ok() ? StatusCode::kOk : error_.code());
}

void StreamLink::emit_closed(StatusCode code) {
  Event ev;
  ev.kind = EventKind::kClosed;
  ev.code = code;
  ev.channel = channel_;
  sink_->emit(ev);
}

}  // namespace lse::comm
