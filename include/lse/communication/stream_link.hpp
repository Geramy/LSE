// One channel over a pair of byte streams — one socket per lane.
//
// The lane split is physical, not a caller discipline: a 400 MB activation on
// the data socket cannot delay a 2 KB plan on the control socket, because they
// are different file descriptors with different windows. Everything a byte
// stream needs and a completion queue does not — framing, partial writes,
// readiness, tag matching, segmentation, teardown ordering — lives here, so
// tcp:// and unix:// are address families and nothing more.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "lse/communication/frame.hpp"
#include "lse/communication/endpoint.hpp"
#include "lse/communication/transport.hpp"
#include "lse/communication/poller.hpp"

namespace lse::comm {

struct LinkOptions {
  std::uint32_t ctrl_ring = kDefaultControlRingBytes;
  std::uint32_t max_inflight = 64;
  std::uint64_t deadline_ns = 30'000'000'000ull;
  std::uint64_t offer_ns = 30'000'000'000ull;
  // 0 = one frame per transfer, however large.
  std::size_t frame_max = 0;
};

[[nodiscard]] Result<LinkOptions> link_options_from(const Endpoint& ep);

class StreamLink final : public ILink, public IReady {
 public:
  // Client half: `fds` are already-created non-blocking sockets with connect()
  // in progress (or complete). `channel_id` is the 128-bit pairing id this side
  // chose.
  StreamLink(Poller& poller, EventSink& sink, std::uint64_t channel,
             Endpoint peer, Capabilities caps, const LinkOptions& opt,
             std::array<int, 2> fds, bool client, std::uint64_t id_hi,
             std::uint64_t id_lo);
  ~StreamLink() override;
  StreamLink(const StreamLink&) = delete;
  StreamLink& operator=(const StreamLink&) = delete;

  Status post(Lane lane, Direction dir, const Transfer& t,
              std::uint64_t op) override;
  Status cancel(std::uint64_t op) override;
  [[nodiscard]] std::size_t credit(Lane lane) const noexcept override;
  [[nodiscard]] const Capabilities& capabilities() const noexcept override {
    return caps_;
  }
  [[nodiscard]] const Endpoint& peer() const noexcept override { return peer_; }
  Status reject(std::uint32_t tag) override;
  Status close() override;
  void abort() noexcept override;
  [[nodiscard]] const Status& last_error() const noexcept override {
    return error_;
  }

  void on_ready(int fd, Readiness r) override;

  // Called once per reactor sweep by the owning transport: fires the offer
  // deadline and finishes a graceful close whose queues have drained.
  void tick(std::uint64_t now_ns);
  [[nodiscard]] std::uint64_t next_timer_ns() const noexcept;

  [[nodiscard]] bool retired() const noexcept { return retired_; }

  // The acceptor read the peer's hello to learn which two sockets belong
  // together, so the ring size it carries is handed over rather than read
  // twice off a socket that has already moved past it.
  void adopt_peer_ring(std::uint32_t bytes) noexcept {
    peer_ctrl_ring_ = std::min(bytes, kMaxControlRingBytes);
  }

 private:
  struct TxOp {
    std::uint64_t op = 0;
    std::uint32_t tag = 0;
    const std::byte* src = nullptr;  // data lane only
    std::size_t total = 0;           // payload bytes in the whole transfer
    std::size_t sent = 0;            // payload bytes already written
    std::size_t hdr_sent = 0;        // bytes of the current frame header
    std::size_t frame_payload = 0;   // payload bytes in the current frame
    std::size_t frame_sent = 0;
    bool framed = false;             // the current frame's header is built
    bool started = false;            // any byte of this op reached the socket
    FrameHeader hdr{};
    // Control lane only: where the pre-formatted frame lives in ctrl_tx.
    std::size_t ctrl_len = 0;
  };

  struct RxOp {
    std::uint64_t op = 0;
    std::uint32_t tag = 0;
    std::byte* dst = nullptr;
    std::size_t capacity = 0;
  };

  struct LaneIo {
    int fd = -1;
    Lane lane = Lane::kControl;
    Interest interest = Interest::kNone;
    bool connecting = false;
    bool hello_sent = false;
    bool hello_read = false;
    std::size_t hello_out_sent = 0;
    std::size_t hello_in_read = 0;
    std::array<std::byte, kHelloBytes> hello_out{};
    std::array<std::byte, kHelloBytes> hello_in{};
    bool eof = false;
    bool write_closed = false;
    // False once the descriptor has been handed back to the poller: a lane
    // that has read its EOF and has nothing left to write is not waited on.
    bool polled = true;

    std::deque<TxOp> tx;
    // Control lane: pre-formatted frames, FIFO, fixed capacity.
    std::vector<std::byte> ctrl_tx;
    std::size_t ctrl_head = 0;
    std::size_t ctrl_tail = 0;
    bool ctrl_refused = false;

    // Control lane receive staging.
    std::vector<std::byte> rx;
    std::size_t rx_len = 0;

    // Data lane receive state machine.
    std::array<std::byte, kFrameHeaderBytes> hdr_in{};
    std::size_t hdr_in_len = 0;
    bool in_frame = false;
    FrameHeader cur{};
    std::size_t frame_left = 0;

    std::deque<RxOp> posted;
    bool inbound_active = false;
    std::uint64_t inbound_op = 0;
    std::uint32_t inbound_tag = 0;
    std::uint64_t inbound_total = 0;
    std::uint64_t inbound_done = 0;
    std::byte* inbound_dst = nullptr;
    bool inbound_discard = false;

    bool offer_pending = false;
    std::uint32_t offer_tag = 0;
    std::uint64_t offer_since_ns = 0;
  };

  LaneIo& lane_of(Lane lane) noexcept {
    return lanes_[lane == Lane::kControl ? 0 : 1];
  }
  const LaneIo& lane_of(Lane lane) const noexcept {
    return lanes_[lane == Lane::kControl ? 0 : 1];
  }

  void build_hello(LaneIo& io);
  void arm(LaneIo& io);
  void service(LaneIo& io, Readiness r);
  void finish_connect(LaneIo& io);
  void drive_hello(LaneIo& io);
  void announce_open_if_ready();

  void drain_tx(LaneIo& io);
  void half_close_if_drained();
  void note_eof(LaneIo& io);
  void drain_rx_control(LaneIo& io);
  void drain_rx_data(LaneIo& io);
  bool bind_inbound(LaneIo& io);

  void complete(Lane lane, Direction dir, std::uint64_t op, std::uint32_t tag,
                std::size_t bytes, StatusCode code);
  void fault(Status why, StatusCode outstanding);
  void retire(StatusCode outstanding);
  void emit_closed(StatusCode code);

  Poller* poller_ = nullptr;
  EventSink* sink_ = nullptr;
  std::uint64_t channel_ = 0;
  Endpoint peer_;
  Capabilities caps_{};
  LinkOptions opt_{};
  std::uint64_t id_hi_ = 0;
  std::uint64_t id_lo_ = 0;
  bool client_ = false;
  bool announced_ = false;
  bool closing_ = false;
  std::uint64_t close_by_ns_ = 0;
  bool retired_ = false;
  std::uint32_t peer_ctrl_ring_ = kDefaultControlRingBytes;
  Status error_;
  std::array<LaneIo, 2> lanes_{};
  std::vector<std::byte> discard_;
};

}  // namespace lse::comm
