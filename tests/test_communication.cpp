// Endpoints, channels and the loop that drives them.
//
// Client and server both run in this process, over a real kernel socket bound
// to an ephemeral port and over a real abstract unix socket, so every case here
// runs on a bare machine with no network and no device. What one box cannot
// prove is named where it matters: a link's real bandwidth, a dmabuf DMA'd
// straight out of GPU memory, a completion queue that has to be armed, and a
// NIC failing halfway through a frame. Those are asserted as declines, not as
// numbers.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "harness.hpp"
#include "lse/communication/capabilities.hpp"
#include "lse/communication/endpoint.hpp"
#include "lse/communication/event.hpp"
#include "lse/communication/reactor.hpp"
#include "lse/communication/transport.hpp"

using namespace lse;
using namespace lse::comm;

namespace {

constexpr std::uint64_t kSweepNs = 200'000;

std::size_t thread_count() {
  std::size_t n = 0;
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::directory_iterator("/proc/self/task", ec)) {
    (void)entry;
    ++n;
  }
  return n;
}

// One reactor plus everything it has been told, with control payloads copied
// out while they are still alive.
struct Pump {
  explicit Pump(Reactor& r) : rx(&r) {}

  Reactor* rx;
  std::vector<Event> log;
  std::vector<std::string> control;
  std::array<Event, 64> buf{};

  std::size_t once(std::uint64_t timeout_ns) {
    auto got = rx->poll(buf, timeout_ns);
    if (!got.ok()) return 0;
    for (std::size_t i = 0; i < *got; ++i) {
      log.push_back(buf[i]);
      if (buf[i].kind == EventKind::kControl && buf[i].data != nullptr) {
        control.emplace_back(reinterpret_cast<const char*>(buf[i].data),
                             buf[i].bytes);
      } else {
        control.emplace_back();
      }
    }
    return *got;
  }

  [[nodiscard]] std::size_t count(EventKind kind) const {
    std::size_t n = 0;
    for (const Event& e : log) {
      if (e.kind == kind) ++n;
    }
    return n;
  }
  [[nodiscard]] const Event* first(EventKind kind) const {
    for (const Event& e : log) {
      if (e.kind == kind) return &e;
    }
    return nullptr;
  }
  [[nodiscard]] const Event* last() const {
    return log.empty() ? nullptr : &log.back();
  }
};

template <typename Pred>
bool pump_until(Pump& a, Pump& b, Pred done, int budget_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    a.once(kSweepNs);
    b.once(kSweepNs);
    if (done()) return true;
  }
  return done();
}

Endpoint must_parse(std::string_view text) {
  auto ep = Endpoint::parse(text);
  LSE_EXPECT(ep.ok());
  if (!ep.ok()) {
    std::printf("       parse failed: %s\n", ep.status().to_string().c_str());
    return Endpoint{};
  }
  return ep.release();
}

std::span<const std::byte> as_bytes(const std::string& s) {
  return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

}  // namespace

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

LSE_TEST(an_endpoint_round_trips_through_its_own_string) {
  const Endpoint a = must_parse("tcp://10.10.13.8:29500?ctrl_ring=65536");
  LSE_EXPECT(a.scheme() == "tcp");
  LSE_EXPECT(a.authority() == "10.10.13.8:29500");
  LSE_EXPECT(a.option("ctrl_ring") == "65536");
  const Endpoint again = must_parse(a.str());
  LSE_EXPECT(again.str() == a.str());

  // A bare name with no "://" is the scheme, and it canonicalises to one form.
  const Endpoint bare = must_parse("unix");
  LSE_EXPECT(bare.scheme() == "unix");
  LSE_EXPECT(bare.str() == "unix://");
  LSE_EXPECT(must_parse(bare.str()).str() == bare.str());

  const Endpoint pipe = must_parse("unix:///run/lse/plan.sock?deadline_ms=500");
  LSE_EXPECT(pipe.path() == "/run/lse/plan.sock");
  LSE_EXPECT(must_parse(pipe.str()).str() == pipe.str());

  // Out of band wins, so a launch flag overrides a stored endpoint without
  // rewriting it.
  auto merged = Endpoint::parse("tcp://127.0.0.1:1?ctrl_ring=4096",
                                {{"ctrl_ring", "8192"}, {"offer_ms", "250"}});
  LSE_EXPECT(merged.ok());
  if (!merged.ok()) return;
  LSE_EXPECT(merged->option("ctrl_ring") == "8192");
  LSE_EXPECT(merged->option("offer_ms") == "250");
  LSE_EXPECT(must_parse(merged->str()).str() == merged->str());
}

LSE_TEST(an_unknown_option_key_is_an_error_not_a_no_op) {
  const Endpoint typo = must_parse("tcp://127.0.0.1:1?ctrl_rng=4096");
  const Status s = check_options(typo, {});
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kInvalidArgument);
  LSE_EXPECT(s.message().find("ctrl_rng") != std::string::npos);
  // And the error lists what the transport does read, so the fix is in the
  // message rather than in the source.
  LSE_EXPECT(s.message().find("ctrl_ring") != std::string::npos);

  auto r = Reactor::create();
  LSE_EXPECT(r.ok());
  if (!r.ok()) return;
  Reactor rx = r.release();
  auto refused = rx.listen(typo);
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.status().message().find("ctrl_rng") != std::string::npos);
}

// One transport instance serves every endpoint of its scheme on a reactor, so a
// check that runs when that instance is created only ever sees the first peer.
// The engine's shape is many peers on one scheme, which makes the SECOND
// endpoint the interesting one.
LSE_TEST(a_typo_on_the_second_peer_of_a_scheme_is_refused_too) {
  auto r = Reactor::create();
  LSE_EXPECT(r.ok());
  if (!r.ok()) return;
  Reactor rx = r.release();

  auto first = rx.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(first.ok());

  auto late_listen = rx.listen(must_parse("tcp://127.0.0.1:0?ctrl_rng=4096"));
  LSE_EXPECT(!late_listen.ok());
  LSE_EXPECT(late_listen.status().message().find("ctrl_rng") !=
             std::string::npos);

  auto late_connect = rx.connect(must_parse("tcp://127.0.0.1:1?deadlin_ms=5"));
  LSE_EXPECT(!late_connect.ok());
  LSE_EXPECT(late_connect.status().message().find("deadlin_ms") !=
             std::string::npos);

  // A knob that IS spelled right still opens on the same instance.
  auto good = rx.listen(must_parse("unix://@lse-comm-late-option?frame_max=64"));
  LSE_EXPECT(good.ok());
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

LSE_TEST(a_reactor_creates_no_threads) {
  const std::size_t before = thread_count();

  auto r = Reactor::create();
  LSE_EXPECT(r.ok());
  if (!r.ok()) return;
  Reactor rx = r.release();
  auto ln = rx.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = rx.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;

  std::array<Event, 32> evs{};
  for (int i = 0; i < 100; ++i) (void)rx.poll(evs, kSweepNs);

  LSE_EXPECT_EQ(thread_count(), before);
  std::printf("       threads before %zu, after %zu\n", before,
              thread_count());
}

LSE_TEST(a_client_and_a_server_move_bytes_over_a_real_socket) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  LSE_EXPECT(sr.ok() && cr.ok());
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  // Port 0: the kernel picks, and Listener::endpoint() reports what it picked,
  // so nothing here races another test for a number.
  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  const std::string dial = ln->endpoint().str();

  auto c = client.connect(must_parse(dial));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  const std::string plan = "kernel=matmul window=[0,512) owner=peer-2";
  auto posted = out.post_control(as_bytes(plan), 0x41);
  LSE_EXPECT(posted.ok());
  if (!posted.ok()) return;

  std::vector<std::byte> payload(1u << 20);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::byte>(i * 7u + 11u);
  }
  std::vector<std::byte> landing(payload.size(), std::byte{0});

  Channel in = server.channel(accepted_channel);
  LSE_EXPECT(in.valid());
  Transfer want;
  want.region = host_region(landing.data(), landing.size());
  want.bytes = landing.size();
  want.tag = 0x77;
  auto recv = in.post_recv(want);
  LSE_EXPECT(recv.ok());

  Transfer give;
  give.region = host_region(payload.data(), payload.size());
  give.bytes = payload.size();
  give.tag = 0x77;
  auto send = out.post_send(give);
  LSE_EXPECT(send.ok());
  if (!send.ok() || !recv.ok()) return;

  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.count(EventKind::kRecvComplete) == 1 &&
                                 cp.count(EventKind::kSendComplete) == 2 &&
                                 sp.count(EventKind::kControl) == 1;
                        },
                        8000));

  LSE_EXPECT_EQ(sp.count(EventKind::kControl), std::size_t{1});
  const Event* ctrl = sp.first(EventKind::kControl);
  if (ctrl != nullptr) {
    LSE_EXPECT_EQ(ctrl->tag, 0x41u);
    for (std::size_t i = 0; i < sp.log.size(); ++i) {
      if (sp.log[i].kind == EventKind::kControl) {
        LSE_EXPECT(sp.control[i] == plan);
      }
    }
  }

  const Event* done = sp.first(EventKind::kRecvComplete);
  LSE_EXPECT(done != nullptr);
  if (done != nullptr) {
    LSE_EXPECT(done->code == StatusCode::kOk);
    LSE_EXPECT_EQ(done->bytes, payload.size());
  }
  LSE_EXPECT_EQ(std::memcmp(landing.data(), payload.data(), payload.size()), 0);
  LSE_EXPECT(out.done(*send));
}

LSE_TEST(a_pipe_name_reaches_the_same_peer_as_an_address_does) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  LSE_EXPECT(sr.ok() && cr.ok());
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  // An abstract name: no filesystem entry, so nothing is left behind by a
  // crash and no test races another for a path.
  const std::string pipe =
      "unix://@lse-comm-" + std::to_string(::getpid()) + "-pipe";
  auto ln = server.listen(must_parse(pipe));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(pipe));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  LSE_EXPECT(sp.first(EventKind::kAccepted) != nullptr);
  if (sp.first(EventKind::kAccepted) == nullptr) return;

  const std::string note = "the caller named a pipe and nothing else changed";
  LSE_EXPECT(out.post_control(as_bytes(note)).ok());
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return sp.count(EventKind::kControl) == 1; },
                        4000));
  for (std::size_t i = 0; i < sp.log.size(); ++i) {
    if (sp.log[i].kind == EventKind::kControl) LSE_EXPECT(sp.control[i] == note);
  }
}

LSE_TEST(a_large_transfer_is_segmented_and_reassembled) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  LSE_EXPECT(sr.ok() && cr.ok());
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  // 4091 is ODL_TB5_STREAM_PAYLOAD_MAX. Running the segmenter at that frame
  // size over a socket is how the DMA-ring path is reviewed before the ring
  // exists.
  auto ln = server.listen(must_parse("unix://@lse-comm-seg-" +
                                     std::to_string(::getpid()) +
                                     "?frame_max=4091"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  constexpr std::size_t kBytes = 4u << 20;
  std::vector<std::byte> src(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    src[i] = static_cast<std::byte>((i * 31u) ^ (i >> 8));
  }
  std::vector<std::byte> dst(kBytes, std::byte{0xAB});

  Channel in = server.channel(accepted_channel);
  Transfer want;
  want.region = host_region(dst.data(), dst.size());
  want.bytes = dst.size();
  want.tag = 3;
  LSE_EXPECT(in.post_recv(want).ok());

  Transfer give;
  give.region = host_region(src.data(), src.size());
  give.bytes = src.size();
  give.tag = 3;
  LSE_EXPECT(out.post_send(give).ok());

  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.count(EventKind::kRecvComplete) == 1 &&
                                 cp.count(EventKind::kSendComplete) == 1;
                        },
                        20000));
  const Event* done = sp.first(EventKind::kRecvComplete);
  LSE_EXPECT(done != nullptr);
  if (done == nullptr) return;
  LSE_EXPECT(done->code == StatusCode::kOk);
  LSE_EXPECT_EQ(done->bytes, kBytes);
  LSE_EXPECT_EQ(std::memcmp(dst.data(), src.data(), kBytes), 0);
  std::printf("       4 MiB reassembled from %zu frames of 4091 B\n",
              (kBytes + 4090) / 4091);
}

LSE_TEST(the_receiving_region_is_written_exactly_once_and_only_where_told) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("unix://@lse-comm-window-" +
                                     std::to_string(::getpid())));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  constexpr std::size_t kGuard = 4096;
  constexpr std::size_t kBody = 64 * 1024;
  std::vector<std::byte> arena(kGuard * 2 + kBody, std::byte{0x5A});
  std::vector<std::byte> body(kBody);
  for (std::size_t i = 0; i < kBody; ++i) {
    body[i] = static_cast<std::byte>(i & 0xFFu);
  }

  Channel in = server.channel(accepted_channel);
  Transfer want;
  want.region = host_region(arena.data(), arena.size());
  want.offset = kGuard;
  want.bytes = kBody;
  want.tag = 9;
  LSE_EXPECT(in.post_recv(want).ok());

  Transfer give;
  give.region = host_region(body.data(), body.size());
  give.bytes = body.size();
  give.tag = 9;
  LSE_EXPECT(out.post_send(give).ok());

  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return sp.count(EventKind::kRecvComplete) == 1; },
                        8000));
  LSE_EXPECT_EQ(std::memcmp(arena.data() + kGuard, body.data(), kBody), 0);
  bool guards_intact = true;
  for (std::size_t i = 0; i < kGuard; ++i) {
    if (arena[i] != std::byte{0x5A}) guards_intact = false;
    if (arena[kGuard + kBody + i] != std::byte{0x5A}) guards_intact = false;
  }
  LSE_EXPECT(guards_intact);
}

LSE_TEST(eight_concurrent_connections_each_get_their_own_bytes) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  const std::string dial = ln->endpoint().str();

  constexpr int kPeers = 8;
  std::vector<Channel> outs;
  for (int i = 0; i < kPeers; ++i) {
    auto c = client.connect(must_parse(dial));
    LSE_EXPECT(c.ok());
    if (!c.ok()) return;
    outs.push_back(c.release());
  }

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(
      sp, cp,
      [&] {
        return sp.count(EventKind::kAccepted) == std::size_t{kPeers} &&
               cp.count(EventKind::kConnected) == std::size_t{kPeers};
      },
      8000));
  LSE_EXPECT_EQ(sp.count(EventKind::kAccepted), std::size_t{kPeers});
  if (sp.count(EventKind::kAccepted) != std::size_t{kPeers}) return;

  for (int i = 0; i < kPeers; ++i) {
    const std::string note = "peer " + std::to_string(i);
    LSE_EXPECT(outs[static_cast<std::size_t>(i)]
                   .post_control(as_bytes(note), static_cast<std::uint32_t>(i))
                   .ok());
  }
  LSE_EXPECT(pump_until(
      sp, cp,
      [&] { return sp.count(EventKind::kControl) == std::size_t{kPeers}; },
      8000));

  // Every channel is distinct and every message landed on the one that sent it.
  std::vector<std::uint64_t> seen;
  for (std::size_t i = 0; i < sp.log.size(); ++i) {
    if (sp.log[i].kind != EventKind::kControl) continue;
    LSE_EXPECT(sp.control[i] == "peer " + std::to_string(sp.log[i].tag));
    seen.push_back(sp.log[i].channel);
  }
  LSE_EXPECT_EQ(seen.size(), std::size_t{kPeers});
  std::sort(seen.begin(), seen.end());
  LSE_EXPECT(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
  LSE_EXPECT_EQ(server.open_channels(), std::size_t{kPeers});
}

// ---------------------------------------------------------------------------
// The two lanes
// ---------------------------------------------------------------------------

LSE_TEST(a_control_message_arrives_while_the_data_lane_is_stalled) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  if (sp.first(EventKind::kAccepted) == nullptr) return;

  // Nobody will ever post a receive for this tag, so the data lane stops dead.
  std::vector<std::byte> bulk(8u << 20, std::byte{1});
  Transfer give;
  give.region = host_region(bulk.data(), bulk.size());
  give.bytes = bulk.size();
  give.tag = 0xBEEF;
  LSE_EXPECT(out.post_send(give).ok());
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return sp.count(EventKind::kDataOffered) == 1; },
                        8000));
  LSE_EXPECT_EQ(sp.count(EventKind::kDataOffered), std::size_t{1});
  const Event* offer = sp.first(EventKind::kDataOffered);
  if (offer != nullptr) LSE_EXPECT_EQ(offer->bytes, bulk.size());

  // The control lane is a different socket, so a plan still crosses.
  for (int i = 0; i < 16; ++i) {
    const std::string plan = "plan " + std::to_string(i);
    LSE_EXPECT(out.post_control(as_bytes(plan),
                                static_cast<std::uint32_t>(i)).ok());
  }
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return sp.count(EventKind::kControl) == 16; },
                        8000));
  LSE_EXPECT_EQ(sp.count(EventKind::kControl), std::size_t{16});
  LSE_EXPECT_EQ(sp.count(EventKind::kClosed), std::size_t{0});
  LSE_EXPECT_EQ(sp.count(EventKind::kRecvComplete), std::size_t{0});
}

LSE_TEST(an_unanswered_offer_faults_the_channel_by_deadline) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("unix://@lse-comm-offer-" +
                                     std::to_string(::getpid()) +
                                     "?offer_ms=200"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  std::vector<std::byte> bulk(1u << 20, std::byte{2});
  Transfer give;
  give.region = host_region(bulk.data(), bulk.size());
  give.bytes = bulk.size();
  give.tag = 5;
  LSE_EXPECT(out.post_send(give).ok());

  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return sp.count(EventKind::kClosed) >= 1; },
                        8000));
  const Event* offered = sp.first(EventKind::kDataOffered);
  LSE_EXPECT(offered != nullptr);
  const Event* closed = sp.first(EventKind::kClosed);
  LSE_EXPECT(closed != nullptr);
  if (closed == nullptr) return;
  LSE_EXPECT(closed->code == StatusCode::kCancelled);
  const Status& why = server.last_error(accepted_channel);
  LSE_EXPECT(why.message().find("tag 5") != std::string::npos);
  std::printf("       %s\n", why.message().c_str());
}

LSE_TEST(a_full_control_ring_refuses_and_then_says_when_it_is_writable) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  const std::string base =
      "unix://@lse-comm-ring-" + std::to_string(::getpid());
  auto ln = server.listen(must_parse(base + "?ctrl_ring=4096&max_inflight=65536"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(
      must_parse(base + "?ctrl_ring=4096&max_inflight=65536"));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.first(EventKind::kConnected) != nullptr; },
                        4000));
  if (cp.first(EventKind::kConnected) == nullptr) return;

  // The server is not polled from here on, so the socket stops draining and the
  // ring is what refuses.
  const std::string chunk(2000, 'x');
  Status refusal;
  int accepted = 0;
  for (int i = 0; i < 200000; ++i) {
    auto posted = out.post_control(as_bytes(chunk));
    if (posted.ok()) {
      ++accepted;
      continue;
    }
    refusal = posted.status();
    break;
  }
  LSE_EXPECT(!refusal.ok());
  LSE_EXPECT(refusal.code() == StatusCode::kOutOfMemory);
  LSE_EXPECT(refusal.message().find("control ring") != std::string::npos);
  LSE_EXPECT(out.credit(Lane::kControl) < chunk.size());
  std::printf("       %d messages of 2000 B before the 4096 B ring refused\n",
              accepted);

  // Draining the peer frees the ring, and the caller is told rather than having
  // to poll for it.
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.count(EventKind::kWritable) >= 1; },
                        20000));
  LSE_EXPECT(cp.count(EventKind::kWritable) >= 1);
  LSE_EXPECT(out.credit(Lane::kControl) > 0);
}

// ---------------------------------------------------------------------------
// Completion, cancellation and death
// ---------------------------------------------------------------------------

LSE_TEST(every_accepted_post_completes_exactly_once) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.first(EventKind::kConnected) != nullptr; },
                        4000));
  if (cp.first(EventKind::kConnected) == nullptr) return;

  constexpr int kPosts = 16;
  std::vector<std::byte> bulk(16u << 20, std::byte{7});
  int posted = 0;
  for (int i = 0; i < kPosts; ++i) {
    Transfer give;
    give.region = host_region(bulk.data(), bulk.size());
    give.bytes = bulk.size();
    give.tag = static_cast<std::uint32_t>(i);
    if (out.post_send(give).ok()) ++posted;
  }
  const std::string plan = "a plan that will not be finished";
  if (out.post_control(as_bytes(plan)).ok()) ++posted;

  // Killed with bytes still moving.
  cp.once(kSweepNs);
  out.abort();
  cp.once(kSweepNs);

  const std::size_t completions =
      cp.count(EventKind::kSendComplete) + cp.count(EventKind::kRecvComplete);
  LSE_EXPECT_EQ(completions, static_cast<std::size_t>(posted));
  LSE_EXPECT_EQ(cp.count(EventKind::kClosed), std::size_t{1});
  const Event* tail = cp.last();
  LSE_EXPECT(tail != nullptr && tail->kind == EventKind::kClosed);
  std::printf("       %d posts, %zu completions, closed last\n", posted,
              completions);
}

LSE_TEST(a_stale_ticket_never_aliases_a_live_one) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  const std::string base = "unix://@lse-comm-slot-" + std::to_string(::getpid()) +
                           "?max_inflight=1";
  auto ln = server.listen(must_parse(base));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(base));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.first(EventKind::kConnected) != nullptr; },
                        4000));
  if (cp.first(EventKind::kConnected) == nullptr) return;

  const std::string one = "first";
  auto a = out.post_control(as_bytes(one));
  LSE_EXPECT(a.ok());
  if (!a.ok()) return;
  const Ticket first = *a;
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.count(EventKind::kSendComplete) == 1; },
                        4000));
  LSE_EXPECT(out.done(first));

  // One slot, so the next post must reuse it — and the old ticket must not name
  // the new transfer.
  const std::string two = "second";
  auto b = out.post_control(as_bytes(two));
  LSE_EXPECT(b.ok());
  if (!b.ok()) return;
  LSE_EXPECT(b->op != first.op);
  LSE_EXPECT(!out.done(first));
  LSE_EXPECT(out.cancel(first).code() == StatusCode::kNotFound);

  // A retired channel id names nothing either.
  const std::uint64_t id = out.id();
  out.abort();
  cp.once(kSweepNs);
  LSE_EXPECT(!client.channel(id).valid());
}

LSE_TEST(a_queued_transfer_can_be_cancelled_and_a_moving_one_cannot) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.first(EventKind::kConnected) != nullptr; },
                        4000));
  if (cp.first(EventKind::kConnected) == nullptr) return;

  std::vector<std::byte> bulk(32u << 20, std::byte{3});
  Transfer first;
  first.region = host_region(bulk.data(), bulk.size());
  first.bytes = bulk.size();
  first.tag = 1;
  auto moving = out.post_send(first);
  LSE_EXPECT(moving.ok());
  Transfer second = first;
  second.tag = 2;
  auto queued = out.post_send(second);
  LSE_EXPECT(queued.ok());
  if (!moving.ok() || !queued.ok()) return;

  // The first is partly on the wire and cannot be un-sent; the second has not
  // started and simply never will.
  LSE_EXPECT(out.cancel(*moving).code() == StatusCode::kOutOfRange);
  LSE_EXPECT_OK(out.cancel(*queued));
  cp.once(kSweepNs);
  bool saw_cancel = false;
  for (const Event& e : cp.log) {
    if (e.kind == EventKind::kSendComplete && e.op == queued->op) {
      saw_cancel = e.code == StatusCode::kCancelled;
    }
  }
  LSE_EXPECT(saw_cancel);
  LSE_EXPECT(out.done(*queued));
  LSE_EXPECT(!out.done(*moving));
}

LSE_TEST(a_peer_that_dies_mid_transfer_completes_every_ticket_then_closes) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  constexpr std::size_t kBytes = 64u << 20;
  std::vector<std::byte> bulk(kBytes, std::byte{9});
  std::vector<std::byte> landing(kBytes, std::byte{0});
  Channel in = server.channel(accepted_channel);
  Transfer want;
  want.region = host_region(landing.data(), landing.size());
  want.bytes = landing.size();
  want.tag = 4;
  LSE_EXPECT(in.post_recv(want).ok());
  Transfer give;
  give.region = host_region(bulk.data(), bulk.size());
  give.bytes = bulk.size();
  give.tag = 4;
  auto sending = out.post_send(give);
  LSE_EXPECT(sending.ok());
  if (!sending.ok()) return;

  // Let some of it move, then kill the far end outright.
  for (int i = 0; i < 8; ++i) {
    sp.once(kSweepNs);
    cp.once(kSweepNs);
  }
  in.abort();

  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.count(EventKind::kClosed) == 1; },
                        20000));
  LSE_EXPECT_EQ(cp.count(EventKind::kClosed), std::size_t{1});
  LSE_EXPECT_EQ(cp.count(EventKind::kSendComplete), std::size_t{1});
  const Event* done = nullptr;
  for (const Event& e : cp.log) {
    if (e.kind == EventKind::kSendComplete) done = &e;
  }
  LSE_EXPECT(done != nullptr);
  if (done != nullptr) {
    LSE_EXPECT(done->code == StatusCode::kIoError);
    // It says how far it got, which is what a caller needs to replay from.
    std::printf("       peer died after %zu of %zu B\n", done->bytes, kBytes);
  }
  const Event* tail = cp.last();
  LSE_EXPECT(tail != nullptr && tail->kind == EventKind::kClosed);
  LSE_EXPECT(!client.last_error(out.id()).message().empty());
}

LSE_TEST(a_graceful_close_delivers_everything_that_was_queued) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.first(EventKind::kAccepted) != nullptr &&
                                 cp.first(EventKind::kConnected) != nullptr;
                        },
                        4000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  const std::uint64_t accepted_channel = accepted->channel;

  constexpr std::size_t kBytes = 4u << 20;
  std::vector<std::byte> src(kBytes, std::byte{0x21});
  std::vector<std::byte> dst(kBytes, std::byte{0});
  Channel in = server.channel(accepted_channel);
  Transfer want;
  want.region = host_region(dst.data(), dst.size());
  want.bytes = dst.size();
  want.tag = 6;
  LSE_EXPECT(in.post_recv(want).ok());

  const std::string last_word = "this is the last thing I will say";
  LSE_EXPECT(out.post_control(as_bytes(last_word)).ok());
  Transfer give;
  give.region = host_region(src.data(), src.size());
  give.bytes = src.size();
  give.tag = 6;
  LSE_EXPECT(out.post_send(give).ok());

  // Closed immediately, with megabytes still queued: graceful means the peer
  // still gets all of it.
  LSE_EXPECT_OK(out.close());
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return sp.count(EventKind::kClosed) >= 1 &&
                                 cp.count(EventKind::kClosed) >= 1;
                        },
                        20000));

  LSE_EXPECT_EQ(sp.count(EventKind::kControl), std::size_t{1});
  for (std::size_t i = 0; i < sp.log.size(); ++i) {
    if (sp.log[i].kind == EventKind::kControl) {
      LSE_EXPECT(sp.control[i] == last_word);
    }
  }
  const Event* landed = sp.first(EventKind::kRecvComplete);
  LSE_EXPECT(landed != nullptr);
  if (landed != nullptr) {
    LSE_EXPECT(landed->code == StatusCode::kOk);
    LSE_EXPECT_EQ(landed->bytes, kBytes);
  }
  LSE_EXPECT_EQ(std::memcmp(dst.data(), src.data(), kBytes), 0);
  LSE_EXPECT_EQ(cp.count(EventKind::kSendComplete), std::size_t{2});
  for (const Event& e : cp.log) {
    if (e.kind == EventKind::kSendComplete) {
      LSE_EXPECT(e.code == StatusCode::kOk);
    }
  }
  const Event* tail = cp.last();
  LSE_EXPECT(tail != nullptr && tail->kind == EventKind::kClosed);
  if (tail != nullptr) LSE_EXPECT(tail->code == StatusCode::kOk);
}

LSE_TEST(a_transfer_that_misses_its_deadline_is_cancelled) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("unix://@lse-comm-deadline-" +
                                     std::to_string(::getpid())));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = client.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel out = c.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.first(EventKind::kConnected) != nullptr; },
                        4000));
  if (cp.first(EventKind::kConnected) == nullptr) return;

  // Nothing will ever carry this tag, so the only thing that ends this receive
  // is its own deadline.
  std::vector<std::byte> landing(4096);
  Transfer want;
  want.region = host_region(landing.data(), landing.size());
  want.bytes = landing.size();
  want.tag = 99;
  want.deadline_ns = steady_now_ns() + 200'000'000ull;
  auto waiting = out.post_recv(want);
  LSE_EXPECT(waiting.ok());
  if (!waiting.ok()) return;

  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.count(EventKind::kRecvComplete) == 1; },
                        8000));
  const Event* late = cp.first(EventKind::kRecvComplete);
  LSE_EXPECT(late != nullptr);
  if (late == nullptr) return;
  LSE_EXPECT(late->code == StatusCode::kCancelled);
  LSE_EXPECT_EQ(late->tag, 99u);
  LSE_EXPECT(out.done(*waiting));
  // The channel survives: only the transfer had a deadline, not the connection.
  LSE_EXPECT_EQ(cp.count(EventKind::kClosed), std::size_t{0});
}

LSE_TEST(a_name_is_resolved_before_the_loop_is_entered_not_inside_it) {
  auto r = Reactor::create();
  if (!r.ok()) return;
  Reactor rx = r.release();

  // getaddrinfo blocks, and nothing inside the loop is allowed to, so an
  // endpoint handed to connect() must already carry a literal address.
  auto refused = rx.connect(must_parse("tcp://localhost:29500"));
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.status().message().find("resolve") != std::string::npos);

  auto resolved = resolve(must_parse("tcp://localhost:29500"), 2000);
  if (!resolved.ok()) {
    std::printf("       skipped: localhost does not resolve here (%s)\n",
                resolved.status().to_string().c_str());
    return;
  }
  LSE_EXPECT(!resolved->empty());
  bool literal = false;
  for (const Endpoint& one : *resolved) {
    std::printf("       localhost -> %s\n", one.str().c_str());
    if (one.authority().find("127.0.0.1") != std::string_view::npos ||
        one.authority().find("::1") != std::string_view::npos) {
      literal = true;
    }
  }
  LSE_EXPECT(literal);
}

// One caller flow — resolve, select, connect — has to be correct for every
// endpoint. A pipe name is not a hostname, so resolving it must hand it back
// untouched; failing to look it up would force the caller to write
// `if (scheme == ...)`, which is the branch this seam exists to remove.
LSE_TEST(resolving_a_pipe_name_hands_it_back_instead_of_failing) {
  for (const char* text : {"unix://@lse-resolve-passthrough",
                           "unix:///run/lse-resolve.sock"}) {
    const Endpoint ep = must_parse(text);
    auto resolved = resolve(ep, 2000);
    LSE_EXPECT(resolved.ok());
    if (!resolved.ok()) {
      std::printf("       %s -> %s\n", text,
                  resolved.status().to_string().c_str());
      continue;
    }
    LSE_EXPECT_EQ(resolved->size(), std::size_t{1});
    LSE_EXPECT((*resolved)[0].str() == ep.str());
    std::printf("       %s -> %s\n", text, (*resolved)[0].str().c_str());
  }
}

// ---------------------------------------------------------------------------
// Capability, selection and declines
// ---------------------------------------------------------------------------

LSE_TEST(a_staged_device_path_costs_twice_what_a_direct_one_does) {
  Capabilities direct;
  direct.device_memory_direct = true;
  direct.registers_memory = true;
  direct.bandwidth_bytes_per_s = 1'000'000'000ull;
  direct.latency_ns = 1000;

  Capabilities staged = direct;
  staged.device_memory_direct = false;

  LSE_EXPECT(direct.moves_device_bytes_in_place());
  LSE_EXPECT(!staged.moves_device_bytes_in_place());

  constexpr std::size_t kBytes = 1u << 20;
  const std::uint64_t d = direct.predicted_ns(kBytes, true);
  const std::uint64_t s = staged.predicted_ns(kBytes, true);
  LSE_EXPECT_EQ(s - direct.latency_ns, 2 * (d - direct.latency_ns));
  // Host bytes are not doubled on either: the rule is about who can DMA out of
  // device memory, not about the link being slow.
  LSE_EXPECT_EQ(direct.predicted_ns(kBytes, false),
                staged.predicted_ns(kBytes, false));

  Capabilities unmeasured;
  unmeasured.bandwidth_bytes_per_s = 0;
  LSE_EXPECT_EQ(unmeasured.predicted_ns(kBytes, false), std::uint64_t{0});
}

LSE_TEST(selection_names_every_candidate_it_passed_over) {
  const std::array<Endpoint, 3> rails{
      must_parse("rdma://10.10.13.8:4791"),
      must_parse("infiniband://nope"),
      must_parse("tcp://10.10.13.8:29500"),
  };
  Requirements need;
  need.largest_message_bytes = 1u << 20;
  auto chosen = select_endpoint(rails, need);
  LSE_EXPECT(chosen.ok());
  if (!chosen.ok()) return;
  LSE_EXPECT_EQ(chosen->index, std::size_t{2});
  LSE_EXPECT(!chosen->reason.empty());
  // The losers are named with the reason each lost, so the decision is
  // inspectable rather than inferred.
  LSE_EXPECT(chosen->reason.find("rdma://") != std::string::npos);
  LSE_EXPECT(chosen->reason.find("infiniband://") != std::string::npos);
  std::printf("       %s\n", chosen->reason.c_str());

  const std::array<Endpoint, 1> nothing{must_parse("rdma://10.10.13.8:4791")};
  auto none = select_endpoint(nothing, need);
  LSE_EXPECT(!none.ok());
  LSE_EXPECT(none.status().code() == StatusCode::kUnimplemented);
}

LSE_TEST(a_transport_that_cannot_take_a_dmabuf_declines_by_name) {
  auto r = Reactor::create();
  if (!r.ok()) return;
  Reactor rx = r.release();
  auto ln = rx.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto c = rx.connect(must_parse(std::string(ln->endpoint().str())));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  Channel ch = c.release();

  RegionRequest exported;
  exported.dmabuf_fd = 3;
  exported.bytes = 4096;
  auto region = rx.register_region(ch, exported);
  LSE_EXPECT(!region.ok());
  LSE_EXPECT(region.status().code() == StatusCode::kUnimplemented);
  LSE_EXPECT(region.status().message().find("dmabuf") != std::string::npos);
  std::printf("       %s\n", region.status().message().c_str());

  // A host range needs no registration at all, and saying so is an answer, not
  // a failure.
  std::vector<std::byte> host(4096);
  RegionRequest plain;
  plain.host = host.data();
  plain.bytes = host.size();
  auto ordinary = rx.register_region(ch, plain);
  LSE_EXPECT(ordinary.ok());
  if (ordinary.ok()) {
    LSE_EXPECT_EQ(ordinary->id, std::uint64_t{0});
    LSE_EXPECT(ordinary->host_addressable());
  }

  LSE_EXPECT(!ch.capabilities().registers_memory);
  LSE_EXPECT(!ch.capabilities().device_memory_direct);
}

LSE_TEST(rdma_declines_by_naming_what_this_machine_is_missing) {
  auto made = create_transport("rdma://10.10.13.8:4791");
  LSE_EXPECT(made.ok());
  if (!made.ok()) {
    std::printf("       skipped: rdma:// is not in this build\n");
    return;
  }
  const Endpoint ep = must_parse("rdma://10.10.13.8:4791");
  const std::string_view why = (*made)->declined(ep);
  if (why.empty()) {
    // A machine that grew a NIC: the seam is the same, the answer is not.
    std::printf("       this machine has a usable verbs fabric\n");
    return;
  }
  std::printf("       %.*s\n", static_cast<int>(why.size()), why.data());
  LSE_EXPECT(!why.empty());

  // And a caller that dials it gets that sentence, not a crash and not a
  // silent demotion onto a slower path.
  auto r = Reactor::create();
  if (!r.ok()) return;
  Reactor rx = r.release();
  auto refused = rx.connect(ep);
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.status().message().find("rdma") != std::string::npos);
  LSE_EXPECT_EQ(rx.open_channels(), std::size_t{0});
}

LSE_TEST(tb5_declines_by_naming_what_this_machine_is_missing) {
  auto made = create_transport("tb5:///dev/odl_tb5_0?stream=7");
  LSE_EXPECT(made.ok());
  if (!made.ok()) {
    std::printf("       skipped: tb5:// is not in this build\n");
    return;
  }
  auto ep = Endpoint::parse("tb5:///dev/odl_tb5_0");
  LSE_EXPECT(ep.ok());
  if (!ep.ok()) return;
  const std::string_view why = (*made)->declined(*ep);
  if (why.empty()) {
    std::printf("       this machine has a usable OdinLink ring\n");
    return;
  }
  std::printf("       %.*s\n", static_cast<int>(why.size()), why.data());
  LSE_EXPECT(why.find("odl_tb5") != std::string_view::npos);
}

LSE_TEST(an_unregistered_scheme_declines_and_lists_what_is_registered) {
  auto missing = create_transport("verbs://nope");
  LSE_EXPECT(!missing.ok());
  LSE_EXPECT(missing.status().code() == StatusCode::kNotFound);
  const std::string& listed = missing.status().message();
  LSE_EXPECT(listed.find("tcp") != std::string::npos);
  LSE_EXPECT(listed.find("unix") != std::string::npos);
  LSE_EXPECT(listed.find("rdma") != std::string::npos);
  LSE_EXPECT(listed.find("tb5") != std::string::npos);

  const std::vector<std::string> schemes = available_transports();
  LSE_EXPECT(schemes.size() >= 4);
  std::string all;
  for (const std::string& s : schemes) {
    if (!all.empty()) all += ", ";
    all += s;
  }
  std::printf("       registered: %s\n", all.c_str());
}

// ---------------------------------------------------------------------------
// Cost, and what this box cannot measure
// ---------------------------------------------------------------------------

LSE_TEST(a_remote_peer_is_reached_by_endpoint_alone) {
  const char* peer = std::getenv("LSE_PEER");
  if (peer == nullptr) {
    std::printf("       skipped: set LSE_PEER=tcp://host:port to dial a peer\n");
    return;
  }
  auto ep = Endpoint::parse(peer, {{"deadline_ms", "2000"}});
  LSE_EXPECT(ep.ok());
  if (!ep.ok()) return;
  auto resolved = resolve(*ep, 2000);
  if (!resolved.ok()) {
    std::printf("       skipped: %s\n", resolved.status().to_string().c_str());
    return;
  }
  auto r = Reactor::create();
  if (!r.ok()) return;
  Reactor rx = r.release();
  auto c = rx.connect(resolved->front());
  if (!c.ok()) {
    std::printf("       skipped: %s\n", c.status().to_string().c_str());
    return;
  }
  Channel ch = c.release();
  Pump p(rx);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline &&
         p.first(EventKind::kConnected) == nullptr &&
         p.first(EventKind::kClosed) == nullptr) {
    p.once(1'000'000);
  }
  if (p.first(EventKind::kConnected) == nullptr) {
    std::printf("       skipped: %s did not answer\n", peer);
    return;
  }
  const std::string hello = "the engine reached this peer by its name alone";
  auto said = ch.post_control(as_bytes(hello));
  LSE_EXPECT(said.ok());
  if (!said.ok()) return;
  const auto sent_by =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < sent_by && !ch.done(*said)) {
    p.once(1'000'000);
  }
  LSE_EXPECT(ch.done(*said));
  // Graceful, not abort: a completed send means the bytes reached the link, not
  // that the peer read them, so leaving without closing would let a reset throw
  // the plan away.
  LSE_EXPECT_OK(ch.close());
  const auto gone_by =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < gone_by &&
         p.count(EventKind::kClosed) == 0) {
    p.once(1'000'000);
  }
  LSE_EXPECT(p.count(EventKind::kClosed) == 1);
  // Deliberately prints no rate: node-1 to node-2 is a WAN VPN today, and a
  // number measured over it would describe the VPN, not a data plane.
  std::printf("       reached %s and handed it a plan\n", peer);
}

LSE_TEST(a_non_blocking_sweep_costs_what_it_costs) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  auto ln = server.listen(must_parse("tcp://127.0.0.1:0"));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  const std::string dial = ln->endpoint().str();
  std::vector<Channel> idle;
  for (int i = 0; i < 4; ++i) {
    auto c = client.connect(must_parse(dial));
    LSE_EXPECT(c.ok());
    if (!c.ok()) return;
    idle.push_back(c.release());
  }
  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] { return cp.count(EventKind::kConnected) == 4; },
                        8000));

  std::array<Event, 16> evs{};
  constexpr int kSweeps = 100000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kSweeps; ++i) (void)client.poll(evs, 0);
  const auto dt = std::chrono::steady_clock::now() - t0;
  const double ns =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()) /
      kSweeps;
  // The only place this number may be quoted from.
  std::printf("       %.0f ns per non-blocking sweep, 4 idle channels\n", ns);
  LSE_EXPECT(ns > 0.0);
}

// epoll reports EPOLLHUP and EPOLLERR whether or not they were asked for, and
// both are level state. A lane that wants neither read nor write — a data lane
// holding an unanswered offer whose write half is shut — must therefore be off
// the poller entirely, or the loop is woken for ever and poll() stops being a
// wait.
LSE_TEST(a_stalled_offer_with_a_hung_up_peer_still_lets_the_loop_sleep) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  const std::string name = "unix://@lse-comm-hup-" + std::to_string(::getpid()) +
                           "?offer_ms=30000&deadline_ms=30000";
  auto ln = server.listen(must_parse(name));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto dialled = client.connect(must_parse(name));
  LSE_EXPECT(dialled.ok());
  if (!dialled.ok()) return;
  Channel out = dialled.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(sp, cp,
                        [&] {
                          return cp.count(EventKind::kConnected) == 1 &&
                                 sp.count(EventKind::kAccepted) == 1;
                        },
                        8000));
  const Event* accepted = sp.first(EventKind::kAccepted);
  LSE_EXPECT(accepted != nullptr);
  if (accepted == nullptr) return;
  Channel in = server.channel(accepted->channel);

  // Offered and never answered: the server stops reading the data lane.
  std::vector<std::byte> payload(4u << 20, std::byte{0x7E});
  Transfer t;
  t.region = host_region(payload.data(), payload.size());
  t.bytes = payload.size();
  t.tag = 5;
  LSE_EXPECT(out.post_send(t).ok());
  LSE_EXPECT(pump_until(
      sp, cp, [&] { return sp.count(EventKind::kDataOffered) == 1; }, 8000));

  // Both halves go down while the offer is still outstanding: the server's data
  // socket now has its write half shut and a FIN from the peer, which is
  // exactly the state epoll reports as a permanent hangup.
  LSE_EXPECT(in.close().ok());
  out.abort();
  for (int i = 0; i < 200; ++i) cp.once(kSweepNs);

  std::array<Event, 16> evs{};
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5; ++i) (void)server.poll(evs, 40'000'000);
  const auto slept =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0)
          .count();
  std::printf("       5 polls of 40 ms slept %lld ms\n",
              static_cast<long long>(slept));
  LSE_EXPECT(slept >= 150);
}

LSE_TEST(a_ticket_on_a_dead_channel_reads_as_done) {
  auto sr = Reactor::create();
  auto cr = Reactor::create();
  if (!sr.ok() || !cr.ok()) return;
  Reactor server = sr.release();
  Reactor client = cr.release();

  const std::string name =
      "unix://@lse-comm-done-" + std::to_string(::getpid());
  auto ln = server.listen(must_parse(name));
  LSE_EXPECT(ln.ok());
  if (!ln.ok()) return;
  auto dialled = client.connect(must_parse(name));
  LSE_EXPECT(dialled.ok());
  if (!dialled.ok()) return;
  Channel out = dialled.release();

  Pump sp(server);
  Pump cp(client);
  LSE_EXPECT(pump_until(
      sp, cp, [&] { return cp.count(EventKind::kConnected) == 1; }, 8000));

  std::vector<std::byte> payload(1u << 20, std::byte{0x41});
  Transfer t;
  t.region = host_region(payload.data(), payload.size());
  t.bytes = payload.size();
  t.tag = 1;
  auto posted = out.post_send(t);
  LSE_EXPECT(posted.ok());
  if (!posted.ok()) return;
  const Ticket ticket = posted.release();
  LSE_EXPECT(!out.done(ticket));

  // The channel dies under an in-flight ticket. Its completion is emitted
  // before kClosed, so nothing on it is outstanding any more and done() must
  // say so — a caller that waits on done() has no other way to stop waiting.
  out.abort();
  LSE_EXPECT(pump_until(
      sp, cp, [&] { return cp.count(EventKind::kClosed) == 1; }, 8000));
  LSE_EXPECT(cp.count(EventKind::kSendComplete) == 1);
  LSE_EXPECT(out.done(ticket));
}

LSE_TEST_MAIN()
