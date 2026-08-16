// Device qualification: the per-device profile, the pairwise link matrix, and
// the cost model both feed.
//
// Everything that does not need hardware is checked on injected profiles, so
// the placement algebra is covered on a machine with no GPU at all. The link
// matrix runs its ranks on threads over the loopback transport, which is what
// makes an ordered-pair matrix testable on a box with one device. The device
// probe itself is checked for what it must never do — invent a number — which
// is assertable with or without a device.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/dist/loopback.hpp"
#include "lse/dist/transport.hpp"
#include "lse/probe/cost_model.hpp"
#include "lse/probe/device_probe.hpp"
#include "lse/probe/link_probe.hpp"
#include "lse/probe/pool.hpp"
#include "lse/probe/profile.hpp"
#include "lse/probe/profile_store.hpp"

using namespace lse;
using namespace lse::probe;

namespace {

namespace fs = std::filesystem;

DeviceId dev(std::string backend, int ordinal) {
  return DeviceId{std::move(backend), ordinal};
}

// A profile with the numbers a caller wants and nothing else, so a cost-model
// test states exactly the hardware it is reasoning about.
DeviceProfile fake_device(const DeviceId& id, double dram, double launch_ns,
                          double flops, math::MatrixElem native) {
  DeviceProfile d;
  d.id = id;
  d.arch = "test";
  d.name = "injected";
  d.dram_bytes_per_s = Measured::measured(dram);
  d.launch_overhead_ns = Measured::measured(launch_ns);
  for (math::MatrixElem e : profiled_operands()) {
    ComputePath p;
    p.operand = e;
    if (e == native) {
      p.executed_as = e;
      p.native = true;
      p.row_key = "test.native";
      p.flops = Measured::measured(flops);
    } else {
      // No row for this operand: the work still runs, in the form the device
      // does have, at that form's rate. Declared, because nothing timed the
      // dequantization in front of it.
      p.executed_as = native;
      p.native = false;
      p.row_key = "test.native";
      p.flops = Measured::declared(flops * 0.5);
    }
    d.paths.push_back(std::move(p));
  }
  return d;
}

LinkProfile fake_link(const DeviceId& a, const DeviceId& b, PathKind kind,
                      double latency_ns, double bandwidth) {
  LinkProfile l;
  l.src = a;
  l.dst = b;
  l.path = kind;
  l.latency_ns = Measured::measured(latency_ns);
  l.bandwidth_bytes_per_s = Measured::measured(bandwidth);
  return l;
}

PoolProfile two_device_pool(double dram_a, double flops_a, double dram_b,
                            double flops_b, double link_latency_ns,
                            double link_bandwidth) {
  PoolProfile pool;
  pool.fingerprint = "injected";
  pool.devices.push_back(fake_device(dev("hrx", 0), dram_a, 2000.0, flops_a,
                                     math::MatrixElem::kBF16));
  pool.devices.push_back(fake_device(dev("hrx", 1), dram_b, 2000.0, flops_b,
                                     math::MatrixElem::kBF16));
  const DeviceId& a = pool.devices[0].id;
  const DeviceId& b = pool.devices[1].id;
  pool.links = {
      fake_link(a, a, PathKind::kSameDevice, 0.0, 0.0),
      fake_link(a, b, PathKind::kPeerDirect, link_latency_ns, link_bandwidth),
      fake_link(b, a, PathKind::kPeerDirect, link_latency_ns, link_bandwidth),
      fake_link(b, b, PathKind::kSameDevice, 0.0, 0.0),
  };
  return pool;
}

std::vector<std::unique_ptr<dist::ITransport>> make_fleet(
    dist::Rank world, const std::string& group) {
  std::vector<std::unique_ptr<dist::ITransport>> out;
  for (dist::Rank r = 0; r < world; ++r) {
    auto t = dist::create_transport("loopback://");
    LSE_EXPECT(t.ok());
    if (!t.ok()) return {};
    dist::TransportConfig cfg;
    cfg.endpoint = "loopback://";
    cfg.rank = r;
    cfg.world_size = world;
    cfg.group_id = group;
    cfg.timeout_ms = 30000;
    LSE_EXPECT_OK((*t)->connect(cfg));
    out.push_back(t.release());
  }
  return out;
}

std::vector<LinkMember> loop_members(dist::Rank world) {
  std::vector<LinkMember> members;
  for (dist::Rank r = 0; r < world; ++r) {
    members.push_back(LinkMember{dev("loop", r), r, "test-host"});
  }
  return members;
}

fs::path scratch_dir() {
  return fs::temp_directory_path() /
         ("lse-probe-" + std::to_string(::getpid()));
}

backend::IBackend* host_backend() {
  static std::unique_ptr<backend::IBackend> be = [] {
    auto b = backend::create_backend("cpu");
    if (!b.ok()) return std::unique_ptr<backend::IBackend>{};
    auto owned = b.release();
    if (!owned->init(0).ok()) return std::unique_ptr<backend::IBackend>{};
    return owned;
  }();
  return be.get();
}

backend::IBackend* device_backend() {
  static std::unique_ptr<backend::IBackend> be = [] {
    auto b = backend::create_backend("hrx");
    if (!b.ok()) return std::unique_ptr<backend::IBackend>{};
    auto owned = b.release();
    if (!owned->init(0).ok()) return std::unique_ptr<backend::IBackend>{};
    return owned;
  }();
  return be.get();
}

}  // namespace

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

LSE_TEST(a_combined_number_inherits_its_weakest_input) {
  LSE_EXPECT(weaker(Provenance::kMeasured, Provenance::kUnknown) ==
             Provenance::kUnknown);
  LSE_EXPECT(weaker(Provenance::kMeasured, Provenance::kDeclared) ==
             Provenance::kDeclared);
  LSE_EXPECT(weaker(Provenance::kDeclared, Provenance::kUnsupported) ==
             Provenance::kUnsupported);
  LSE_EXPECT(!Measured::unknown().known());
  LSE_EXPECT(!Measured::unsupported().known());
  LSE_EXPECT(Measured::declared(1.0).known());
  LSE_EXPECT(Measured::measured(1.0).known());
  // An unknown carries no value to be mistaken for one.
  LSE_EXPECT_EQ(Measured::unknown().value, 0.0);
}

// ---------------------------------------------------------------------------
// The two-term link model
// ---------------------------------------------------------------------------

LSE_TEST(the_fit_separates_latency_from_bandwidth) {
  // Points generated from a known link: 5 us of latency and 25 GB/s.
  constexpr double kLatency = 5000.0;
  constexpr double kBandwidth = 25e9;
  LinkProfile l;
  l.src = dev("hrx", 0);
  l.dst = dev("hrx", 1);
  l.path = PathKind::kPeerDirect;
  for (std::size_t bytes : default_transfer_sizes()) {
    l.points.push_back(TransferPoint{
        bytes, kLatency + static_cast<double>(bytes) * 1e9 / kBandwidth});
  }
  fit_link(l);
  LSE_EXPECT(l.latency_ns.provenance == Provenance::kMeasured);
  LSE_EXPECT(l.bandwidth_bytes_per_s.provenance == Provenance::kMeasured);
  LSE_EXPECT_NEAR(l.latency_ns.value, kLatency, 1.0);
  LSE_EXPECT(std::fabs(l.bandwidth_bytes_per_s.value - kBandwidth) <
             kBandwidth * 1e-6);
  LSE_EXPECT(l.fit_error < 1e-6);

  // A single average GB/s would price a small transfer wrong by the whole
  // latency, which is exactly the transfer a placement decision turns on.
  const Measured small = l.cost_ns(64);
  LSE_EXPECT(small.known());
  LSE_EXPECT(small.value > kLatency);
  LSE_EXPECT(small.value < kLatency * 1.01);
}

LSE_TEST(one_size_cannot_separate_the_two_terms_and_says_so) {
  LinkProfile l;
  l.points.push_back(TransferPoint{4096, 1000.0});
  l.points.push_back(TransferPoint{4096, 1100.0});
  fit_link(l);
  LSE_EXPECT(!l.latency_ns.known());
  LSE_EXPECT(!l.bandwidth_bytes_per_s.known());
  LSE_EXPECT(!l.cost_ns(4096).known());
}

LSE_TEST(a_path_kind_comes_from_capabilities_not_from_a_name) {
  const LinkMember a{dev("hrx", 0), 0, "node-1"};
  const LinkMember b{dev("hrx", 1), 1, "node-1"};
  const LinkMember far{dev("hrx", 2), 2, "node-2"};

  dist::Capabilities direct;
  direct.device_memory_direct = true;
  dist::Capabilities staged;
  staged.device_memory_direct = false;

  LSE_EXPECT(classify_path(a, a, direct) == PathKind::kSameDevice);
  LSE_EXPECT(classify_path(a, b, direct) == PathKind::kPeerDirect);
  LSE_EXPECT(classify_path(a, b, staged) == PathKind::kHostStaged);
  LSE_EXPECT(classify_path(a, far, direct) == PathKind::kRdmaDirect);
  LSE_EXPECT(classify_path(a, far, staged) == PathKind::kRdmaStaged);
}

// ---------------------------------------------------------------------------
// Cost model, on injected profiles
// ---------------------------------------------------------------------------

LSE_TEST(compute_cost_is_the_roofline_plus_the_dispatch) {
  const PoolProfile pool =
      two_device_pool(200e9, 50e12, 200e9, 50e12, 5000.0, 25e9);
  const CostModel model(pool);
  const DeviceId a = dev("hrx", 0);

  // Memory bound: 200 MB at 200 GB/s is 1 ms, and 2 GFLOP at 50 TFLOP/s is
  // 40 us, so the stream is what it costs.
  Work stream;
  stream.bytes_streamed = 200u << 20;
  stream.flops = 2'000'000'000ull;
  stream.operand = math::MatrixElem::kBF16;
  const Cost c = model.compute_cost(stream, a);
  LSE_EXPECT(c.known());
  LSE_EXPECT_NEAR(c.ns, static_cast<double>(200u << 20) * 1e9 / 200e9 + 2000.0,
                  1.0);

  // A pure byte move needs no matrix rate at all.
  Work bytes_only;
  bytes_only.bytes_streamed = 1u << 20;
  bytes_only.launches = 0;
  const Cost b = model.compute_cost(bytes_only, a);
  LSE_EXPECT(b.known());
  LSE_EXPECT_NEAR(b.ns, static_cast<double>(1u << 20) * 1e9 / 200e9, 1.0);
}

LSE_TEST(an_unknown_term_makes_the_answer_unknown_not_a_default) {
  PoolProfile pool = two_device_pool(200e9, 50e12, 200e9, 50e12, 5000.0, 25e9);
  // Device 1 was admitted but its probe could not reach the roofline.
  pool.devices[1].dram_bytes_per_s = Measured::unknown();
  pool.devices[1].paths.clear();
  const CostModel model(pool);

  Work w = matmul_work(1, 4096, 4096, DType::kBF16);
  LSE_EXPECT(model.compute_cost(w, dev("hrx", 0)).known());
  LSE_EXPECT(!model.compute_cost(w, dev("hrx", 1)).known());
  // A device nobody measured is not a device the model will move work to, at
  // any queue depth.
  for (std::uint32_t depth : {1u, 32u}) {
    const auto d = model.should_offload(w, dev("hrx", 0), dev("hrx", 1),
                                        w.bytes_moved, 4096 * 4, depth);
    LSE_EXPECT(!d.relocate);
    LSE_EXPECT(!d.moved.known());
    LSE_EXPECT(d.reason.find("unknown") != std::string_view::npos);
  }

  // A device that is not in the pool at all is unknown, never zero.
  LSE_EXPECT(!model.compute_cost(w, dev("cuda", 0)).known());
  LSE_EXPECT(!model.transfer_cost(1024, dev("hrx", 0), dev("cuda", 0)).known());
}

LSE_TEST(a_pipeline_bubble_is_what_converts_a_link_cost_into_a_rate) {
  // Eight stages with eight items in flight still idles almost half the
  // pipeline; thirty-two items brings it under a fifth. This is the term that
  // makes a queue depth part of a placement decision rather than an
  // afterthought.
  LSE_EXPECT_NEAR(bubble_fraction(8, 8), 7.0 / 15.0, 1e-12);
  LSE_EXPECT(bubble_fraction(8, 8) > 0.46);
  LSE_EXPECT(bubble_fraction(8, 32) < 0.20);
  // One stage never bubbles, and depth never makes it worse.
  LSE_EXPECT_EQ(bubble_fraction(1, 1), 0.0);
  LSE_EXPECT(bubble_fraction(8, 1) > bubble_fraction(8, 64));
}

LSE_TEST(placement_follows_throughput_at_depth_not_a_latency_threshold) {
  // A peer eight times the roofline, behind a link that costs about as much as
  // the operation itself. The round trip makes one request slower; with a queue
  // behind it, the send overlaps the previous item's compute and only the
  // slowest stage still counts.
  const PoolProfile pool =
      two_device_pool(200e9, 50e12, 1600e9, 400e12, 5000.0, 100e9);
  const CostModel model(pool);
  const DeviceId home = dev("hrx", 0);
  const DeviceId peer = dev("hrx", 1);

  Work w;
  w.bytes_streamed = 16u << 20;
  w.operand = math::MatrixElem::kBF16;
  constexpr std::size_t kBytes = 4u << 20;

  const auto alone = model.should_offload(w, home, peer, kBytes, kBytes, 1);
  LSE_EXPECT(alone.stay.known() && alone.moved.known());
  LSE_EXPECT(!alone.relocate);
  std::printf("       depth 1:  stay %.1f/ms, move %.1f/ms (item latency "
              "%.1f us) -> %s\n",
              alone.stay.items_per_s / 1e3, alone.moved.items_per_s / 1e3,
              alone.moved.item_latency_ns / 1e3,
              alone.relocate ? "move" : "stay");

  const auto deep = model.should_offload(w, home, peer, kBytes, kBytes, 32);
  LSE_EXPECT(deep.relocate);
  std::printf("       depth 32: stay %.1f/ms, move %.1f/ms (item latency "
              "%.1f us, bubble %.0f%%) -> %s\n",
              deep.stay.items_per_s / 1e3, deep.moved.items_per_s / 1e3,
              deep.moved.item_latency_ns / 1e3, deep.moved.bubble * 100.0,
              deep.relocate ? "move" : "stay");

  // The two rules disagree, which is the whole point: the same operation, the
  // same measured numbers, opposite answers at depth 1 and depth 32.
  LSE_EXPECT(alone.moved.items_per_s < alone.stay.items_per_s);
  LSE_EXPECT(deep.moved.items_per_s > deep.stay.items_per_s);
  // Per-item latency is unchanged by depth; it is throughput that moved.
  LSE_EXPECT_NEAR(alone.moved.item_latency_ns, deep.moved.item_latency_ns,
                  1e-6);
  // ...and the item got slower even though the pool got faster. That trade is
  // the thing a caller has to be able to see.
  LSE_EXPECT(deep.moved.item_latency_ns > deep.stay.item_latency_ns);

  // Depth cannot rescue a link that is itself the bottleneck: a stage that is
  // slower than staying put stays slower however many items are behind it.
  const PoolProfile crawling =
      two_device_pool(200e9, 50e12, 1600e9, 400e12, 5000.0, 2e9);
  const CostModel crawl(crawling);
  LSE_EXPECT(!crawl.should_offload(w, home, peer, kBytes, kBytes, 1).relocate);
  LSE_EXPECT(!crawl.should_offload(w, home, peer, kBytes, kBytes, 256).relocate);

  // Staying put is one stage, so it never bubbles and its throughput is flat
  // in depth — the baseline the comparison is against.
  LSE_EXPECT_NEAR(alone.stay.items_per_s, deep.stay.items_per_s, 1e-6);
  LSE_EXPECT_EQ(alone.stay.bubble, 0.0);
}

LSE_TEST(a_heterogeneous_split_follows_measured_throughput) {
  // Twice the roofline on device 1, everything else equal. Splitting a
  // stream-bound op equally would leave the fast device idle for a third of
  // the pass.
  PoolProfile pool = two_device_pool(200e9, 50e12, 400e9, 100e12, 0.0, 1e30);
  pool.devices[0].launch_overhead_ns = Measured::measured(0.0);
  pool.devices[1].launch_overhead_ns = Measured::measured(0.0);
  const CostModel model(pool);
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);

  Work w;
  w.bytes_streamed = 1u << 30;
  w.operand = math::MatrixElem::kBF16;
  w.launches = 1;

  const DeviceId set[] = {a, b};
  const auto shares = model.split(w, set, a);
  LSE_EXPECT_EQ(shares.size(), 2u);
  double total = 0.0;
  for (const auto& s : shares) total += s.fraction;
  LSE_EXPECT_NEAR(total, 1.0, 1e-9);
  LSE_EXPECT_NEAR(shares[0].fraction, 1.0 / 3.0, 1e-4);
  LSE_EXPECT_NEAR(shares[1].fraction, 2.0 / 3.0, 1e-4);
  // Equal per-item time is the criterion, so both members report the same one.
  LSE_EXPECT_NEAR(shares[0].per_item.ns, shares[1].per_item.ns, 1e-6);
  std::printf("       split: %.3f / %.3f, both at %.1f us per item\n",
              shares[0].fraction, shares[1].fraction,
              shares[0].per_item.ns / 1e3);

  // Put the fast device behind a link that has to carry its share and its
  // share shrinks — the link is part of its throughput, not a separate story.
  PoolProfile costly = pool;
  costly.links[1] = fake_link(a, b, PathKind::kRdmaStaged, 10'000.0, 150e9);
  costly.links[2] = fake_link(b, a, PathKind::kRdmaStaged, 10'000.0, 150e9);
  Work moved = w;
  moved.bytes_moved = 1u << 30;
  const CostModel costly_model(costly);
  const auto shallow = costly_model.split(moved, set, a, 1);
  std::printf("       slow link, depth 1:  %.3f / %.3f\n",
              shallow[0].fraction, shallow[1].fraction);
  LSE_EXPECT(shallow[1].fraction < shares[1].fraction);
  LSE_EXPECT(shallow[0].fraction > shares[0].fraction);

  // ...and it wins most of that share back once there is a queue, because the
  // carry of the next item overlaps the compute of the current one. The
  // latency-bound answer above is the depth-1 limit of this, not a different
  // model.
  const auto deep = costly_model.split(moved, set, a, 32);
  std::printf("       slow link, depth 32: %.3f / %.3f\n", deep[0].fraction,
              deep[1].fraction);
  LSE_EXPECT(deep[1].fraction > shallow[1].fraction);
  double deep_total = 0.0;
  for (const auto& s : deep) deep_total += s.fraction;
  LSE_EXPECT_NEAR(deep_total, 1.0, 1e-9);
  LSE_EXPECT(deep[0].per_item.ns < shallow[0].per_item.ns);
}

LSE_TEST(a_device_without_the_operand_form_is_still_a_pool_member) {
  // Device 0 has a bf16 matrix core, device 1 does not have the fp8 one
  // either — neither has fp8, and neither is disqualified by that. The
  // checkpoint's format never changes; only the rate does.
  const PoolProfile pool =
      two_device_pool(200e9, 50e12, 200e9, 50e12, 1000.0, 100e9);
  const CostModel model(pool);
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);

  LSE_EXPECT(model.runs_natively(math::MatrixElem::kBF16, a));
  LSE_EXPECT(!model.runs_natively(math::MatrixElem::kFp8, a));

  Work fp8;
  fp8.flops = 100'000'000'000ull;
  fp8.operand = math::MatrixElem::kFp8;
  fp8.bytes_streamed = 1u << 20;
  const Cost native_cost = model.compute_cost(
      Work{fp8.bytes_streamed, fp8.flops, math::MatrixElem::kBF16, 0, 1}, a);
  const Cost fallback_cost = model.compute_cost(fp8, a);
  LSE_EXPECT(fallback_cost.known());
  // A fallback is priced, and priced worse — which is what makes a split
  // proportional rather than equal.
  LSE_EXPECT(fallback_cost.ns > native_cost.ns);
  // ...and it is labelled as an inference, not as something that was timed.
  LSE_EXPECT(fallback_cost.provenance == Provenance::kDeclared);

  const DeviceId set[] = {a, b};
  const auto shares = model.split(fp8, set, a);
  double total = 0.0;
  for (const auto& s : shares) total += s.fraction;
  LSE_EXPECT_NEAR(total, 1.0, 1e-9);
  for (const auto& s : shares) LSE_EXPECT(s.fraction > 0.0);

  // The bytes a shard costs to move are the checkpoint's own, and they do not
  // depend on which member receives it. Nothing widens a tensor to suit a
  // device that lacks its operand form — the format on the wire and in memory
  // is the same for both members, and only the kernel behind it differs.
  const Work bf16 = matmul_work(1, 4096, 4096, DType::kBF16);
  LSE_EXPECT_EQ(bf16.bytes_moved, dtype_storage_bytes(DType::kBF16, 4096 * 4096));
  LSE_EXPECT_EQ(model.transfer_cost(bf16.bytes_moved, a, b).ns,
                model.transfer_cost(bf16.bytes_moved, b, a).ns);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

LSE_TEST(a_profile_survives_a_round_trip_and_a_stale_key_is_a_miss) {
  PoolProfile pool = two_device_pool(227e9, 48e12, 200e9, 50e12, 5000.0, 25e9);
  pool.fingerprint = "0123456789abcdef";
  pool.devices[0].arch = "gfx1151";
  pool.devices[0].name = "Radeon 8060S Graphics";
  MatrixRowRate row;
  row.key = "wmma.f32.16x16x16.bf16";
  row.acc = math::MatrixElem::kF32;
  row.operand = math::MatrixElem::kBF16;
  row.m = row.n = row.k_step = 16;
  row.relative = 100;
  row.support = RowSupport::kMeasured;
  row.flops = Measured::measured(48e12);
  pool.devices[0].rows.push_back(row);
  MatrixRowRate absent = row;
  absent.key = "wmma.f32.16x16x16.fp8";
  absent.operand = math::MatrixElem::kFp8;
  absent.support = RowSupport::kAbsent;
  absent.flops = Measured::unsupported();
  pool.devices[0].rows.push_back(absent);
  pool.links[1].points.push_back(TransferPoint{4096, 5163.84});

  const std::string text = serialize_pool_profile(pool);
  auto back = parse_pool_profile(text);
  LSE_EXPECT(back.ok());
  if (!back.ok()) return;
  LSE_EXPECT_EQ(back->devices.size(), pool.devices.size());
  LSE_EXPECT_EQ(back->links.size(), pool.links.size());
  LSE_EXPECT(back->devices[0].name == "Radeon 8060S Graphics");
  LSE_EXPECT(back->devices[0].arch == "gfx1151");
  LSE_EXPECT_EQ(back->devices[0].rows.size(), 2u);
  LSE_EXPECT(back->devices[0].rows[1].support == RowSupport::kAbsent);
  LSE_EXPECT(back->devices[0].rows[1].flops.provenance ==
             Provenance::kUnsupported);
  LSE_EXPECT(back->devices[0].dram_bytes_per_s.value == 227e9);
  LSE_EXPECT(back->devices[0].paths.size() == pool.devices[0].paths.size());
  LSE_EXPECT(back->links[1].points.size() == 1u);
  LSE_EXPECT(back->links[1].path == PathKind::kPeerDirect);

  const fs::path dir = scratch_dir();
  LSE_EXPECT_OK(save_pool_profile(pool, dir.string()));
  auto hit = load_pool_profile(pool.fingerprint, dir.string());
  LSE_EXPECT(hit.ok());
  if (hit.ok()) {
    LSE_EXPECT(hit->fingerprint == pool.fingerprint);
    LSE_EXPECT_EQ(hit->devices.size(), 2u);
  }
  // Re-probe on change, not on a timer: a different pool is a different key,
  // and there is no entry under it.
  auto miss = load_pool_profile("fedcba9876543210", dir.string());
  LSE_EXPECT(!miss.ok());
  LSE_EXPECT(miss.status().code() == StatusCode::kNotFound);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---------------------------------------------------------------------------
// The link matrix, over the loopback transport
// ---------------------------------------------------------------------------

LSE_TEST(the_link_probe_fills_every_ordered_pair) {
  constexpr dist::Rank kWorld = 4;
  auto fleet = make_fleet(kWorld, "probe-links");
  if (fleet.empty()) return;
  const std::vector<LinkMember> members = loop_members(kWorld);

  std::vector<std::vector<LinkProfile>> matrices(kWorld);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  LinkProbeConfig cfg;
  cfg.latency_reps = 16;
  for (dist::Rank r = 0; r < kWorld; ++r) {
    threads.emplace_back([&, r] {
      const auto i = static_cast<std::size_t>(r);
      auto got = probe_links(*fleet[i], members, cfg);
      if (!got.ok()) {
        std::printf("       rank %d: %s\n", r, got.status().to_string().c_str());
        ++failures;
        return;
      }
      matrices[i] = got.release();
    });
  }
  for (std::thread& t : threads) t.join();
  LSE_EXPECT_EQ(failures.load(), 0);
  if (failures.load() != 0) return;

  const auto n = static_cast<std::size_t>(kWorld);
  for (std::size_t r = 0; r < n; ++r) LSE_EXPECT_EQ(matrices[r].size(), n * n);

  const std::vector<LinkProfile>& m = matrices[0];
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const LinkProfile& l = m[i * n + j];
      LSE_EXPECT(l.src == members[i].id);
      LSE_EXPECT(l.dst == members[j].id);
      if (i == j) {
        LSE_EXPECT(l.path == PathKind::kSameDevice);
        // Moving bytes to where they already are costs nothing, and that is a
        // fact rather than a missing measurement.
        LSE_EXPECT(l.cost_ns(1u << 20).known());
        LSE_EXPECT_EQ(l.cost_ns(1u << 20).value, 0.0);
        continue;
      }
      // One process, one address space: the ranks share a machine and the
      // transport says a buffer needs no bounce, so this is a peer copy.
      LSE_EXPECT(l.path == PathKind::kPeerDirect);
      LSE_EXPECT(l.latency_ns.provenance == Provenance::kMeasured);
      LSE_EXPECT(l.latency_ns.value > 0.0);
      LSE_EXPECT(l.bandwidth_bytes_per_s.provenance == Provenance::kMeasured);
      LSE_EXPECT(l.bandwidth_bytes_per_s.value > 0.0);
      // One point per size, plus the round-trip anchor that pins the intercept.
      LSE_EXPECT(l.points.size() == default_transfer_sizes().size() + 1);
      // Both terms present means a small transfer and a big one are priced
      // differently, which is the whole reason they are separate.
      LSE_EXPECT(l.cost_ns(1u << 22).value > l.cost_ns(64).value);
    }
  }
  std::printf("       loopback 4 ranks: 0->1 %.2f us + %.2f GB/s (fit err %.1f%%)\n",
              m[1].latency_ns.value / 1e3,
              m[1].bandwidth_bytes_per_s.value / 1e9, m[1].fit_error * 100.0);

  // Every rank came away with the same matrix; a placement decision made on
  // one rank has to agree with the one made on another.
  for (std::size_t r = 1; r < n; ++r) {
    for (std::size_t k = 0; k < n * n; ++k) {
      LSE_EXPECT(matrices[r][k].path == m[k].path);
      LSE_EXPECT(matrices[r][k].src == m[k].src);
      LSE_EXPECT(matrices[r][k].dst == m[k].dst);
    }
  }
}

LSE_TEST(qualifying_a_pool_measures_once_and_then_reads_it_back) {
  constexpr dist::Rank kWorld = 2;
  auto fleet = make_fleet(kWorld, "probe-pool");
  if (fleet.empty()) return;
  const fs::path dir = scratch_dir() / "pool";

  std::vector<PoolMember> members;
  for (dist::Rank r = 0; r < kWorld; ++r) {
    PoolMember m;
    m.id = dev("loop", r);
    m.rank = r;
    m.host = "test-host";
    // Rank 0 drives whatever backend this build has; rank 1 stands for a peer
    // this process cannot drive, which must not stop the pool qualifying.
    if (r == 0) m.backend = host_backend();
    members.push_back(m);
  }

  PoolOptions opts;
  opts.profile_dir = dir.string();
  opts.links.latency_reps = 8;

  const auto run_all = [&](std::vector<PoolProfile>& out) {
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (dist::Rank r = 0; r < kWorld; ++r) {
      threads.emplace_back([&, r] {
        const auto i = static_cast<std::size_t>(r);
        auto got = qualify_pool(members, fleet[i].get(), opts);
        if (!got.ok()) {
          std::printf("       rank %d: %s\n", r,
                      got.status().to_string().c_str());
          ++failures;
          return;
        }
        out[i] = got.release();
      });
    }
    for (std::thread& t : threads) t.join();
    return failures.load();
  };

  std::vector<PoolProfile> first(kWorld);
  LSE_EXPECT_EQ(run_all(first), 0);
  if (first[0].devices.empty()) return;
  LSE_EXPECT_EQ(first[0].devices.size(), 2u);
  LSE_EXPECT_EQ(first[0].links.size(), 4u);
  LSE_EXPECT(!first[0].fingerprint.empty());
  // Every rank keys the same pool the same way, or they would take different
  // branches through a collective probe and hang.
  LSE_EXPECT(first[0].fingerprint == first[1].fingerprint);
  LSE_EXPECT(fs::exists(profile_path(dir.string(), first[0].fingerprint)));

  std::vector<PoolProfile> second(kWorld);
  LSE_EXPECT_EQ(run_all(second), 0);
  LSE_EXPECT(second[0].fingerprint == first[0].fingerprint);
  LSE_EXPECT_EQ(second[0].devices.size(), first[0].devices.size());
  LSE_EXPECT(second[0].links[1].path == first[0].links[1].path);
  // Rank 1 stands for a peer nobody could probe, so the pool is qualified but
  // not complete — and it reports that rather than presenting a hole as a
  // measurement.
  LSE_EXPECT(!first[0].complete());
  std::printf("%s", first[0].describe().c_str());

  std::error_code ec;
  fs::remove_all(scratch_dir(), ec);
}

// ---------------------------------------------------------------------------
// The probe never invents a number
// ---------------------------------------------------------------------------

namespace {

// The one property every profile must have, whatever produced it.
void expect_no_invented_numbers(const DeviceProfile& d) {
  const Measured scalars[] = {d.dram_bytes_per_s, d.launch_overhead_ns,
                              d.h2d_bytes_per_s, d.d2h_bytes_per_s};
  for (const Measured& m : scalars) {
    if (!m.known()) LSE_EXPECT_EQ(m.value, 0.0);
    if (m.known()) LSE_EXPECT(m.value > 0.0);
  }
  for (const MatrixRowRate& r : d.rows) {
    if (r.support == RowSupport::kAbsent) {
      LSE_EXPECT(r.flops.provenance == Provenance::kUnsupported);
      LSE_EXPECT_EQ(r.flops.value, 0.0);
    }
    if (r.support == RowSupport::kUnverified ||
        r.support == RowSupport::kDeclared) {
      // Nothing ran, so nothing may claim to have been measured.
      LSE_EXPECT(r.flops.provenance != Provenance::kMeasured);
    }
    if (r.support == RowSupport::kMeasured) LSE_EXPECT(r.flops.value > 0.0);
  }
  for (const ComputePath& p : d.paths) {
    if (p.native) {
      LSE_EXPECT(p.flops.provenance == Provenance::kMeasured);
      LSE_EXPECT(p.executed_as == p.operand);
    } else if (p.flops.known()) {
      // A fallback rate is an inference about a kernel nobody timed, and it
      // must say so.
      LSE_EXPECT(p.flops.provenance == Provenance::kDeclared);
    } else {
      LSE_EXPECT_EQ(p.flops.value, 0.0);
    }
  }
}

}  // namespace

LSE_TEST(a_probe_that_cannot_reach_a_number_leaves_it_unknown) {
  backend::IBackend* be = host_backend();
  if (be == nullptr) {
    std::printf("       skipped: no cpu backend in this build\n");
    return;
  }
  auto probe = create_device_probe(*be);
  // No emitter and no compiler: there is no matrix core to rate here, and the
  // probe says which parts it cannot reach rather than filling them in.
  LSE_EXPECT(!probe->declined().empty());
  std::printf("       cpu probe declines: %s\n",
              std::string(probe->declined()).c_str());

  auto profiled = probe_device(*be);
  LSE_EXPECT(profiled.ok());
  if (!profiled.ok()) return;
  expect_no_invented_numbers(*profiled);
  LSE_EXPECT(profiled->id.backend == "cpu");
  LSE_EXPECT(profiled->h2d_bytes_per_s.provenance == Provenance::kMeasured);
  LSE_EXPECT(profiled->paths.empty());

  // Which means a matmul on it has no price, and the cost model says unknown
  // rather than picking a number that would look like a decision.
  PoolProfile pool;
  pool.fingerprint = "cpu-only";
  pool.devices.push_back(*profiled);
  pool.links.push_back(
      fake_link(profiled->id, profiled->id, PathKind::kSameDevice, 0.0, 0.0));
  const CostModel model(pool);
  LSE_EXPECT(!model.compute_cost(matmul_work(1, 512, 512, DType::kBF16),
                                 profiled->id)
                  .known());
  std::printf("       cpu: %.2f GB/s h2d, %.2f GB/s stream, %.1f ns dispatch\n",
              profiled->h2d_bytes_per_s.value / 1e9,
              profiled->dram_bytes_per_s.value / 1e9,
              profiled->launch_overhead_ns.value);
}

LSE_TEST(the_device_probe_measures_the_roofline_and_the_matrix_rows) {
  backend::IBackend* be = device_backend();
  if (be == nullptr) {
    std::printf("       skipped: no live device\n");
    return;
  }
  auto probe = create_device_probe(*be);
  if (probe->name() != "hrx") {
    std::printf("       skipped: backend registered no device probe\n");
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto profiled = probe_device(*be);
  const double probe_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0)
          .count();
  LSE_EXPECT(profiled.ok());
  if (!profiled.ok()) {
    std::printf("       %s\n", profiled.status().to_string().c_str());
    return;
  }
  std::printf("       probe took %.0f ms\n", probe_ms);
  expect_no_invented_numbers(*profiled);
  std::printf("       %s: %.1f GB/s stream, %.2f us dispatch, %.1f/%.1f GB/s h2d/d2h\n",
              profiled->arch.c_str(), profiled->dram_bytes_per_s.value / 1e9,
              profiled->launch_overhead_ns.value / 1e3,
              profiled->h2d_bytes_per_s.value / 1e9,
              profiled->d2h_bytes_per_s.value / 1e9);
  for (const MatrixRowRate& r : profiled->rows) {
    std::printf("         %-32s %-10s %8.2f TFLOP/s (table ratio %d)\n",
                r.key.c_str(), std::string(to_string(r.support)).c_str(),
                r.flops.value / 1e12, r.relative);
  }
  for (const ComputePath& p : profiled->paths) {
    std::printf("         %-5s -> %-5s %-9s %8.2f TFLOP/s (%s)\n",
                std::string(to_string(p.operand)).c_str(),
                std::string(to_string(p.executed_as)).c_str(),
                p.native ? "native" : "fallback", p.flops.value / 1e12,
                std::string(to_string(p.flops.provenance)).c_str());
  }
  if (!probe->declined().empty()) {
    std::printf("       declined: %s\n",
                std::string(probe->declined()).c_str());
  }

  LSE_EXPECT(profiled->dram_bytes_per_s.provenance == Provenance::kMeasured);
  LSE_EXPECT(profiled->launch_overhead_ns.provenance == Provenance::kMeasured);
  // Every row of the table this ISA generation owns is here, including the
  // ones it does not have: an omitted row and an absent one read the same to
  // the cost model, and only one of them is true.
  LSE_EXPECT(!profiled->rows.empty());
  LSE_EXPECT_EQ(profiled->paths.size(), profiled_operands().size());

  // fp8 has no row on this generation. That does not remove the device from
  // the pool — it gives it a fallback path, at a rate that is labelled as one.
  const ComputePath* fp8 = profiled->path_for(math::MatrixElem::kFp8);
  LSE_EXPECT(fp8 != nullptr);
  if (fp8 != nullptr && !fp8->native) {
    LSE_EXPECT(fp8->executed_as != math::MatrixElem::kFp8);
    LSE_EXPECT(fp8->flops.provenance != Provenance::kMeasured);
  }

  // The measured roofline is what the model must price a weight stream with.
  PoolProfile pool;
  pool.fingerprint = "live";
  pool.devices.push_back(*profiled);
  pool.links.push_back(
      fake_link(profiled->id, profiled->id, PathKind::kSameDevice, 0.0, 0.0));
  const CostModel model(pool);
  Work stream;
  stream.bytes_streamed = 1425u << 20;   // one decode token of bf16 weights
  const Cost token = model.compute_cost(stream, profiled->id);
  LSE_EXPECT(token.known());
  std::printf("       a 1.425 GB weight stream costs %.2f ms here\n",
              token.ns / 1e6);
}

LSE_TEST_MAIN()
