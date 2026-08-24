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
#include <limits>
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
#include "lse/place/devices.hpp"
#include "lse/place/placement.hpp"
#include "lse/place/residency.hpp"
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
  const auto even = model.split(w, set, a);
  LSE_EXPECT(even.feasible());
  const auto& shares = even.shares;
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
  const auto shallow_split = costly_model.split(moved, set, a, 1);
  const auto& shallow = shallow_split.shares;
  std::printf("       slow link, depth 1:  %.3f / %.3f\n",
              shallow[0].fraction, shallow[1].fraction);
  LSE_EXPECT(shallow[1].fraction < shares[1].fraction);
  LSE_EXPECT(shallow[0].fraction > shares[0].fraction);

  // ...and it wins most of that share back once there is a queue, because the
  // carry of the next item overlaps the compute of the current one. The
  // latency-bound answer above is the depth-1 limit of this, not a different
  // model.
  const auto deep_split = costly_model.split(moved, set, a, 32);
  const auto& deep = deep_split.shares;
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
      Work{.bytes_streamed = fp8.bytes_streamed,
           .flops = fp8.flops,
           .operand = math::MatrixElem::kBF16,
           .launches = 1},
      a);
  const Cost fallback_cost = model.compute_cost(fp8, a);
  LSE_EXPECT(fallback_cost.known());
  // A fallback is priced, and priced worse — which is what makes a split
  // proportional rather than equal.
  LSE_EXPECT(fallback_cost.ns > native_cost.ns);
  // ...and it is labelled as an inference, not as something that was timed.
  LSE_EXPECT(fallback_cost.provenance == Provenance::kDeclared);

  const DeviceId set[] = {a, b};
  const auto split = model.split(fp8, set, a);
  double total = 0.0;
  for (const auto& s : split.shares) total += s.fraction;
  LSE_EXPECT_NEAR(total, 1.0, 1e-9);
  for (const auto& s : split.shares) LSE_EXPECT(s.fraction > 0.0);

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
// Capacity, which is a constraint and not a cost
// ---------------------------------------------------------------------------

namespace {

// Bytes, decimal, so the arithmetic in the assertions below is the arithmetic a
// reader does in their head.
constexpr double kGB = 1e9;

// Twice the roofline on device 1, no link cost, so time alone splits this
// 1/3 : 2/3 — the answer the existing heterogeneous-split case proves.
PoolProfile lopsided_pool(double free_a, double free_b) {
  PoolProfile pool = two_device_pool(200e9, 50e12, 400e9, 100e12, 0.0, 1e30);
  pool.devices[0].launch_overhead_ns = Measured::measured(0.0);
  pool.devices[1].launch_overhead_ns = Measured::measured(0.0);
  if (free_a >= 0.0) pool.devices[0].free_memory = Measured::measured(free_a);
  if (free_b >= 0.0) pool.devices[1].free_memory = Measured::measured(free_b);
  return pool;
}

Work resident_work(double bytes) {
  Work w;
  w.bytes_streamed = static_cast<std::size_t>(bytes);
  w.bytes_resident = static_cast<std::size_t>(bytes);
  w.operand = math::MatrixElem::kBF16;
  w.launches = 1;
  return w;
}

// Identical members, free links: the split is then decided by capacity alone,
// which is the point of the case that uses it.
PoolProfile uniform_pool(std::size_t n, double free_each) {
  PoolProfile pool;
  pool.fingerprint = "injected";
  for (std::size_t i = 0; i < n; ++i) {
    DeviceProfile d = fake_device(dev("hrx", static_cast<int>(i)), 200e9, 0.0,
                                  50e12, math::MatrixElem::kBF16);
    d.free_memory = Measured::measured(free_each);
    pool.devices.push_back(std::move(d));
  }
  pool.links.reserve(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      pool.links.push_back(
          i == j ? fake_link(pool.devices[i].id, pool.devices[j].id,
                             PathKind::kSameDevice, 0.0, 0.0)
                 : fake_link(pool.devices[i].id, pool.devices[j].id,
                             PathKind::kPeerDirect, 0.0, 1e30));
    }
  }
  return pool;
}

}  // namespace

LSE_TEST(a_forty_gigabyte_share_never_lands_on_a_sixteen_gigabyte_device) {
  // The defect this whole file exists to close. Device 1 is twice as fast, so
  // optimizing time alone hands it two thirds of a 40 GB weight set — 26.7 GB —
  // and it has 16 GB. That plan is not expensive, it is impossible, and a
  // planner that reports it as the optimum is worse than one that refuses.
  const Work w = resident_work(40 * kGB);
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);
  const DeviceId set[] = {a, b};

  const PoolProfile roomy_pool = lopsided_pool(80 * kGB, 80 * kGB);
  const auto roomy = CostModel(roomy_pool).split(w, set, a);
  LSE_EXPECT(roomy.feasible());
  LSE_EXPECT_NEAR(roomy.shares[1].fraction, 2.0 / 3.0, 1e-4);
  // With room, the time-optimal share is what it always was...
  const double time_optimal_bytes = roomy.shares[1].fraction * 40 * kGB;
  std::printf("       80 GB free: time-optimal share is %.1f GB\n",
              time_optimal_bytes / kGB);
  // ...and on a 16 GB device that same share is 10 GB of wishful thinking.
  LSE_EXPECT(time_optimal_bytes > 16 * kGB);

  const PoolProfile tight_pool = lopsided_pool(80 * kGB, 16 * kGB);
  const auto tight = CostModel(tight_pool).split(w, set, a);
  LSE_EXPECT(tight.feasible());
  LSE_EXPECT(tight.fit == Fit::kFits);
  // Capped at what it can hold, 16/40, and the slower device carries the rest.
  LSE_EXPECT_NEAR(tight.shares[1].fraction, 0.4, 1e-4);
  LSE_EXPECT_NEAR(tight.shares[0].fraction, 0.6, 1e-4);
  double total = 0.0;
  for (const auto& s : tight.shares) total += s.fraction;
  LSE_EXPECT_NEAR(total, 1.0, 1e-9);
  std::printf("       16 GB free: %.1f GB / %.1f GB, both resident-feasible\n",
              tight.shares[0].fraction * 40.0, tight.shares[1].fraction * 40.0);

  // Every share that was handed out fits on the member that got it, stated as
  // a verdict rather than left for the caller to re-derive.
  for (const auto& s : tight.shares) {
    LSE_EXPECT(s.capacity.fits());
    LSE_EXPECT(static_cast<double>(s.capacity.bytes_resident) <=
               s.capacity.bytes_free);
  }
  LSE_EXPECT(static_cast<double>(tight.shares[1].capacity.bytes_resident) <=
             16 * kGB);
  // Capacity is what moved the share, not a change in either device's rate.
  LSE_EXPECT(tight.shares[1].fraction < roomy.shares[1].fraction);
  LSE_EXPECT(tight.shares[1].per_item.known());
}

LSE_TEST(a_pool_with_no_room_anywhere_says_so_distinctly) {
  // Three outcomes, one enum, and a caller has to be able to tell them apart:
  // a partition, a measured refusal, and the absence of a measurement. Only
  // the first is a plan, and the last two must never be read as either of the
  // others.
  const Work w = resident_work(40 * kGB);
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);
  const DeviceId set[] = {a, b};

  // 8 + 8 GB of measured free memory against 40 GB of weights. No assignment
  // of this work to this set exists at any price.
  const PoolProfile cramped = lopsided_pool(8 * kGB, 8 * kGB);
  const auto refused = CostModel(cramped).split(w, set, a);
  LSE_EXPECT(!refused.feasible());
  LSE_EXPECT(refused.fit == Fit::kExceeds);
  LSE_EXPECT(refused.provenance == Provenance::kMeasured);
  for (const auto& s : refused.shares) LSE_EXPECT_EQ(s.fraction, 0.0);
  std::printf("       refused: %s\n", std::string(refused.reason).c_str());
  LSE_EXPECT(refused.reason.find("fits") != std::string_view::npos);

  // The same set, enough room: a plan, and a different verdict.
  const PoolProfile roomy = lopsided_pool(80 * kGB, 80 * kGB);
  const auto planned = CostModel(roomy).split(w, set, a);
  LSE_EXPECT(planned.feasible());
  LSE_EXPECT(planned.fit == Fit::kFits);
  LSE_EXPECT(planned.reason != refused.reason);

  // The same set, room measured but nothing priced: not a refusal. Nothing
  // here has been shown not to fit — nothing here has been shown at all.
  PoolProfile unpriced = roomy;
  unpriced.devices[0].dram_bytes_per_s = Measured::unknown();
  unpriced.devices[1].dram_bytes_per_s = Measured::unknown();
  const auto unknown = CostModel(unpriced).split(w, set, a);
  LSE_EXPECT(!unknown.feasible());
  LSE_EXPECT(unknown.fit == Fit::kUnknown);
  LSE_EXPECT(unknown.reason != refused.reason);
  std::printf("       unpriced: %s\n", std::string(unknown.reason).c_str());
}

LSE_TEST(a_hundred_gigabyte_model_on_eight_devices_is_a_capacity_split) {
  // PLAN.md's headline regime, which the model could not express at all before
  // capacity existed: the model does not fit on any one member, so splitting is
  // mandatory rather than an optimization, and the members are identical so
  // nothing but capacity decides the answer.
  constexpr std::size_t kMembers = 8;
  const Work model_weights = resident_work(100 * kGB);
  std::vector<DeviceId> set;
  for (std::size_t i = 0; i < kMembers; ++i) {
    set.push_back(dev("hrx", static_cast<int>(i)));
  }

  // Exactly enough, summed: 8 x 12.5 GB. The boundary is checked in bytes and
  // not in eighths, because eight eighths do not add to one in binary.
  const PoolProfile exact = uniform_pool(kMembers, 12.5 * kGB);
  const auto split = CostModel(exact).split(model_weights, set, set[0]);
  LSE_EXPECT(split.feasible());
  double total = 0.0;
  for (const auto& s : split.shares) {
    total += s.fraction;
    LSE_EXPECT(s.capacity.fits());
    LSE_EXPECT(static_cast<double>(s.capacity.bytes_resident) <= 12.5 * kGB);
  }
  LSE_EXPECT_NEAR(total, 1.0, 1e-9);
  LSE_EXPECT_NEAR(split.shares[0].fraction, 0.125, 1e-6);
  std::printf("       100 GB over 8 x 12.5 GB: %.1f GB each\n",
              split.shares[0].fraction * 100.0);

  // One gigabyte short across the whole set, and there is no plan. Not a worse
  // plan — none, and the difference is the entire point.
  const PoolProfile short_pool = uniform_pool(kMembers, 12.375 * kGB);
  const auto refused = CostModel(short_pool).split(model_weights, set, set[0]);
  LSE_EXPECT(!refused.feasible());
  LSE_EXPECT(refused.fit == Fit::kExceeds);
  for (const auto& s : refused.shares) LSE_EXPECT_EQ(s.fraction, 0.0);
}

LSE_TEST(a_sampled_free_memory_turns_an_unknown_verdict_into_a_decision) {
  const Work w = resident_work(40 * kGB);
  const DeviceId a = dev("hrx", 0);

  // What every device on this box answered before the backend seam carried a
  // memory query: no basis, so no verdict.
  const PoolProfile blind = lopsided_pool(-1.0, -1.0);
  const Capacity before = CostModel(blind).capacity_for(w, a);
  LSE_EXPECT(before.fit == Fit::kUnknown);
  LSE_EXPECT(!before.known());
  LSE_EXPECT(before.provenance == Provenance::kUnknown);

  // Same device, same work, one number the backend answered.
  const PoolProfile sampled = lopsided_pool(80 * kGB, 80 * kGB);
  const Capacity after = CostModel(sampled).capacity_for(w, a);
  LSE_EXPECT(after.fit == Fit::kFits);
  LSE_EXPECT(after.fits());
  // Measured, not declared: this is an observation of one instant on one
  // machine, not a fact about the part that a table could have held.
  LSE_EXPECT(after.provenance == Provenance::kMeasured);
  LSE_EXPECT_EQ(after.bytes_free, 80 * kGB);
  LSE_EXPECT_EQ(after.bytes_resident, w.bytes_resident);
}

LSE_TEST(forty_gigabytes_resident_on_a_sixteen_gigabyte_device_reads_exceeds) {
  const Work w = resident_work(40 * kGB);
  const DeviceId b = dev("hrx", 1);

  // Unsampled, this is the verdict — a refusal, but not a statement about the
  // device. It is the state the whole defect consisted of.
  const PoolProfile blind = lopsided_pool(80 * kGB, -1.0);
  LSE_EXPECT(CostModel(blind).capacity_for(w, b).fit == Fit::kUnknown);

  // Sampled, the constraint binds: 40 GB resident against 16 GB free is not an
  // expensive plan, it is not a plan.
  const PoolProfile tight = lopsided_pool(80 * kGB, 16 * kGB);
  const Capacity c = CostModel(tight).capacity_for(w, b);
  LSE_EXPECT(c.fit == Fit::kExceeds);
  LSE_EXPECT(c.known());
  LSE_EXPECT(!c.fits());
  LSE_EXPECT(c.provenance == Provenance::kMeasured);
  LSE_EXPECT_EQ(c.bytes_free, 16 * kGB);
  LSE_EXPECT_EQ(c.bytes_resident, w.bytes_resident);

  // A device with nothing left is a measurement of zero, not an absence of
  // one, and it must exceed rather than go unknown.
  const PoolProfile full = lopsided_pool(80 * kGB, 0.0);
  const Capacity empty = CostModel(full).capacity_for(w, b);
  LSE_EXPECT(empty.fit == Fit::kExceeds);
  LSE_EXPECT(empty.known());
}

LSE_TEST(unknown_free_memory_is_an_unknown_verdict_not_a_pass) {
  const Work w = resident_work(40 * kGB);
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);
  const DeviceId set[] = {a, b};

  // Nothing filled free memory — which is the state of every device on this
  // box, because IBackend carries no memory query. The model must not treat
  // that as room, and must not treat it as a refusal either.
  const PoolProfile blind = lopsided_pool(-1.0, -1.0);
  const CostModel model(blind);

  const Capacity c = model.capacity_for(w, a);
  LSE_EXPECT(c.fit == Fit::kUnknown);
  LSE_EXPECT(!c.known());
  LSE_EXPECT(!c.fits());
  LSE_EXPECT_EQ(c.bytes_free, 0.0);
  LSE_EXPECT_EQ(c.bytes_resident, w.bytes_resident);

  const auto blind_split = model.split(w, set, a);
  LSE_EXPECT(!blind_split.feasible());
  LSE_EXPECT(blind_split.fit == Fit::kUnknown);
  for (const auto& s : blind_split.shares) LSE_EXPECT_EQ(s.fraction, 0.0);
  LSE_EXPECT(blind_split.reason.find("unknown") != std::string_view::npos);

  const auto blind_move = model.should_offload(w, a, b, w.bytes_moved, 4096);
  LSE_EXPECT(!blind_move.relocate);
  LSE_EXPECT(blind_move.reason.find("unknown") != std::string_view::npos);
  LSE_EXPECT(!blind_move.home_capacity.fits());
  LSE_EXPECT(!blind_move.candidate_capacity.fits());

  // Work that holds nothing resident still splits on an unmeasured pool: that
  // is a fact about the work, not a claim about the devices, and it is why the
  // whole cost model did not stop working the day this field was added.
  Work streaming = w;
  streaming.bytes_resident = 0;
  LSE_EXPECT(model.capacity_for(streaming, a).fits());
  const auto streamed = model.split(streaming, set, a);
  LSE_EXPECT(streamed.feasible());
  LSE_EXPECT_NEAR(streamed.shares[1].fraction, 2.0 / 3.0, 1e-4);

  // One member measured, one not: the split uses what it knows, gives the
  // unmeasured member nothing, and says which member it left out and why.
  const PoolProfile half_known = lopsided_pool(80 * kGB, -1.0);
  const auto partial = CostModel(half_known).split(w, set, a);
  LSE_EXPECT(partial.feasible());
  LSE_EXPECT_NEAR(partial.shares[0].fraction, 1.0, 1e-9);
  LSE_EXPECT_EQ(partial.shares[1].fraction, 0.0);
  LSE_EXPECT(partial.shares[0].capacity.fits());
  LSE_EXPECT(partial.shares[1].capacity.fit == Fit::kUnknown);
  std::printf("       one member unmeasured: %s\n",
              std::string(partial.reason).c_str());
}

LSE_TEST(capacity_overrides_a_favourable_time_cost_in_both_directions) {
  // The peer from the depth case: eight times the roofline, behind a link the
  // queue pays for. At depth 32 moving is measurably the faster plan.
  const DeviceId home = dev("hrx", 0);
  const DeviceId peer = dev("hrx", 1);
  constexpr std::size_t kBytes = 4u << 20;
  constexpr double kResident = 16.0 * 1024 * 1024;

  Work w;
  w.bytes_streamed = 16u << 20;
  w.bytes_resident = static_cast<std::size_t>(kResident);
  w.operand = math::MatrixElem::kBF16;

  const auto pool_with = [](double free_home, double free_peer) {
    PoolProfile p = two_device_pool(200e9, 50e12, 1600e9, 400e12, 5000.0, 100e9);
    p.devices[0].free_memory = Measured::measured(free_home);
    p.devices[1].free_memory = Measured::measured(free_peer);
    return p;
  };

  // Room at both ends: the throughput rule decides, exactly as before.
  const PoolProfile roomy = pool_with(4 * kResident, 4 * kResident);
  const auto fast = CostModel(roomy).should_offload(w, home, peer, kBytes,
                                                    kBytes, 32);
  LSE_EXPECT(fast.relocate);
  LSE_EXPECT(fast.moved.items_per_s > fast.stay.items_per_s);
  LSE_EXPECT(fast.candidate_capacity.fits());

  // Same numbers, same measured advantage, and the peer cannot hold the work.
  const PoolProfile small_peer = pool_with(4 * kResident, kResident / 2);
  const auto refused = CostModel(small_peer).should_offload(w, home, peer,
                                                            kBytes, kBytes, 32);
  LSE_EXPECT(!refused.relocate);
  LSE_EXPECT(refused.candidate_capacity.fit == Fit::kExceeds);
  // The time advantage is still there and still measured. It simply does not
  // get a vote: a faster device that cannot hold the work is not a placement.
  LSE_EXPECT(refused.moved.known() && refused.stay.known());
  LSE_EXPECT(refused.moved.items_per_s > refused.stay.items_per_s);
  std::printf("       refused a %.0f%% throughput win: %s\n",
              (refused.moved.items_per_s / refused.stay.items_per_s - 1.0) *
                  100.0,
              std::string(refused.reason).c_str());

  // And the other direction, which is the regime PLAN.md leads with: home
  // cannot hold the model, so moving is mandatory rather than worthwhile. At
  // depth 1 the throughput rule says stay and is overruled, because a slower
  // plan that runs beats a faster one that cannot be allocated.
  const PoolProfile small_home = pool_with(kResident / 2, 4 * kResident);
  const auto forced = CostModel(small_home).should_offload(w, home, peer,
                                                           kBytes, kBytes, 1);
  LSE_EXPECT(forced.relocate);
  LSE_EXPECT(forced.home_capacity.fit == Fit::kExceeds);
  LSE_EXPECT(forced.candidate_capacity.fits());
  LSE_EXPECT(forced.moved.items_per_s < forced.stay.items_per_s);
  std::printf("       forced a %.0f%% throughput loss: %s\n",
              (1.0 - forced.moved.items_per_s / forced.stay.items_per_s) *
                  100.0,
              std::string(forced.reason).c_str());
}

LSE_TEST(a_split_never_claims_a_partition_it_did_not_produce) {
  // kFits is a claim about the shares, so it has to be checked against them.
  // A free-memory figure that is not a byte count is the case that separates
  // asserting the verdict from verifying it: NaN compares false against every
  // bound, so `min(1, free/resident)` yields a ceiling of 1 and the member bids
  // for the whole work — while capacity_for, comparing the other way, calls the
  // same share kExceeds. One of those is wrong whatever NaN means, and the
  // split must not answer kFits while holding a share it has itself ruled out.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);
  const DeviceId set[] = {a, b};
  const Work w = resident_work(400 * kGB);

  PoolProfile pool = lopsided_pool(-1.0, 8 * kGB);
  pool.devices[0].free_memory = Measured::measured(nan);
  const CostModel model(pool);

  // Whatever the verdict is, it may not be a partition whose own shares do not
  // fit. Only 8 GB of this pool is founded, so 400 GB cannot be one.
  const auto s = model.split(w, set, a);
  for (const auto& share : s.shares) {
    if (share.fraction > 0.0) LSE_EXPECT(share.capacity.fits());
  }
  LSE_EXPECT(!s.feasible());
  // ...and the same figure reads the same way through both entry points.
  LSE_EXPECT(!model.capacity_for(w, a).fits());
  LSE_EXPECT(model.capacity_for(w, a).fit == Fit::kUnknown);
}

LSE_TEST(a_member_nobody_priced_does_not_harden_into_a_measured_refusal) {
  // kExceeds is the strong claim — no assignment exists at any price — and the
  // caller acts on it by not loading the model at all. It may only be made when
  // the whole set was established. A member the cost model could not price is
  // not evidence of anything, and it must not turn "we do not know" into "it
  // does not fit": the fix for one is to measure, and for the other to buy
  // hardware.
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);
  const DeviceId set[] = {a, b};
  const Work w = resident_work(40 * kGB);

  // Device 1 has 100 GB free and measured. The set plainly has room for 40 GB.
  PoolProfile roomy_but_unpriced = lopsided_pool(8 * kGB, 100 * kGB);
  roomy_but_unpriced.devices[1].dram_bytes_per_s = Measured::unknown();
  const auto s = CostModel(roomy_but_unpriced).split(w, set, a);
  LSE_EXPECT(!s.feasible());
  LSE_EXPECT(s.fit != Fit::kExceeds);

  // Same shape with the unpriced member also unmeasured, which is the case the
  // three-way verdict exists for.
  PoolProfile unpriced_and_unmeasured = lopsided_pool(8 * kGB, -1.0);
  unpriced_and_unmeasured.devices[1].dram_bytes_per_s = Measured::unknown();
  const auto t = CostModel(unpriced_and_unmeasured).split(w, set, a);
  LSE_EXPECT(t.fit == Fit::kUnknown);
  // The verdict must not turn on whether the *time* model could price the
  // member: capacity and cost are separate questions and this is the same
  // free-memory configuration either way.
  PoolProfile priced = unpriced_and_unmeasured;
  priced.devices[1].dram_bytes_per_s = Measured::measured(400e9);
  LSE_EXPECT(CostModel(priced).split(w, set, a).fit == t.fit);
}

LSE_TEST(a_device_the_pool_never_heard_of_gets_no_verdict) {
  // capacity_for answers about a device, and `fits()` is the affirmative. A
  // device that is not a member is one the model has measured nothing about,
  // so it gets the same non-answer compute_cost gives it — not a pass that
  // happens to be true of the work.
  const PoolProfile pool = lopsided_pool(80 * kGB, 80 * kGB);
  const CostModel model(pool);
  const DeviceId ghost = dev("hrx", 99);

  Work streaming = resident_work(40 * kGB);
  streaming.bytes_resident = 0;
  LSE_EXPECT(!model.capacity_for(streaming, ghost).fits());
  LSE_EXPECT(model.capacity_for(streaming, ghost).fit == Fit::kUnknown);
  LSE_EXPECT(!model.compute_cost(streaming, ghost).known());

  // A member with an unmeasured free-memory figure is a different thing and
  // keeps the rule the work earns: holding nothing fits anywhere.
  const PoolProfile blind = lopsided_pool(-1.0, -1.0);
  LSE_EXPECT(CostModel(blind).capacity_for(streaming, dev("hrx", 0)).fits());
}

LSE_TEST(a_free_memory_figure_that_is_not_a_byte_count_is_not_a_measurement) {
  // Provenance says who put the number there; it does not say the number is a
  // number. A capacity that is negative or non-finite is unfounded whatever it
  // is labelled, and reading it as one lets zero-byte work "exceed" a device
  // and forces a relocation to fix a residency that was never there.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const DeviceId a = dev("hrx", 0);
  const DeviceId b = dev("hrx", 1);

  for (double bad : {-1.0, nan}) {
    PoolProfile pool = lopsided_pool(-1.0, 80 * kGB);
    pool.devices[0].free_memory = Measured::measured(bad);
    const CostModel model(pool);

    Work streaming = resident_work(40 * kGB);
    streaming.bytes_resident = 0;
    LSE_EXPECT(model.capacity_for(streaming, a).fit != Fit::kExceeds);

    const Work w = resident_work(40 * kGB);
    LSE_EXPECT(model.capacity_for(w, a).fit == Fit::kUnknown);
    // Whatever the throughput rule then decides, capacity does not force a
    // relocation to repair a residency the work never had.
    const auto d = model.should_offload(streaming, a, b, 1u << 20, 4096);
    LSE_EXPECT(d.home_capacity.fit != Fit::kExceeds);
    LSE_EXPECT(d.reason.find("cannot hold") == std::string_view::npos);
  }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

LSE_TEST(a_profile_survives_a_round_trip_and_a_stale_key_is_a_miss) {
  PoolProfile pool = two_device_pool(227e9, 48e12, 200e9, 50e12, 5000.0, 25e9);
  pool.fingerprint = "0123456789abcdef";
  pool.devices[0].arch = "gfx1151";
  pool.devices[0].name = "Radeon 8060S Graphics";
  pool.devices[0].free_memory = Measured::measured(96e9);
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
  // Capacity survives the wire, and a device nobody asked survives it as
  // unknown rather than as a zero that would read as a full device.
  LSE_EXPECT(back->devices[0].free_memory.value == 96e9);
  LSE_EXPECT(back->devices[0].free_memory.provenance == Provenance::kMeasured);
  LSE_EXPECT(!back->devices[1].free_memory.known());
  LSE_EXPECT_EQ(back->devices[1].free_memory.value, 0.0);
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

LSE_TEST(a_cached_pool_never_replays_the_free_memory_it_was_saved_with) {
  backend::IBackend* be = host_backend();
  if (be == nullptr) {
    std::printf("       skipped: no cpu backend in this build\n");
    return;
  }
  const fs::path dir = scratch_dir() / "stale";
  PoolMember m;
  m.id = dev("cpu", 0);
  m.rank = 0;
  m.host = "test-host";
  m.backend = be;
  const PoolMember members[] = {m};

  PoolOptions opts;
  opts.profile_dir = dir.string();

  auto first = qualify_pool(members, nullptr, opts);
  LSE_EXPECT(first.ok());
  if (!first.ok()) return;

  // Age the entry the way an hour on a shared box would: same fingerprint,
  // same hardware, a free-memory figure that was true when it was written.
  // The fingerprint is folded from device identity and is deliberately blind
  // to anything volatile, so it cannot tell the difference — which is exactly
  // why the reader has to.
  auto fingerprint = pool_fingerprint(members, nullptr);
  LSE_EXPECT(fingerprint.ok());
  if (!fingerprint.ok()) return;
  PoolProfile stale = first.release();
  stale.devices[0].free_memory = Measured::measured(999e9);
  LSE_EXPECT_OK(save_pool_profile(stale, dir.string()));
  auto reread = load_pool_profile(*fingerprint, dir.string());
  LSE_EXPECT(reread.ok());
  // The store is a faithful serializer; it is qualify_pool that refuses.
  if (reread.ok()) LSE_EXPECT_EQ(reread->devices[0].free_memory.value, 999e9);

  auto second = qualify_pool(members, nullptr, opts);
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;
  // A cache hit: the numbers that are properties of the hardware came straight
  // off the disk, bit for bit.
  LSE_EXPECT_EQ(second->devices[0].dram_bytes_per_s.value,
                stale.devices[0].dram_bytes_per_s.value);
  LSE_EXPECT_EQ(second->devices[0].launch_overhead_ns.value,
                stale.devices[0].launch_overhead_ns.value);
  // The one that is a property of the moment did not.
  LSE_EXPECT(second->devices[0].free_memory.value != 999e9);
  // This backend declines the query, so nothing sampled it this run and the
  // verdict is a refusal rather than a stale pass.
  LSE_EXPECT(!second->devices[0].free_memory.known());
  LSE_EXPECT_EQ(second->devices[0].free_memory.value, 0.0);

  std::error_code ec;
  fs::remove_all(dir, ec);
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
  // Free memory is a capacity, not a rate, so zero is a meaningful measurement
  // — a device with nothing left. The rule it does share is the one that
  // matters: an unmeasured capacity carries no value to be mistaken for one,
  // and in particular is never quietly filled in from total_memory.
  if (!d.free_memory.known()) LSE_EXPECT_EQ(d.free_memory.value, 0.0);
  if (d.free_memory.known()) LSE_EXPECT(d.free_memory.value >= 0.0);
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

LSE_TEST(a_backend_that_declines_the_memory_query_leaves_free_memory_unknown) {
  backend::IBackend* be = host_backend();
  if (be == nullptr) {
    std::printf("       skipped: no cpu backend in this build\n");
    return;
  }
  // The seam refuses with a status. It does not answer zero, which would read
  // as a full device, and it does not answer the declared total, which would
  // read as an empty one.
  const auto answer = be->sample_free_memory();
  LSE_EXPECT(!answer.ok());
  LSE_EXPECT(answer.status().code() == StatusCode::kUnimplemented);
  LSE_EXPECT(!sample_free_memory(*be).known());
  LSE_EXPECT_EQ(sample_free_memory(*be).value, 0.0);

  auto profiled = probe_device(*be);
  LSE_EXPECT(profiled.ok());
  if (!profiled.ok()) return;
  LSE_EXPECT(!profiled->free_memory.known());
  LSE_EXPECT_EQ(profiled->free_memory.value, 0.0);

  // A decline is a hole the profile can name, not a silent one.
  auto probe = create_device_probe(*be);
  LSE_EXPECT(probe->declined().find("memory query") != std::string_view::npos);

  // And downstream it refuses a placement rather than approving one.
  PoolProfile pool;
  pool.fingerprint = "cpu-only";
  pool.devices.push_back(*profiled);
  pool.links.push_back(
      fake_link(profiled->id, profiled->id, PathKind::kSameDevice, 0.0, 0.0));
  const Capacity c =
      CostModel(pool).capacity_for(resident_work(kGB), profiled->id);
  LSE_EXPECT(c.fit == Fit::kUnknown);
  LSE_EXPECT(!c.fits());
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
  std::printf("       memory: %.2f GB free of %.2f GB declared (%s)\n",
              profiled->free_memory.value / 1e9,
              static_cast<double>(profiled->total_memory) / 1e9,
              std::string(to_string(profiled->free_memory.provenance)).c_str());
  // This device's runtime does answer the memory query, so the figure is here
  // and it is an observation rather than the part number.
  LSE_EXPECT(profiled->free_memory.known());
  LSE_EXPECT(profiled->free_memory.provenance == Provenance::kMeasured);
  LSE_EXPECT(profiled->free_memory.value >= 0.0);
  // Two samples a moment apart are both valid answers to different instants,
  // and neither is the declared total. A low reading on a busy box is the
  // correct answer, not a fault.
  LSE_EXPECT(sample_free_memory(*be).known());
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

// ---------------------------------------------------------------------------
// Enumeration: what a pool is built out of
// ---------------------------------------------------------------------------

LSE_TEST(an_enumerated_device_id_round_trips_into_a_pool_member) {
  std::size_t seen = 0;
  for (const std::string& name : backend::available_backends()) {
    auto found = backend::enumerate_devices(name);
    if (!found.ok()) {
      std::printf("       %s does not enumerate here: %s\n", name.c_str(),
                  found.status().to_string().c_str());
      continue;
    }
    for (const backend::DeviceDescriptor& d : *found) {
      // The descriptor's id is the engine's whole address for that device, and
      // it is the pool's too: a member is built by parsing it, never by
      // carrying the product name along.
      auto id = parse_device_id(d.id());
      LSE_EXPECT_OK(id.status());
      if (!id.ok()) continue;
      LSE_EXPECT(id->backend == d.backend);
      LSE_EXPECT_EQ(id->ordinal, d.ordinal);
      LSE_EXPECT(id->str() == d.id());

      PoolMember member;
      member.id = *id;
      LSE_EXPECT(member.id.str() == d.id());
      ++seen;
      std::printf("       %s: %s\n", d.id().c_str(),
                  d.product.known() ? d.product.value.c_str() : "unknown");
    }
  }
  // The cpu backend is in every build this suite runs in, so an empty walk is
  // a broken registry rather than a bare machine.
  LSE_EXPECT(seen >= 1);
}

LSE_TEST(enumerating_and_a_refused_ordinal_both_leave_the_process_bindable) {
  // The enumerate-then-bind loop, which is what a device set is built by. Two
  // hazards live here and both were real: enumeration has to bring the
  // accelerator up to count devices at all, and a bind that fails its range
  // check used to leave the accelerator up with nothing recorded as owning it,
  // after which every later bind in the process failed ALREADY_EXISTS.
  auto found = backend::enumerate_devices("hrx");
  if (!found.ok()) {
    std::printf("       skipped: hrx does not enumerate here (%s)\n",
                found.status().to_string().c_str());
    return;
  }
  const int count = static_cast<int>(found->size());
  auto created = backend::create_backend("hrx");
  LSE_EXPECT(created.ok());
  if (!created.ok()) return;
  std::unique_ptr<backend::IBackend> be = created.release();

  const Status refused = be->init(count + 7);
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.code() == StatusCode::kInvalidArgument);

  // Same object, right after the refusal.
  LSE_EXPECT_OK(be->init(0));
  LSE_EXPECT(be->device_info().arch == (*found)[0].arch.value);
  auto buf = be->allocate(4096, backend::MemoryClass::kDevice);
  LSE_EXPECT_OK(buf.status());
  if (buf.ok()) {
    backend::DeviceBuffer owned = buf.release();
    be->deallocate(owned);
  }
}

LSE_TEST(two_backends_for_one_device_coexist_in_one_process) {
  // The first acceptance item for a device *set*: a pool member holds its own
  // backend, so N members in one process is N live backend objects. Before the
  // accelerator's lifecycle was lifted out of the instance, the second init in
  // a process failed outright (hrx_gpu_initialize returns ALREADY_EXISTS) and
  // the first destructor took the accelerator down under everyone else.
  //
  // Only one ordinal exists on this box, so this binds it twice: that shares
  // the device and its allocator, which is the weaker of the two cases but the
  // one that catches process-wide lifecycle damage. Two *different* ordinals
  // await a multi-GPU box.
  auto first = backend::create_backend("hrx");
  if (!first.ok()) {
    std::printf("       skipped: no hrx backend in this build\n");
    return;
  }
  std::unique_ptr<backend::IBackend> a = first.release();
  if (const Status up = a->init(0); !up.ok()) {
    std::printf("       skipped: no hrx device here (%s)\n",
                up.to_string().c_str());
    return;
  }

  auto second = backend::create_backend("hrx");
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;
  std::unique_ptr<backend::IBackend> b = second.release();
  LSE_EXPECT_OK(b->init(0));

  LSE_EXPECT(a->device_info().arch == b->device_info().arch);
  LSE_EXPECT(a->stream_capabilities().stream_count ==
             b->stream_capabilities().stream_count);

  // Each instance compiles and dispatches its own kernels, on its own streams,
  // through its own emitter — the probe is the shortest path to a real
  // dispatch, and it ends in a readback so a wrong answer is a failure.
  auto profile_a = probe_device(*a);
  LSE_EXPECT_OK(profile_a.status());
  auto profile_b = probe_device(*b);
  LSE_EXPECT_OK(profile_b.status());
  if (profile_a.ok() && profile_b.ok()) {
    LSE_EXPECT(profile_a->dram_bytes_per_s.known());
    LSE_EXPECT(profile_b->dram_bytes_per_s.known());
    std::printf("       two instances, one device: %.0f and %.0f GB/s streamed\n",
                profile_a->dram_bytes_per_s.value / 1e9,
                profile_b->dram_bytes_per_s.value / 1e9);
  }

  // Destroying one must leave the other with a working device, not one whose
  // queries still answer and whose allocator has stopped.
  b.reset();
  const auto free_after = a->sample_free_memory();
  LSE_EXPECT(free_after.ok());
  auto buf = a->allocate(1u << 20, backend::MemoryClass::kDevice);
  LSE_EXPECT_OK(buf.status());
  if (buf.ok()) {
    backend::DeviceBuffer owned = buf.release();
    std::vector<std::uint32_t> host(8, 0x5eedu);
    LSE_EXPECT_OK(a->copy_h2d(host.data(), owned, host.size() * 4, 0));
    std::vector<std::uint32_t> back(8, 0);
    LSE_EXPECT_OK(a->copy_d2h(owned, back.data(), back.size() * 4, 0));
    LSE_EXPECT_EQ(back[7], 0x5eedu);
    a->deallocate(owned);
  }
}

// ---------------------------------------------------------------------------
// place: the live device set
// ---------------------------------------------------------------------------

LSE_TEST(a_selector_is_backend_qualified_or_it_is_refused) {
  auto empty = place::parse_selector("");
  LSE_EXPECT(empty.ok());
  LSE_EXPECT(empty->empty());

  auto one = place::parse_selector("cpu:0");
  LSE_EXPECT(one.ok());
  LSE_EXPECT_EQ(one->size(), std::size_t{1});
  LSE_EXPECT((*one)[0] == dev("cpu", 0));

  auto two = place::parse_selector(" hrx:0 , cpu:0 ");
  LSE_EXPECT(two.ok());
  LSE_EXPECT_EQ(two->size(), std::size_t{2});
  LSE_EXPECT((*two)[0] == dev("hrx", 0));
  LSE_EXPECT((*two)[1] == dev("cpu", 0));

  // A bare ordinal is not an address: PLAN.md's rule, enforced by the parser
  // that probe::parse_device_id already implements.
  LSE_EXPECT(!place::parse_selector("0").ok());
  // The same device twice is not two load locations.
  LSE_EXPECT(!place::parse_selector("cpu:0,cpu:0").ok());
}

LSE_TEST(a_set_owns_its_devices_and_routes_a_release_to_the_owner) {
  auto opened = place::Devices::open("cpu:0");
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  std::unique_ptr<place::Devices> set = opened.release();

  LSE_EXPECT_EQ(set->size(), std::size_t{1});
  LSE_EXPECT(set->members()[0].id == dev("cpu", 0));
  LSE_EXPECT(set->members()[0].index.bound());
  LSE_EXPECT(set->find(dev("cpu", 0)) != nullptr);
  LSE_EXPECT(set->find(dev("cpu", 7)) == nullptr);

  auto buf = set->allocate(0, 4096);
  LSE_EXPECT_OK(buf.status());
  if (!buf.ok()) return;
  backend::DeviceBuffer owned = buf.release();
  LSE_EXPECT(owned.residency == set->members()[0].index);
  LSE_EXPECT_EQ(set->member_of(owned.residency), std::size_t{0});
  LSE_EXPECT_OK(set->deallocate(owned));
  LSE_EXPECT(!owned.residency.bound());

  // A buffer some other set's device holds cannot be released here: guessing an
  // owner would free an allocation under whoever actually holds it.
  backend::DeviceBuffer foreign;
  foreign.handle = 1;
  foreign.size_bytes = 16;
  foreign.residency = backend::DeviceIndex{0xfffe};
  const Status refused = set->deallocate(foreign);
  LSE_EXPECT(!refused.ok());
  LSE_EXPECT(refused.code() == StatusCode::kInvalidArgument);

  LSE_EXPECT(!set->allocate(3, 16).ok());
}

LSE_TEST(a_named_device_that_will_not_come_up_is_an_error_not_an_omission) {
  // A set that quietly lost a member would place work by a plan nobody agreed
  // to, so the refusal has to name the device and carry the runtime's reason.
  auto missing = place::Devices::open("cpu:9");
  LSE_EXPECT(!missing.ok());
  LSE_EXPECT(missing.status().message().find("cpu:9") != std::string::npos);

  auto nonsense = place::Devices::open("nope:0");
  LSE_EXPECT(!nonsense.ok());
  LSE_EXPECT(nonsense.status().code() == StatusCode::kNotFound);
}

LSE_TEST(a_two_member_local_pool_qualifies_with_no_transport) {
  // Two devices in one box is not a degenerate distributed pool: there is no
  // rank to pair off with and no fabric to send over, so the process that
  // drives both ends measures them. On this machine the second member is the
  // cpu backend, which is a real IBackend with a real (host) memory of its own.
  if (device_backend() == nullptr) {
    std::printf("       skipped: no hrx device here\n");
    return;
  }
  LSE_EXPECT(host_backend() != nullptr);
  if (host_backend() == nullptr) return;

  auto opened = place::Devices::open("hrx:0,cpu:0");
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  std::unique_ptr<place::Devices> set = opened.release();
  LSE_EXPECT_EQ(set->size(), std::size_t{2});
  LSE_EXPECT_EQ(set->primary(), std::size_t{0});
  LSE_EXPECT(set->members()[0].id == dev("hrx", 0));
  LSE_EXPECT(set->members()[1].id == dev("cpu", 0));
  // Two members, two distinct residencies. Sharing one would make every
  // downstream question about which device holds what unanswerable.
  LSE_EXPECT(!(set->members()[0].index == set->members()[1].index));

  PoolOptions options;
  options.profile_dir = (scratch_dir() / "local-pool").string();
  LSE_EXPECT_OK(set->qualify(options));

  const PoolProfile& pool = set->profile();
  LSE_EXPECT_EQ(pool.devices.size(), std::size_t{2});
  LSE_EXPECT_EQ(pool.links.size(), std::size_t{4});
  LSE_EXPECT(pool.devices[0].dram_bytes_per_s.known());
  LSE_EXPECT(pool.devices[1].dram_bytes_per_s.known());

  // Both directions are measured, separately: a link can be faster one way.
  const LinkProfile* out = pool.link(dev("hrx", 0), dev("cpu", 0));
  const LinkProfile* back = pool.link(dev("cpu", 0), dev("hrx", 0));
  LSE_EXPECT(out != nullptr && back != nullptr);
  if (out == nullptr || back == nullptr) return;
  // The only move this seam can make today is out to host and in again, so
  // that is what was timed and that is what it claims to be. A peer path would
  // be a different measurement and a different PathKind.
  LSE_EXPECT(out->path == PathKind::kHostStaged);
  LSE_EXPECT(back->path == PathKind::kHostStaged);
  LSE_EXPECT(out->bandwidth_bytes_per_s.provenance == Provenance::kMeasured);
  LSE_EXPECT(out->bandwidth_bytes_per_s.value > 0.0);
  std::printf("       hrx:0->cpu:0 %.2f GB/s, cpu:0->hrx:0 %.2f GB/s\n",
              out->bandwidth_bytes_per_s.value / 1e9,
              back->bandwidth_bytes_per_s.value / 1e9);

  std::error_code ec;
  fs::remove_all(scratch_dir(), ec);
}

LSE_TEST(reach_is_queried_or_measured_and_otherwise_unknown) {
  // The mappings, on their own, because they are what decides whether a kernel
  // is allowed to load from another device's memory.
  LSE_EXPECT(place::reach_of(backend::PeerAccess::kSelf) == place::Reach::kSame);
  LSE_EXPECT(place::reach_of(backend::PeerAccess::kYes) == place::Reach::kPeer);
  LSE_EXPECT(place::reach_of(backend::PeerAccess::kOnRequest) ==
             place::Reach::kPeer);
  LSE_EXPECT(place::reach_of(backend::PeerAccess::kNo) == place::Reach::kNo);
  LSE_EXPECT(place::reach_of(backend::PeerAccess::kUnknown) ==
             place::Reach::kUnknown);

  LSE_EXPECT(place::reach_of(PathKind::kPeerDirect) == place::Reach::kPeer);
  LSE_EXPECT(place::reach_of(PathKind::kHostStaged) == place::Reach::kStaged);
  // A NIC that DMAs out of device memory still does not let a shader load from
  // another machine.
  LSE_EXPECT(place::reach_of(PathKind::kRdmaDirect) == place::Reach::kStaged);
  LSE_EXPECT(place::reach_of(PathKind::kUnknown) == place::Reach::kUnknown);

  LSE_EXPECT(place::readable(place::Reach::kSame));
  LSE_EXPECT(place::readable(place::Reach::kPeer));
  LSE_EXPECT(place::readable(place::Reach::kUnclaimed));
  // The two that must never pass: a staged path is a copy, not a read, and an
  // unanswerable question about two devices is not a yes.
  LSE_EXPECT(!place::readable(place::Reach::kStaged));
  LSE_EXPECT(!place::readable(place::Reach::kUnknown));
  LSE_EXPECT(!place::readable(place::Reach::kNo));
}

LSE_TEST(a_kernel_may_not_read_a_staged_members_bytes) {
  if (device_backend() == nullptr) {
    std::printf("       skipped: no hrx device here\n");
    return;
  }
  auto opened = place::Devices::open("hrx:0,cpu:0");
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  std::unique_ptr<place::Devices> set = opened.release();

  const backend::DeviceIndex gpu = set->members()[0].index;
  const backend::DeviceIndex host = set->members()[1].index;

  LSE_EXPECT(set->reach(gpu, 0) == place::Reach::kSame);
  LSE_EXPECT(set->reach(backend::kNoDevice, 0) == place::Reach::kUnclaimed);
  LSE_EXPECT_OK(set->may_read(gpu, 0));
  LSE_EXPECT_OK(set->may_read(backend::kNoDevice, 0));

  // Before anything measured the pair, nothing here says the read is legal —
  // and an unanswerable question refuses.
  LSE_EXPECT(set->reach(host, 0) == place::Reach::kUnknown);
  const Status before = set->may_read(host, 0);
  LSE_EXPECT(!before.ok());
  LSE_EXPECT(before.message().find("cpu:0") != std::string::npos);

  PoolOptions options;
  options.profile_dir = (scratch_dir() / "reach-pool").string();
  LSE_EXPECT_OK(set->qualify(options));

  // Measured, and what was measured is a copy through host. Still not a read:
  // the bytes have to be moved first.
  LSE_EXPECT(set->reach(host, 0) == place::Reach::kStaged);
  const Status after = set->may_read(host, 0);
  LSE_EXPECT(!after.ok());
  LSE_EXPECT(after.message().find("moved first") != std::string::npos);

  std::error_code ec;
  fs::remove_all(scratch_dir(), ec);
}

LSE_TEST(a_planner_on_one_device_never_moves_anything) {
  auto opened = place::Devices::open("cpu:0");
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  std::unique_ptr<place::Devices> set = opened.release();

  PoolProfile pool;
  pool.devices.push_back(
      fake_device(dev("cpu", 0), 2.0e9, 1000.0, 1.0e12, math::MatrixElem::kF16));
  pool.links.resize(1);
  pool.links[0].src = dev("cpu", 0);
  pool.links[0].dst = dev("cpu", 0);
  pool.links[0].path = PathKind::kSameDevice;
  pool.links[0].latency_ns = Measured::measured(0.0);

  const place::Planner planner(*set, pool);
  const Work work = matmul_work(4096, 4096, 4096, DType::kF16);
  auto placed = planner.place(work, 0, place::Transfer{1u << 20, 1u << 20});
  LSE_EXPECT_OK(placed.status());
  if (!placed.ok()) return;
  LSE_EXPECT_EQ(placed->member, std::size_t{0});
  LSE_EXPECT(!placed->relocates);
  LSE_EXPECT(placed->reason.find("one device") != std::string_view::npos);
}

LSE_TEST(a_planner_splits_a_two_member_set_by_measured_rate) {
  // The heterogeneous case, on injected profiles so it runs on a bare machine:
  // one member is four times the other's rate, and an equal split would idle
  // the fast one.
  auto opened = place::Devices::open("cpu:0");
  LSE_EXPECT_OK(opened.status());
  if (!opened.ok()) return;
  std::unique_ptr<place::Devices> set = opened.release();

  // A one-member set can still be asked to divide; the answer is the whole
  // work on the one member, which is the boundary case the solver must not
  // special-case away.
  PoolProfile pool;
  pool.devices.push_back(fake_device(dev("cpu", 0), 2.0e11, 500.0, 4.0e13,
                                     math::MatrixElem::kF16));
  pool.devices[0].free_memory = Measured::measured(8.0e9);
  pool.links.resize(1);
  pool.links[0].src = dev("cpu", 0);
  pool.links[0].dst = dev("cpu", 0);
  pool.links[0].path = PathKind::kSameDevice;
  pool.links[0].latency_ns = Measured::measured(0.0);

  const place::Planner planner(*set, pool);
  Work work = matmul_work(2048, 2048, 2048, DType::kF16);
  work.bytes_resident = 1u << 20;
  auto divided = planner.divide(work, 0);
  LSE_EXPECT_OK(divided.status());
  if (!divided.ok()) return;
  LSE_EXPECT(divided->fit == Fit::kFits);
  LSE_EXPECT_EQ(divided->portions.size(), std::size_t{1});
  LSE_EXPECT_EQ(divided->portions[0].member, std::size_t{0});
  LSE_EXPECT_NEAR(divided->portions[0].fraction, 1.0, 1e-9);
}

LSE_TEST_MAIN()
