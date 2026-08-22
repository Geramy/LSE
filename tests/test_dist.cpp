// Transports, wire codecs and collectives.
//
// Everything here runs on one device. The loopback transport is what makes a
// world size of 2, 4 or 8 testable on a machine with one GPU; what it cannot
// prove is called out in each test that is limited by it.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/dist/adapter.hpp"
#include "lse/dist/codec.hpp"
#include "lse/dist/collective.hpp"
#include "lse/dist/collectives.hpp"
#include "lse/dist/loopback.hpp"
#include "lse/dist/peer.hpp"
#include "lse/dist/quick_reduce.hpp"
#include "lse/dist/transport.hpp"
#include "lse/quant/block_codec.hpp"
#include "lse/quant/traits.hpp"

using namespace lse;
using namespace lse::dist;

namespace {

constexpr DType kLossy[] = {DType::kQ8, DType::kQ6, DType::kQ4};
constexpr DType kWireFormats[] = {DType::kF16, DType::kQ8, DType::kQ6,
                                  DType::kQ4};

// Deterministic, block-varying magnitudes: consecutive blocks differ by ~4x so
// a per-block scale is actually exercised.
std::vector<float> make_signal(std::size_t n, unsigned seed) {
  std::vector<float> v(n);
  std::uint32_t s = seed * 2654435761u + 1u;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
    const float block_scale =
        1.0f + 3.0f * static_cast<float>((i / kCodecBlockElems) % 4);
    v[i] = (u * 2.0f - 1.0f) * block_scale;
  }
  return v;
}

// The bound codec_block_tolerance() states: worst error inside a block over
// that block's own magnitude.
double block_relative_error(const std::vector<float>& ref,
                            const std::vector<float>& got) {
  double worst = 0.0;
  for (std::size_t b = 0; b + kCodecBlockElems <= ref.size();
       b += kCodecBlockElems) {
    double absmax = 0.0;
    double err = 0.0;
    for (std::size_t i = b; i < b + kCodecBlockElems; ++i) {
      absmax = std::max(absmax, std::fabs(static_cast<double>(ref[i])));
      err = std::max(err, std::fabs(static_cast<double>(ref[i]) -
                                    static_cast<double>(got[i])));
    }
    if (absmax > 0.0) worst = std::max(worst, err / absmax);
  }
  return worst;
}

std::vector<std::uint8_t> host_wire(DType kind,
                                    const std::vector<float>& src) {
  std::vector<std::uint8_t> wire(codec_wire_bytes(kind, src.size()));
  const Status s = host_encode(kind, src, wire);
  LSE_EXPECT_OK(s);
  return wire;
}

// A live HRX device, or null when this build or machine has none.
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

// One transport per rank, all in the same group.
std::vector<std::unique_ptr<ITransport>> make_fleet(Rank world,
                                                    const std::string& group) {
  std::vector<std::unique_ptr<ITransport>> out;
  for (Rank r = 0; r < world; ++r) {
    auto t = create_transport("loopback://");
    LSE_EXPECT(t.ok());
    if (!t.ok()) return {};
    TransportConfig cfg;
    cfg.endpoint = "loopback://";
    cfg.rank = r;
    cfg.world_size = world;
    cfg.group_id = group;
    cfg.timeout_ms = 20000;
    LSE_EXPECT_OK((*t)->connect(cfg));
    out.push_back(t.release());
  }
  return out;
}

CommBuffer host_view(std::vector<float>& v) {
  CommBuffer b;
  b.host = v.data();
  b.bytes = v.size() * sizeof(float);
  b.dtype = DType::kF32;
  return b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

LSE_TEST(codec_wire_format_is_the_existing_block_layout) {
  // Block width is quant's, which is also QuickReduce's, and the wire length
  // is core's dtype geometry rather than a second table.
  LSE_EXPECT_EQ(kCodecBlockElems, 32u);
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kQ8, 32u), sizeof(quant::BlockQ8));
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kQ6, 32u), sizeof(quant::BlockQ6));
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kQ4, 32u), sizeof(quant::BlockQ4));
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kF16, 32u), 64u);

  // The device codec's block geometry is the on-disk one, not a copy of it.
  LSE_EXPECT_EQ(quant::BlockCodec<quant::Q8>::kBytes, sizeof(quant::BlockQ8));
  LSE_EXPECT_EQ(quant::BlockCodec<quant::Q6>::kBytes, sizeof(quant::BlockQ6));
  LSE_EXPECT_EQ(quant::BlockCodec<quant::Q4>::kBytes, sizeof(quant::BlockQ4));

  // A payload that is not a whole number of blocks has no wire length.
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kQ4, 33u), 0u);
  LSE_EXPECT_EQ(codec_wire_bytes(DType::kQ4, 64u),
                2u * sizeof(quant::BlockQ4));
}

LSE_TEST(host_codec_bytes_match_the_quant_scheme_pack) {
  constexpr std::size_t kN = 32 * 64;
  const std::vector<float> src = make_signal(kN, 7);

  const auto check = [&]<class Scheme>(DType kind) {
    std::vector<typename Scheme::Block> blocks(kN / kCodecBlockElems);
    Scheme::quantize_row(src.data(), blocks.data(), kN);
    const std::vector<std::uint8_t> wire = host_wire(kind, src);
    LSE_EXPECT_EQ(wire.size(), blocks.size() * sizeof(typename Scheme::Block));
    LSE_EXPECT_EQ(std::memcmp(wire.data(), blocks.data(), wire.size()), 0);
  };
  check.template operator()<quant::Q8>(DType::kQ8);
  check.template operator()<quant::Q6>(DType::kQ6);
  check.template operator()<quant::Q4>(DType::kQ4);
}

LSE_TEST(codec_round_trip_error_is_ordered_q8_q6_q4) {
  constexpr std::size_t kN = 32 * 512;
  const std::vector<float> src = make_signal(kN, 11);
  double err[3] = {};

  for (int i = 0; i < 3; ++i) {
    const DType kind = kLossy[i];
    const std::vector<std::uint8_t> wire = host_wire(kind, src);
    std::vector<float> back(kN);
    LSE_EXPECT_OK(host_decode(kind, wire, back));
    err[i] = block_relative_error(src, back);
    std::printf("       %s: block max_rel %.3e (bound %.3e, %.1f bits/elem)\n",
                std::string(to_string(kind)).c_str(), err[i],
                codec_block_tolerance(kind),
                8.0 * static_cast<double>(codec_wire_bytes(kind, 32u)) / 32.0);
    LSE_EXPECT(err[i] <= codec_block_tolerance(kind));
  }
  // Q6 is the middle of the three, which is the claim that makes it the
  // fidelity/performance pick.
  LSE_EXPECT(err[0] < err[1]);
  LSE_EXPECT(err[1] < err[2]);

  std::vector<float> back(kN);
  const std::vector<std::uint8_t> wire = host_wire(DType::kF16, src);
  LSE_EXPECT_OK(host_decode(DType::kF16, wire, back));
  const double f16_err = block_relative_error(src, back);
  std::printf("       f16: block max_rel %.3e (bound %.3e, 16.0 bits/elem)\n",
              f16_err, codec_block_tolerance(DType::kF16));
  LSE_EXPECT(f16_err <= codec_block_tolerance(DType::kF16));
  LSE_EXPECT(f16_err < err[0]);
}

LSE_TEST(codec_round_trip_holds_against_a_double_reference) {
  // The decoded value must be a scale times an integer code, and the residual
  // must be the quantization step and nothing else — this catches a codec that
  // is self-consistent but not actually quantizing.
  constexpr std::size_t kN = 32 * 16;
  const std::vector<float> src = make_signal(kN, 3);
  for (DType kind : kLossy) {
    const std::vector<std::uint8_t> wire = host_wire(kind, src);
    std::vector<float> back(kN);
    LSE_EXPECT_OK(host_decode(kind, wire, back));
    for (std::size_t b = 0; b < kN; b += kCodecBlockElems) {
      double absmax = 0.0;
      for (std::size_t i = b; i < b + kCodecBlockElems; ++i) {
        absmax = std::max(absmax, std::fabs(static_cast<double>(src[i])));
      }
      const double f16_eps = codec_block_tolerance(DType::kF16);
      const double max_q =
          1.0 / (2.0 * (codec_block_tolerance(kind) - f16_eps));
      const double step = absmax / max_q;
      for (std::size_t i = b; i < b + kCodecBlockElems; ++i) {
        const double d = std::fabs(static_cast<double>(src[i]) -
                                   static_cast<double>(back[i]));
        // Half a step, plus the fp16 rounding of the scale applied to the
        // value's own magnitude.
        LSE_EXPECT(d <= step * 0.5 +
                            std::fabs(static_cast<double>(src[i])) * f16_eps +
                            1e-9);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Device codec
// ---------------------------------------------------------------------------

// Two live devices and a set over them. Built here rather than taken from a
// scheduler because this file sits below the layer that opens pools, and
// because a peer transport has to be exercised on real separate devices --
// two ranks on one device would pass without ever crossing a link.
struct TwoDevices : backend::IDeviceSet {
  std::vector<std::unique_ptr<backend::IBackend>> owned;

  std::size_t size() const noexcept override { return owned.size(); }
  backend::IBackend& device(std::size_t i) const override { return *owned[i]; }
  std::size_t primary() const noexcept override { return 0; }
  std::size_t member_of(backend::DeviceIndex d) const noexcept override {
    for (std::size_t i = 0; i < owned.size(); ++i) {
      if (owned[i]->device_index() == d) return i;
    }
    return owned.size();
  }
  Status may_read(backend::DeviceIndex held, std::size_t) const override {
    if (!held.bound() || member_of(held) < owned.size()) return OkStatus();
    return LSE_ERROR(kInvalidArgument, "residency outside this set");
  }
};

TwoDevices* two_devices() {
  static TwoDevices set = [] {
    TwoDevices s;
    for (int i = 0; i < 2; ++i) {
      auto b = backend::create_backend("hrx");
      if (!b.ok()) return TwoDevices{};
      auto owned = b.release();
      if (!owned->init(i).ok()) return TwoDevices{};
      s.owned.push_back(std::move(owned));
    }
    return s;
  }();
  return set.owned.size() == 2 ? &set : nullptr;
}

LSE_TEST(peer_transport_moves_device_memory_between_two_gpus) {
  TwoDevices* set = two_devices();
  if (set == nullptr) {
    std::printf("       skipped: fewer than two live devices\n");
    return;
  }
  constexpr std::size_t kN = 32 * 512;
  constexpr std::size_t kBytes = kN * sizeof(float);
  const std::vector<float> src = make_signal(kN, 7);

  auto a = set->device(0).allocate(kBytes, backend::MemoryClass::kDevice);
  auto b = set->device(1).allocate(kBytes, backend::MemoryClass::kDevice);
  LSE_EXPECT(a.ok() && b.ok());
  if (!a.ok() || !b.ok()) return;
  backend::DeviceBuffer from = a.release();
  backend::DeviceBuffer to = b.release();
  LSE_EXPECT_OK(set->device(0).copy_h2d(src.data(), from, kBytes, 0));

  PeerTransport t0, t1;
  t0.bind(*set, 0);
  t1.bind(*set, 1);
  TransportConfig c0;
  c0.rank = 0;
  c0.world_size = 2;
  c0.group_id = "peer-move";
  TransportConfig c1 = c0;
  c1.rank = 1;
  LSE_EXPECT_OK(t0.connect(c0));
  LSE_EXPECT_OK(t1.connect(c1));

  // The link the transport measured, not one anybody declared. A pool that
  // reached peer-direct reports tens of GB/s here; one that fell back to host
  // staging reports an order of magnitude less, and the collective selector
  // reads exactly this number.
  std::printf("       peer link: %.1f GB/s, %llu ns latency\n",
              static_cast<double>(t0.capabilities().bandwidth_bytes_per_s) / 1e9,
              static_cast<unsigned long long>(t0.capabilities().latency_ns));
  LSE_EXPECT(t0.capabilities().bandwidth_bytes_per_s > 1);

  CommBuffer send{};
  send.device = &from;
  send.bytes = kBytes;
  CommBuffer recv{};
  recv.device = &to;
  recv.bytes = kBytes;

  std::thread sender([&] {
    auto h = t0.send(send, 1, 0);
    if (h.ok()) (void)t0.wait(h.release());
  });
  auto got = t1.recv(recv, 0, 0);
  sender.join();
  LSE_EXPECT_OK(got.status());

  std::vector<float> back(kN, 0.0f);
  LSE_EXPECT_OK(set->device(1).copy_d2h(to, back.data(), kBytes, 0));
  LSE_EXPECT(std::memcmp(back.data(), src.data(), kBytes) == 0);
}

LSE_TEST(a_peer_allocation_can_be_imported_and_read_where_it_is_not_owned) {
  TwoDevices* set = two_devices();
  if (set == nullptr) {
    std::printf("       skipped: fewer than two live devices\n");
    return;
  }
  constexpr std::size_t kN = 1024;
  constexpr std::size_t kBytes = kN * sizeof(float);
  const std::vector<float> src = make_signal(kN, 11);

  auto owned = set->device(0).allocate(kBytes, backend::MemoryClass::kDevice);
  LSE_EXPECT(owned.ok());
  if (!owned.ok()) return;
  backend::DeviceBuffer theirs = owned.release();
  LSE_EXPECT_OK(set->device(0).copy_h2d(src.data(), theirs, kBytes, 0));

  // Device 1 importing device 0's allocation. Peer access alone does not make
  // a foreign handle bindable; this is what does, and before the AMDGPU HAL
  // learned to accept an allocation it can reach but does not own, it refused
  // with PERMISSION_DENIED.
  auto view = set->device(1).import_peer(theirs);
  if (!view.ok()) {
    std::printf("       skipped: %s\n", view.status().to_string().c_str());
    return;
  }
  backend::DeviceBuffer imported = view.release();
  LSE_EXPECT(imported.size_bytes == kBytes);

  // Read it back through the importing device, which is the claim that
  // matters: these bytes live on the other GPU.
  std::vector<float> back(kN, 0.0f);
  LSE_EXPECT_OK(set->device(1).copy_d2h(imported, back.data(), kBytes, 0));
  LSE_EXPECT(std::memcmp(back.data(), src.data(), kBytes) == 0);
}

LSE_TEST(device_codec_is_bit_identical_to_the_host_codec) {
  backend::IBackend* be = device_backend();
  if (be == nullptr) {
    std::printf("       skipped: no live device\n");
    return;
  }
  auto engine = create_codec_engine(*be);
  if (engine->name() != "hrx") {
    std::printf("       skipped: backend registered no device codec\n");
    return;
  }
  constexpr std::size_t kN = 32 * 256;
  const std::vector<float> src = make_signal(kN, 23);
  CommBuffer s;
  s.host = const_cast<float*>(src.data());
  s.bytes = src.size() * sizeof(float);

  for (DType kind : kWireFormats) {
    if (!engine->supports(kind)) {
      std::printf("       %s: declined on device (%s)\n",
                  std::string(to_string(kind)).c_str(),
                  std::string(engine->declined(kind)).c_str());
      continue;
    }
    const std::size_t wb = codec_wire_bytes(kind, kN);
    std::vector<std::uint8_t> dev_wire(wb, 0xCD);
    CommBuffer w;
    w.host = dev_wire.data();
    w.bytes = wb;
    w.dtype = DType::kU8;
    LSE_EXPECT_OK(engine->encode(kind, s, w, kN));

    const std::vector<std::uint8_t> ref_wire = host_wire(kind, src);
    LSE_EXPECT_EQ(std::memcmp(dev_wire.data(), ref_wire.data(), wb), 0);

    std::vector<float> dev_back(kN, -7.0f);
    CommBuffer d;
    d.host = dev_back.data();
    d.bytes = dev_back.size() * sizeof(float);
    LSE_EXPECT_OK(engine->decode(kind, w, d, kN));

    std::vector<float> ref_back(kN);
    LSE_EXPECT_OK(host_decode(kind, ref_wire, ref_back));
    LSE_EXPECT_EQ(std::memcmp(dev_back.data(), ref_back.data(),
                              kN * sizeof(float)),
                  0);
    std::printf("       %s: %zu wire bytes, device == host bit-for-bit\n",
                std::string(to_string(kind)).c_str(), wb);
  }
}

LSE_TEST(device_codec_throughput_is_measured_not_assumed) {
  backend::IBackend* be = device_backend();
  if (be == nullptr) {
    std::printf("       skipped: no live device\n");
    return;
  }
  auto engine = create_codec_engine(*be);
  if (engine->name() != "hrx") {
    std::printf("       skipped: backend registered no device codec\n");
    return;
  }
  std::printf("       device: %s\n", be->device_info().arch.c_str());
  // 16 MB of payload, well past this device's 2 MB L2, so the number is the
  // DRAM-resident rate a real collective segment would see rather than a
  // cache-resident one.
  constexpr std::size_t kN = 4u << 20;
  for (DType kind : kWireFormats) {
    if (!engine->supports(kind)) continue;
    LSE_EXPECT_OK(engine->calibrate(kind, kN));
    const std::uint64_t rate = engine->throughput_bytes_per_s(kind);
    LSE_EXPECT(rate > 0);
    std::printf(
        "       %s: %.2f GB/s payload round trip, %.2f GB/s of memory "
        "traffic\n",
        std::string(to_string(kind)).c_str(),
        static_cast<double>(rate) / 1e9,
        // Each round trip moves the payload twice and the wire twice.
        static_cast<double>(rate) / 1e9 *
            (2.0 + 2.0 * static_cast<double>(codec_wire_bytes(kind, 32u)) /
                       (32.0 * sizeof(float))));
  }
}

LSE_TEST(an_unsupported_codec_declines_rather_than_guessing) {
  // bf16 is a dtype the engine knows and has no BlockCodec specialization for,
  // which is the same position fp8 is in on this target. Asking for it must
  // refuse, name the reason, and leave the destination untouched.
  auto host = make_host_codec_engine();
  LSE_EXPECT(!host->supports(DType::kBF16));
  LSE_EXPECT(!host->declined(DType::kBF16).empty());

  constexpr std::size_t kN = 32 * 4;
  const std::vector<float> src = make_signal(kN, 5);
  std::vector<std::uint8_t> wire(kN * 2, 0xA5);
  const std::vector<std::uint8_t> before = wire;

  CommBuffer s;
  s.host = const_cast<float*>(src.data());
  s.bytes = src.size() * sizeof(float);
  CommBuffer w;
  w.host = wire.data();
  w.bytes = wire.size();
  w.dtype = DType::kU8;

  const Status st = host->encode(DType::kBF16, s, w, kN);
  LSE_EXPECT(!st.ok());
  LSE_EXPECT(st.code() == StatusCode::kUnimplemented);
  LSE_EXPECT_EQ(std::memcmp(wire.data(), before.data(), wire.size()), 0);

  backend::IBackend* be = device_backend();
  if (be == nullptr) return;
  auto engine = create_codec_engine(*be);
  LSE_EXPECT(!engine->supports(DType::kBF16));
  const Status dst = engine->encode(DType::kBF16, s, w, kN);
  LSE_EXPECT(!dst.ok());
  LSE_EXPECT_EQ(std::memcmp(wire.data(), before.data(), wire.size()), 0);
  std::printf("       device declines bf16: %s\n",
              std::string(engine->declined(DType::kBF16)).c_str());
  // Every format that DOES have a specialization is offered on this device.
  for (DType f : kWireFormats) LSE_EXPECT(engine->supports(f));

  // A selector offered only a declining codec must not plan a compressed run.
  CollectiveCost cost;
  cost.caps.bandwidth_bytes_per_s = 1'000'000'000ull;
  cost.caps.latency_ns = 1000;
  cost.world_size = 2;
  cost.elems = 1u << 20;
  const CollectivePlan plan = select_all_reduce(cost, engine.get());
  LSE_EXPECT(plan.algo != CollectiveAlgo::kTwoShot ||
             plan.wire != DType::kBF16);
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

LSE_TEST(loopback_registers_under_its_own_scheme) {
  const std::vector<std::string> schemes = available_transports();
  LSE_EXPECT(std::find(schemes.begin(), schemes.end(), "loopback") !=
             schemes.end());
  auto ok = create_transport("loopback://ranks");
  LSE_EXPECT(ok.ok());
  auto bad = create_transport("infiniband://nope");
  LSE_EXPECT(!bad.ok());
  LSE_EXPECT(bad.status().code() == StatusCode::kNotFound);
}

LSE_TEST(loopback_capabilities_are_measured_from_the_machine) {
  auto fleet = make_fleet(2, "caps");
  if (fleet.empty()) return;
  const Capabilities& caps = fleet[0]->capabilities();
  LSE_EXPECT(!caps.native_collectives);
  LSE_EXPECT(caps.bandwidth_bytes_per_s > 0);
  LSE_EXPECT(caps.latency_bound_threshold() > 0);
  std::printf(
      "       loopback: %.1f GB/s, %llu ns, latency-bound below %zu B\n",
      static_cast<double>(caps.bandwidth_bytes_per_s) / 1e9,
      static_cast<unsigned long long>(caps.latency_ns),
      caps.latency_bound_threshold());
}

LSE_TEST(ring_all_reduce_over_loopback_sums_every_rank) {
  constexpr Rank kWorld = 4;
  constexpr std::size_t kN = 512;
  auto fleet = make_fleet(kWorld, "ring");
  if (fleet.empty()) return;

  std::vector<std::vector<float>> data;
  for (Rank r = 0; r < kWorld; ++r) {
    data.push_back(make_signal(kN, 40u + static_cast<unsigned>(r)));
  }
  std::vector<double> want(kN, 0.0);
  for (Rank r = 0; r < kWorld; ++r) {
    for (std::size_t i = 0; i < kN; ++i) {
      want[i] += static_cast<double>(data[static_cast<std::size_t>(r)][i]);
    }
  }

  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  for (Rank r = 0; r < kWorld; ++r) {
    threads.emplace_back([&, r] {
      CommBuffer b = host_view(data[static_cast<std::size_t>(r)]);
      if (!ring_all_reduce_over(*fleet[static_cast<std::size_t>(r)], b,
                                ReduceOp::kSum)
               .ok()) {
        ++failures;
      }
    });
  }
  for (std::thread& t : threads) t.join();
  LSE_EXPECT_EQ(failures.load(), 0);

  for (Rank r = 0; r < kWorld; ++r) {
    for (std::size_t i = 0; i < kN; ++i) {
      LSE_EXPECT_NEAR(data[static_cast<std::size_t>(r)][i], want[i], 1e-4);
    }
  }
}

// ---------------------------------------------------------------------------
// Two-shot compressed all-reduce
// ---------------------------------------------------------------------------

namespace {

struct TwoShotError {
  // Worst block error over the sum of the ranks' block magnitudes. This is the
  // quantity the two-shot bounds: one codec hop on each rank's contribution
  // (|sum of those| <= tol * sum of magnitudes) plus one on the result
  // (|that| <= tol * |result| <= tol * sum of magnitudes), so 2*tol whatever
  // the world size.
  double vs_inputs = 1e9;
  // The same error over the magnitude of the answer. Reported, not bounded:
  // random-sign summands cancel, so the answer can be much smaller than the
  // inputs and this ratio is data-dependent rather than a property of the
  // algorithm.
  double vs_output = 1e9;
};

// Runs the two-shot on `world` threads and measures against the exact sum
// computed in double.
TwoShotError run_two_shot(Rank world, DType kind, std::size_t n,
                          const std::string& group) {
  auto fleet = make_fleet(world, group);
  if (fleet.empty()) return {};

  std::vector<std::vector<float>> data;
  for (Rank r = 0; r < world; ++r) {
    data.push_back(make_signal(n, 100u + static_cast<unsigned>(r)));
  }
  const std::vector<std::vector<float>> inputs = data;
  std::vector<float> want(n, 0.0f);
  {
    std::vector<double> acc(n, 0.0);
    for (Rank r = 0; r < world; ++r) {
      for (std::size_t i = 0; i < n; ++i) {
        acc[i] += static_cast<double>(data[static_cast<std::size_t>(r)][i]);
      }
    }
    for (std::size_t i = 0; i < n; ++i) want[i] = static_cast<float>(acc[i]);
  }

  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  std::vector<std::unique_ptr<CodecEngine>> engines;
  for (Rank r = 0; r < world; ++r) engines.push_back(make_host_codec_engine());

  for (Rank r = 0; r < world; ++r) {
    threads.emplace_back([&, r] {
      const auto i = static_cast<std::size_t>(r);
      CommBuffer b = host_view(data[i]);
      const Status s = two_shot_all_reduce(*fleet[i], *engines[i], kind, b,
                                           ReduceOp::kSum);
      if (!s.ok()) {
        std::printf("       rank %d: %s\n", r, s.to_string().c_str());
        ++failures;
      }
    });
  }
  for (std::thread& t : threads) t.join();
  if (failures.load() != 0) return {};

  // Every rank must hold the same answer — a collective that differs by rank
  // is broken however small its error is.
  for (Rank r = 1; r < world; ++r) {
    const auto i = static_cast<std::size_t>(r);
    if (std::memcmp(data[0].data(), data[i].data(), n * sizeof(float)) != 0) {
      std::printf("       rank %d disagrees with rank 0\n", r);
      return {};
    }
  }

  TwoShotError out{0.0, 0.0};
  for (std::size_t b = 0; b + kCodecBlockElems <= n; b += kCodecBlockElems) {
    double err = 0.0;
    double out_absmax = 0.0;
    for (std::size_t i = b; i < b + kCodecBlockElems; ++i) {
      err = std::max(err, std::fabs(static_cast<double>(want[i]) -
                                    static_cast<double>(data[0][i])));
      out_absmax =
          std::max(out_absmax, std::fabs(static_cast<double>(want[i])));
    }
    double in_absmax_sum = 0.0;
    for (Rank r = 0; r < world; ++r) {
      double m = 0.0;
      for (std::size_t i = b; i < b + kCodecBlockElems; ++i) {
        m = std::max(m, std::fabs(static_cast<double>(
                            inputs[static_cast<std::size_t>(r)][i])));
      }
      in_absmax_sum += m;
    }
    if (in_absmax_sum > 0.0) {
      out.vs_inputs = std::max(out.vs_inputs, err / in_absmax_sum);
    }
    if (out_absmax > 0.0) {
      out.vs_output = std::max(out.vs_output, err / out_absmax);
    }
  }
  return out;
}

}  // namespace

LSE_TEST(two_shot_all_reduce_matches_a_sequential_sum) {
  constexpr std::size_t kN = 32 * 64;  // divides by 2, 4 and 8 whole blocks
  for (Rank world : {2, 4, 8}) {
    for (DType kind : kWireFormats) {
      const TwoShotError e =
          run_two_shot(world, kind, kN,
                       "ts-" + std::string(to_string(kind)) + "-" +
                           std::to_string(world));
      std::printf(
          "       world %d %-3s: max_rel %.3e vs inputs (bound %.3e), "
          "%.3e vs answer\n",
          world, std::string(to_string(kind)).c_str(), e.vs_inputs,
          2.0 * codec_block_tolerance(kind), e.vs_output);
      // Two codec hops per element, so twice the single-block bound.
      LSE_EXPECT(e.vs_inputs <= 2.0 * codec_block_tolerance(kind));
    }
  }
}

LSE_TEST(two_shot_error_does_not_grow_with_the_world_size) {
  // The property that makes two-shot the right shape for a lossy wire: each
  // element crosses the codec exactly twice whatever the world size, unlike a
  // ring, where it crosses 2*(world-1) times and its bound would grow with the
  // world. Held against the inputs, the bound must be flat in `world`.
  constexpr std::size_t kN = 32 * 64;
  const double bound = 2.0 * codec_block_tolerance(DType::kQ4);
  double prev = 0.0;
  for (Rank world : {2, 4, 8}) {
    const TwoShotError e =
        run_two_shot(world, DType::kQ4, kN, "grow" + std::to_string(world));
    std::printf("       q4 world %d: %.3e vs inputs (flat bound %.3e)\n", world,
                e.vs_inputs, bound);
    LSE_EXPECT(e.vs_inputs <= bound);
    // A ring would multiply the hop count by (world-1); this must not.
    if (prev > 0.0) LSE_EXPECT(e.vs_inputs <= prev * 1.25);
    prev = e.vs_inputs;
  }
}

LSE_TEST(two_shot_refuses_a_payload_it_cannot_segment) {
  auto fleet = make_fleet(2, "shape");
  if (fleet.empty()) return;
  std::vector<float> odd(48);  // 1.5 blocks, not 2 whole blocks per rank
  CommBuffer b = host_view(odd);
  auto engine = make_host_codec_engine();
  const Status s =
      two_shot_all_reduce(*fleet[0], *engine, DType::kQ4, b, ReduceOp::kSum);
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

LSE_TEST(selection_follows_capabilities_and_the_cost_model) {
  auto engine = make_host_codec_engine();
  LSE_EXPECT_OK(engine->calibrate(DType::kQ4, 0));

  CollectiveCost cost;
  cost.world_size = 4;
  cost.caps.bandwidth_bytes_per_s = 25'000'000'000ull;  // 25 GB/s fabric
  cost.caps.latency_ns = 5000;
  cost.caps.device_memory_direct = true;

  // A native transport is delegated to and never second-guessed.
  {
    CollectiveCost native = cost;
    native.caps.native_collectives = true;
    native.elems = 1u << 20;
    LSE_EXPECT(select_all_reduce(native, engine.get()).algo ==
               CollectiveAlgo::kNative);
  }

  // Small payload: latency dominates, so compression buys nothing.
  {
    CollectiveCost small = cost;
    small.elems = 32;  // 128 B, far below the 125 KB bandwidth-delay product
    const CollectivePlan p = select_all_reduce(small, engine.get());
    LSE_EXPECT(p.algo == CollectiveAlgo::kRing);
    std::printf("       small: %s (%s)\n",
                std::string(to_string(p.algo)).c_str(),
                std::string(p.reason).c_str());
  }

  // Large payload on a link slow relative to this machine's codec: the
  // compressed wire wins. The threshold is derived from the engine's own
  // measured rate, so the test asserts the trade exists rather than pinning a
  // number that is really a property of the host.
  const std::uint64_t codec_rate =
      engine->throughput_bytes_per_s(DType::kQ4);
  LSE_EXPECT(codec_rate > 0);
  std::printf("       host codec: %.2f GB/s (q4 round trip)\n",
              static_cast<double>(codec_rate) / 1e9);
  {
    CollectiveCost big = cost;
    big.caps.bandwidth_bytes_per_s = codec_rate / 16;
    big.elems = 1u << 22;  // 16 MB
    const CollectivePlan p = select_all_reduce(big, engine.get());
    LSE_EXPECT(p.algo == CollectiveAlgo::kTwoShot);
    std::printf("       slow link (%.2f GB/s): %s %s (%s)\n",
                static_cast<double>(big.caps.bandwidth_bytes_per_s) / 1e9,
                std::string(to_string(p.algo)).c_str(),
                std::string(to_string(p.wire)).c_str(),
                std::string(p.reason).c_str());
  }

  // A link far faster than the codec: compressing is a loss, so it must not.
  {
    CollectiveCost fast = cost;
    fast.caps.bandwidth_bytes_per_s = codec_rate * 64;
    fast.caps.latency_ns = 0;
    fast.elems = 1u << 22;
    const CollectivePlan p = select_all_reduce(fast, engine.get());
    LSE_EXPECT(p.algo == CollectiveAlgo::kRing);
  }

  // A non-linear reduction never gets a lossy wire.
  {
    CollectiveCost prod = cost;
    prod.caps.bandwidth_bytes_per_s = codec_rate / 16;
    prod.elems = 1u << 22;
    prod.op = ReduceOp::kProd;
    LSE_EXPECT(select_all_reduce(prod, engine.get()).algo ==
               CollectiveAlgo::kRing);
  }

  // No engine at all is the same as no compression, never a failure.
  {
    CollectiveCost none = cost;
    none.caps.bandwidth_bytes_per_s = codec_rate / 16;
    none.elems = 1u << 22;
    LSE_EXPECT(select_all_reduce(none, nullptr).algo == CollectiveAlgo::kRing);
  }

  // World size 1 short-circuits.
  {
    CollectiveCost solo = cost;
    solo.world_size = 1;
    solo.elems = 1u << 22;
    const CollectivePlan p = select_all_reduce(solo, engine.get());
    LSE_EXPECT(p.algo == CollectiveAlgo::kRing);
  }
}

LSE_TEST(all_reduce_runs_whatever_the_selector_planned) {
  constexpr Rank kWorld = 4;
  constexpr std::size_t kN = 32 * 64;
  auto fleet = make_fleet(kWorld, "select-run");
  if (fleet.empty()) return;

  std::vector<std::vector<float>> data;
  for (Rank r = 0; r < kWorld; ++r) {
    data.push_back(make_signal(kN, 200u + static_cast<unsigned>(r)));
  }
  std::vector<double> want(kN, 0.0);
  for (Rank r = 0; r < kWorld; ++r) {
    for (std::size_t i = 0; i < kN; ++i) {
      want[i] += static_cast<double>(data[static_cast<std::size_t>(r)][i]);
    }
  }

  std::vector<std::unique_ptr<CodecEngine>> engines;
  for (Rank r = 0; r < kWorld; ++r) engines.push_back(make_host_codec_engine());

  CommBuffer probe = host_view(data[0]);
  CollectiveContext ctx0{fleet[0].get(), engines[0].get()};
  auto planned = plan_all_reduce(ctx0, probe, ReduceOp::kSum);
  LSE_EXPECT(planned.ok());
  if (!planned.ok()) return;
  std::printf("       plan: %s/%s (%s)\n",
              std::string(to_string(planned->algo)).c_str(),
              std::string(to_string(planned->wire)).c_str(),
              std::string(planned->reason).c_str());

  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  for (Rank r = 0; r < kWorld; ++r) {
    threads.emplace_back([&, r] {
      const auto i = static_cast<std::size_t>(r);
      CommBuffer b = host_view(data[i]);
      CollectiveContext ctx{fleet[i].get(), engines[i].get()};
      if (!all_reduce(ctx, b, ReduceOp::kSum).ok()) ++failures;
    });
  }
  for (std::thread& t : threads) t.join();
  LSE_EXPECT_EQ(failures.load(), 0);

  // Whatever was chosen, the answer holds to that plan's tolerance.
  const double tol = planned->algo == CollectiveAlgo::kTwoShot
                         ? 2.0 * codec_block_tolerance(planned->wire) *
                               static_cast<double>(kWorld)
                         : 1e-5;
  std::vector<float> want_f(kN);
  for (std::size_t i = 0; i < kN; ++i) want_f[i] = static_cast<float>(want[i]);
  for (Rank r = 0; r < kWorld; ++r) {
    const auto i = static_cast<std::size_t>(r);
    LSE_EXPECT(block_relative_error(want_f, data[i]) <= tol);
  }
}

LSE_TEST(a_pinned_plan_runs_through_the_same_entry_point) {
  // The loopback's link is a memcpy at tens of GB/s, so its cost model will
  // never choose to compress — correctly. The executor still has to dispatch a
  // two-shot plan when one is chosen, which is what this pins.
  constexpr Rank kWorld = 4;
  constexpr std::size_t kN = 32 * 64;
  auto fleet = make_fleet(kWorld, "pinned");
  if (fleet.empty()) return;

  std::vector<std::vector<float>> data;
  for (Rank r = 0; r < kWorld; ++r) {
    data.push_back(make_signal(kN, 300u + static_cast<unsigned>(r)));
  }
  std::vector<float> want(kN, 0.0f);
  for (std::size_t i = 0; i < kN; ++i) {
    double acc = 0.0;
    for (Rank r = 0; r < kWorld; ++r) {
      acc += static_cast<double>(data[static_cast<std::size_t>(r)][i]);
    }
    want[i] = static_cast<float>(acc);
  }

  std::vector<std::unique_ptr<CodecEngine>> engines;
  for (Rank r = 0; r < kWorld; ++r) engines.push_back(make_host_codec_engine());

  CollectivePlan plan;
  plan.algo = CollectiveAlgo::kTwoShot;
  plan.wire = DType::kQ6;

  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  for (Rank r = 0; r < kWorld; ++r) {
    threads.emplace_back([&, r] {
      const auto i = static_cast<std::size_t>(r);
      CommBuffer b = host_view(data[i]);
      CollectiveContext ctx{fleet[i].get(), engines[i].get()};
      if (!run_all_reduce(ctx, plan, b, ReduceOp::kSum).ok()) ++failures;
    });
  }
  for (std::thread& t : threads) t.join();
  LSE_EXPECT_EQ(failures.load(), 0);
  for (Rank r = 0; r < kWorld; ++r) {
    LSE_EXPECT(block_relative_error(want, data[static_cast<std::size_t>(r)]) <=
               2.0 * codec_block_tolerance(DType::kQ6) *
                   static_cast<double>(kWorld));
  }

  // The same entry point with a ring plan is exact.
  CollectivePlan ring;
  ring.algo = CollectiveAlgo::kRing;
  std::vector<std::vector<float>> fresh;
  for (Rank r = 0; r < kWorld; ++r) {
    fresh.push_back(make_signal(kN, 300u + static_cast<unsigned>(r)));
  }
  threads.clear();
  for (Rank r = 0; r < kWorld; ++r) {
    threads.emplace_back([&, r] {
      const auto i = static_cast<std::size_t>(r);
      CommBuffer b = host_view(fresh[i]);
      CollectiveContext ctx{fleet[i].get(), engines[i].get()};
      if (!run_all_reduce(ctx, ring, b, ReduceOp::kSum).ok()) ++failures;
    });
  }
  for (std::thread& t : threads) t.join();
  LSE_EXPECT_EQ(failures.load(), 0);
  for (Rank r = 0; r < kWorld; ++r) {
    for (std::size_t i = 0; i < kN; ++i) {
      LSE_EXPECT_NEAR(fresh[static_cast<std::size_t>(r)][i], want[i], 1e-5);
    }
  }
}

LSE_TEST_MAIN()
