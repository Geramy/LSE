// The trace module: records in, Perfetto bytes out, decoded back with a
// dependency-free protobuf walker so the assertions are on the wire and not on
// the writer's own opinion of what it wrote.
//
// No device is needed for any of it. Where one answers, the last two cases run
// real dispatches through the same seam the scheduler hook will use and leave a
// .pftrace behind.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/trace/collector.hpp"
#include "lse/trace/export.hpp"
#include "lse/trace/record.hpp"

using namespace lse;
using namespace lse::trace;

namespace {

// ---------------------------------------------------------------------------
// A protobuf reader that vendors nothing: enough to walk a pftrace field by
// field, which is the only way to assert that the field NUMBERS are right.
// ---------------------------------------------------------------------------

struct Cursor {
  const std::byte* p = nullptr;
  const std::byte* end = nullptr;

  [[nodiscard]] bool done() const noexcept { return p >= end; }

  std::uint64_t varint() {
    std::uint64_t value = 0;
    int shift = 0;
    while (p < end) {
      const auto byte = static_cast<std::uint8_t>(*p++);
      value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) break;
      shift += 7;
    }
    return value;
  }
};

struct Field {
  std::uint32_t number = 0;
  std::uint32_t wire = 0;
  std::uint64_t value = 0;         // wire 0 and wire 1 (raw bits)
  const std::byte* data = nullptr;  // wire 2
  std::size_t size = 0;
};

void walk(const std::byte* begin, const std::byte* end, auto&& fn) {
  Cursor c{begin, end};
  while (!c.done()) {
    const std::uint64_t key = c.varint();
    Field f;
    f.number = static_cast<std::uint32_t>(key >> 3);
    f.wire = static_cast<std::uint32_t>(key & 7);
    switch (f.wire) {
      case 0:
        f.value = c.varint();
        break;
      case 1: {
        std::uint64_t bits = 0;
        for (int i = 0; i < 8 && c.p < c.end; ++i) {
          bits |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(*c.p++))
                  << (8 * i);
        }
        f.value = bits;
        break;
      }
      case 2: {
        const std::uint64_t len = c.varint();
        f.data = c.p;
        f.size = static_cast<std::size_t>(len);
        c.p += len;
        break;
      }
      default:
        return;  // no group or fixed32 field is written by this exporter
    }
    fn(f);
  }
}

std::string text_of(const Field& f) {
  return std::string(reinterpret_cast<const char*>(f.data), f.size);
}

double double_of(const Field& f) {
  double out = 0.0;
  std::memcpy(&out, &f.value, sizeof(out));
  return out;
}

struct DecodedTrack {
  std::uint64_t uuid = 0;
  std::string name;
  std::uint64_t parent = 0;
  bool counter = false;
};

struct DecodedEvent {
  std::uint64_t ts = 0;
  std::uint32_t clock = 0;
  std::uint32_t type = 0;
  std::uint64_t track = 0;
  std::string name;
  std::vector<std::uint64_t> flows;
  std::map<std::string, std::uint64_t> uints;
  std::map<std::string, std::string> strings;
  std::map<std::string, double> doubles;
  std::uint64_t counter_value = 0;
  bool has_counter = false;
};

struct DecodedTrace {
  std::vector<DecodedTrack> tracks;
  std::vector<DecodedEvent> events;
  std::vector<std::pair<std::uint32_t, std::uint64_t>> clock_snapshot;
  std::vector<std::uint32_t> sequence_ids;
  std::size_t packets = 0;

  [[nodiscard]] const DecodedTrack* track(std::string_view name) const {
    const auto it = std::find_if(tracks.begin(), tracks.end(),
                                 [&](const DecodedTrack& t) { return t.name == name; });
    return it == tracks.end() ? nullptr : &*it;
  }
  [[nodiscard]] std::vector<const DecodedEvent*> on(std::uint64_t uuid) const {
    std::vector<const DecodedEvent*> out;
    for (const DecodedEvent& e : events) {
      if (e.track == uuid) out.push_back(&e);
    }
    return out;
  }
  // A group instant and the dispatches attributed to it share the kernel's
  // name, so a lookup has to say which track it means.
  [[nodiscard]] const DecodedEvent* named_on(std::uint64_t uuid,
                                             std::string_view name) const {
    const auto it = std::find_if(events.begin(), events.end(),
                                 [&](const DecodedEvent& e) {
                                   return e.track == uuid && e.name == name &&
                                          e.type == pftrace::kSliceBegin;
                                 });
    return it == events.end() ? nullptr : &*it;
  }
};

void decode_annotation(const Field& field, DecodedEvent& event) {
  std::string name;
  std::optional<std::uint64_t> uint_value;
  std::optional<std::string> string_value;
  std::optional<double> double_value;
  walk(field.data, field.data + field.size, [&](const Field& f) {
    if (f.number == pftrace::kAnnotationName) name = text_of(f);
    if (f.number == pftrace::kAnnotationUint) uint_value = f.value;
    if (f.number == pftrace::kAnnotationString) string_value = text_of(f);
    if (f.number == pftrace::kAnnotationDouble) double_value = double_of(f);
  });
  if (name.empty()) return;
  if (uint_value) event.uints[name] = *uint_value;
  if (string_value) event.strings[name] = *string_value;
  if (double_value) event.doubles[name] = *double_value;
}

DecodedTrace decode(const std::vector<std::byte>& bytes) {
  DecodedTrace out;
  walk(bytes.data(), bytes.data() + bytes.size(), [&](const Field& top) {
    if (top.number != pftrace::kTracePacket || top.wire != 2) return;
    ++out.packets;
    std::uint64_t ts = 0;
    std::uint32_t clock = 0;
    const std::size_t before = out.events.size();
    walk(top.data, top.data + top.size, [&](const Field& f) {
      switch (f.number) {
        case pftrace::kPacketTimestamp:
          ts = f.value;
          break;
        case pftrace::kPacketClockId:
          clock = static_cast<std::uint32_t>(f.value);
          break;
        case pftrace::kPacketSequenceId:
          out.sequence_ids.push_back(static_cast<std::uint32_t>(f.value));
          break;
        case pftrace::kPacketTrackDescriptor: {
          DecodedTrack track;
          walk(f.data, f.data + f.size, [&](const Field& d) {
            if (d.number == pftrace::kTrackUuid) track.uuid = d.value;
            if (d.number == pftrace::kTrackName) track.name = text_of(d);
            if (d.number == pftrace::kTrackParentUuid) track.parent = d.value;
            if (d.number == pftrace::kTrackCounter) track.counter = true;
          });
          out.tracks.push_back(std::move(track));
          break;
        }
        case pftrace::kPacketClockSnapshot: {
          walk(f.data, f.data + f.size, [&](const Field& s) {
            if (s.number != pftrace::kClockSnapshotClock) return;
            std::uint32_t id = 0;
            std::uint64_t value = 0;
            walk(s.data, s.data + s.size, [&](const Field& c) {
              if (c.number == pftrace::kClockId) id = static_cast<std::uint32_t>(c.value);
              if (c.number == pftrace::kClockTimestamp) value = c.value;
            });
            out.clock_snapshot.emplace_back(id, value);
          });
          break;
        }
        case pftrace::kPacketTrackEvent: {
          DecodedEvent event;
          walk(f.data, f.data + f.size, [&](const Field& e) {
            switch (e.number) {
              case pftrace::kEventType:
                event.type = static_cast<std::uint32_t>(e.value);
                break;
              case pftrace::kEventTrackUuid:
                event.track = e.value;
                break;
              case pftrace::kEventName:
                event.name = text_of(e);
                break;
              case pftrace::kEventFlowIds:
                event.flows.push_back(e.value);
                break;
              case pftrace::kEventCounterValue:
                event.counter_value = e.value;
                event.has_counter = true;
                break;
              case pftrace::kEventDebugAnnotations:
                decode_annotation(e, event);
                break;
              default:
                break;
            }
          });
          out.events.push_back(std::move(event));
          break;
        }
        default:
          break;
      }
    });
    // The packet's timestamp and clock id belong to the event it carried; they
    // are read as sibling fields of the TrackEvent, so they are applied after.
    if (out.events.size() > before) {
      out.events.back().ts = ts;
      out.events.back().clock = clock;
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

constexpr std::uint64_t kCacheKey = 0x51c0ffee51c0ffeeull;
constexpr std::uint64_t kSignature = 7497904328953780393ull;

GroupRecord fused_group() {
  GroupRecord group;
  group.id = GroupId{kCacheKey, kSignature};
  group.entry_name = "lse_fused_" + std::to_string(kSignature);
  group.arch = "gfx1151";
  group.source_hash = 0xabcdef0123456789ull;
  group.nodes = {
      NodeRef{0, "linear", "[1,1024]", ""},
      NodeRef{1, "silu", "[1,1024]", "silu"},
      NodeRef{2, "mul", "[1,1024]", ""},
  };
  return group;
}

DispatchRecord dispatch_at(std::uint64_t begin, std::uint64_t end) {
  DispatchRecord rec;
  rec.group = GroupId{kCacheKey, kSignature};
  rec.host_begin_ns = begin;
  rec.host_end_ns = end;
  rec.stream = 0;
  rec.node_count = 3;
  rec.bindings = 4;
  rec.workgroup_count = 8;
  rec.workgroup_size = 256;
  rec.elements = 1024;
  return rec;
}

backend::DeviceClock gfx1151_clock() {
  backend::DeviceClock clock;
  clock.domain = backend::ClockDomain::kDeviceAgent;
  clock.ticks_per_second = 99'810'000;  // measured on this part, not rounded
  clock.valid_bits = 64;
  clock.ordinal = 0;
  return clock;
}

// One backend for the process, as the engine has: a device brought up twice in
// one process is not what any caller does, and the second bring-up is what fails.
backend::IBackend* live_backend() {
  static std::unique_ptr<backend::IBackend> shared = [] {
    auto backend = backend::create_default_backend();
    if (!backend.ok()) {
      std::printf("       no backend: %s\n", backend.status().message().c_str());
      return std::unique_ptr<backend::IBackend>{};
    }
    auto owned = backend.release();
    if (const Status status = owned->init(0); !status.ok()) {
      std::printf("       backend declined to come up: %s\n",
                  status.message().c_str());
      return std::unique_ptr<backend::IBackend>{};
    }
    return owned;
  }();
  return shared.get();
}

}  // namespace

LSE_TEST(a_dispatch_record_round_trips_through_the_writer) {
  TraceData data;
  data.groups.push_back(fused_group());
  DispatchRecord rec = dispatch_at(1'000'000, 1'009'740);
  rec.window = backend::WorkRange{512, 256};
  rec.sequence = 41;
  data.dispatches.push_back(rec);

  ExportOptions options;
  options.backend_name = "hrx";
  options.device_clock = gfx1151_clock();
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;

  const DecodedTrace trace = decode(*bytes);
  const DecodedTrack* host = trace.track("lse host");
  LSE_EXPECT(host != nullptr);
  if (host == nullptr) return;

  const DecodedEvent* slice = trace.named_on(host->uuid, fused_group().entry_name);
  LSE_EXPECT(slice != nullptr);
  if (slice == nullptr) return;

  // The slice is named from the group table, which is the correlation doing its
  // job: the record itself carries no name.
  LSE_EXPECT_EQ(slice->type, pftrace::kSliceBegin);
  LSE_EXPECT_EQ(slice->ts, rec.host_begin_ns);
  LSE_EXPECT_EQ(slice->clock, pftrace::kClockMonotonic);
  // Identities travel as fixed-width hex, because trace_processor surfaces an
  // annotation's uint64 in a signed column and kCacheKey is above 2^63: as a
  // number it would read negative in the viewer and positive in the engine.
  LSE_EXPECT(slice->strings.at(std::string(annotation::kGroupId)) ==
             "0x51c0ffee51c0ffee");
  LSE_EXPECT(slice->strings.at(std::string(annotation::kSignature)) ==
             "0x680de842f72758a9");
  LSE_EXPECT(slice->uints.count(std::string(annotation::kGroupId)) == 0);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kSequence)), 41ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kWindowBegin)), 512ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kWindowCount)), 256ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kNodeCount)), 3ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kWorkgroups)), 8ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kWorkgroupSize)), 256ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kElements)), 1024ull);
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kDeviceClockHz)), 99'810'000ull);
  LSE_EXPECT(slice->strings.at(std::string(annotation::kClockDomain)) == "host-steady");
  LSE_EXPECT(slice->strings.at(std::string(annotation::kDevice)) == "hrx:0");

  // The end of the slice carries the host end, so the duration survives.
  const auto on_host = trace.on(host->uuid);
  const auto last_end = std::find_if(on_host.rbegin(), on_host.rend(),
                                     [](const DecodedEvent* e) {
                                       return e->type == pftrace::kSliceEnd;
                                     });
  LSE_EXPECT(last_end != on_host.rend());
  if (last_end != on_host.rend()) LSE_EXPECT_EQ((*last_end)->ts, rec.host_end_ns);

  // Every TrackEvent packet carries a sequence id: without one the event is
  // silently dropped, which is the failure this assertion exists to catch.
  LSE_EXPECT_EQ(trace.sequence_ids.size(), trace.packets);
  for (const std::uint32_t id : trace.sequence_ids) {
    LSE_EXPECT_EQ(id, options.sequence_id);
  }

  // The group table travels once, with the node SET a dispatch cannot carry.
  const DecodedTrack* kernels = trace.track("lse kernels");
  LSE_EXPECT(kernels != nullptr);
  const auto on_kernels = kernels == nullptr ? std::vector<const DecodedEvent*>{}
                                             : trace.on(kernels->uuid);
  LSE_EXPECT_EQ(on_kernels.size(), 1u);
  if (on_kernels.empty()) return;
  LSE_EXPECT_EQ(on_kernels[0]->type, pftrace::kInstant);
  LSE_EXPECT(on_kernels[0]->strings.at("lse.node.1") == "silu[1,1024] silu");
  LSE_EXPECT(on_kernels[0]->strings.at(std::string(annotation::kSourceHash)) ==
             "0xabcdef0123456789");
}

LSE_TEST(nested_spans_two_lanes_a_flow_and_a_counter_are_structurally_valid) {
  TraceData data;
  data.groups.push_back(fused_group());

  SpanRecord token;
  token.kind = SpanKind::kToken;
  token.name = "token 0";
  token.begin_ns = 1'000'000;
  token.end_ns = 2'000'000;
  data.spans.push_back(token);

  SpanRecord phase;
  phase.kind = SpanKind::kPhase;
  phase.name = "phase decode";
  phase.begin_ns = 1'010'000;
  phase.end_ns = 1'900'000;
  phase.depth = 1;
  data.spans.push_back(phase);

  SpanRecord wait;
  wait.kind = SpanKind::kHostWait;
  wait.name = "host-wait";
  wait.begin_ns = 1'500'000;
  wait.end_ns = 1'580'000;
  wait.depth = 2;
  data.spans.push_back(wait);

  // A dispatch with a real device span, so the device lane has something on it.
  DispatchRecord rec = dispatch_at(1'020'000, 1'029'740);
  rec.device.clock = gfx1151_clock();
  rec.device.begin_tick = 1'000'000'000;
  rec.device.end_tick = 1'000'007'980;  // 7980 ticks = 79.95 us
  data.dispatches.push_back(rec);

  CounterSet experts;
  experts.kind = CounterKind::kRouterExperts;
  experts.instance = 3;
  experts.bins = {12, 37, 5, 9};
  experts.total = 63;
  experts.host_ns = 1'900'000;
  data.counters.push_back(experts);

  ExportOptions options;
  options.backend_name = "hrx";
  options.device_clock = gfx1151_clock();
  ClockAnchor anchor;
  anchor.host_ns = 1'000'000;
  anchor.device = backend::DeviceTimestamp{999'900'000, gfx1151_clock()};
  anchor.uncertainty_ns = 120;
  options.anchor = anchor;

  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  // Kept on disk so the full shape — nesting, two lanes, a flow and a counter —
  // can be opened in a viewer and queried with trace_processor, not only
  // asserted here.
  if (write_pftrace("build-tr/lse_shape.pftrace", data, options).ok()) {
    std::printf("       wrote build-tr/lse_shape.pftrace\n");
  }
  const DecodedTrace trace = decode(*bytes);

  // Two lanes, both parented to the process track, plus the kernels lane.
  const DecodedTrack* host = trace.track("lse host");
  const DecodedTrack* device = trace.track("hrx:0 stream 0");
  const DecodedTrack* root = trace.track("lse");
  LSE_EXPECT(host != nullptr);
  LSE_EXPECT(device != nullptr);
  LSE_EXPECT(root != nullptr);
  if (host == nullptr || device == nullptr || root == nullptr) return;
  LSE_EXPECT_EQ(host->parent, root->uuid);
  LSE_EXPECT_EQ(device->parent, root->uuid);

  // One descriptor per uuid, ever: rocprofv3 emits two for one uuid and
  // trace_processor calls that a producer bug.
  std::vector<std::uint64_t> uuids;
  for (const DecodedTrack& t : trace.tracks) uuids.push_back(t.uuid);
  std::sort(uuids.begin(), uuids.end());
  LSE_EXPECT(std::adjacent_find(uuids.begin(), uuids.end()) == uuids.end());

  // Nesting is begin/begin/.../end/end on ONE track uuid. Walking the host
  // lane's events must never close a slice that was not the innermost open one,
  // and depth must reach 3 (token > phase > dispatch or wait).
  int depth = 0;
  int deepest = 0;
  for (const DecodedEvent* e : trace.on(host->uuid)) {
    if (e->type == pftrace::kSliceBegin) {
      ++depth;
      deepest = std::max(deepest, depth);
    } else if (e->type == pftrace::kSliceEnd) {
      --depth;
      LSE_EXPECT(depth >= 0);
    }
  }
  LSE_EXPECT_EQ(depth, 0);
  LSE_EXPECT_EQ(deepest, 3);

  // The device lane is on its own clock, correlated by a snapshot. Without the
  // snapshot the device timestamps would be uninterpretable.
  const auto on_device = trace.on(device->uuid);
  LSE_EXPECT_EQ(on_device.size(), 2u);
  if (on_device.size() == 2) {
    // Placed on the host clock against the anchor, because no custom clock id
    // survives trace_processor: >=128 invalidates the snapshot and 64..127 is
    // read as an incremental delta. The DURATION is still the device's own —
    // 7980 ticks at 99.81 MHz is 79.95 us — and only the POSITION is anchored.
    LSE_EXPECT_EQ(on_device[0]->clock, pftrace::kClockMonotonic);
    LSE_EXPECT_NEAR(static_cast<double>(on_device[1]->ts - on_device[0]->ts),
                    79'951.9, 2.0);
    // 100,000 ticks after the anchor's tick, so ~1.0019 ms after its host read.
    LSE_EXPECT_NEAR(static_cast<double>(on_device[0]->ts - anchor.host_ns),
                    1'001'902.6, 2.0);
    LSE_EXPECT_EQ(
        on_device[0]->uints.at(std::string(annotation::kAnchorUncertainty)),
        anchor.uncertainty_ns);
    LSE_EXPECT(on_device[0]->strings.at(std::string(annotation::kClockDomain)) ==
               "device-agent");
  }
  // steady_clock is MONOTONIC and a Perfetto trace's default clock is BOOTTIME.
  // Without a measured pair relating them, trace_processor drops every packet
  // with clock_sync_failure_unknown_source_clock. This is that pair.
  bool has_monotonic = false;
  bool has_boottime = false;
  for (const auto& [id, value] : trace.clock_snapshot) {
    if (id == pftrace::kClockMonotonic) has_monotonic = value != 0;
    if (id == pftrace::kClockBoottime) has_boottime = value != 0;
  }
  LSE_EXPECT(has_monotonic);
  LSE_EXPECT(has_boottime);

  // The flow is the dispatch-to-group link: the same id on the group's instant
  // and on the dispatch's slice, which needs no naming convention.
  const DecodedEvent* kernel_instant = nullptr;
  const DecodedEvent* dispatch_slice = trace.named_on(host->uuid, fused_group().entry_name);
  for (const DecodedEvent& e : trace.events) {
    if (e.type == pftrace::kInstant) kernel_instant = &e;
  }
  LSE_EXPECT(kernel_instant != nullptr);
  LSE_EXPECT(dispatch_slice != nullptr);
  if (kernel_instant != nullptr && dispatch_slice != nullptr) {
    LSE_EXPECT_EQ(kernel_instant->flows.size(), 1u);
    LSE_EXPECT_EQ(dispatch_slice->flows.size(), 1u);
    LSE_EXPECT_EQ(kernel_instant->flows.at(0), dispatch_slice->flows.at(0));
    LSE_EXPECT_EQ(dispatch_slice->flows.at(0), kCacheKey);
  }

  // One counter track per bin, each with a CounterDescriptor, carrying the
  // distribution rather than a scalar derived from it.
  int counter_tracks = 0;
  for (const DecodedTrack& t : trace.tracks) {
    if (t.counter) ++counter_tracks;
  }
  LSE_EXPECT_EQ(counter_tracks, 4);
  const DecodedTrack* hot = trace.track("router experts [3] #1");
  LSE_EXPECT(hot != nullptr);
  if (hot != nullptr) {
    const auto samples = trace.on(hot->uuid);
    LSE_EXPECT_EQ(samples.size(), 1u);
    if (!samples.empty()) {
      LSE_EXPECT(samples[0]->has_counter);
      LSE_EXPECT_EQ(samples[0]->counter_value, 37ull);
      LSE_EXPECT_EQ(samples[0]->type, pftrace::kCounter);
    }
  }
}

LSE_TEST(an_unknown_device_duration_stays_unknown_through_export) {
  TraceData data;
  data.groups.push_back(fused_group());
  DispatchRecord rec = dispatch_at(5'000'000, 5'009'000);
  LSE_EXPECT(!rec.device.known());
  LSE_EXPECT(!rec.device.duration_ns().ok());
  data.dispatches.push_back(rec);

  ExportOptions options;
  options.backend_name = "hrx";
  // The clock is known — HRX publishes rate, domain and width — and the tick is
  // not. Publishing one without the other is exactly the state this asserts.
  options.device_clock = gfx1151_clock();
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);

  const DecodedTrack* host = trace.track("lse host");
  LSE_EXPECT(host != nullptr);
  if (host == nullptr) return;
  const DecodedEvent* slice = trace.named_on(host->uuid, fused_group().entry_name);
  LSE_EXPECT(slice != nullptr);
  if (slice == nullptr) return;
  const std::string key(annotation::kDeviceDuration);
  // A string, never a number: a query for a duration must not get a 0 that no
  // clock produced.
  LSE_EXPECT(slice->strings.count(key) == 1);
  LSE_EXPECT(slice->strings.at(key) == "unknown");
  LSE_EXPECT(slice->doubles.count(key) == 0);
  LSE_EXPECT(slice->uints.count(key) == 0);
  // The rate still travels, so a reader can see what a tick WOULD have been
  // counted on.
  LSE_EXPECT_EQ(slice->uints.at(std::string(annotation::kDeviceClockHz)),
                99'810'000ull);
  // And no device lane exists at all: an unplaceable span is not drawn.
  LSE_EXPECT(trace.track("hrx:0 stream 0") == nullptr);
}

LSE_TEST(a_device_span_refuses_to_be_placed_without_a_correlation_pair) {
  TraceData data;
  data.groups.push_back(fused_group());
  DispatchRecord rec = dispatch_at(5'000'000, 5'009'000);
  rec.device.clock = gfx1151_clock();
  rec.device.begin_tick = 1'000'000'000;
  rec.device.end_tick = 1'000'007'980;
  data.dispatches.push_back(rec);

  ExportOptions options;
  options.backend_name = "hrx";
  options.device_clock = gfx1151_clock();
  // No anchor: the duration is known, the position is not.
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);
  LSE_EXPECT(trace.track("hrx:0 stream 0") == nullptr);
  // The host clock pair is always there; nothing else is, because nothing
  // anchored the device counter.
  LSE_EXPECT_EQ(trace.clock_snapshot.size(), 2u);
  const DecodedTrack* host = trace.track("lse host");
  LSE_EXPECT(host != nullptr);
  if (host == nullptr) return;
  const DecodedEvent* slice = trace.named_on(host->uuid, fused_group().entry_name);
  LSE_EXPECT(slice != nullptr);
  if (slice == nullptr) return;
  // The duration still travels as a number, because two ticks on one clock are
  // subtractable without any anchor at all.
  LSE_EXPECT_NEAR(slice->doubles.at(std::string(annotation::kDeviceDuration)),
                  79'951.9, 1.0);
}

LSE_TEST(a_device_span_from_an_unrelated_clock_is_refused) {
  DeviceSpan span;
  span.clock = gfx1151_clock();
  span.begin_tick = 100;
  span.end_tick = 50;  // 64 valid bits: a decrease is the clock rolling back
  LSE_EXPECT(!span.duration_ns().ok());

  backend::DeviceClock other = gfx1151_clock();
  other.ordinal = 1;
  LSE_EXPECT(!backend::nanoseconds_between(
                  backend::DeviceTimestamp{10, gfx1151_clock()},
                  backend::DeviceTimestamp{20, other})
                  .ok());
}

LSE_TEST(the_collector_normalizes_dispatches_to_groups_and_never_to_nodes) {
  Collector& collector = Collector::instance();
  collector.clear();
  collector.record_group(fused_group());

  // One group, many dispatches: sibling fusion and lds_fold make this the
  // ordinary case, not an edge case.
  for (int i = 0; i < 8; ++i) {
    DispatchRecord rec = dispatch_at(static_cast<std::uint64_t>(10'000 + i * 100),
                                     static_cast<std::uint64_t>(10'090 + i * 100));
    record_dispatch(rec);
  }
  const TraceData data = collector.snapshot();
  LSE_EXPECT_EQ(data.dispatches.size(), 8u);
  LSE_EXPECT_EQ(data.groups.size(), 1u);
  // The sequence is assigned by the collector, monotonic, and is what a vendor
  // profiler carries as an external correlation id.
  for (std::size_t i = 0; i < data.dispatches.size(); ++i) {
    LSE_EXPECT_EQ(data.dispatches[i].sequence, i);
  }
  const GroupRecord* group = find_group(data, GroupId{kCacheKey, kSignature});
  LSE_EXPECT(group != nullptr);
  if (group != nullptr) LSE_EXPECT_EQ(group->nodes.size(), 3u);
  collector.clear();
}

LSE_TEST(the_group_table_never_loses_a_source_hash_it_already_had) {
  Collector& collector = Collector::instance();
  collector.clear();
  collector.record_group(fused_group());
  // The memory-hit path cannot reach a source hash: jit_cache computes it inside
  // get_or_compile and try_get never sees it.
  GroupRecord without = fused_group();
  without.source_hash.reset();
  collector.record_group(without);
  const TraceData data = collector.snapshot();
  LSE_EXPECT_EQ(data.groups.size(), 1u);
  if (!data.groups.empty()) {
    LSE_EXPECT(data.groups[0].source_hash.has_value());
  }
  collector.clear();
}

LSE_TEST(the_ring_counts_what_it_overwrote) {
  Collector& collector = Collector::instance();
  collector.clear();
  const Collector::Capacity cap = collector.capacity();
  const std::size_t depth = cap.dispatches;
  for (std::size_t i = 0; i < depth + 64; ++i) {
    record_dispatch(dispatch_at(i, i + 1));
  }
  const TraceData data = collector.snapshot();
  LSE_EXPECT_EQ(data.dispatches.size(), depth);
  LSE_EXPECT_EQ(data.dispatches_dropped, 64ull);
  // Oldest first, and the oldest surviving record is the 64th.
  LSE_EXPECT_EQ(data.dispatches.front().sequence, 64ull);
  collector.clear();
}

LSE_TEST(scoped_spans_nest_and_carry_their_clock_domain) {
  Collector& collector = Collector::instance();
  collector.clear();
  {
    Span token(SpanKind::kToken, "token 0");
    {
      Span phase(SpanKind::kPhase, "phase decode");
      Span wait(SpanKind::kHostWait, "host-wait");
    }
  }
  const TraceData data = collector.snapshot();
  LSE_EXPECT_EQ(data.spans.size(), 3u);
  if (data.spans.size() != 3) return;
  // Sorted by (begin asc, end desc), which is the nesting order.
  LSE_EXPECT(data.spans[0].name == "token 0");
  LSE_EXPECT(data.spans[1].name == "phase decode");
  LSE_EXPECT(data.spans[2].name == "host-wait");
  LSE_EXPECT_EQ(data.spans[0].depth, 0u);
  LSE_EXPECT_EQ(data.spans[1].depth, 1u);
  LSE_EXPECT_EQ(data.spans[2].depth, 2u);
  for (const SpanRecord& span : data.spans) {
    LSE_EXPECT(span.clock == backend::ClockDomain::kHostSteady);
    LSE_EXPECT(span.end_ns >= span.begin_ns);
  }
  collector.clear();
}

LSE_TEST(routing_skew_is_computed_by_the_reader_and_refuses_without_data) {
  CounterSet empty;
  LSE_EXPECT(!routing_skew(empty).ok());
  CounterSet zeros;
  zeros.bins = {0, 0, 0, 0};
  LSE_EXPECT(!routing_skew(zeros).ok());

  CounterSet balanced;
  balanced.bins = {10, 10, 10, 10};
  auto flat = routing_skew(balanced);
  LSE_EXPECT_OK(flat.status());
  if (flat.ok()) LSE_EXPECT_NEAR(*flat, 1.0, 1e-9);

  CounterSet hot;
  hot.bins = {0, 40, 0, 0};
  auto skewed = routing_skew(hot);
  LSE_EXPECT_OK(skewed.status());
  if (skewed.ok()) LSE_EXPECT_NEAR(*skewed, 4.0, 1e-9);
}

LSE_TEST(the_always_on_path_costs_nanoseconds) {
  Collector& collector = Collector::instance();
  collector.clear();
  const DispatchRecord rec = dispatch_at(1, 2);
  // Warm the lane so the one allocation it makes is not in the measurement.
  record_dispatch(rec);

  constexpr int kReps = 200'000;
  double best = 1e30;
  for (int trial = 0; trial < 5; ++trial) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) record_dispatch(rec);
    const double ns = static_cast<double>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count()) /
                      kReps;
    best = std::min(best, ns);
  }
  std::printf("       record_dispatch: %.2f ns/record (best of 5 x %d)\n", best,
              kReps);
  // The engine's own measured launch cost is ~974 ns per dispatch (launch=0.011 s
  // over 11291 launches). A bound of 100 ns keeps this under 10% of that even on
  // a loaded machine, and the printed figure is the number that matters.
  LSE_EXPECT(best < 100.0);
  collector.clear();
}

LSE_TEST(tier_two_declines_cleanly_and_tier_one_does_not) {
  backend::IBackend* backend = live_backend();
  if (backend == nullptr) {
    std::printf("       no backend: skipped\n");
    return;
  }
  auto tracer = create_tracer(*backend);
  LSE_EXPECT(tracer != nullptr);
  if (tracer == nullptr) return;
  std::printf("       tracer for backend '%s': %s\n",
              std::string(backend->name()).c_str(),
              std::string(tracer->name()).c_str());

  // Tier 1 is served unconditionally: its host half needs no backend.
  LSE_EXPECT_OK(tracer->supports(Tier::kDispatch));

  // Tier 2 declines, and the decline names what is missing rather than saying
  // "unavailable".
  const Status deep = tracer->supports(Tier::kDeepCounters);
  LSE_EXPECT(!deep.ok());
  LSE_EXPECT_EQ(static_cast<int>(deep.code()),
                static_cast<int>(StatusCode::kUnimplemented));
  LSE_EXPECT(deep.message().size() > 32);
  std::printf("       tier 2: %s\n", deep.message().c_str());
  auto session = tracer->open_deep(DeepRequest{});
  LSE_EXPECT(!session.ok());

  // The device half of tier 1 declines too, for a reason that is one function
  // wide, and the record stays UNKNOWN rather than picking up a host number.
  DispatchRecord rec = dispatch_at(1'000, 2'000);
  const Status device = tracer->resolve_device_span(rec);
  LSE_EXPECT(!device.ok());
  LSE_EXPECT(!rec.device.known());
  LSE_EXPECT(!rec.device.duration_ns().ok());
  LSE_EXPECT_EQ(rec.device.begin_tick, 0ull);
  std::printf("       device span: %s\n", device.message().c_str());

  // The clock is a separate question and may well be answered.
  auto clock = tracer->device_clock();
  if (clock.ok()) {
    std::printf("       device clock: %s, %llu Hz, %u bits\n",
                std::string(backend::clock_domain_name(clock->domain)).c_str(),
                static_cast<unsigned long long>(clock->ticks_per_second),
                static_cast<unsigned>(clock->valid_bits));
    LSE_EXPECT(clock->known());
  } else {
    std::printf("       device clock: %s\n", clock.status().message().c_str());
  }
  auto anchor = tracer->clock_anchor();
  std::printf("       clock anchor: %s\n",
              anchor.ok() ? "available" : anchor.status().message().c_str());

  // Collector::resolve_device_spans reports the tracer's own decline once, not
  // once per record.
  Collector& collector = Collector::instance();
  collector.clear();
  record_dispatch(rec);
  std::size_t resolved = 0;
  const Status batch = collector.resolve_device_spans(*tracer, &resolved);
  LSE_EXPECT(!batch.ok());
  LSE_EXPECT_EQ(resolved, 0u);
  collector.clear();
}

LSE_TEST(real_dispatches_on_a_live_device_export_a_valid_trace) {
  backend::IBackend* backend = live_backend();
  if (backend == nullptr) {
    std::printf("       no backend: skipped\n");
    return;
  }
  const graph::IKernelCompiler* compiler = backend->compiler();
  if (compiler == nullptr || !compiler->available()) {
    std::printf("       no device compiler: skipped\n");
    return;
  }
  const std::string entry = "lse_trace_probe";
  const std::string source =
      "#include <hip/hip_runtime.h>\n"
      "struct LseConstants { unsigned int count; };\n"
      "extern \"C\" __global__ void " + entry +
      "(float* __restrict__ out, LseConstants kc) {\n"
      "  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (i < kc.count) out[i] = out[i] * 1.0009765625f + 1.0f;\n"
      "}\n";
  auto code = compiler->compile(source, backend->device_info().arch);
  if (!code.ok()) {
    std::printf("       compile declined: %s\n", code.status().message().c_str());
    return;
  }
  auto handle = backend->load_executable(entry, *code);
  LSE_EXPECT_OK(handle.status());
  if (!handle.ok()) return;
  constexpr std::size_t kElements = 4096;
  auto buffer = backend->allocate(kElements * sizeof(float),
                                 backend::MemoryClass::kDevice);
  LSE_EXPECT_OK(buffer.status());
  if (!buffer.ok()) return;
  backend::DeviceBuffer out = buffer.release();

  Collector& collector = Collector::instance();
  collector.clear();

  // The group table, once, exactly as the hook at the JIT's compile edge would
  // write it.
  const std::uint64_t signature = 0x7f3ac0de7f3ac0deull;
  GroupRecord group;
  group.id = GroupId{signature ^ 0x9e3779b97f4a7c15ull, signature};
  group.entry_name = entry;
  group.arch = backend->device_info().arch;
  group.nodes = {NodeRef{0, "custom", "[4096]", "trace.probe"}};
  collector.record_group(group);

  backend::LaunchDims dims;
  dims.workgroup_size[0] = 256;
  dims.workgroup_count[0] = static_cast<std::uint32_t>(kElements / 256);
  const backend::BufferRef bindings[] = {{&out, 0, out.size_bytes}};
  const auto count = static_cast<std::uint32_t>(kElements);
  std::array<std::byte, sizeof(std::uint32_t)> constants{};
  std::memcpy(constants.data(), &count, sizeof(count));
  backend::DispatchArgs args;
  args.bindings = bindings;
  args.constants = constants;

  constexpr int kTokens = 4;
  constexpr int kPerToken = 32;
  Status launch_status;
  {
    Span run(SpanKind::kPhase, "phase decode");
    for (int token = 0; token < kTokens && launch_status.ok(); ++token) {
      Span tok(SpanKind::kToken, "token " + std::to_string(token));
      for (int i = 0; i < kPerToken && launch_status.ok(); ++i) {
        // Exactly the shape the requested hook has: the clock is read twice
        // around launch and both readings are already needed for launch_ns, so
        // recording costs no extra clock read.
        const auto t0 = std::chrono::steady_clock::now();
        launch_status = backend->launch(*handle, dims, args);
        const auto t1 = std::chrono::steady_clock::now();
        DispatchRecord rec;
        rec.group = group.id;
        rec.host_begin_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t0.time_since_epoch())
                .count());
        rec.host_end_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                t1.time_since_epoch())
                .count());
        rec.stream = 0;
        rec.node_count = 1;
        rec.bindings = 1;
        rec.workgroup_count = dims.workgroup_count[0];
        rec.workgroup_size = dims.workgroup_size[0];
        rec.elements = count;
        record_dispatch(rec);
      }
      Span wait(SpanKind::kHostWait, "synchronize");
      launch_status = backend->synchronize();
    }
  }
  LSE_EXPECT_OK(launch_status);

  auto tracer = create_tracer(*backend);
  std::size_t resolved = 0;
  const Status device = collector.resolve_device_spans(*tracer, &resolved);
  std::printf("       device spans resolved: %zu of %llu (%s)\n", resolved,
              static_cast<unsigned long long>(kTokens * kPerToken),
              device.ok() ? "ok" : device.message().c_str());

  const TraceData data = collector.snapshot();
  LSE_EXPECT_EQ(data.dispatches.size(),
                static_cast<std::size_t>(kTokens * kPerToken));
  LSE_EXPECT_EQ(data.spans.size(), static_cast<std::size_t>(1 + 2 * kTokens));

  double host_total = 0.0;
  for (const DispatchRecord& rec : data.dispatches) {
    host_total += static_cast<double>(rec.host_duration_ns());
    // Whatever the backend answered, no host number reached the device field.
    LSE_EXPECT(!rec.device.known());
  }
  std::printf("       %zu real dispatches, host submit mean %.0f ns\n",
              data.dispatches.size(),
              host_total / static_cast<double>(data.dispatches.size()));

  ExportOptions options;
  options.backend_name = std::string(backend->name());
  if (auto clock = tracer->device_clock(); clock.ok()) {
    options.device_clock = *clock;
  }
  if (auto anchor = tracer->clock_anchor(); anchor.ok()) {
    options.anchor = *anchor;
  }
  const std::string path = "build-tr/lse_live.pftrace";
  const Status written = write_pftrace(path, data, options);
  if (!written.ok()) {
    // No build-tr in this checkout is not a failure of the writer.
    std::printf("       write declined: %s\n", written.message().c_str());
  } else {
    std::printf("       wrote %s\n", path.c_str());
  }

  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);
  LSE_EXPECT(trace.track("lse host") != nullptr);
  LSE_EXPECT(trace.track("lse kernels") != nullptr);
  // No device lane: HRX declines the tick, so nothing may be drawn there.
  LSE_EXPECT(trace.track(options.backend_name + ":0 stream 0") == nullptr);
  int begins = 0;
  int ends = 0;
  for (const DecodedEvent& e : trace.events) {
    if (e.type == pftrace::kSliceBegin) ++begins;
    if (e.type == pftrace::kSliceEnd) ++ends;
  }
  LSE_EXPECT_EQ(begins, ends);
  LSE_EXPECT_EQ(begins, 1 + 2 * kTokens + kTokens * kPerToken);
  backend->deallocate(out);
  collector.clear();
}

// A duration read the way a viewer reads one: pair each begin with the end that
// closes it on the same track. The writer's opinion of a duration is not in the
// bytes — only two timestamps are — so this is the only honest way to assert one.
std::map<std::string, std::uint64_t> slice_durations(const DecodedTrace& trace,
                                                     std::uint64_t uuid) {
  std::map<std::string, std::uint64_t> out;
  std::vector<std::pair<std::string, std::uint64_t>> open;
  for (const DecodedEvent* e : trace.on(uuid)) {
    if (e->type == pftrace::kSliceBegin) {
      open.emplace_back(e->name, e->ts);
    } else if (e->type == pftrace::kSliceEnd && !open.empty()) {
      out[open.back().first] = e->ts - open.back().second;
      open.pop_back();
    }
  }
  return out;
}

// A device duration is a number only when a device clock produced it, and the
// clock has to arrive with it. ClockDomain::kHostSystem is the case that matters:
// a vendor runtime publishing a host clock under a device-scoped name still
// yields a real measurement, but of submission order, and a trace that files it
// under lse.device_duration_ns without saying so is the substitution this module
// exists to prevent, one indirection removed.
LSE_TEST(a_device_duration_carries_the_clock_it_was_counted_on) {
  const auto host_masquerading = [] {
    backend::DeviceClock clock;
    clock.domain = backend::ClockDomain::kHostSystem;
    clock.ticks_per_second = 1'000'000'000;
    clock.valid_bits = 64;
    return clock;
  }();

  TraceData data;
  data.groups.push_back(fused_group());
  DispatchRecord agent = dispatch_at(1'000'000, 1'000'160);
  agent.device.clock = gfx1151_clock();
  agent.device.begin_tick = 1'000'000;
  agent.device.end_tick = 1'008'000;
  data.dispatches.push_back(agent);
  DispatchRecord host = dispatch_at(2'000'000, 2'000'160);
  host.sequence = 1;
  host.device.clock = host_masquerading;
  host.device.begin_tick = 2'000'000;
  host.device.end_tick = 2'080'000;
  data.dispatches.push_back(host);
  DispatchRecord unknown = dispatch_at(3'000'000, 3'000'160);
  unknown.sequence = 2;
  data.dispatches.push_back(unknown);

  ExportOptions options;
  options.backend_name = "hrx";
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);
  const DecodedTrack* lane = trace.track("lse host");
  LSE_EXPECT(lane != nullptr);
  if (lane == nullptr) return;

  std::map<std::uint64_t, const DecodedEvent*> by_sequence;
  for (const DecodedEvent* e : trace.on(lane->uuid)) {
    if (e->type != pftrace::kSliceBegin) continue;
    const auto it = e->uints.find(std::string(annotation::kSequence));
    if (it != e->uints.end()) by_sequence[it->second] = e;
  }
  LSE_EXPECT_EQ(by_sequence.size(), std::size_t{3});

  const std::string domain_key{annotation::kDeviceClockDomain};
  const std::string duration_key{annotation::kDeviceDuration};

  // Read off the device's own counter: a number, labelled device-agent.
  const DecodedEvent* first = by_sequence[0];
  LSE_EXPECT(first != nullptr && first->doubles.count(duration_key) == 1);
  LSE_EXPECT(first != nullptr && first->strings.count(duration_key) == 0);
  if (first != nullptr) {
    LSE_EXPECT(first->strings.at(domain_key) == "device-agent");
  }

  // Read off a host clock wearing a device-scoped name: still a number, and the
  // trace says which clock it came from rather than leaving the reader to assume.
  const DecodedEvent* second = by_sequence[1];
  LSE_EXPECT(second != nullptr && second->doubles.count(duration_key) == 1);
  if (second != nullptr) {
    LSE_EXPECT_EQ(second->doubles.at(duration_key), 80'000.0);
    LSE_EXPECT(second->strings.at(domain_key) == "host-system");
    // The submission span's own domain is a separate key and keeps its own value.
    LSE_EXPECT(second->strings.at(std::string(annotation::kClockDomain)) ==
               "host-steady");
  }

  // Unreadable: the string "unknown", and no domain key at all, because there is
  // no clock to name.
  const DecodedEvent* third = by_sequence[2];
  LSE_EXPECT(third != nullptr && third->strings.count(duration_key) == 1);
  if (third != nullptr) {
    LSE_EXPECT(third->strings.at(duration_key) == "unknown");
    LSE_EXPECT_EQ(third->strings.count(domain_key), std::size_t{0});
    LSE_EXPECT_EQ(third->doubles.count(duration_key), std::size_t{0});
  }
}

// Every host timestamp here is emitted on CLOCK_MONOTONIC. A record that says it
// came from another host clock is refused, not quietly placed on that timeline —
// CLOCK_REALTIME differs from MONOTONIC by the epoch, so the only symptom of
// placing one would be a trace that looks wrong.
LSE_TEST(a_host_record_from_another_clock_is_refused_not_placed) {
  ExportOptions options;

  TraceData span_data;
  SpanRecord span;
  span.kind = SpanKind::kToken;
  span.name = "token 0";
  span.begin_ns = 1'000'000;
  span.end_ns = 2'000'000;
  span.clock = backend::ClockDomain::kHostSystem;
  span_data.spans.push_back(span);
  auto span_bytes = encode_pftrace(span_data, options);
  LSE_EXPECT(!span_bytes.ok());

  TraceData dispatch_data;
  DispatchRecord rec = dispatch_at(1'000'000, 1'000'160);
  rec.host_clock = backend::ClockDomain::kUnknown;
  dispatch_data.dispatches.push_back(rec);
  auto dispatch_bytes = encode_pftrace(dispatch_data, options);
  LSE_EXPECT(!dispatch_bytes.ok());

  TraceData counter_data;
  CounterSet counters;
  counters.bins = {1, 2, 3};
  counters.total = 6;
  counters.host_ns = 1'000'000;
  counters.clock = backend::ClockDomain::kHostSystem;
  counter_data.counters.push_back(counters);
  auto counter_bytes = encode_pftrace(counter_data, options);
  LSE_EXPECT(!counter_bytes.ok());

  // The same records on steady_clock, which is what MONOTONIC is, all go.
  span_data.spans[0].clock = backend::ClockDomain::kHostSteady;
  dispatch_data.dispatches[0].host_clock = backend::ClockDomain::kHostSteady;
  counter_data.counters[0].clock = backend::ClockDomain::kHostSteady;
  LSE_EXPECT_OK(encode_pftrace(span_data, options).status());
  LSE_EXPECT_OK(encode_pftrace(dispatch_data, options).status());
  LSE_EXPECT_OK(encode_pftrace(counter_data, options).status());
}

// A span whose end precedes its begin is not a duration. Emitted as-is it
// becomes |end - begin| at a position it never occupied, and it also truncates
// the span that really did enclose it, so a bad record corrupts a good one.
LSE_TEST(a_reversed_span_claims_no_duration_and_does_not_truncate_its_parent) {
  TraceData data;
  SpanRecord token;
  token.kind = SpanKind::kToken;
  token.name = "token 0";
  token.begin_ns = 1'000'000;
  token.end_ns = 2'000'000;
  data.spans.push_back(token);

  SpanRecord reversed;
  reversed.kind = SpanKind::kPhase;
  reversed.name = "reversed";
  reversed.begin_ns = 1'500'000;
  reversed.end_ns = 1'400'000;  // before its own begin
  reversed.depth = 1;
  data.spans.push_back(reversed);

  ExportOptions options;
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);
  const DecodedTrack* lane = trace.track("lse host");
  LSE_EXPECT(lane != nullptr);
  if (lane == nullptr) return;

  const std::map<std::string, std::uint64_t> durations =
      slice_durations(trace, lane->uuid);
  // The reversed span claims nothing rather than a fabricated 100 us.
  LSE_EXPECT_EQ(durations.at("reversed"), std::uint64_t{0});
  // And the span that encloses it keeps the duration it was recorded with.
  LSE_EXPECT_EQ(durations.at("token 0"), std::uint64_t{1'000'000});

  // No slice may end before it began: that is an event out of timestamp order in
  // the packet stream, which is what a viewer refuses to reconstruct.
  std::uint64_t previous = 0;
  for (const DecodedEvent& e : trace.events) {
    LSE_EXPECT(e.ts >= previous);
    previous = e.ts;
  }
}

// A snapshot can hold a compiled kernel and no dispatch yet. Timestamp 0 on
// MONOTONIC is boot, so placing the group table there would put it however long
// the box has been up before the rest of the trace.
LSE_TEST(a_group_table_with_no_slices_is_not_placed_at_boot) {
  TraceData data;
  data.groups.push_back(fused_group());
  data.host_clocks = read_host_clocks();
  LSE_EXPECT(data.host_clocks.known());

  ExportOptions options;
  auto bytes = encode_pftrace(data, options);
  LSE_EXPECT_OK(bytes.status());
  if (!bytes.ok()) return;
  const DecodedTrace trace = decode(*bytes);
  const DecodedTrack* kernels = trace.track("lse kernels");
  LSE_EXPECT(kernels != nullptr);
  if (kernels == nullptr) return;
  const std::vector<const DecodedEvent*> events = trace.on(kernels->uuid);
  LSE_EXPECT_EQ(events.size(), std::size_t{1});
  if (events.empty()) return;
  LSE_EXPECT_EQ(events[0]->type, pftrace::kInstant);
  LSE_EXPECT_EQ(events[0]->ts, data.host_clocks.monotonic_ns);
  LSE_EXPECT(events[0]->ts > 0);
}

LSE_TEST_MAIN()
