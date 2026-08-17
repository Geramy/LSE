#include "lse/trace/export.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <map>
#include <utility>

namespace lse::trace {

namespace {

using Bytes = std::vector<std::byte>;

void varint(Bytes& out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value)));
}

void tag(Bytes& out, std::uint32_t field, std::uint32_t wire) {
  varint(out, (static_cast<std::uint64_t>(field) << 3) | wire);
}

void field_u64(Bytes& out, std::uint32_t field, std::uint64_t value) {
  tag(out, field, 0);
  varint(out, value);
}

void field_f64(Bytes& out, std::uint32_t field, double value) {
  tag(out, field, 1);
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xff));
  }
}

void field_bytes(Bytes& out, std::uint32_t field, const Bytes& value) {
  tag(out, field, 2);
  varint(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

void field_str(Bytes& out, std::uint32_t field, std::string_view value) {
  tag(out, field, 2);
  varint(out, value.size());
  for (const char c : value) {
    out.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
  }
}

Bytes annotation_uint(std::string_view name, std::uint64_t value) {
  Bytes out;
  field_str(out, pftrace::kAnnotationName, name);
  field_u64(out, pftrace::kAnnotationUint, value);
  return out;
}

Bytes annotation_str(std::string_view name, std::string_view value) {
  Bytes out;
  field_str(out, pftrace::kAnnotationName, name);
  field_str(out, pftrace::kAnnotationString, value);
  return out;
}

// A 64-bit identity, as a fixed-width hex string.
//
// Not a uint: trace_processor stores DebugAnnotation.uint_value in a signed
// int64 column, so any id above 2^63 comes back out negative. The bits survive,
// but a correlation key that reads as -2229922388438369077 in the viewer and
// 16216821685271182539 in the engine's own log is a key someone will eventually
// mis-join. Hex reads identically in both, and a join is an equality test, which
// a string serves exactly as well as a number.
Bytes annotation_id(std::string_view name, std::uint64_t value) {
  char text[19] = {};
  std::snprintf(text, sizeof(text), "0x%016llx",
                static_cast<unsigned long long>(value));
  return annotation_str(name, text);
}

Bytes annotation_double(std::string_view name, double value) {
  Bytes out;
  field_str(out, pftrace::kAnnotationName, name);
  field_f64(out, pftrace::kAnnotationDouble, value);
  return out;
}

// A TrackEvent with a name, optional flow id and annotations.
Bytes track_event(std::uint32_t type, std::uint64_t uuid, std::string_view name,
                  std::uint64_t flow, const std::vector<Bytes>& annotations) {
  Bytes out;
  field_u64(out, pftrace::kEventType, type);
  field_u64(out, pftrace::kEventTrackUuid, uuid);
  if (!name.empty()) field_str(out, pftrace::kEventName, name);
  if (flow != 0) field_u64(out, pftrace::kEventFlowIds, flow);
  for (const Bytes& annotation : annotations) {
    field_bytes(out, pftrace::kEventDebugAnnotations, annotation);
  }
  return out;
}

Bytes slice_end_event(std::uint64_t uuid) {
  Bytes out;
  field_u64(out, pftrace::kEventType, pftrace::kSliceEnd);
  field_u64(out, pftrace::kEventTrackUuid, uuid);
  return out;
}

Bytes counter_event(std::uint64_t uuid, std::uint64_t value,
                    std::uint64_t total, backend::ClockDomain clock) {
  Bytes out;
  field_u64(out, pftrace::kEventType, pftrace::kCounter);
  field_u64(out, pftrace::kEventTrackUuid, uuid);
  field_u64(out, pftrace::kEventCounterValue, value);
  field_bytes(out, pftrace::kEventDebugAnnotations,
              annotation_uint(annotation::kCounterTotal, total));
  // Every timestamp in this format says which clock it came from, and a counter
  // sample is timestamped like anything else.
  field_bytes(out, pftrace::kEventDebugAnnotations,
              annotation_str(annotation::kClockDomain,
                             backend::clock_domain_name(clock)));
  return out;
}

Bytes track_descriptor(std::uint64_t uuid, std::string_view name,
                       std::uint64_t parent, bool counter) {
  Bytes out;
  field_u64(out, pftrace::kTrackUuid, uuid);
  field_str(out, pftrace::kTrackName, name);
  if (parent != 0) field_u64(out, pftrace::kTrackParentUuid, parent);
  // An empty CounterDescriptor is enough to make the track a counter track.
  if (counter) field_bytes(out, pftrace::kTrackCounter, Bytes{});
  return out;
}

// One emitted TracePacket carrying a TrackEvent. Every timestamp in this
// exporter is CLOCK_MONOTONIC — including a device span's, which is placed there
// against the anchor — so no event needs to name a clock of its own.
struct Event {
  std::uint64_t ts = 0;
  Bytes payload;
};

// A slice to place on one track, before it is turned into a begin/end pair.
struct Item {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  Bytes begin_payload;
  std::uint64_t uuid = 0;
};

// Turns one track's slices into a properly nested begin/end sequence.
//
// Sorting by (begin asc, end desc) is a depth-first order over a properly
// nested set, which is what scoped spans produce. A caller that hands in two
// slices which merely overlap has produced something a timeline cannot show, so
// the enclosing one is closed early rather than emitted crossing: every slice's
// begin stays honest and only the outer duration shortens.
void nest(std::vector<Item>& items, std::vector<Event>& out) {
  if (items.empty()) return;
  // A slice whose end precedes its begin is not a duration. Emitted as-is it
  // becomes |end - begin| at the wrong position — a number no clock produced —
  // and, because it then sorts as if it enclosed nothing, it also truncates
  // whatever slice really did enclose it. So it is collapsed to zero length
  // here: the record still appears where it was recorded and claims no
  // duration. The record structs guard their own duration_ns() the same way,
  // and this is the same rule applied at the only point every slice passes.
  for (Item& item : items) {
    if (item.end < item.begin) item.end = item.begin;
  }
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) {
                     if (a.begin != b.begin) return a.begin < b.begin;
                     return a.end > b.end;
                   });
  std::vector<std::pair<std::uint64_t, std::uint64_t>> open;  // (end, uuid)
  for (Item& item : items) {
    while (!open.empty() && open.back().first <= item.begin) {
      out.push_back(
          Event{open.back().first, slice_end_event(open.back().second)});
      open.pop_back();
    }
    while (!open.empty() && open.back().first < item.end) {
      out.push_back(Event{item.begin, slice_end_event(open.back().second)});
      open.pop_back();
    }
    out.push_back(Event{item.begin, std::move(item.begin_payload)});
    open.emplace_back(item.end, item.uuid);
  }
  while (!open.empty()) {
    out.push_back(Event{open.back().first, slice_end_event(open.back().second)});
    open.pop_back();
  }
}

std::uint64_t fnv1a(std::string_view text) noexcept {
  std::uint64_t hash = 0xcbf29ce484222325ull;
  for (const char c : text) {
    hash ^= static_cast<std::uint8_t>(c);
    hash *= 0x100000001b3ull;
  }
  return hash;
}

// Where a device tick falls on CLOCK_MONOTONIC, given a measured anchor.
//
// The scale is ticks over rate in double, for the same reason
// nanoseconds_between does it: truncating first loses a microsecond-scale span on
// a ~100 MHz counter. Refuses rather than clamps when the tick lands before the
// host clock's origin — a negative timestamp is not a placement.
std::optional<std::uint64_t> device_tick_to_host_ns(std::uint64_t tick,
                                                    const ClockAnchor& anchor) {
  const double ticks = static_cast<double>(tick) -
                       static_cast<double>(anchor.device.tick);
  const double offset_ns =
      ticks * 1e9 / static_cast<double>(anchor.device.clock.ticks_per_second);
  const double placed = static_cast<double>(anchor.host_ns) + offset_ns;
  if (placed < 0.0) return std::nullopt;
  return static_cast<std::uint64_t>(placed);
}

std::string group_slice_name(const TraceData& data, GroupId group) {
  if (const GroupRecord* record = find_group(data, group);
      record != nullptr && !record->entry_name.empty()) {
    return record->entry_name;
  }
  // The table did not hold this group. Still nameable, and the name says which
  // id to go looking for rather than reading as an anonymous slice.
  return "group " + std::to_string(group.cache_key);
}

}  // namespace

std::uint64_t track_uuid(std::string_view label) noexcept {
  // Never 0: uuid 0 is "no track".
  return fnv1a(label) | 1ull;
}

std::string host_lane_label(const ExportOptions& options, std::uint32_t thread) {
  if (thread == 0) return options.process_name + " host";
  return options.process_name + " host " + std::to_string(thread);
}

std::string device_lane_label(const ExportOptions& options,
                              std::uint8_t device_ordinal,
                              std::uint32_t stream) {
  return options.backend_name + ":" + std::to_string(device_ordinal) +
         " stream " + std::to_string(stream);
}

std::string kernel_lane_label(const ExportOptions& options) {
  return options.process_name + " kernels";
}

Result<std::vector<std::byte>> encode_pftrace(const TraceData& data,
                                              const ExportOptions& options) {
  if (options.sequence_id == 0) {
    return LSE_ERROR(kInvalidArgument,
                     "trusted_packet_sequence_id 0 is not a sequence: a "
                     "TrackEvent packet without one is silently dropped");
  }

  // Every host timestamp in this trace is emitted on CLOCK_MONOTONIC, which is
  // steady_clock and nothing else. A record that says it was counted on another
  // host clock cannot go on that timeline: CLOCK_REALTIME differs from MONOTONIC
  // by the epoch, so it would land decades away and the only symptom would be a
  // timeline that looks wrong. The domain is on the record for exactly this
  // reason, so it is checked rather than trusted to be the default — placing an
  // uncorrelated clock is the comparison this module refuses to make.
  const auto refuse_domain = [](std::string_view what, backend::ClockDomain d) {
    return LSE_ERROR(kInvalidArgument, what, " is counted on ",
                     std::string(backend::clock_domain_name(d)),
                     ", and every host timestamp in a pftrace here goes on "
                     "CLOCK_MONOTONIC. Correlating that clock to MONOTONIC is a "
                     "measurement nobody has taken, so it is refused rather "
                     "than placed");
  };
  for (const SpanRecord& span : data.spans) {
    if (span.clock != backend::ClockDomain::kHostSteady) {
      return refuse_domain("span '" + span.name + "'", span.clock);
    }
  }
  for (const DispatchRecord& rec : data.dispatches) {
    if (rec.host_clock != backend::ClockDomain::kHostSteady) {
      return refuse_domain(
          "the host span of dispatch " + std::to_string(rec.sequence),
          rec.host_clock);
    }
  }
  for (const CounterSet& counters : data.counters) {
    if (counters.clock != backend::ClockDomain::kHostSteady) {
      return refuse_domain(
          std::string(to_string(counters.kind)) + " counters", counters.clock);
    }
  }

  const std::uint64_t root = track_uuid(options.process_name);
  // uuid -> descriptor, so a uuid is described exactly once. rocprofv3 emits two
  // descriptors for one uuid and trace_processor calls that a producer bug; it
  // is one here by construction.
  std::map<std::uint64_t, Bytes> descriptors;
  descriptors.emplace(root, track_descriptor(root, options.process_name, 0, false));

  const auto lane = [&](const std::string& label, bool counter) {
    const std::uint64_t uuid = track_uuid(label);
    descriptors.try_emplace(uuid, track_descriptor(uuid, label, root, counter));
    return uuid;
  };

  // Device ticks are absolute on a counter with its own origin, so they can be
  // placed on the host timeline only against a correlation pair. Without one the
  // duration still travels, as an annotation, but no device slice is drawn:
  // a device slice positioned by a host clock is exactly the substitution this
  // module exists to prevent.
  const bool can_place_device =
      options.anchor.has_value() && options.anchor->known() &&
      options.anchor->device.clock.known();

  std::map<std::uint64_t, std::vector<Item>> tracks;
  std::vector<Event> events;

  for (const SpanRecord& span : data.spans) {
    const std::uint64_t uuid = lane(host_lane_label(options, span.thread), false);
    std::vector<Bytes> annotations;
    annotations.push_back(
        annotation_str(annotation::kClockDomain,
                       backend::clock_domain_name(span.clock)));
    if (span.group.valid()) {
      annotations.push_back(annotation_id(annotation::kGroupId, span.group.cache_key));
      annotations.push_back(annotation_id(annotation::kSignature, span.group.signature));
    }
    tracks[uuid].push_back(
        Item{span.begin_ns, span.end_ns,
             track_event(pftrace::kSliceBegin, uuid, span.name,
                         span.group.cache_key, annotations),
             uuid});
  }

  for (const DispatchRecord& rec : data.dispatches) {
    const std::string name = group_slice_name(data, rec.group);
    const backend::DeviceClock& clock =
        rec.device.known() ? rec.device.clock : options.device_clock;

    std::vector<Bytes> annotations;
    annotations.push_back(annotation_id(annotation::kGroupId, rec.group.cache_key));
    annotations.push_back(annotation_id(annotation::kSignature, rec.group.signature));
    annotations.push_back(annotation_uint(annotation::kSequence, rec.sequence));
    annotations.push_back(annotation_uint(annotation::kStream, rec.stream));
    annotations.push_back(annotation_str(
        annotation::kDevice,
        options.backend_name + ":" + std::to_string(rec.device_ordinal)));
    annotations.push_back(annotation_str(
        annotation::kClockDomain, backend::clock_domain_name(rec.host_clock)));
    annotations.push_back(
        annotation_uint(annotation::kDeviceClockHz, clock.ticks_per_second));
    if (auto device_ns = rec.device.duration_ns(); device_ns.ok()) {
      annotations.push_back(annotation_double(annotation::kDeviceDuration, *device_ns));
      // The domain travels with the number. kClockDomain above is the submission
      // span's, so without this key a duration counted on a vendor's
      // device-scoped HOST clock would sit under a device-duration key with
      // nothing in the trace saying so.
      annotations.push_back(
          annotation_str(annotation::kDeviceClockDomain,
                         backend::clock_domain_name(rec.device.clock.domain)));
    } else {
      // A string, not 0: a query asking for a number must not silently get one
      // that no clock produced.
      annotations.push_back(annotation_str(annotation::kDeviceDuration, "unknown"));
    }
    annotations.push_back(annotation_uint(annotation::kNodeCount, rec.node_count));
    annotations.push_back(annotation_uint(annotation::kBindings, rec.bindings));
    annotations.push_back(annotation_uint(annotation::kWorkgroups, rec.workgroup_count));
    annotations.push_back(annotation_uint(annotation::kWorkgroupSize, rec.workgroup_size));
    annotations.push_back(annotation_uint(annotation::kElements, rec.elements));
    annotations.push_back(annotation_uint(annotation::kWindowBegin, rec.window.begin));
    annotations.push_back(annotation_uint(annotation::kWindowCount, rec.window.count));
    annotations.push_back(annotation_uint(annotation::kIsPhase, rec.is_phase ? 1 : 0));

    // The submission, on the host lane where it belongs: this span is the host
    // recording a packet into a command buffer, not the device running it.
    const std::uint64_t host = lane(host_lane_label(options, rec.thread), false);
    tracks[host].push_back(
        Item{rec.host_begin_ns, rec.host_end_ns,
             track_event(pftrace::kSliceBegin, host, name, rec.group.cache_key,
                         annotations),
             host});

    if (!rec.device.known() || !can_place_device) continue;
    // Same domain, same device, same rate, or the two counters are unrelated and
    // the anchor says nothing about this tick.
    if (!rec.device.clock.same_source_as(options.anchor->device.clock)) continue;
    const auto begin = device_tick_to_host_ns(rec.device.begin_tick, *options.anchor);
    const auto end = device_tick_to_host_ns(rec.device.end_tick, *options.anchor);
    if (!begin.has_value() || !end.has_value()) continue;
    const std::uint64_t device =
        lane(device_lane_label(options, rec.device_ordinal, rec.stream), false);
    std::vector<Bytes> device_annotations;
    device_annotations.push_back(annotation_id(annotation::kGroupId, rec.group.cache_key));
    device_annotations.push_back(annotation_uint(annotation::kSequence, rec.sequence));
    device_annotations.push_back(annotation_str(
        annotation::kClockDomain,
        backend::clock_domain_name(rec.device.clock.domain)));
    device_annotations.push_back(annotation_uint(annotation::kAnchorUncertainty,
                                                 options.anchor->uncertainty_ns));
    tracks[device].push_back(
        Item{*begin, *end,
             track_event(pftrace::kSliceBegin, device, name,
                         rec.group.cache_key, device_annotations),
             device});
  }

  // The group table sits at the start of the timeline, one instant per emitted
  // kernel, carrying the node SET a dispatch cannot carry per launch.
  std::uint64_t first_ts = 0;
  for (const auto& [uuid, items] : tracks) {
    for (const Item& item : items) {
      if (first_ts == 0 || item.begin < first_ts) first_ts = item.begin;
    }
  }
  // A snapshot can hold a group table and no slices at all — the kernel was
  // compiled and nothing has been dispatched yet. Timestamp 0 on MONOTONIC is
  // boot, so the table would sit however many seconds the box has been up before
  // the rest of the trace and stretch every viewer's default zoom over that gap.
  // The snapshot's own clock reading is the closest honest position for it.
  if (first_ts == 0) first_ts = data.host_clocks.monotonic_ns;
  if (!data.groups.empty()) {
    const std::uint64_t kernels = lane(kernel_lane_label(options), false);
    for (const GroupRecord& group : data.groups) {
      std::vector<Bytes> annotations;
      annotations.push_back(annotation_id(annotation::kGroupId, group.id.cache_key));
      annotations.push_back(annotation_id(annotation::kSignature, group.id.signature));
      annotations.push_back(annotation_uint(annotation::kNodeCount, group.nodes.size()));
      annotations.push_back(annotation_uint(annotation::kIsPhase, group.is_phase ? 1 : 0));
      if (!group.arch.empty()) {
        annotations.push_back(annotation_str(annotation::kArch, group.arch));
      }
      if (group.source_hash.has_value()) {
        annotations.push_back(
            annotation_id(annotation::kSourceHash, *group.source_hash));
      } else {
        annotations.push_back(annotation_str(annotation::kSourceHash, "unknown"));
      }
      for (const NodeRef& node : group.nodes) {
        std::string text = node.kind + node.shape;
        if (!node.primitive.empty()) text += " " + node.primitive;
        annotations.push_back(annotation_str(
            std::string(annotation::kNodePrefix) + std::to_string(node.index),
            text));
      }
      events.push_back(Event{first_ts,
                             track_event(pftrace::kInstant, kernels,
                                         group.entry_name, group.id.cache_key,
                                         annotations)});
    }
  }

  for (const CounterSet& counters : data.counters) {
    for (std::size_t bin = 0; bin < counters.bins.size(); ++bin) {
      const std::string label = std::string(to_string(counters.kind)) + " [" +
                                std::to_string(counters.instance) + "] #" +
                                std::to_string(bin);
      const std::uint64_t uuid = lane(label, true);
      // The denominator travels with the sample: top-k routing puts one token in
      // k bins, so the bins do not sum to the tokens routed and a reader cannot
      // recover the total from them.
      events.push_back(Event{
          counters.host_ns,
          counter_event(uuid, counters.bins[bin], counters.total,
                        counters.clock)});
    }
  }

  for (auto& [uuid, items] : tracks) nest(items, events);

  // Timestamp order overall, with each track's own nesting order preserved: a
  // stable sort on the timestamp alone leaves events that share a timestamp in
  // the order they were produced, and within a track that is the nesting order.
  std::stable_sort(events.begin(), events.end(),
                   [](const Event& a, const Event& b) { return a.ts < b.ts; });

  Bytes out;
  const auto packet = [&](const Bytes& body) { field_bytes(out, pftrace::kTracePacket, body); };

  {
    // First packet on the sequence. The cleared-state flag matters the moment
    // anything interned is added: without it an interned name resolves to NULL
    // and trace_processor reports no error at all.
    Bytes body;
    field_u64(body, pftrace::kPacketSequenceId, options.sequence_id);
    field_u64(body, pftrace::kPacketSequenceFlags, pftrace::kSeqIncrementalStateCleared);
    field_bytes(body, pftrace::kPacketTrackDescriptor, descriptors.at(root));
    packet(body);
  }

  {
    // The load-bearing packet, and the reason a standalone trace opens at all.
    // Every timestamp here is CLOCK_MONOTONIC (steady_clock); a Perfetto trace's
    // default clock is BOOTTIME, and MONOTONIC is a *known* clock id but not a
    // convertible one — trace_processor needs a snapshot pairing the two, or it
    // drops every packet with "clock_sync_failure_unknown_source_clock". The
    // offset is the machine's accumulated suspend time, so it is measured, not
    // assumed.
    const HostClockPair host = data.host_clocks.known() ? data.host_clocks
                                                        : read_host_clocks();
    if (!host.known()) {
      return LSE_ERROR(kInternal,
                       "cannot read CLOCK_MONOTONIC and CLOCK_BOOTTIME: without "
                       "the offset between them these timestamps cannot be "
                       "placed on a trace timeline");
    }
    Bytes host_pair;
    Bytes monotonic;
    field_u64(monotonic, pftrace::kClockId, pftrace::kClockMonotonic);
    field_u64(monotonic, pftrace::kClockTimestamp, host.monotonic_ns);
    field_bytes(host_pair, pftrace::kClockSnapshotClock, monotonic);
    Bytes boottime;
    field_u64(boottime, pftrace::kClockId, pftrace::kClockBoottime);
    field_u64(boottime, pftrace::kClockTimestamp, host.boottime_ns);
    field_bytes(host_pair, pftrace::kClockSnapshotClock, boottime);

    Bytes body;
    field_u64(body, pftrace::kPacketTimestamp, host.monotonic_ns);
    field_u64(body, pftrace::kPacketClockId, pftrace::kClockMonotonic);
    field_u64(body, pftrace::kPacketSequenceId, options.sequence_id);
    field_bytes(body, pftrace::kPacketClockSnapshot, host_pair);
    packet(body);
  }

  for (const auto& [uuid, descriptor] : descriptors) {
    if (uuid == root) continue;
    Bytes body;
    field_u64(body, pftrace::kPacketSequenceId, options.sequence_id);
    field_bytes(body, pftrace::kPacketTrackDescriptor, descriptor);
    packet(body);
  }

  for (const Event& event : events) {
    Bytes body;
    field_u64(body, pftrace::kPacketTimestamp, event.ts);
    // steady_clock is CLOCK_MONOTONIC and a packet with no clock id is read as
    // BOOTTIME. The two differ by every suspend the machine has had, and nothing
    // warns, so this varint is not optional.
    field_u64(body, pftrace::kPacketClockId, pftrace::kClockMonotonic);
    field_u64(body, pftrace::kPacketSequenceId, options.sequence_id);
    field_bytes(body, pftrace::kPacketTrackEvent, event.payload);
    packet(body);
  }

  return out;
}

Status write_pftrace(const std::string& path, const TraceData& data,
                     const ExportOptions& options) {
  auto bytes = encode_pftrace(data, options);
  if (!bytes.ok()) return bytes.status();
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) return LSE_ERROR(kIoError, "cannot open '", path, "' for writing");
  file.write(reinterpret_cast<const char*>(bytes->data()),
             static_cast<std::streamsize>(bytes->size()));
  if (!file) return LSE_ERROR(kIoError, "short write to '", path, "'");
  return OkStatus();
}

}  // namespace lse::trace
