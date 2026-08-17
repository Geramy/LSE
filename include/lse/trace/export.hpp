// Export: TraceData -> a Perfetto protobuf trace (.pftrace).
//
// WHY THIS FORMAT AND NOT A BESPOKE ONE. rocprofv3 already writes pftrace
// natively (`-f {csv,json,pftrace,otf2,rocpd}`), so our spans and its kernel
// and HSA tracks land on ONE timeline under ONE clock — and clock correlation
// is precisely what a bespoke format gets wrong. Two traces concatenate
// byte-wise into a merged one. trace_processor gives SQL over it and
// ui.perfetto.dev opens it today, which no format invented here would.
//
// Every requirement maps onto a primitive that already exists:
//
//   nesting (token > phase > group > dispatch)  TrackEvent slices, begin/end
//                                              on ONE track_uuid
//   device and stream lanes                    TrackDescriptor per lane,
//                                              parented to the process track
//   dispatch <-> group correlation             flow ids, plus the group id as
//                                              a debug annotation for SQL
//   counters (router experts)                  counter tracks, one per bin
//   group signature, work window, clock domain debug annotations
//
// FlatBuffers is NOT used here. It stays what it already is in this repo: the
// control-plane plan codec and the persisted profile store. A trace is a
// timeline artifact for a viewer, and the viewer already exists.
//
// The writer emits raw varints. It vendors no protobuf library and no Perfetto
// SDK — 13 MB of tracing runtime to produce forty bytes of varints is a bad
// trade — so the field numbers below ARE the schema of record. They were read
// out of perfetto_trace.proto (perfetto 0.57.2) rather than remembered, and
// each one is exercised by tests/test_trace.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/trace/record.hpp"

namespace lse::trace {

// Perfetto wire constants. Append-only, like the schema they describe.
namespace pftrace {

// Trace
inline constexpr std::uint32_t kTracePacket = 1;

// TracePacket
inline constexpr std::uint32_t kPacketTimestamp = 8;
inline constexpr std::uint32_t kPacketSequenceId = 10;   // REQUIRED on every
                                                         // TrackEvent packet:
                                                         // without it the event
                                                         // is silently dropped
inline constexpr std::uint32_t kPacketTrackEvent = 11;
inline constexpr std::uint32_t kPacketSequenceFlags = 13;
inline constexpr std::uint32_t kPacketClockId = 58;      // absent means
                                                         // BOOTTIME, which is
                                                         // NOT steady_clock
inline constexpr std::uint32_t kPacketTrackDescriptor = 60;
inline constexpr std::uint32_t kPacketClockSnapshot = 6;

// TracePacket.sequence_flags bits
inline constexpr std::uint32_t kSeqIncrementalStateCleared = 1;

// TrackDescriptor
inline constexpr std::uint32_t kTrackUuid = 1;
inline constexpr std::uint32_t kTrackName = 2;
inline constexpr std::uint32_t kTrackParentUuid = 5;
inline constexpr std::uint32_t kTrackCounter = 8;

// TrackEvent
inline constexpr std::uint32_t kEventDebugAnnotations = 4;
inline constexpr std::uint32_t kEventType = 9;
inline constexpr std::uint32_t kEventTrackUuid = 11;
inline constexpr std::uint32_t kEventName = 23;
inline constexpr std::uint32_t kEventCounterValue = 30;
inline constexpr std::uint32_t kEventFlowIds = 47;

// TrackEvent.Type
inline constexpr std::uint32_t kSliceBegin = 1;
inline constexpr std::uint32_t kSliceEnd = 2;
inline constexpr std::uint32_t kInstant = 3;
inline constexpr std::uint32_t kCounter = 4;

// DebugAnnotation
inline constexpr std::uint32_t kAnnotationName = 10;
inline constexpr std::uint32_t kAnnotationUint = 3;
inline constexpr std::uint32_t kAnnotationDouble = 5;
inline constexpr std::uint32_t kAnnotationString = 6;

// ClockSnapshot / ClockSnapshot.Clock
inline constexpr std::uint32_t kClockSnapshotClock = 1;
inline constexpr std::uint32_t kClockId = 1;
inline constexpr std::uint32_t kClockTimestamp = 2;
inline constexpr std::uint32_t kClockUnitMultiplierNs = 3;

// Builtin clock ids. std::chrono::steady_clock is CLOCK_MONOTONIC, and a packet
// with no clock id is read as BOOTTIME — the two differ by every suspend the
// machine has had, and nothing warns. So kMonotonic goes on every host packet.
inline constexpr std::uint32_t kClockMonotonic = 3;
inline constexpr std::uint32_t kClockBoottime = 6;
// THERE IS NO CUSTOM CLOCK FOR THE DEVICE COUNTER, and the reason was measured
// rather than assumed. Three encodings were tried against trace_processor v56.1:
//
//   - a global custom id (>= 128, which the proto comments describe as global):
//     rejected outright, `invalid_clock_snapshots: 1`, and it takes the host
//     clocks down with it — every packet in the trace is then dropped.
//   - a sequence-scoped id (64..127): accepted, then treated as an INCREMENTAL
//     delta clock whatever is_incremental says, so absolute ticks accumulate —
//     an 80 us kernel came out as a 10.0 s one.
//   - a builtin id: impossible anyway. unit_multiplier_ns is an integer and the
//     gfx1151 agent tick is 10.019 ns.
//
// So device ticks are converted to CLOCK_MONOTONIC by the exporter, against the
// measured ClockAnchor, and emitted on kClockMonotonic like everything else. This
// is also what rocprofiler does with its own agent timestamps — it emits kernel
// dispatches directly on BOOTTIME — which is what keeps our device lane
// comparable with its kernel track in a merged trace. The device duration stays
// exact (a subtraction of two ticks on one clock); only the POSITION is anchored,
// and the anchor's measured uncertainty travels with it as an annotation.

}  // namespace pftrace

// The debug-annotation keys this exporter writes.
//
// They appear in trace_processor's `args` table as `debug.<key>` with dots
// flattened to underscores, so `lse.group_id` is queried as
// `debug.lse_group_id`. The `lse.` prefix is what keeps them from colliding with
// rocprofiler's own (begin_ns, corr_id, grid_size, ...) in a merged trace.
//
// TYPES ARE PART OF THE CONTRACT. The three 64-bit identities — group id,
// signature, source hash — travel as fixed-width hex strings ("0x%016llx"),
// because trace_processor stores an annotation's uint64 in a signed int64 column
// and anything above 2^63 comes back negative. Quantities (counts, geometry,
// stream, sequence) travel as numbers. A duration that no clock produced travels
// as the string "unknown" and never as 0.
namespace annotation {
inline constexpr std::string_view kGroupId = "lse.group_id";
inline constexpr std::string_view kSignature = "lse.signature";
inline constexpr std::string_view kSequence = "lse.sequence";
inline constexpr std::string_view kStream = "lse.stream";
inline constexpr std::string_view kDevice = "lse.device";
inline constexpr std::string_view kClockDomain = "lse.clock_domain";
inline constexpr std::string_view kDeviceDuration = "lse.device_duration_ns";
// Which clock the number in kDeviceDuration was counted on, and the reason it is
// a separate key from kClockDomain: kClockDomain on a dispatch's host slice
// describes the SUBMISSION span, so it says "host-steady" there whatever the
// device half came from. ClockDomain::kHostSystem exists because vendor runtimes
// really do publish a host clock under a device-scoped name, and a duration
// derived from one is a submission-order number wearing a kernel duration's key.
// It is still worth exporting — it is a real measurement of something — but it
// must arrive labelled, so this key travels with every kDeviceDuration that is a
// number, and is absent exactly when the duration is "unknown".
inline constexpr std::string_view kDeviceClockDomain = "lse.device_clock_domain";
inline constexpr std::string_view kDeviceClockHz = "lse.device_clock_hz";
// How wide the host interval was that the anchor's device read happened inside.
// Present on a device slice because that slice's POSITION is only as good as
// this number, while its duration does not depend on it at all.
inline constexpr std::string_view kAnchorUncertainty = "lse.anchor_uncertainty_ns";
inline constexpr std::string_view kNodeCount = "lse.node_count";
inline constexpr std::string_view kBindings = "lse.bindings";
inline constexpr std::string_view kWorkgroups = "lse.workgroups";
inline constexpr std::string_view kWorkgroupSize = "lse.workgroup_size";
inline constexpr std::string_view kElements = "lse.elements";
inline constexpr std::string_view kWindowBegin = "lse.window_begin";
inline constexpr std::string_view kWindowCount = "lse.window_count";
inline constexpr std::string_view kIsPhase = "lse.is_phase";
inline constexpr std::string_view kSourceHash = "lse.source_hash";
inline constexpr std::string_view kArch = "lse.arch";
inline constexpr std::string_view kNodePrefix = "lse.node.";  // + index
inline constexpr std::string_view kCounterTotal = "lse.counter_total";
}  // namespace annotation

struct ExportOptions {
  // Lane labels are built the way the whole engine addresses a device:
  // "<backend>:<ordinal>", never a bare ordinal.
  std::string backend_name = "device";
  std::string process_name = "lse";
  // Declared in the trace when known, so a reader can see what a device tick
  // would have been counted on even in a trace that has no device ticks.
  backend::DeviceClock device_clock{};
  // Without an anchor an absolute device tick cannot be placed on the host
  // timeline at all, so device slices are NOT emitted and the device duration
  // travels as an annotation instead. Refusing to place them is the point: a
  // device slice positioned by a host clock is the substitution this whole
  // module exists to prevent.
  std::optional<ClockAnchor> anchor{};
  // rocprofv3 uses sequence ids 1 and 2, and two producers sharing one id
  // corrupt each other's incremental state in a merged trace.
  std::uint32_t sequence_id = 0x15e;
};

// Stable track uuid for a lane label, so redeploying a writer does not renumber
// the tracks of an otherwise identical trace.
[[nodiscard]] std::uint64_t track_uuid(std::string_view label) noexcept;

// The lane a dispatch's device execution belongs to, and the one its submission
// belongs to. Exposed because they are part of the mapping, not an internal
// detail: a consumer that wants "the device lane for stream 2" must compute the
// same label the writer did.
// One host lane per recording thread. Spans from two threads interleave, and a
// timeline track holds one nesting, so merging them would report a bogus
// hierarchy rather than a busy one.
[[nodiscard]] std::string host_lane_label(const ExportOptions& options,
                                          std::uint32_t thread);
[[nodiscard]] std::string device_lane_label(const ExportOptions& options,
                                            std::uint8_t device_ordinal,
                                            std::uint32_t stream);
[[nodiscard]] std::string kernel_lane_label(const ExportOptions& options);

[[nodiscard]] Result<std::vector<std::byte>> encode_pftrace(
    const TraceData& data, const ExportOptions& options);

[[nodiscard]] Status write_pftrace(const std::string& path,
                                   const TraceData& data,
                                   const ExportOptions& options);

}  // namespace lse::trace
