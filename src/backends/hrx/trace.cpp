// The HRX tracing adapter.
//
// Tier 1's host half needs nothing from a backend. Its device half needs the
// backend to have stamped the dispatch, and HRX cannot: the device-timestamp
// capability on IBackend exists, HRX publishes the clock and DECLINES the tick,
// and this file forwards that decline with the reason attached rather than
// filling the field from a host clock.
//
// Tier 2 is declined by name. rocprofiler-sdk is the adapter that goes here, and
// what it needs from the engine is already in the schema: the join key is
// DispatchRecord::sequence, pushed as an external correlation id, so nothing has
// to parse "lse_fused_<signature>" out of a kernel name.
#include <memory>
#include <string>

#include "lse/backend/backend.hpp"
#include "lse/trace/collector.hpp"

namespace lse::backend {

namespace {

class HrxTracer final : public trace::ITrace {
 public:
  explicit HrxTracer(IBackend& backend)
      : portable_(trace::make_portable_tracer(backend)) {}

  std::string_view name() const noexcept override { return "hrx"; }

  // Both clock questions are already answered correctly from the IBackend seam —
  // kDeviceAgent, 99,810,000 Hz, 64 valid bits on gfx1151, queried from the
  // agent rather than assumed, and an anchor the moment a tick becomes readable.
  // Delegated rather than copied so there is one implementation of the
  // two-host-reads-around-one-device-read measurement.
  Result<DeviceClock> device_clock() const override {
    return portable_->device_clock();
  }

  Result<trace::ClockAnchor> clock_anchor() const override {
    return portable_->clock_anchor();
  }

  Status resolve_device_span(trace::DispatchRecord&) override {
    // Naming what is missing, because the gap is one function wide and everything
    // under it is built: IREE's AMDGPU HAL implements
    // iree_hal_device_queue_timestamp with PM4 emitters, a 64-byte timestamp
    // record ABI carrying correlation ids, and a per-queue-family
    // timestamp_frequency_hz. libhrx forwards none of it.
    //
    // hrx_event_record must never be substituted: it stamps the HOST clock
    // before hrx_stream_flush, so it times host recording order, not device
    // execution. On a command buffer flushed every 16 dispatches that is
    // submission jitter reported as a kernel duration.
    return LSE_ERROR(kUnimplemented,
                     "hrx stamps no dispatch timestamps: needs "
                     "hrx_stream_timestamp() forwarding "
                     "iree_hal_device_queue_timestamp(), which the AMDGPU HAL "
                     "already implements. Device duration stays unknown until "
                     "then; hrx_event_elapsed_time is a host clock taken before "
                     "the flush and is not a substitute");
  }

  Status supports(trace::Tier tier) const override {
    if (tier == trace::Tier::kDispatch) return OkStatus();
    return LSE_ERROR(kUnimplemented,
                     "no rocprofiler adapter is bound to hrx yet. The join key "
                     "it needs is already in the schema: DispatchRecord::"
                     "sequence, pushed through "
                     "rocprofiler_push_external_correlation_id, which lands in "
                     "every kernel-dispatch record and removes any need to "
                     "match kernel names");
  }

  Result<std::unique_ptr<trace::IDeepSession>> open_deep(
      const trace::DeepRequest&) override {
    return LSE_ERROR(kUnimplemented,
                     "no rocprofiler adapter is bound to hrx yet");
  }

 private:
  std::unique_ptr<trace::ITrace> portable_;
};

std::unique_ptr<trace::ITrace> make_hrx_tracer(IBackend& backend) {
  return std::make_unique<HrxTracer>(backend);
}

const trace::TraceRegistrar _lse_hrx_trace_registrar{"hrx", make_hrx_tracer};

}  // namespace

}  // namespace lse::backend
