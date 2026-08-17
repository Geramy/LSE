#include <chrono>
#include <map>
#include <mutex>
#include <string>

#include "lse/trace/collector.hpp"

namespace lse::trace {

namespace {

std::uint64_t host_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// What the IBackend seam alone can answer. This is not a fallback that pretends:
// it forwards the two device-clock questions and declines the rest by name.
class PortableTracer final : public ITrace {
 public:
  explicit PortableTracer(backend::IBackend& backend)
      : backend_(backend), backend_name_(backend.name()) {}

  std::string_view name() const noexcept override { return "portable"; }

  Result<backend::DeviceClock> device_clock() const override {
    return backend_.device_clock();
  }

  Result<ClockAnchor> clock_anchor() const override {
    // Two host reads around one device read: the device tick happened somewhere
    // inside that interval, and its width is the uncertainty. Reported rather
    // than assumed, because the interval is a syscall wide on some runtimes and
    // a few nanoseconds wide on others.
    const std::uint64_t before = host_now_ns();
    auto tick = backend_.sample_device_time();
    const std::uint64_t after = host_now_ns();
    if (!tick.ok()) return tick.status();
    ClockAnchor anchor;
    anchor.host_ns = before + (after - before) / 2;
    anchor.device = tick.release();
    anchor.uncertainty_ns = after - before;
    return anchor;
  }

  Status resolve_device_span(DispatchRecord&) override {
    return LSE_ERROR(kUnimplemented, "backend '", backend_name_,
                     "' does not stamp dispatch timestamps: a device span has "
                     "to be written by whatever submitted the packet, and "
                     "sampling a clock afterwards cannot recover one");
  }

  Status supports(Tier tier) const override {
    // Tier 1 is served: the host span needs no backend at all. Whether a device
    // span comes with it is resolve_device_span's question, not this one.
    if (tier == Tier::kDispatch) return OkStatus();
    return LSE_ERROR(kUnimplemented, "no deep-counter adapter for backend '",
                     backend_name_,
                     "': hardware counters come from a vendor profiler bound to "
                     "this backend, and none is registered");
  }

  Result<std::unique_ptr<IDeepSession>> open_deep(const DeepRequest&) override {
    return LSE_ERROR(kUnimplemented, "no deep-counter adapter for backend '",
                     backend_name_, "'");
  }

 private:
  backend::IBackend& backend_;
  std::string backend_name_;
};

struct Registry {
  std::mutex mu;
  std::map<std::string, TraceFactory, std::less<>> factories;
};

Registry& registry() {
  static Registry r;
  return r;
}

}  // namespace

void register_tracer(std::string_view backend_name, TraceFactory factory) {
  Registry& r = registry();
  std::lock_guard lock(r.mu);
  r.factories.emplace(std::string(backend_name), factory);
}

std::unique_ptr<ITrace> create_tracer(backend::IBackend& backend) {
  Registry& r = registry();
  TraceFactory factory = nullptr;
  {
    std::lock_guard lock(r.mu);
    const auto it = r.factories.find(backend.name());
    if (it != r.factories.end()) factory = it->second;
  }
  if (factory != nullptr) {
    if (auto tracer = factory(backend)) return tracer;
  }
  return make_portable_tracer(backend);
}

std::unique_ptr<ITrace> make_portable_tracer(backend::IBackend& backend) {
  return std::make_unique<PortableTracer>(backend);
}

}  // namespace lse::trace
