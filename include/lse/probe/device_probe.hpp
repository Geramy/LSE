// Measuring one device, behind a seam.
//
// A backend that can generate kernels registers a probe that records the bodies
// in probe_kernels.hpp and times them on its own hardware. A backend that
// cannot gets the portable probe, which measures what its own API can answer
// and leaves the rest kUnknown. Neither invents anything: a probe that cannot
// reach a number DECLINES and names what it could not reach, exactly as the
// codec engine declines a wire format it has no spelling for.
//
// A second backend joins by registering a factory. No caller changes.
#pragma once

#include <memory>
#include <string_view>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/probe/profile.hpp"

namespace lse::probe {

class IDeviceProbe {
 public:
  virtual ~IDeviceProbe() = default;

  // Fills what it can of `out`, leaving every field it did not obtain exactly
  // as it found it. Returns non-ok only when the probe could not run at all;
  // a field it merely could not reach stays kUnknown and is not an error.
  virtual Status run(backend::IBackend& backend, DeviceProfile& out) = 0;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  // Empty when this probe reaches every field it reports on. Otherwise names
  // what it cannot measure here, so an unknown in the profile is diagnosable
  // rather than a silent hole.
  [[nodiscard]] virtual std::string_view declined() const noexcept = 0;
};

using DeviceProbeFactory =
    std::unique_ptr<IDeviceProbe> (*)(backend::IBackend&);

void register_device_probe(std::string_view backend_name,
                           DeviceProbeFactory factory);

// The registered probe for this backend, or the portable one.
[[nodiscard]] std::unique_ptr<IDeviceProbe> create_device_probe(
    backend::IBackend& backend);

struct DeviceProbeRegistrar {
  DeviceProbeRegistrar(std::string_view backend_name,
                       DeviceProbeFactory factory) {
    register_device_probe(backend_name, factory);
  }
};

// Identity plus the transfer rates any backend's own allocate/copy can answer.
// When the backend has neither an emitter nor a compiler it cannot run a
// generated kernel at all, so its work is host work: the same probe bodies then
// run under env::Cpu and the numbers describe the host, which is the truth for
// that backend.
[[nodiscard]] std::unique_ptr<IDeviceProbe> make_portable_device_probe(
    backend::IBackend& backend);

// What is free on this device right now, or kUnknown when the backend's
// runtime declines to say. kMeasured, never kDeclared: unlike every other
// number in a profile this one is not a property of the part, it is an
// observation of this device in this process at this instant, and it changes
// under any other process on the box.
//
// Split out of the probe because it is the one field a memoized profile may
// not replay — see qualify_pool, which drops it on a cache hit and calls this
// instead. Cheap enough to redo: one driver query, no kernel, no allocation.
[[nodiscard]] Measured sample_free_memory(const backend::IBackend& backend);

// Identity from the backend, then the probe. The whole per-device deliverable.
[[nodiscard]] Result<DeviceProfile> probe_device(backend::IBackend& backend);

// Identity only: the fields a DeviceInfo already answers, with no measurement.
// Split out so a profile read from disk can be checked against the live device
// without re-measuring anything.
[[nodiscard]] DeviceProfile device_identity(const backend::IBackend& backend);

}  // namespace lse::probe
