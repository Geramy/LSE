// The answer to "how much of this compiled kernel can actually be resident,
// and what says no first", assembled from the MEASURED code object rather than
// an emitter's prediction.
//
// This is the measured half of the occupancy decision. Before a kernel is
// compiled, `opt::occupancy` runs over a `KernelDemand` whose register count is
// unknown and the register arm drops out of the minimum. After it is compiled,
// the code object states its own VGPR count, its workgroup segment, and (where
// the toolchain reports it) whether it spilled — and THIS is the number a
// bandwidth diagnosis must be argued from, because a kernel that is not resident
// enough to keep the memory pipe fed is bound by occupancy, not by the DRAM.
//
// Nothing here names a vendor, an ISA or a dialect: it is a pairing of the
// measured `backend::KernelResources` with the `opt::DeviceCapacity`, the same
// `opt::occupancy` model a pre-compile decision uses, plus the launch grid that
// tells whether the kernel even has enough work to matter.
#pragma once

#include <cstdint>
#include <string>

#include "lse/backend/resources.hpp"
#include "lse/opt/occupancy.hpp"

namespace lse::opt {

// What one compiled kernel was launched with, the part the code object does not
// state for itself. A code object without launch bounds does not say how many
// workgroups it was run as, so the grid is carried by the caller, exactly as
// `KernelDemand::measured` carries the thread count.
struct LaunchGeometry {
  // Threads per workgroup the kernel ran with. `KernelDemand::measured` takes
  // this as its `threads` argument; a required_workgroup_size from the object
  // overrides it when the object fixes its own geometry.
  std::uint32_t threads = 0;
  // How many workgroups the launch actually ran. 0 means "not stated" — the
  // census then reports the kernel as unrun rather than guessing an occupancy
  // it was never seated with.
  std::uint32_t workgroups = 0;
};

// The measured occupancy of one compiled kernel on one device. Every field is
// either a fact the code object stated, a fact the device stated, or the
// minimum of them — never a guess. `occupancy` is the same struct a pre-compile
// `opt::occupancy` call returns, so a diagnosis and the fusion policy that
// produced the kernel argue from the identical arithmetic.
struct KernelCensus {
  backend::KernelResources resources;
  LaunchGeometry launch;
  Occupancy occupancy;

  // The device's capacity the occupancy was computed against, carried so the
  // report can show WHICH facts bound the answer instead of just the number.
  DeviceCapacity capacity;

  // True when the code object fixed its own launch geometry and it differs from
  // what the caller ran it with — a kernel seated at a smaller workgroup than
  // it requested is a different residency than the object was tuned for.
  [[nodiscard]] bool geometry_mismatched() const noexcept;

  // The one-line human form: the resources, the achieved occupancy, and which
  // limit binds, in a shape that reads the same whether it is a fusion that
  // fit or a decode GEMV that cannot feed the pipe.
  [[nodiscard]] std::string describe() const;
};

// Assemble the census from the measured object and the launch it ran with.
// `cap` is the device's capacity (`DeviceCapacity::of(info)`); `r` is the
// kernel's measured resources (`read_code_object_resources(object).front()`);
// `launch` is the threads-per-workgroup and workgroup count it ran at. The
// thread count the occupancy model sees is the object's required_workgroup_size
// when the object states one, else `launch.threads` — the object's own geometry
// is the source of truth for what it was compiled to seat.
[[nodiscard]] KernelCensus census(const DeviceCapacity& cap,
                                  const backend::KernelResources& r,
                                  const LaunchGeometry& launch);

}  // namespace lse::opt
