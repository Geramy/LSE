#include "lse/opt/kernel_census.hpp"

#include <sstream>

namespace lse::opt {

bool KernelCensus::geometry_mismatched() const noexcept {
  if (!resources.required_workgroup_size.known()) return false;
  const auto& req = resources.required_workgroup_size.value;
  return req[0] != launch.threads;
}

std::string KernelCensus::describe() const {
  std::ostringstream os;
  os << resources.entry << " [measured] ";
  auto fact = [&os](const char* name,
                    const backend::DeviceFact<std::uint32_t>& f) {
    os << name << '=';
    if (f.known()) {
      os << f.value;
    } else {
      os << '-';
    }
  };
  fact("vgpr", resources.vector_registers);
  fact("sgpr", resources.scalar_registers);
  fact("lds", resources.workgroup_segment_bytes);
  fact("priv", resources.private_segment_bytes);
  os << " spill=" << to_string(resources.spilled());
  if (resources.vector_spills.known()) os << " vspill=" << resources.vector_spills.value;
  if (resources.scalar_spills.known()) os << " spspill=" << resources.scalar_spills.value;
  os << "  launch=";
  if (launch.workgroups != 0) {
    os << launch.workgroups << "wg x " << launch.threads;
  } else {
    os << "(not run) x " << launch.threads;
  }
  if (geometry_mismatched()) os << " [mismatched required-wg]";
  os << " -> ";
  os << occupancy.describe();
  return os.str();
}

KernelCensus census(const DeviceCapacity& cap, const backend::KernelResources& r,
                    const LaunchGeometry& launch) {
  KernelCensus c;
  c.resources = r;
  c.launch = launch;
  c.capacity = cap;
  // The object's own geometry wins when it states one: a kernel compiled with a
  // fixed workgroup size is seated by that size, not by whatever the caller
  // happened to pass, and seating it at a smaller size would overstate its
  // residency.
  const std::uint32_t threads =
      r.required_workgroup_size.known() && r.required_workgroup_size.value[0] != 0
          ? r.required_workgroup_size.value[0]
          : launch.threads;
  c.occupancy = occupancy(cap, KernelDemand::measured(threads, r));
  return c;
}

}  // namespace lse::opt
