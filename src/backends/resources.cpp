#include "lse/backend/resources.hpp"

#include <sstream>

namespace lse::backend {

namespace {

// A fact renders as its value or as the reason there is no value, never as a
// blank or a zero — either of which reads as a measurement.
void field(std::ostringstream& os, const char* label,
           const DeviceFact<std::uint32_t>& fact, const char* unit = "") {
  os << "  " << label << ": ";
  if (!fact.known()) {
    os << to_string(fact.source) << '\n';
    return;
  }
  os << fact.value << unit;
  if (fact.source == FactSource::kDeclared) os << " (declared)";
  os << '\n';
}

}  // namespace

std::string KernelResources::describe() const {
  std::ostringstream os;
  os << (entry.empty() ? "<unnamed kernel>" : entry) << '\n';
  field(os, "vgprs        ", vector_registers);
  field(os, "sgprs        ", scalar_registers);
  field(os, "agprs        ", accum_registers);
  field(os, "group segment", workgroup_segment_bytes, " B");
  field(os, "private seg  ", private_segment_bytes, " B");
  field(os, "vgpr spills  ", vector_spills);
  field(os, "sgpr spills  ", scalar_spills);
  field(os, "kernarg      ", kernarg_segment_bytes, " B");
  field(os, "max flat wg  ", max_flat_workgroup_size);
  field(os, "wavefront    ", wavefront_size);
  os << "  reqd wg size : ";
  if (required_workgroup_size.known()) {
    os << required_workgroup_size.value[0] << 'x'
       << required_workgroup_size.value[1] << 'x'
       << required_workgroup_size.value[2] << '\n';
  } else {
    os << to_string(required_workgroup_size.source) << '\n';
  }
  os << "  spilled      : " << to_string(spilled()) << '\n';
  return os.str();
}

std::string ArchFacts::describe() const {
  std::ostringstream os;
  field(os, "vgprs/SIMD          ", vector_registers_per_simd);
  field(os, "vgpr granule        ", vector_register_alloc_granule);
  field(os, "vgprs/wave (addr)   ", vector_registers_addressable_per_wave);
  field(os, "sgprs/SIMD          ", scalar_registers_per_simd);
  field(os, "sgpr granule        ", scalar_register_alloc_granule);
  field(os, "sgprs/wave (addr)   ", scalar_registers_addressable_per_wave);
  field(os, "LDS/workgroup (addr)", lds_bytes_addressable_per_workgroup, " B");
  field(os, "LDS banks           ", lds_banks);
  field(os, "max flat workgroup  ", max_flat_workgroup_size);
  field(os, "wave slots/SIMD     ", wave_slots_per_simd);
  field(os, "SIMDs/LDS pool      ", simds_per_lds_pool);
  field(os, "LDS/pool            ", lds_bytes_per_pool, " B");
  field(os, "LDS alloc granule   ", lds_alloc_granule_bytes, " B");
  return os.str();
}

}  // namespace lse::backend
