#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include "lse/backend/resources.hpp"

namespace lse::graph::detail {

// Allocation metadata describes fixed bytes, not executed memory traffic.
inline std::string private_memory_diagnostic(
    const backend::KernelResources& r, std::string_view arch,
    std::uint64_t artifact_key, std::uint64_t source_hash) {
  const bool private_allocation =
      r.private_segment_bytes.known() && r.private_segment_bytes.value != 0;
  if (r.entry.empty() ||
      (!private_allocation && r.spilled() != backend::SpillState::kSpilled)) {
    return {};
  }
  std::ostringstream os;
  os << "[kernel-resources] entry=" << r.entry << " arch=" << arch
     << " artifact=" << artifact_key << " source_hash=" << source_hash;
  const auto fact = [&os](const char* name, const auto& value) {
    os << ' ' << name << '=';
    if (value.known()) os << value.value;
    else os << to_string(value.source);
  };
  fact("private_B_per_workitem", r.private_segment_bytes);
  os << " private_source=" << to_string(r.private_segment_bytes.source);
  fact("LDS_B_per_workgroup", r.workgroup_segment_bytes);
  os << " spill=" << to_string(r.spilled());
  fact("vgpr_spills", r.vector_spills);
  fact("sgpr_spills", r.scalar_spills);
  os << '\n';
  return os.str();
}

}  // namespace lse::graph::detail
