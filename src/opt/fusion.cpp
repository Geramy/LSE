#include "lse/opt/fusion.hpp"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/opt/measurements.hpp"

namespace lse::opt {

namespace {

// Settled decisions never promote again. Measured private-memory or spill
// regressions can demote once; ordinary residency feedback stays deferred.
struct Settled {
  std::mutex mu;
  std::unordered_map<std::string, bool> verdicts;
};

Settled& settled() {
  static Settled s;
  return s;
}

}  // namespace

FusionVerdict admit_fusion(const DeviceCapacity& cap,
                           const FusionCandidate& candidate) {
  FusionVerdict v;
  KernelDemand fused =
      KernelDemand::counted(candidate.threads, candidate.fused_scratch_bytes);
  KernelDemand solo = KernelDemand::counted(candidate.threads,
                                            candidate.worst_solo_scratch_bytes);

  const KernelMeasurements& known = KernelMeasurements::instance();
  backend::KernelResources fused_r;
  if (!candidate.fused_entry.empty()) {
    fused_r = known.lookup(candidate.fused_entry);
    fused.spill = fused_r.spilled();
    fused.private_bytes_per_workitem = fused_r.private_segment_bytes;
  }

  std::vector<backend::KernelResources> solo_r;
  solo_r.reserve(candidate.solo_entries.size());
  bool all_measured = fused_r.workgroup_segment_bytes.known() &&
                      !candidate.solo_entries.empty();
  bool all_private = !candidate.solo_entries.empty();
  bool all_clean = !candidate.solo_entries.empty();
  bool any_spilled = false;
  std::uint32_t max_private = 0;
  backend::FactSource private_source = backend::FactSource::kQueried;
  for (const std::string& entry : candidate.solo_entries) {
    auto r = known.lookup(entry);
    all_measured &= r.workgroup_segment_bytes.known();
    all_private &= r.private_segment_bytes.known();
    if (r.private_segment_bytes.known()) {
      max_private = std::max(max_private, r.private_segment_bytes.value);
      if (r.private_segment_bytes.source == backend::FactSource::kDeclared) {
        private_source = backend::FactSource::kDeclared;
      }
    }
    all_clean &= r.spilled() == backend::SpillState::kNone;
    any_spilled |= r.spilled() == backend::SpillState::kSpilled;
    solo_r.push_back(std::move(r));
  }
  if (all_private) {
    solo.private_bytes_per_workitem = {max_private, private_source};
  }
  solo.spill = any_spilled ? backend::SpillState::kSpilled
               : all_clean ? backend::SpillState::kNone
                           : backend::SpillState::kUnknown;

  // Residency uses all measured launches or all emitter counts. Never compare
  // measured LDS on one side against predicted LDS on the other.
  if (all_measured) {
    v.measured = true;
    v.fused = occupancy(cap, KernelDemand::measured(candidate.threads, fused_r));
    bool first = true;
    for (const backend::KernelResources& r : solo_r) {
      const Occupancy one =
          occupancy(cap, KernelDemand::measured(candidate.threads, r));
      if (first || one.workgroups_per_pool < v.unfused.workgroups_per_pool ||
          (one.workgroups_per_pool == v.unfused.workgroups_per_pool &&
           !one.exact && v.unfused.exact)) {
        v.unfused = one;
      }
      first = false;
    }
  } else {
    v.fused = occupancy(cap, fused);
    v.unfused = occupancy(cap, solo);
  }
  // Private allocation is a separate fact, not a residency limit. Sequential
  // launches reuse private storage, so compare the largest known solo request;
  // one unknown solo leaves the aggregate unknown rather than supplying zero.
  v.unfused.private_bytes_per_workitem = solo.private_bytes_per_workitem;
  v.unfused.spill = solo.spill;

  v.admit = !cap.usable() || prefer(v.fused, v.unfused);
  if (!candidate.fused_entry.empty()) {
    Settled& s = settled();
    const std::lock_guard lock(s.mu);
    auto [it, inserted] =
        s.verdicts.emplace(std::string(candidate.fused_entry), v.admit);
    const bool private_regression =
        private_allocation_preference(v.fused, v.unfused) ==
            AllocationPreference::kIncumbent ||
        (v.fused.workgroups_per_pool == v.unfused.workgroups_per_pool &&
         private_allocation_preference(v.fused, v.unfused, true) ==
             AllocationPreference::kIncumbent);
    const bool spill_regression =
        v.fused.spill == backend::SpillState::kSpilled &&
        v.unfused.spill == backend::SpillState::kNone;
    if (!inserted && it->second && !v.admit && v.unfused.seated() &&
        (private_regression || spill_regression)) {
      it->second = false;
      v.demoted = true;
    }
    v.recommendation_deferred = it->second != v.admit;
    v.admit = it->second;
  }
  return v;
}

std::string FusionVerdict::describe() const {
  std::ostringstream os;
  os << "fusion " << (admit ? "admitted" : "refused");
  if (demoted) {
    os << (fused.spill == backend::SpillState::kSpilled &&
                   unfused.spill == backend::SpillState::kNone
               ? " (reported spills demoted)"
               : " (private allocation demoted)");
  }
  if (recommendation_deferred) os << " (new recommendation deferred)";
  os << ", " << (measured ? "measured residency" : "counted residency")
     << "\n  fused: " << fused.describe()
     << "  solo: " << unfused.describe();
  return os.str();
}

}  // namespace lse::opt
