#include "lse/opt/fusion.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "lse/opt/measurements.hpp"

namespace lse::opt {

namespace {

// One verdict per arrangement per process.
//
// A group is emitted once per step, so an answer that moves between steps
// changes the kernel a running model is dispatching. Freezing it means a
// measurement arriving mid-run affects the NEXT process and never this one,
// which is also the termination argument: within a process the answer is a
// constant, and across processes each arrangement is scored from its own
// persisted measurement, so an arrangement can change the answer at most once
// — the first run in which its measurement exists before the decision is
// taken.
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

  // THE ESTIMATE, both sides. What the emitter says each arrangement's body
  // will declare, counted the same way from the same source — which is the
  // only thing that makes the comparison below mean anything.
  KernelDemand fused =
      KernelDemand::counted(candidate.threads, candidate.fused_scratch_bytes);
  KernelDemand solo = KernelDemand::counted(candidate.threads,
                                            candidate.worst_solo_scratch_bytes);

  // What these same arrangements measured last time they were built, each
  // keyed on its OWN entry name so an arrangement's score never depends on
  // which arrangement was chosen.
  const KernelMeasurements& known = KernelMeasurements::instance();
  backend::KernelResources fused_r;
  if (!candidate.fused_entry.empty()) {
    fused_r = known.lookup(candidate.fused_entry);
    // Spilling is not a comparison — it disqualifies an arrangement on its own
    // — so it needs no counterpart and is taken whenever it is known.
    if (fused_r.any()) fused.spill = fused_r.spilled();
  }

  // THE MEASUREMENT, all of it or none of it. A measured workgroup segment on
  // one side of the comparison against an emitter's estimate on the other is a
  // comparison of two different quantities, and the verdict then turns on that
  // difference rather than on residency.
  std::vector<backend::KernelResources> solo_r;
  bool all_measured = fused_r.workgroup_segment_bytes.known() &&
                      !candidate.solo_entries.empty();
  if (all_measured) {
    solo_r.reserve(candidate.solo_entries.size());
    for (const std::string& e : candidate.solo_entries) {
      backend::KernelResources r = known.lookup(e);
      if (!r.workgroup_segment_bytes.known()) {
        all_measured = false;
        break;
      }
      solo_r.push_back(std::move(r));
    }
  }

  if (all_measured) {
    v.measured = true;
    v.fused = occupancy(cap, KernelDemand::measured(candidate.threads, fused_r));
    // The unfused arrangement's residency is the TIGHTEST of the launches it
    // is made of — the one that seats fewest — and each is scored whole, from
    // one kernel's own registers beside its own scratch. Taking the worst of
    // each across different kernels would describe a launch that does not
    // exist.
    bool first = true;
    for (const backend::KernelResources& r : solo_r) {
      const Occupancy one =
          occupancy(cap, KernelDemand::measured(candidate.threads, r));
      if (first || !prefer(one, v.unfused)) v.unfused = one;
      first = false;
    }
  } else {
    v.fused = occupancy(cap, fused);
    v.unfused = occupancy(cap, solo);
  }

  // A device that answers nothing about its own capacity cannot refuse
  // anything on residency grounds: the arrangement that was emitted before any
  // of this existed is the one it keeps.
  v.admit = !cap.usable() || prefer(v.fused, v.unfused);

  if (!candidate.fused_entry.empty()) {
    Settled& s = settled();
    const std::lock_guard lock(s.mu);
    v.admit = s.verdicts.emplace(std::string(candidate.fused_entry), v.admit)
                  .first->second;
  }
  return v;
}

}  // namespace lse::opt
