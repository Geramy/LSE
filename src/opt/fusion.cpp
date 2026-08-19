#include "lse/opt/fusion.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

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

  KernelDemand fused;
  fused.threads = candidate.threads;
  fused.lds_bytes = candidate.fused_scratch_bytes;

  // What the same arrangement measured last time it was built, keyed on the
  // fused kernel's OWN entry name so an arrangement's score never depends on
  // which arrangement was chosen.
  //
  // ONLY THE SPILL STATE IS TAKEN, and the reason is worth stating because the
  // richer version is wrong: this is a COMPARISON, and a measurement put into
  // one side of it has to be the same quantity as the estimate on the other.
  // The fused kernel's measured workgroup segment is the whole body's, every
  // stage's own staging included, while the unfused side can only be estimated
  // from the panels — so substituting it compares two different quantities and
  // flips the verdict on that difference alone. Measured, not feared: doing so
  // made the 4B answer "The first part of the" where it had answered " Paris.".
  // Spilling is not a comparison — it disqualifies an arrangement on its own —
  // so it needs no counterpart and is safe to take.
  if (!candidate.fused_entry.empty()) {
    const backend::KernelResources r =
        KernelMeasurements::instance().lookup(candidate.fused_entry);
    if (r.any()) {
      v.measured = true;
      fused.spill = r.spilled();
    }
  }

  KernelDemand solo;
  solo.threads = candidate.threads;
  solo.lds_bytes = candidate.worst_solo_scratch_bytes;

  v.fused = occupancy(cap, fused);
  v.unfused = occupancy(cap, solo);
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
