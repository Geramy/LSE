#include "lse/opt/arrangement.hpp"

#include <sstream>

namespace lse::opt {

ArrangementCost arrangement_cost(const DeviceCapacity& cap,
                                 const Arrangement& a) {
  ArrangementCost out;
  out.residency = occupancy(cap, a.demand);
  if (!a.traffic.stated || a.traffic.workgroups == 0) return out;
  out.stated = true;
  out.launch_bytes =
      a.traffic.total_bytes() * static_cast<std::uint64_t>(a.traffic.workgroups);
  out.charged_bytes = static_cast<double>(out.launch_bytes);
  const double share =
      cap.residency_bandwidth.share(out.residency.workgroups_per_pool);
  if (share > 0.0) {
    out.charged_bytes /= share;
    out.charged = true;
  }
  return out;
}

std::string ArrangementCost::describe() const {
  std::ostringstream os;
  if (!stated) {
    os << "unpriced (no traffic stated)\n";
    return os.str();
  }
  os << launch_bytes << " B over the grid, charged "
     << static_cast<std::uint64_t>(charged_bytes) << " B"
     << (charged ? "" : " (residency uncharged)") << ", " << residency.describe();
  return os.str();
}

bool prefer(const ArrangementCost& candidate, const ArrangementCost& incumbent) {
  // A spilling kernel is a different regime, and that judgement is not this
  // file's to re-derive: opt::prefer already owns it, and owning it in one
  // place is what stops the two rules from disagreeing.
  const bool cand_spilled =
      candidate.residency.spill == backend::SpillState::kSpilled;
  const bool inc_spilled =
      incumbent.residency.spill == backend::SpillState::kSpilled;
  if (cand_spilled != inc_spilled) return inc_spilled;

  if (!candidate.stated || !candidate.residency.seated()) return false;
  if (!incumbent.stated || !incumbent.residency.seated()) return true;

  if (candidate.charged_bytes < incumbent.charged_bytes) return true;
  if (candidate.charged_bytes > incumbent.charged_bytes) return false;

  // The same traffic, collected at the same rate. Residency decides next, on
  // its own terms.
  const std::uint32_t cand = candidate.residency.workgroups_per_pool;
  const std::uint32_t inc = incumbent.residency.workgroups_per_pool;
  if (cand != inc) return cand > inc;
  // Still equal. The smaller request wins: the room it leaves on the pool is
  // what a later fusion has to spend, and two shapes that move the same bytes
  // and seat the same workgroups differ in nothing else. This is where the two
  // preference rules part company — the fusion gate hands a tie to the
  // candidate so a scratch-neutral merge is admissible, and here a tie is
  // between two shapes of the SAME launch, where taking the cheaper one costs
  // nothing.
  if (candidate.residency.scratch_request.known() &&
      incumbent.residency.scratch_request.known()) {
    return candidate.residency.scratch_request.value <=
           incumbent.residency.scratch_request.value;
  }
  return opt::prefer(candidate.residency, incumbent.residency);
}

std::size_t best_arrangement(const DeviceCapacity& cap,
                             std::span<const Arrangement> candidates) {
  std::size_t best = candidates.size();
  ArrangementCost incumbent;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const ArrangementCost cost = arrangement_cost(cap, candidates[i]);
    if (!cost.stated || !cost.residency.seated()) continue;
    if (best == candidates.size() || prefer(cost, incumbent)) {
      best = i;
      incumbent = cost;
    }
  }
  return best;
}

}  // namespace lse::opt
