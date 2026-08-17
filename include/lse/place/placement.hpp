// Where a piece of work runs, decided from what it costs and where its bytes
// already are.
//
// Two inputs and no third: probe::CostModel prices the work on each member, and
// residency says which member already holds the operands. There is no policy
// knob here and no device name — a caller states the work, the planner answers
// with a member of the set. That is PLAN.md's rule ("devices are capable load
// locations, never names") expressed as a call.
//
// It refuses rather than guesses. Work the cost model cannot price stays where
// its bytes are and the reason says so; that is the answer, not a fallback, and
// it is why wiring this in cannot move a single-device run's work anywhere.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/place/devices.hpp"
#include "lse/probe/cost_model.hpp"

namespace lse::place {

// Bytes a relocation would have to move. The caller supplies both because only
// it knows what the work reads from elsewhere and what its result owes back;
// a planner that assumed either would be inventing the term that decides the
// answer.
struct Transfer {
  std::size_t out = 0;    // to the candidate before it can start
  std::size_t back = 0;   // to whoever asked, after it finishes
};

struct Assignment {
  std::size_t member = 0;           // index into the set
  backend::DeviceIndex device{};    // that member's residency token
  // Whether the chosen member can hold what the work needs resident. Reported
  // whatever the verdict, so "staying is faster" is distinguishable from
  // "nobody measured whether staying is even possible".
  probe::Capacity capacity;
  // True when the work's operands have to be moved for this to run.
  bool relocates = false;
  // Why this member and not another, in words a report can print.
  std::string_view reason;
};

// One member's share of a divided operation.
struct Portion {
  std::size_t member = 0;
  double fraction = 0.0;
  probe::Cost per_item;
  probe::Capacity capacity;
};

class Planner {
 public:
  // The pool must be the one measured for this set: shares are resolved back to
  // members by DeviceId, and a profile of some other pool would name members
  // this set does not hold.
  Planner(const Devices& devices, const probe::PoolProfile& pool) noexcept;

  [[nodiscard]] const probe::CostModel& model() const noexcept { return model_; }

  // Where work whose operands are resident on member `home` should run.
  //
  // A set of one answers `home` without pricing anything — there is nowhere
  // else, so asking the cost model would be a measurement spent on a decision
  // that is already made.
  [[nodiscard]] Result<Assignment> place(
      const probe::Work& work, std::size_t home, Transfer moved,
      std::uint32_t depth = probe::kDefaultQueueDepth) const;

  // How to divide one operation across the whole set, proportionally to what
  // each member measurably achieves on it. Empty when the split is not
  // feasible; `fit` says whether that is "does not fit" or "nobody measured".
  struct Division {
    std::vector<Portion> portions;
    probe::Fit fit = probe::Fit::kUnknown;
    probe::Provenance provenance = probe::Provenance::kUnknown;
    std::string_view reason;

    [[nodiscard]] bool feasible() const noexcept {
      return fit == probe::Fit::kFits && !portions.empty();
    }
  };
  [[nodiscard]] Result<Division> divide(
      const probe::Work& work, std::size_t home,
      std::uint32_t depth = probe::kDefaultQueueDepth) const;

 private:
  const Devices* devices_;
  probe::CostModel model_;
};

}  // namespace lse::place
