// Whether to emit a run of stages as ONE launch or as one launch each.
//
// THE RULE: a fused run must seat at least as many workgroups on one scratch
// pool as the tightest of the launches it replaces. Fitting the per-workgroup
// cap is not the question and never was — the measured failure this exists for
// summed two panels to 64000 bytes, fit a 65536-byte cap, and seated 2
// workgroups per pool where the arrangement it replaced seated 4. A 141-token
// prefill went from 9.4 s to 126.8 s.
//
// The counted quantities are bytes, threads and workgroups. Nothing here needs
// to know the instruction set or the language the body will be written in,
// which is what puts it in the engine.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "lse/opt/occupancy.hpp"

namespace lse::opt {

struct FusionCandidate {
  // The workgroup size both arrangements would launch at.
  std::uint32_t threads = 0;
  // Workgroup scratch the merged body is predicted to request, and the largest
  // predicted among the launches the merge replaces — the tightest of them,
  // which is what the unfused arrangement's residency actually is.
  //
  // BOTH MUST BE THE SAME QUANTITY, counted the same way from the same source.
  // Pricing the unfused arrangement as the fused one's hoisted panel is what
  // made every comparison a tie and left this gate unable to fire at all.
  std::uint32_t fused_scratch_bytes = 0;
  std::uint32_t worst_solo_scratch_bytes = 0;
  // Entry name the merged body would be compiled as, and the entry names the
  // launches it replaces would be compiled as. Empty asks nothing.
  //
  // A measured workgroup segment replaces the prediction only when EVERY one
  // of these is on record, fused side and solo side alike. Measured, not
  // feared: substituting the fused kernel's measured segment against an
  // estimated solo figure compares two different quantities and flips the
  // verdict on that difference alone — it made the 4B answer "The first part
  // of the" where it had answered " Paris.". The fused entry's SPILL state is
  // taken whenever it is known, because spilling disqualifies an arrangement
  // on its own and so needs no counterpart to be compared against.
  std::string_view fused_entry;
  std::span<const std::string> solo_entries;
};

struct FusionVerdict {
  bool admit = false;
  Occupancy fused;
  Occupancy unfused;
  // Whether both sides were scored from previous compiles rather than from the
  // emitter's prediction.
  bool measured = false;
};

[[nodiscard]] FusionVerdict admit_fusion(const DeviceCapacity& cap,
                                         const FusionCandidate& candidate);

}  // namespace lse::opt
