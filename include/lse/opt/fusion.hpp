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
#include <string_view>

#include "lse/opt/occupancy.hpp"

namespace lse::opt {

struct FusionCandidate {
  // The workgroup size both arrangements would launch at.
  std::uint32_t threads = 0;
  // Workgroup scratch the merged body is predicted to request.
  std::uint32_t fused_scratch_bytes = 0;
  // The largest scratch request among the launches the merge replaces. That is
  // the tightest of them, and the tightest is what the unfused arrangement's
  // residency actually is.
  std::uint32_t worst_solo_scratch_bytes = 0;
  // Entry name the merged body would be compiled as. When a previous compile
  // of THAT SAME kernel is on record, its measured scratch replaces the
  // prediction and its spill state is decisive. Empty asks nothing.
  std::string_view fused_entry;
};

struct FusionVerdict {
  bool admit = false;
  Occupancy fused;
  Occupancy unfused;
  // Whether a previous compile of the fused kernel was consulted, rather than
  // the emitter's prediction.
  bool measured = false;
};

[[nodiscard]] FusionVerdict admit_fusion(const DeviceCapacity& cap,
                                         const FusionCandidate& candidate);

}  // namespace lse::opt
