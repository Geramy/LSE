#include "lse/opt/occupancy.hpp"

#include <sstream>

#include "lse/backend/backend.hpp"

namespace lse::opt {

namespace {

std::uint32_t align_up(std::uint32_t v, std::uint32_t step) noexcept {
  if (step <= 1) return v;
  return ((v + step - 1) / step) * step;
}

void fact_line(std::ostringstream& os, const char* name,
               const backend::DeviceFact<std::uint32_t>& f) {
  os << ' ' << name << '=';
  if (f.known()) {
    os << f.value;
  } else {
    os << '-';
  }
}

}  // namespace

bool DeviceCapacity::usable() const noexcept {
  return wave_slots_per_simd.known() && simds_per_lds_pool.known() &&
         wavefront_size.known() && wave_slots_per_simd.value != 0 &&
         simds_per_lds_pool.value != 0 && wavefront_size.value != 0;
}

std::string DeviceCapacity::describe() const {
  std::ostringstream os;
  os << "capacity";
  fact_line(os, "slots/simd", wave_slots_per_simd);
  fact_line(os, "vgpr/simd", vector_registers_per_simd);
  fact_line(os, "vgpr-granule", vector_register_alloc_granule);
  fact_line(os, "lds/pool", lds_bytes_per_pool);
  fact_line(os, "simds/pool", simds_per_lds_pool);
  fact_line(os, "lds-granule", lds_alloc_granule_bytes);
  fact_line(os, "lds/wg-cap", lds_bytes_addressable_per_workgroup);
  fact_line(os, "max-threads", max_flat_workgroup_size);
  fact_line(os, "wavefront", wavefront_size);
  os << '\n';
  return os.str();
}

DeviceCapacity DeviceCapacity::of(const backend::DeviceInfo& info) {
  DeviceCapacity c;
  const backend::ArchFacts& f = info.arch_facts;
  c.wave_slots_per_simd = f.wave_slots_per_simd;
  c.vector_registers_per_simd = f.vector_registers_per_simd;
  c.vector_register_alloc_granule = f.vector_register_alloc_granule;
  c.lds_bytes_per_pool = f.lds_bytes_per_pool;
  c.simds_per_lds_pool = f.simds_per_lds_pool;
  c.lds_alloc_granule_bytes = f.lds_alloc_granule_bytes;
  c.lds_bytes_addressable_per_workgroup = f.lds_bytes_addressable_per_workgroup;
  c.max_flat_workgroup_size = f.max_flat_workgroup_size;
  // The runtime's own answers, where the ISA table had nothing to say. These
  // are queried from the device rather than declared for the part number.
  if (!c.lds_bytes_addressable_per_workgroup.known() &&
      info.lds_bytes_per_workgroup != 0) {
    c.lds_bytes_addressable_per_workgroup =
        backend::DeviceFact<std::uint32_t>::queried(
            info.lds_bytes_per_workgroup);
  }
  if (!c.max_flat_workgroup_size.known() &&
      info.max_threads_per_workgroup != 0) {
    c.max_flat_workgroup_size = backend::DeviceFact<std::uint32_t>::queried(
        info.max_threads_per_workgroup);
  }
  if (info.wavefront_size != 0) {
    c.wavefront_size =
        backend::DeviceFact<std::uint32_t>::queried(info.wavefront_size);
  }
  return c;
}

KernelDemand KernelDemand::measured(std::uint32_t threads,
                                    const backend::KernelResources& r) {
  KernelDemand d;
  d.threads = threads;
  // What the compiler emitted, not what an emitter predicted it would.
  d.lds_bytes = r.workgroup_segment_bytes.value_or(0u);
  d.vector_registers = r.vector_registers;
  d.spill = r.spilled();
  return d;
}

Occupancy occupancy(const DeviceCapacity& cap, const KernelDemand& demand) {
  Occupancy occ;
  occ.spill = demand.spill;
  occ.scratch_request_bytes = demand.lds_bytes;
  if (!cap.usable() || demand.threads == 0) return occ;

  const std::uint32_t wave = cap.wavefront_size.value;
  const std::uint32_t simds = cap.simds_per_lds_pool.value;
  const std::uint32_t slots = cap.wave_slots_per_simd.value;

  if (cap.max_flat_workgroup_size.known() &&
      demand.threads > cap.max_flat_workgroup_size.value) {
    return occ;
  }
  if (cap.lds_bytes_addressable_per_workgroup.known() &&
      demand.lds_bytes > cap.lds_bytes_addressable_per_workgroup.value) {
    return occ;
  }

  const std::uint32_t waves_per_wg = (demand.threads + wave - 1) / wave;
  if (waves_per_wg == 0 || waves_per_wg > slots * simds) return occ;

  bool exact = true;

  // WAVE SLOTS. Always countable once the capacity is usable at all.
  occ.waves_by_slots = slots;
  std::uint32_t wgs = (slots * simds) / waves_per_wg;
  OccupancyLimit binding = OccupancyLimit::kWaveSlots;

  // VECTOR REGISTERS. waves = file / align_up(request, granule), per SIMD. The
  // arm drops out entirely when the kernel has not been compiled yet or the
  // target's file size is unknown — a guess here is the failure this replaces.
  if (demand.vector_registers.known() &&
      cap.vector_registers_per_simd.known() &&
      cap.vector_register_alloc_granule.known() &&
      demand.vector_registers.value != 0) {
    const std::uint32_t alloc = align_up(demand.vector_registers.value,
                                         cap.vector_register_alloc_granule.value);
    const std::uint32_t by_regs =
        alloc == 0 ? slots : cap.vector_registers_per_simd.value / alloc;
    occ.waves_by_registers = by_regs;
    occ.registers_counted = true;
    if (by_regs == 0) return occ;
    // A workgroup's waves are spread over the pool's SIMDs, so a per-SIMD wave
    // budget becomes a workgroup budget the same way the slot budget does.
    const std::uint32_t by_regs_wgs = (by_regs * simds) / waves_per_wg;
    if (by_regs_wgs < wgs) {
      wgs = by_regs_wgs;
      binding = OccupancyLimit::kVectorRegisters;
    }
  }
  // A kernel that has not been compiled yet reports no register count, and
  // that is not an inexact answer — it is the normal pre-compile regime, it
  // applies alike to every arrangement being compared, and `registers_counted`
  // already says the arm did not run. Only a fact that was SUBSTITUTED, or an
  // arm dropped in the direction that UNDERSTATES a request, makes the figure
  // an upper bound; see the scratch arm below.

  // WORKGROUP SCRATCH. Zero bytes asks nothing of the pool, which is a real
  // answer and not a missing one.
  if (demand.lds_bytes != 0) {
    if (cap.lds_bytes_per_pool.known()) {
      // An unknown granule means the request is charged unrounded, which
      // UNDER-states what it costs — so the answer is marked inexact and a
      // decision that would admit on it is held to strict improvement.
      if (!cap.lds_alloc_granule_bytes.known()) exact = false;
      const std::uint32_t granule = cap.lds_alloc_granule_bytes.value_or(1u);
      const std::uint32_t charged = align_up(demand.lds_bytes, granule);
      const std::uint32_t by_lds =
          charged == 0 ? wgs : cap.lds_bytes_per_pool.value / charged;
      occ.waves_by_scratch = (by_lds * waves_per_wg) / simds;
      if (by_lds == 0) return occ;
      if (by_lds < wgs) {
        wgs = by_lds;
        binding = OccupancyLimit::kWorkgroupScratch;
      }
    } else {
      exact = false;
    }
  }

  occ.workgroups_per_pool = wgs;
  occ.waves_per_simd = (wgs * waves_per_wg) / simds;
  // A workgroup smaller than the pool's SIMD count seats whole workgroups
  // without filling one wave per SIMD; the count is still one wave deep.
  if (occ.waves_per_simd == 0) occ.waves_per_simd = 1;
  occ.binding = binding;
  occ.exact = exact;
  return occ;
}

std::string Occupancy::describe() const {
  std::ostringstream os;
  os << waves_per_simd << " waves/simd, " << workgroups_per_pool
     << " wgs/pool, bound by " << to_string(binding) << " (slots ";
  auto arm = [&os](std::uint32_t v) {
    if (v == kNoLimit) {
      os << '-';
    } else {
      os << v;
    }
  };
  arm(waves_by_slots);
  os << ", vgpr ";
  arm(waves_by_registers);
  os << ", lds ";
  arm(waves_by_scratch);
  os << "), spill " << to_string(spill) << (exact ? ", exact" : ", inexact")
     << '\n';
  return os.str();
}

bool prefer(const Occupancy& candidate, const Occupancy& incumbent) {
  // A kernel whose values went to memory is a different regime. No residency
  // figure trades against it, in either direction.
  if (candidate.spill == backend::SpillState::kSpilled &&
      incumbent.spill != backend::SpillState::kSpilled) {
    return false;
  }
  if (incumbent.spill == backend::SpillState::kSpilled &&
      candidate.spill != backend::SpillState::kSpilled) {
    return true;
  }
  if (!candidate.seated()) return false;
  if (!incumbent.seated()) return true;
  if (candidate.workgroups_per_pool > incumbent.workgroups_per_pool) return true;
  if (candidate.workgroups_per_pool < incumbent.workgroups_per_pool) return false;
  // A tie. Trustworthy when both figures were counted from answered facts, and
  // otherwise only when the candidate asks for no more scratch than the
  // incumbent — an unmeasured granule cannot separate two equal requests, but
  // it can hide a step between two different ones.
  if (candidate.exact && incumbent.exact) return true;
  return candidate.scratch_request_bytes <= incumbent.scratch_request_bytes;
}

}  // namespace lse::opt
