// ISA fallbacks for fields HRX does not report.
//
// Live device numbers come from HRX (name, arch, memory, CUs, workgroup
// size, warp size, LDS). apply_arch_defaults never overwrites a non-zero
// runtime value. The tables below fill zeros: family rows are ISA-fixed
// (wavefront, matrix core, load width); board rows are last-resort CU/L2
// counts for when no device is attached.
//
// PROVENANCE: gfx1151 CUs/L2 measured on the development box (Radeon 8060S).
// Other board rows are from AMD public specs and are unverified here.
#pragma once

#include <string_view>

#include "lse/backends/hrx/device_info.hpp"

namespace lse::backend {

struct FamilyIsa {
  ArchFamily family;
  std::uint16_t max_threads_per_workgroup;
  std::uint32_t lds_bytes_per_workgroup;
  std::uint32_t l2_cache_bytes;
  // Wavefront slots per CORE, which is `simds_per_cu` times the slots one SIMD
  // holds. Filing the per-SIMD number here is the 2x error that hides a
  // register-bound kernel, so the two fields move together.
  std::uint8_t max_waves_per_cu;
  std::uint8_t wavefront_size;
  // RDNA pairs two CUs into a WGP that shares one LDS block; CDNA gives each
  // CU its own. See DeviceInfo::cus_per_lds_pool for the gfx1151 measurement.
  std::uint8_t cus_per_lds_pool;
  // SIMDs in one CU. RDNA is 2, GCN and CDNA are 4. With cus_per_lds_pool this
  // is what turns a per-SIMD budget into a per-pool one.
  std::uint8_t simds_per_cu;
  // The workgroup-scratch block a pool allocates from, AND the CU count that
  // figure is stated for. THE TWO ARE ONE FACT and must be read together: this
  // is NOT lds_bytes_per_workgroup times cus_per_lds_pool in general — that
  // identity holds on RDNA by coincidence and breaks where the per-workgroup
  // cap is itself the whole block — so the block size cannot be rescaled to a
  // pool of a different shape. 131072 on RDNA is the block behind a TWO-CU
  // WGP; read at cus_per_lds_pool = 1 it is a silent factor of two in every
  // residency answer, which is why the pair is spelled out rather than
  // assumed.
  std::uint32_t lds_bytes_per_pool;
  std::uint8_t lds_bytes_per_pool_cus;
  MatrixCore matrix_core;
  bool has_bf16_arith;
  bool matrix_core_bf16;
  bool has_dot4_i8;
  // dot8-insts, RDNA3 and later only. See AmdDeviceInfo::has_dot4_iu8.
  bool has_dot4_iu8;
  std::uint8_t max_load_bytes;
  std::uint8_t max_store_bytes;
};

struct BoardFallback {
  std::string_view arch;
  std::uint16_t compute_units;
  std::uint32_t l2_cache_bytes;
  bool unified_memory;
};

// max_waves_per_cu is slots-per-SIMD times simds_per_cu: 16 x 2 on every RDNA
// generation, 10 x 4 on CDNA1, 8 x 4 on CDNA2 and later. RDNA2/3/4 and CDNA2
// previously carried a per-SIMD figure under the per-CU name and so undercounted
// by 2x and 1.25x; the wave slots the occupancy model divides by are derived
// from this pair, so a wrong cell here is now a wrong occupancy rather than an
// unread number.
inline constexpr FamilyIsa kFamilyIsa[] = {
    {ArchFamily::kRdna2, 1024, 65536, 4u << 20, 32, 32, 2, 2, 131072, 2,
     MatrixCore::kNone, false, false, true, false, 16, 16},
    {ArchFamily::kRdna3, 1024, 65536, 4u << 20, 32, 32, 2, 2, 131072, 2,
     MatrixCore::kWMMA, true, true, true, true, 16, 16},
    {ArchFamily::kRdna35, 1024, 65536, 2u << 20, 32, 32, 2, 2, 131072, 2,
     MatrixCore::kWMMA, true, true, true, true, 16, 16},
    {ArchFamily::kRdna4, 1024, 65536, 4u << 20, 32, 32, 2, 2, 131072, 2,
     MatrixCore::kWMMA, true, true, true, true, 16, 16},
    {ArchFamily::kCdna1, 1024, 65536, 8u << 20, 40, 64, 1, 4, 65536, 1,
     MatrixCore::kMFMA, false, false, true, false, 16, 16},
    {ArchFamily::kCdna2, 1024, 65536, 8u << 20, 32, 64, 1, 4, 65536, 1,
     MatrixCore::kMFMA, true, true, true, false, 16, 16},
    {ArchFamily::kCdna3, 1024, 65536, 4u << 20, 32, 64, 1, 4, 65536, 1,
     MatrixCore::kMFMA, true, true, true, false, 16, 16},
    {ArchFamily::kCdna4, 1024, 65536, 4u << 20, 32, 64, 1, 4, 65536, 1,
     MatrixCore::kMFMA, true, true, true, false, 16, 16},
};

// Step a workgroup-scratch request is rounded up to before it is charged
// against the pool. Not in any ISA table and not answered by any runtime query
// reachable from here, so this holds ONLY rows that were measured by device
// co-residency, and a target with no row leaves the fact unknown. On gfx1151
// the 1024 is uniquely determined: 5200 B seats 21 workgroups per pool, which
// requires rounding to 6144 — a 512 B granule would predict 23.
struct LdsGranule {
  std::string_view arch;
  std::uint32_t bytes;
};

inline constexpr LdsGranule kLdsGranule[] = {
    {"gfx1151", 1024},
};

inline const LdsGranule* lds_granule(std::string_view arch) noexcept {
  for (const LdsGranule& g : kLdsGranule) {
    if (arch.rfind(g.arch, 0) == 0) return &g;
  }
  return nullptr;
}

inline constexpr BoardFallback kBoardFallback[] = {
    {"gfx1151", 40, 2u << 20, true},
    {"gfx1100", 96, 4u << 20, false},
    {"gfx1201", 64, 4u << 20, false},
    {"gfx90a", 110, 8u << 20, false},
    {"gfx942", 304, 4u << 20, false},
    {"gfx950", 256, 4u << 20, false},
};

inline const FamilyIsa* family_isa(ArchFamily family) noexcept {
  for (const FamilyIsa& p : kFamilyIsa) {
    if (p.family == family) return &p;
  }
  return nullptr;
}

inline const BoardFallback* board_fallback(std::string_view arch) noexcept {
  for (const BoardFallback& b : kBoardFallback) {
    if (arch.rfind(b.arch, 0) == 0) return &b;
  }
  return nullptr;
}

// The four capacity facts an occupancy answer needs and no compiler or runtime
// reachable from here reports: wave slots per SIMD, the SIMDs behind one
// workgroup-scratch block, that block's size, and the step a request is rounded
// up to. Declared for the part, and left UNKNOWN where nothing measured them —
// comgr's MaxWavesPerCU (40) and EUsPerCU (4) are identical on every target it
// knows and contradicted by this device's runtime, so they are not a source.
inline void apply_residency_facts(DeviceInfo& info) {
  const FamilyIsa* isa = family_isa(arch_family(info.arch));
  if (isa != nullptr && isa->simds_per_cu != 0) {
    info.arch_facts.wave_slots_per_simd = DeviceFact<std::uint32_t>::declared(
        isa->max_waves_per_cu / isa->simds_per_cu);
    const std::uint32_t cus = info.cus_per_lds_pool != 0
                                  ? info.cus_per_lds_pool
                                  : isa->cus_per_lds_pool;
    info.arch_facts.simds_per_lds_pool = DeviceFact<std::uint32_t>::declared(
        static_cast<std::uint32_t>(isa->simds_per_cu) * cus);
    // The block size and the CU count it was stated for travel together. The
    // SIMD count above follows the live part, so a part that pairs its cores
    // differently from the family row would otherwise inherit a block sized
    // for the row's pairing and divide it among a pool of another shape — a
    // factor of two that cancels on this part and on no argument. Nothing
    // states the block for a pool of that shape, so the fact stays UNKNOWN and
    // the scratch limit drops out of the min instead of being invented.
    if (isa->lds_bytes_per_pool != 0 && cus == isa->lds_bytes_per_pool_cus) {
      info.arch_facts.lds_bytes_per_pool =
          DeviceFact<std::uint32_t>::declared(isa->lds_bytes_per_pool);
    }
  }
  if (const LdsGranule* g = lds_granule(info.arch); g != nullptr) {
    info.arch_facts.lds_alloc_granule_bytes =
        DeviceFact<std::uint32_t>::declared(g->bytes);
  }
}

inline void apply_arch_defaults(DeviceInfo& info, AmdDeviceInfo& amd) {
  const ArchFamily family = arch_family(info.arch);
  const FamilyIsa* isa = family_isa(family);
  const BoardFallback* board = board_fallback(info.arch);

  if (info.compute_units == 0 && board != nullptr) {
    info.compute_units = board->compute_units;
  }
  if (info.max_threads_per_workgroup == 0 && isa != nullptr) {
    info.max_threads_per_workgroup = isa->max_threads_per_workgroup;
  }
  if (info.lds_bytes_per_workgroup == 0 && isa != nullptr) {
    info.lds_bytes_per_workgroup = isa->lds_bytes_per_workgroup;
  }
  if (amd.l2_cache_bytes == 0) {
    if (board != nullptr && board->l2_cache_bytes != 0) {
      amd.l2_cache_bytes = board->l2_cache_bytes;
    } else if (isa != nullptr) {
      amd.l2_cache_bytes = isa->l2_cache_bytes;
    }
  }
  if (info.max_waves_per_cu == 0 && isa != nullptr) {
    info.max_waves_per_cu = isa->max_waves_per_cu;
  }
  // Not "if zero": HRX has no query for it, so the ISA row is the only source.
  if (isa != nullptr) info.cus_per_lds_pool = isa->cus_per_lds_pool;
  if (info.wavefront_size == 0 && isa != nullptr) {
    info.wavefront_size = isa->wavefront_size;
  }
  if (amd.matrix_core == MatrixCore::kNone && isa != nullptr) {
    amd.matrix_core = isa->matrix_core;
  }
  if (!info.unified_memory && board != nullptr) {
    info.unified_memory = board->unified_memory;
  }
  if (isa != nullptr) {
    amd.has_bf16_arith = isa->has_bf16_arith;
    amd.matrix_core_bf16 = isa->matrix_core_bf16;
    amd.has_dot4_i8 = isa->has_dot4_i8;
    amd.has_dot4_iu8 = isa->has_dot4_iu8;
    if (amd.max_load_bytes == 0) amd.max_load_bytes = isa->max_load_bytes;
    if (amd.max_store_bytes == 0) amd.max_store_bytes = isa->max_store_bytes;
  }
  info.wavefront_size = select_wavefront(
      info.arch, static_cast<std::uint8_t>(info.wavefront_size));
  apply_residency_facts(info);
}

}  // namespace lse::backend
