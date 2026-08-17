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
  std::uint8_t max_waves_per_cu;
  std::uint8_t wavefront_size;
  MatrixCore matrix_core;
  bool has_bf16_arith;
  bool matrix_core_bf16;
  bool has_dot4_i8;
  std::uint8_t max_load_bytes;
  std::uint8_t max_store_bytes;
};

struct BoardFallback {
  std::string_view arch;
  std::uint16_t compute_units;
  std::uint32_t l2_cache_bytes;
  bool unified_memory;
};

inline constexpr FamilyIsa kFamilyIsa[] = {
    {ArchFamily::kRdna2, 1024, 65536, 4u << 20, 16, 32, MatrixCore::kNone,
     false, false, true, 16, 16},
    {ArchFamily::kRdna3, 1024, 65536, 4u << 20, 16, 32, MatrixCore::kWMMA,
     true, true, true, 16, 16},
    {ArchFamily::kRdna35, 1024, 65536, 2u << 20, 32, 32, MatrixCore::kWMMA,
     true, true, true, 16, 16},
    {ArchFamily::kRdna4, 1024, 65536, 4u << 20, 16, 32, MatrixCore::kWMMA,
     true, true, true, 16, 16},
    {ArchFamily::kCdna1, 1024, 65536, 8u << 20, 40, 64, MatrixCore::kMFMA,
     false, false, true, 16, 16},
    {ArchFamily::kCdna2, 1024, 65536, 8u << 20, 40, 64, MatrixCore::kMFMA,
     true, true, true, 16, 16},
    {ArchFamily::kCdna3, 1024, 65536, 4u << 20, 32, 64, MatrixCore::kMFMA,
     true, true, true, 16, 16},
    {ArchFamily::kCdna4, 1024, 65536, 4u << 20, 32, 64, MatrixCore::kMFMA,
     true, true, true, 16, 16},
};

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
    if (amd.max_load_bytes == 0) amd.max_load_bytes = isa->max_load_bytes;
    if (amd.max_store_bytes == 0) amd.max_store_bytes = isa->max_store_bytes;
  }
  info.wavefront_size = select_wavefront(
      info.arch, static_cast<std::uint8_t>(info.wavefront_size));
}

}  // namespace lse::backend
