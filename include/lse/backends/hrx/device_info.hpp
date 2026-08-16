// AMD GPU specifics, published as a DeviceInfo extension.
//
// These are the numbers a tile chooser needs and nobody else can interpret: a
// wavefront, an LDS budget and a matrix-core generation mean nothing on a
// device that has none of them. The HRX backend owns the block; code that
// already targets AMD reads it with device_extension<AmdDeviceInfo>().
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "lse/backend/backend.hpp"

namespace lse::backend {

enum class ArchFamily : std::uint8_t {
  kUnknown,
  kRdna2,   // gfx10
  kRdna3,   // gfx110x
  kRdna35,  // gfx115x
  kRdna4,   // gfx12
  kCdna1,   // gfx908
  kCdna2,   // gfx90a
  kCdna3,   // gfx94x
  kCdna4,   // gfx95x
};

[[nodiscard]] inline ArchFamily arch_family(std::string_view arch) noexcept {
  if (arch.rfind("gfx115", 0) == 0) return ArchFamily::kRdna35;
  if (arch.rfind("gfx110", 0) == 0) return ArchFamily::kRdna3;
  if (arch.rfind("gfx12", 0) == 0) return ArchFamily::kRdna4;
  if (arch.rfind("gfx10", 0) == 0) return ArchFamily::kRdna2;
  if (arch.rfind("gfx90a", 0) == 0) return ArchFamily::kCdna2;
  if (arch.rfind("gfx908", 0) == 0) return ArchFamily::kCdna1;
  if (arch.rfind("gfx94", 0) == 0) return ArchFamily::kCdna3;
  if (arch.rfind("gfx95", 0) == 0) return ArchFamily::kCdna4;
  return ArchFamily::kUnknown;
}

[[nodiscard]] inline std::string_view arch_family_name(ArchFamily f) noexcept {
  switch (f) {
    case ArchFamily::kRdna2: return "rdna2";
    case ArchFamily::kRdna3: return "rdna3";
    case ArchFamily::kRdna35: return "rdna3.5";
    case ArchFamily::kRdna4: return "rdna4";
    case ArchFamily::kCdna1: return "cdna1";
    case ArchFamily::kCdna2: return "cdna2";
    case ArchFamily::kCdna3: return "cdna3";
    case ArchFamily::kCdna4: return "cdna4";
    case ArchFamily::kUnknown: return "unknown";
  }
  return "unknown";
}

[[nodiscard]] inline bool family_is_cdna(ArchFamily f) noexcept {
  return f == ArchFamily::kCdna1 || f == ArchFamily::kCdna2 ||
         f == ArchFamily::kCdna3 || f == ArchFamily::kCdna4;
}

[[nodiscard]] inline bool wavefront_legal(std::string_view arch,
                                          std::uint8_t wave) noexcept {
  if (wave != 32 && wave != 64) return false;
  switch (arch_family(arch)) {
    case ArchFamily::kRdna2:
    case ArchFamily::kRdna3:
    case ArchFamily::kRdna35:
      return wave == 32;
    case ArchFamily::kCdna1:
    case ArchFamily::kCdna2:
    case ArchFamily::kCdna3:
    case ArchFamily::kCdna4:
      return wave == 64;
    case ArchFamily::kRdna4:
    case ArchFamily::kUnknown:
      return true;
  }
  return false;
}

// LSE_WAVEFRONT=32|64 wins on RDNA4. Other families ignore an illegal ask.
[[nodiscard]] inline std::uint8_t select_wavefront(
    std::string_view arch, std::uint8_t runtime_wave) noexcept {
  const std::uint8_t fallback = family_is_cdna(arch_family(arch)) ? 64 : 32;
  std::uint8_t chosen =
      wavefront_legal(arch, runtime_wave) ? runtime_wave : fallback;
  if (const char* env = std::getenv("LSE_WAVEFRONT")) {
    const int want = std::atoi(env);
    if (want == 32 || want == 64) {
      const auto wave = static_cast<std::uint8_t>(want);
      if (wavefront_legal(arch, wave)) chosen = wave;
    }
  }
  return chosen;
}

enum class MatrixCore : std::uint8_t {
  kNone = 0,
  kWMMA,   // RDNA3 / 3.5 / 4
  kMFMA,   // CDNA2 / 3 / 4
};

struct AmdDeviceInfo {
  static constexpr std::string_view kExtensionId = "amd";

  std::uint32_t l2_cache_bytes = 0;
  std::uint32_t lds_bytes_per_workgroup = 0;
  std::uint32_t clock_khz = 0;               // HRX CLOCK_RATE, 0 if unknown
  std::uint8_t max_waves_per_cu = 0;
  // Selected launch width. RDNA3/3.5: 32. CDNA: 64. RDNA4: 32 or 64
  // (LSE_WAVEFRONT picks; default 32).
  std::uint8_t wavefront_size = 0;
  MatrixCore matrix_core = MatrixCore::kNone;
  // ALU bf16, which is not the same question as whether the *matrix core* has
  // a bf16 operand form: CDNA1 has MFMA and no bf16 at all, and the two could
  // diverge again. A kernel choosing a matrix instruction asks the second one.
  bool has_bf16_arith = false;
  bool matrix_core_bf16 = false;
  bool has_dot4_i8 = false;
  std::uint8_t max_load_bytes = 0;
  std::uint8_t max_store_bytes = 0;
};

// HRX reports the live arch; the width is an ISA constant of that arch.
// Unknown gfx* still gets 16 — every GCN/RDNA/CDNA part we ship for has
// dwordx4. Non-AMD or empty arch falls back to a scalar dword.
// Workgroup LDS budget from the live device. 0 if unknown or not AMD —
// the Lds object then does not refuse allocations.
[[nodiscard]] inline std::uint32_t workgroup_lds_bytes(
    const DeviceInfo* info) noexcept {
  if (info == nullptr) return 0;
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(*info);
  return amd != nullptr ? amd->lds_bytes_per_workgroup : 0;
}

[[nodiscard]] inline std::uint8_t max_load_bytes(const DeviceInfo& info) noexcept {
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(info);
  if (amd != nullptr && amd->max_load_bytes != 0) return amd->max_load_bytes;
  if (info.arch.rfind("gfx", 0) == 0) return 16;
  return 4;
}

[[nodiscard]] inline std::uint8_t max_store_bytes(const DeviceInfo& info) noexcept {
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(info);
  if (amd != nullptr && amd->max_store_bytes != 0) return amd->max_store_bytes;
  if (info.arch.rfind("gfx", 0) == 0) return 16;
  return 4;
}

// How many `elem_bytes` values fit in one load. 1, 2, or 4 — those are the
// vector widths the emitter knows how to spell.
[[nodiscard]] inline std::uint32_t max_vec_elems(
    const DeviceInfo* info, std::uint32_t elem_bytes) noexcept {
  if (info == nullptr || elem_bytes == 0) return 1;
  const std::uint32_t n = max_load_bytes(*info) / elem_bytes;
  if (n >= 4) return 4;
  if (n >= 2) return 2;
  return 1;
}

// How many workgroups of `threads` can co-reside on one CU given an LDS
// request. 0 means the request cannot be satisfied — including when `info` is
// not an AMD device, since then no AMD tile is legal on it.
[[nodiscard]] inline std::uint32_t occupancy_per_cu(
    const DeviceInfo& info, std::uint32_t threads,
    std::uint32_t lds_bytes) noexcept {
  const AmdDeviceInfo* amd = device_extension<AmdDeviceInfo>(info);
  if (amd == nullptr || threads == 0 || amd->wavefront_size == 0) return 0;
  if (threads > info.max_threads_per_workgroup) return 0;
  if (lds_bytes > amd->lds_bytes_per_workgroup) return 0;
  const std::uint32_t waves =
      (threads + amd->wavefront_size - 1) / amd->wavefront_size;
  const std::uint32_t wave_limit = amd->max_waves_per_cu / waves;
  if (lds_bytes == 0) return wave_limit;
  const std::uint32_t lds_limit = amd->lds_bytes_per_workgroup / lds_bytes;
  return wave_limit < lds_limit ? wave_limit : lds_limit;
}

}  // namespace lse::backend
