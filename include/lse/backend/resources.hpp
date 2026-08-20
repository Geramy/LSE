// What a compiler can report: per-target capacity, and per-kernel usage.
//
// Backend-agnostic on purpose. A backend reports these; the engine decides
// what to do about them. Nothing here names a dialect, a vendor or an ISA.
//
// UNKNOWN IS NOT ZERO. Every value is a DeviceFact because "the toolchain does
// not report this" and "the toolchain reports this and it is zero" are
// different facts, and arithmetic that reads them the same is wrong in exactly
// the direction that looks fine. Measured instance: loomc's code objects carry
// no spill counts at all, while the HIP path emits them even when zero — a
// spill count defaulted to 0 would have declared every Loom kernel clean.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "lse/core/enum_names.hpp"

namespace lse::backend {

// Where a described fact came from. A reporter reports what it was told and
// nothing else: a property nothing reachable can answer stays kUnknown and
// carries no number to be mistaken for one, because at the point a policy
// reads it a plausible substitute is indistinguishable from a real answer.
// The same discipline as probe::Provenance, restated here because this header
// sits below probe in the build and must not depend on it.
#define LSE_FACT_SOURCE_LIST(X)                                             \
  X(kUnknown, "unknown")      /* nothing reachable here answers it */       \
  X(kInapplicable, "n/a")     /* the device has no such property at all */  \
  X(kDeclared, "declared")    /* a table answers for this part number */    \
  X(kQueried, "queried")      /* the device's own runtime answered */

LSE_DECLARE_ENUM(FactSource, std::uint8_t, LSE_FACT_SOURCE_LIST)

template <typename T>
struct DeviceFact {
  // Meaningless unless known(). Default-constructed rather than left
  // uninitialized so a reader that ignores the source gets a zero it can spot,
  // not whatever was on the stack.
  T value{};
  FactSource source = FactSource::kUnknown;

  [[nodiscard]] bool known() const noexcept {
    return source == FactSource::kQueried || source == FactSource::kDeclared;
  }

  // The value if it was answered, else what the caller says to use instead.
  // The substitute is spelled at the call site so it cannot be mistaken for a
  // measurement further down.
  [[nodiscard]] T value_or(T fallback) const {
    return known() ? value : std::move(fallback);
  }

  static DeviceFact queried(T v) { return {std::move(v), FactSource::kQueried}; }
  static DeviceFact declared(T v) {
    return {std::move(v), FactSource::kDeclared};
  }
  static DeviceFact inapplicable() { return {T{}, FactSource::kInapplicable}; }
};

// Whether the register allocator spilled. Three answers, not a bool: a
// toolchain that reports no spill counts has said nothing, and reading that as
// "did not spill" is how a spilling kernel comes to look merely slow.
#define LSE_SPILL_STATE_LIST(X)                                            \
  X(kUnknown, "unknown")  /* the toolchain reports no spill counts */      \
  X(kNone, "none")        /* reported, and zero */                         \
  X(kSpilled, "spilled")  /* reported, and non-zero */

LSE_DECLARE_ENUM(SpillState, std::uint8_t, LSE_SPILL_STATE_LIST)

// Everything a compiler reports about ONE compiled kernel. Names carry their
// unit: counts are registers per lane, sizes are bytes.
struct KernelResources {
  // The symbol these numbers belong to. A code object may define several.
  std::string entry;

  // Registers the kernel was allocated. Occupancy is a function of this and
  // the target's register file — but that arithmetic is the engine's policy,
  // not this struct's.
  DeviceFact<std::uint32_t> vector_registers;
  DeviceFact<std::uint32_t> scalar_registers;
  // Separate accumulation file where the architecture has one, kInapplicable
  // where it does not, and never a 0 standing in for either.
  DeviceFact<std::uint32_t> accum_registers;

  // Workgroup-shared scratch the kernel statically requests. This is the
  // number to admit a fusion against — it is what the compiler actually
  // emitted, not what the emitter predicted it would.
  DeviceFact<std::uint32_t> workgroup_segment_bytes;
  // Per-lane private memory: spill destination, but also alloca'd arrays and
  // call frames. Non-zero is NOT evidence of spilling — measured in this
  // tree's own cache, a kernel holds 192 B of private segment with both spill
  // counts reported and zero. Ask spilled() instead.
  DeviceFact<std::uint32_t> private_segment_bytes;
  // Values the allocator could not keep in registers. A spilling kernel is a
  // different performance regime, not a slightly worse one, so these are
  // reported apart from the scratch that holds them.
  DeviceFact<std::uint32_t> vector_spills;
  DeviceFact<std::uint32_t> scalar_spills;

  // Launch-shaped facts the object states about itself.
  DeviceFact<std::uint32_t> kernarg_segment_bytes;
  DeviceFact<std::uint32_t> max_flat_workgroup_size;
  DeviceFact<std::uint32_t> wavefront_size;
  // Present only when the object fixes its own launch geometry.
  DeviceFact<std::array<std::uint32_t, 3>> required_workgroup_size;

  [[nodiscard]] SpillState spilled() const noexcept {
    if (!vector_spills.known() && !scalar_spills.known()) {
      return SpillState::kUnknown;
    }
    const std::uint32_t v = vector_spills.known() ? vector_spills.value : 0;
    const std::uint32_t s = scalar_spills.known() ? scalar_spills.value : 0;
    return (v != 0 || s != 0) ? SpillState::kSpilled : SpillState::kNone;
  }

  // Whether anything at all was reported. A default-constructed instance means
  // "nothing measured", which is what a toolchain with no metadata produces.
  [[nodiscard]] bool any() const noexcept {
    return vector_registers.known() || scalar_registers.known() ||
           accum_registers.known() || workgroup_segment_bytes.known() ||
           private_segment_bytes.known() || vector_spills.known() ||
           scalar_spills.known() || kernarg_segment_bytes.known() ||
           max_flat_workgroup_size.known() || wavefront_size.known() ||
           required_workgroup_size.known();
  }

  [[nodiscard]] std::string describe() const;
};

// Per-target capacity: what one SIMD has to allocate from, and in what steps.
// The complement of KernelResources — that says what one kernel took, this
// says what there is to take. An occupancy figure is a pure function of the
// pair, and that function is the engine's policy, not this struct's.
//
// EVERY FIELD NAMES ITS UNIT. The one failure this must not have is a per-SIMD
// number filed under a per-core name; that reads as a plain 2x error in
// whichever direction the reader guessed.
struct ArchFacts {
  // Register file of one SIMD, in whole registers of one lane each. Bytes are
  // this times the wavefront size times 4 on every AMD target checked.
  DeviceFact<std::uint32_t> vector_registers_per_simd;
  // Step the allocator rounds a kernel's request up to. Occupancy moves in
  // these, not in single registers.
  DeviceFact<std::uint32_t> vector_register_alloc_granule;
  // Ceiling one wave can name, which is not the file size.
  DeviceFact<std::uint32_t> vector_registers_addressable_per_wave;

  DeviceFact<std::uint32_t> scalar_registers_per_simd;
  DeviceFact<std::uint32_t> scalar_register_alloc_granule;
  DeviceFact<std::uint32_t> scalar_registers_addressable_per_wave;

  // The most one workgroup may request. NOT the size of the pool those
  // requests are served from — where the design pairs cores behind one
  // physical block the pool is larger, and conflating the two is how a fusion
  // that "fits" halves residency.
  DeviceFact<std::uint32_t> lds_bytes_addressable_per_workgroup;
  DeviceFact<std::uint32_t> lds_banks;
  DeviceFact<std::uint32_t> max_flat_workgroup_size;

  // Residency capacity. None of these four is in any compiler's ISA table or
  // any runtime query reachable from here, so they are declared for the part
  // and stay unknown where nothing measured them — an occupancy answer then
  // degrades rather than inventing one.
  //
  // Wavefronts one SIMD can hold. Deliberately NOT derived from a per-core
  // "max waves" figure without dividing by the SIMDs in that core; the 2x
  // error that produces is the signature failure of this whole area.
  DeviceFact<std::uint32_t> wave_slots_per_simd;
  // SIMDs drawing on one workgroup-scratch block, and that block's size. The
  // block is not the per-workgroup cap times anything in general: the identity
  // holds on RDNA by coincidence and breaks on parts with a larger cap.
  DeviceFact<std::uint32_t> simds_per_lds_pool;
  DeviceFact<std::uint32_t> lds_bytes_per_pool;
  // Step a scratch request is rounded up to before it is charged against the
  // block. Costs a whole resident workgroup at the sizes where it bites.
  DeviceFact<std::uint32_t> lds_alloc_granule_bytes;

  [[nodiscard]] bool any() const noexcept {
    return vector_registers_per_simd.known() ||
           vector_register_alloc_granule.known() ||
           vector_registers_addressable_per_wave.known() ||
           scalar_registers_per_simd.known() ||
           scalar_register_alloc_granule.known() ||
           scalar_registers_addressable_per_wave.known() ||
           lds_bytes_addressable_per_workgroup.known() || lds_banks.known() ||
           max_flat_workgroup_size.known() || wave_slots_per_simd.known() ||
           simds_per_lds_pool.known() || lds_bytes_per_pool.known() ||
           lds_alloc_granule_bytes.known();
  }

  [[nodiscard]] std::string describe() const;
};

// What share of its streaming rate this device's memory system delivers to a
// launch at a given residency.
//
// WHY IT IS A SEPARATE FACT AND NOT A SLOPE. Residency buys memory-level
// parallelism, and parallelism buys bandwidth only until the memory system is
// saturated. Neither end of that is a slope: measured on gfx1151 the rate falls
// 27% between eight resident workgroups per pool and two, and almost none of
// that fall happens in the top half of the range. No law reproduces the shape,
// so it is measured and tabulated, exactly as the scratch allocation granule is,
// and a part nobody measured leaves it unknown rather than getting a curve
// derived from another part's.
//
// DIMENSIONLESS ON PURPOSE. An absolute rate would let a caller compute a time,
// but a comparison between two arrangements OF ONE LAUNCH divides by the same
// rate on both sides, so the rate cancels and only the share does not. Carrying
// the share alone is what lets residency and traffic be compared without
// anybody having to measure a bandwidth first.
struct ResidencyBandwidth {
  // Indexed by whole workgroups resident on one scratch pool. Percent of the
  // rate the same kernel reaches where residency stops binding; 0 means that
  // point was not measured. Index 0 is unused — no workgroups move no bytes.
  static constexpr std::size_t kPoints = 9;
  std::array<std::uint8_t, kPoints> percent{};
  FactSource source = FactSource::kUnknown;

  [[nodiscard]] bool known() const noexcept {
    return source == FactSource::kQueried || source == FactSource::kDeclared;
  }

  // The share at `workgroups`, or 0 when nothing was measured at all.
  //
  // Past the last point the curve is read flat: it saturates, and reading a
  // saturated curve flat is what it says rather than an extrapolation. Between
  // points the answer is the nearest measured one BELOW, which understates what
  // the launch collects — the direction that charges a thin arrangement more
  // rather than less, so an unmeasured point can only make the model keep
  // residency it might not have needed, never spend residency it did need.
  // BELOW the lowest measured point there is nothing to fall back to and the
  // scan turns around, so a residency thinner than anything measured is priced
  // at the thinnest that was. That is the one direction this cannot be
  // conservative in, and the answer is to measure that point, not to invent it.
  [[nodiscard]] double share(std::uint32_t workgroups) const noexcept {
    if (!known() || workgroups == 0) return 0.0;
    std::size_t at = workgroups < kPoints ? workgroups : kPoints - 1;
    while (at > 0 && percent[at] == 0) --at;
    while (at < kPoints && percent[at] == 0) ++at;
    if (at >= kPoints || percent[at] == 0) return 0.0;
    return static_cast<double>(percent[at]) / 100.0;
  }
};

}  // namespace lse::backend
