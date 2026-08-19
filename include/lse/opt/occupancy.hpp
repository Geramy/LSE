// How many of a launch's workgroups a device can hold at once, and which
// resource says no first.
//
// THE RULE: occupancy is the MINIMUM over every limit that applies — wave
// slots, vector registers, workgroup scratch. A model that counts one of them
// answers a different question, and answers it confidently. Two failures in
// this tree came from exactly that: a kernel pinned at 12 waves/SIMD by 102
// vector registers while its scratch was cut 36% for nothing, and a fused run
// admitted because 64000 bytes FIT a 65536-byte cap while seating half as many
// workgroups per pool as the two groups it replaced.
//
// UNITS, because every 2x error here is a unit error. `waves_per_simd` is the
// figure a compiler prints as "Occupancy"; `workgroups_per_pool` is how many
// whole workgroups co-reside on the SIMDs that share one scratch block. They
// are not interchangeable and both are reported.
//
// Nothing here names a vendor, an ISA or a dialect: it is arithmetic over
// backend::ArchFacts and backend::KernelResources, which is why it is engine
// policy and not a backend's.
#pragma once

#include <cstdint>
#include <string>

#include "lse/backend/resources.hpp"

namespace lse::backend {
struct DeviceInfo;
}

namespace lse::opt {

// What one scratch pool's worth of SIMDs has to give out. Every field names
// its unit; a per-SIMD number filed under a per-core name is the one mistake
// this must not make.
struct DeviceCapacity {
  // Wavefronts one SIMD can hold, whatever they cost. The ceiling everything
  // else is a min against.
  backend::DeviceFact<std::uint32_t> wave_slots_per_simd;

  // Register file of one SIMD in whole per-lane registers, and the step the
  // allocator rounds a request up to. Occupancy moves in granules.
  backend::DeviceFact<std::uint32_t> vector_registers_per_simd;
  backend::DeviceFact<std::uint32_t> vector_register_alloc_granule;

  // The scratch block, and how many SIMDs draw from it. NOT the per-workgroup
  // cap: where a design pairs cores behind one block the pool is larger, and
  // reading the cap as the pool is how a fusion that "fits" halves residency.
  backend::DeviceFact<std::uint32_t> lds_bytes_per_pool;
  backend::DeviceFact<std::uint32_t> simds_per_lds_pool;
  backend::DeviceFact<std::uint32_t> lds_alloc_granule_bytes;

  // Ceilings on one workgroup's own request.
  backend::DeviceFact<std::uint32_t> lds_bytes_addressable_per_workgroup;
  backend::DeviceFact<std::uint32_t> max_flat_workgroup_size;
  backend::DeviceFact<std::uint32_t> wavefront_size;

  // Enough answered to seat anything at all: the wave arithmetic needs the
  // wavefront size, the slot count and the pool's SIMD count. Everything else
  // may be unknown, and a limit whose facts are unknown drops out of the min
  // rather than guessing a value for it.
  [[nodiscard]] bool usable() const noexcept;

  [[nodiscard]] std::string describe() const;

  // Everything the device says about itself: queried facts where the toolchain
  // answered, the family row where it did not, unknown where neither does.
  static DeviceCapacity of(const backend::DeviceInfo& info);
};

// What one launch asks for. `vector_registers` is unknown before the kernel is
// compiled, which is the normal case at a fusion decision — the register limit
// then drops out of the min and `registers_counted` says so.
struct KernelDemand {
  std::uint32_t threads = 0;
  std::uint32_t lds_bytes = 0;
  backend::DeviceFact<std::uint32_t> vector_registers;
  backend::SpillState spill = backend::SpillState::kUnknown;

  // The demand a previously compiled object justifies: its measured registers,
  // its measured scratch, and whether it spilled. `threads` stays the caller's,
  // since a code object without launch bounds does not state its own.
  static KernelDemand measured(std::uint32_t threads,
                               const backend::KernelResources& r);
};

// Which resource produced the answer. kNothing means no limit bound: every
// fact that could constrain it was unknown, so the figure is a ceiling and not
// a count.
#define LSE_OCC_LIMIT_LIST(X)                                                \
  X(kNothing, "nothing")            /* no limit was countable */             \
  X(kIllegal, "illegal")            /* the request cannot be seated at all */\
  X(kWaveSlots, "wave-slots")                                                \
  X(kVectorRegisters, "vector-registers")                                    \
  X(kWorkgroupScratch, "workgroup-scratch")

LSE_DECLARE_ENUM(OccupancyLimit, std::uint8_t, LSE_OCC_LIMIT_LIST)

struct Occupancy {
  // Wavefronts resident per SIMD — the unit a compiler prints as "Occupancy".
  std::uint32_t waves_per_simd = 0;
  // Whole workgroups resident on the SIMDs sharing one scratch pool. What a
  // fusion decision spends.
  std::uint32_t workgroups_per_pool = 0;

  // Each arm on its own, so a disagreement with a compiler's own number can be
  // attributed rather than argued about. kNoLimit where the arm did not count.
  static constexpr std::uint32_t kNoLimit = 0xFFFFFFFFu;
  std::uint32_t waves_by_slots = kNoLimit;
  std::uint32_t waves_by_registers = kNoLimit;
  std::uint32_t waves_by_scratch = kNoLimit;

  OccupancyLimit binding = OccupancyLimit::kIllegal;

  // The scratch the launch asked for, carried so a comparison can tell a tie
  // it can trust from a tie an unknown allocation granule might be hiding: two
  // arrangements asking for the SAME bytes round the same way whatever the
  // granule is, and two asking for different bytes do not.
  std::uint32_t scratch_request_bytes = 0;

  // A SPILLING KERNEL IS A DIFFERENT REGIME, NOT A LOWER NUMBER. Occupancy
  // says how many waves fit; it says nothing about a wave that goes to memory
  // for a value it should have held. Carried beside the count, never folded
  // into it, and `prefer` treats it as decisive.
  backend::SpillState spill = backend::SpillState::kUnknown;

  // Whether the answer had to substitute for a fact nobody answered, or drop a
  // limit in the direction that understates a request. False means the number
  // is an upper bound, and `prefer` then demands strict improvement rather than
  // admitting a tie between two upper bounds.
  //
  // A missing REGISTER count does not make it false: before a kernel is
  // compiled nobody has one, that applies alike to both sides of a comparison,
  // and `registers_counted` already records it.
  bool exact = false;
  // Whether the register arm took part. False before a kernel is compiled.
  bool registers_counted = false;

  [[nodiscard]] bool seated() const noexcept {
    return workgroups_per_pool != 0;
  }

  [[nodiscard]] std::string describe() const;
};

// THE model. Every limit that can be counted from the facts is counted, the
// answer is their minimum, and a limit whose facts are unknown degrades out of
// the min instead of contributing a plausible number.
[[nodiscard]] Occupancy occupancy(const DeviceCapacity& cap,
                                  const KernelDemand& demand);

// Would `candidate` be at least as good an arrangement as `incumbent`?
//
// Spilling decides first and outright: no occupancy figure buys back a kernel
// whose values live in memory. Then residency, in workgroups per pool. Ties go
// to the candidate, which is what makes a scratch-neutral fusion admissible —
// merging two stages that already share their staged row costs nothing, and a
// rule that demanded strict improvement would refuse it.
//
// When either side is inexact — an allocation granule nobody measured, a pool
// size nobody reported — a tie is only trustworthy when the candidate asks for
// no more scratch than the incumbent, since a rounding step that cannot be
// counted still cannot separate two equal requests. Otherwise the candidate
// must be strictly better, because equality between two upper bounds is not
// equality.
[[nodiscard]] bool prefer(const Occupancy& candidate,
                          const Occupancy& incumbent);

}  // namespace lse::opt
