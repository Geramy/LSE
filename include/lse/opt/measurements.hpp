// What the toolchain reported about a kernel the last time it was built.
//
// A fusion decision is made before the kernel exists, so it cannot know the
// register count of the thing it is deciding about. It can know what the SAME
// arrangement measured last time: the JIT cache persists the numbers beside
// the object, so they survive a restart and arrive without recompiling.
//
// WHY THIS TERMINATES. Every arrangement is looked up under ITS OWN identity —
// the entry name it would be compiled as — not under "the kernel we are
// running". So the occupancy of the fused arrangement rests on the fused
// object's measurement and the occupancy of the unfused one on the unfused
// objects', and each of those is a fixed property of a compiled object. The
// decision is therefore an argmax over a finite set of arrangements whose
// scores do not depend on which arrangement was chosen, and an arrangement can
// change the answer at most once: the first time its own measurement arrives.
// Reading the CURRENT kernel's measurement to decide the NEXT kernel's shape is
// what would oscillate, and is what this deliberately does not do.
#pragma once

#include <cstdint>
#include <string_view>

#include "lse/backend/resources.hpp"

namespace lse::opt {

// Process-wide, because the thing it mirrors is: one on-disk kernel cache
// shared by every device set and every session in this process.
class KernelMeasurements {
 public:
  static KernelMeasurements& instance() noexcept;

  // Identity is the entry name the object defines, which a decision site can
  // spell before any text exists.
  void record(std::string_view entry, const backend::KernelResources& r);

  // Nothing recorded is nothing known — never a row of zeros.
  [[nodiscard]] backend::KernelResources lookup(std::string_view entry) const;
  [[nodiscard]] bool known(std::string_view entry) const;

  [[nodiscard]] std::size_t size() const noexcept;
  void clear();

 private:
  KernelMeasurements() = default;
};

}  // namespace lse::opt
