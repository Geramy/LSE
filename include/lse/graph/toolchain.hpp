// One dialect's codegen pair, as a device declares it.
//
// A device does not have "an" emitter and "a" compiler: it has a set of
// dialects it can generate and build, and the two halves of one dialect belong
// together — text an emitter wrote is only ever fed to the compiler declared
// beside it. Keeping them in one struct is what makes a second dialect an extra
// entry rather than a second pair of accessors and a rule about which to use.
//
// The list is declared by the backend, in preference order. No build flag
// selects a dialect and none is default-off: a caller either takes the first
// entry, which is what a caller with no opinion wants, or looks one up and
// finds it absent, which is the whole of the negotiation. The lookup is at the
// bottom of this file.
#pragma once

#include <optional>
#include <span>

#include "lse/graph/dialect_source.hpp"

namespace lse::graph {

class IKernelEmitter;
class IKernelCompiler;

struct KernelToolchain {
  Dialect dialect = Dialect::kHip;
  // Either half may be null: a device that can build a dialect it cannot
  // generate, or generate one it cannot build, still describes itself here.
  const IKernelEmitter* emitter = nullptr;
  const IKernelCompiler* compiler = nullptr;
};

// --- the negotiation -------------------------------------------------------
// Two questions and no third: what does this device prefer, and does it have
// the one I want. Nothing here substitutes a dialect for another, because the
// two halves of a toolchain belong together and text one emitter wrote is only
// ever fed to the compiler declared beside it.

// What a caller with no opinion gets: the device's own first choice. nullptr
// when the device declares no dialect at all, which is a device with no
// codegen rather than a device that refused.
[[nodiscard]] constexpr const KernelToolchain* preferred_toolchain(
    std::span<const KernelToolchain> declared) noexcept {
  return declared.empty() ? nullptr : &declared.front();
}

// The declared toolchain for `dialect`, or nullptr when this device does not
// declare it. Absence is the whole answer.
[[nodiscard]] constexpr const KernelToolchain* find_toolchain(
    std::span<const KernelToolchain> declared, Dialect dialect) noexcept {
  for (const KernelToolchain& tc : declared) {
    if (tc.dialect == dialect) return &tc;
  }
  return nullptr;
}

// Which dialect a run would rather its kernels were generated in, or nothing.
// A runtime value naming a resource, the way a device selector names a device:
// unset is the ordinary case and every dialect it can name is already built.
using DialectPreference = std::optional<Dialect>;

// A preference resolved against what one device declares. A named dialect the
// device does not declare degrades to that device's first choice, because the
// preference names a resource a heterogeneous set may hold on some members and
// not others — refusing there would make naming a dialect a way to fail a run
// on the members that lack it.
[[nodiscard]] constexpr const KernelToolchain* resolve_toolchain(
    std::span<const KernelToolchain> declared,
    DialectPreference want) noexcept {
  if (want.has_value()) {
    if (const KernelToolchain* tc = find_toolchain(declared, *want)) return tc;
  }
  return preferred_toolchain(declared);
}

}  // namespace lse::graph
