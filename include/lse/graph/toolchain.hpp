// One dialect's codegen pair, as a device declares it.
//
// A device does not have "an" emitter and "a" compiler: it has a set of
// dialects it can generate and build, and the two halves of one dialect belong
// together — text an emitter wrote is only ever fed to the compiler declared
// beside it. Keeping them in one struct is what makes a second dialect an extra
// entry rather than a second pair of accessors and a rule about which to use.
//
// The list is declared by the backend, in preference order. Nothing selects a
// dialect by name or by build flag: a caller either takes the first entry,
// which is what a caller with no opinion wants, or looks one up and finds it
// absent, which is the whole of the negotiation.
#pragma once

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

}  // namespace lse::graph
