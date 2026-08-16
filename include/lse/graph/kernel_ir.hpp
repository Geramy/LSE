// The kernel IR, under the name the graph layer has always used for it.
//
// The IR itself lives in `lse::ir` — `include/lse/ir/` holds the types,
// values, ops, regions, the affine index vocabulary, the passes and the
// lowering. `kir` is an alias for that namespace, kept because a kernel writes
// `kir::f32` and `kir::Val` and the IR moving out of `graph/` is not a reason
// to edit every kernel.
//
// Two IRs live in this engine and they are at different levels. This one is
// the KERNEL IR: values, ops and regions inside one device function. The other
// is the tensor-level DAG in `lse::graph` — Node, Partitioner, Scheduler,
// Program — which is about which kernels exist and in what order. They must
// not be conflated, which is why they are now in different directories.
#pragma once

#include "lse/ir/recorder.hpp"

namespace lse::graph {

namespace kir = ::lse::ir;

}  // namespace lse::graph
