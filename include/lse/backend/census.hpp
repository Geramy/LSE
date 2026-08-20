// What a compiler actually EMITTED, counted off the object it produced.
//
// KernelResources says what the kernel was allocated. This says what it was
// told to do: how many memory accesses of what width, how much arithmetic, and
// how the waits are spaced against the loads. Those are counts, so they are
// engine input — a policy that reads them never learns an instruction name.
// Deriving them from the ISA is the backend's half; deciding anything from them
// is not.
//
// A STATIC COUNT IS NOT A DYNAMIC ONE. Everything here counts instructions in
// the body, not executions of them. An access under a backward branch runs an
// unknown number of times — measured on this tree's own contraction kernels, a
// staging loop's trip count is per-lane divergent, so no static reading of the
// object recovers it — and an access under an exec-mask branch may run zero
// times. Each access class therefore reports how much of it sits under a
// backward branch, so a consumer can tell an exact reading from a partial one
// instead of assuming. Where nothing loops, static and dynamic agree.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lse/backend/resources.hpp"

namespace lse::backend {

// Accesses of one class, split by the width one instruction moves.
//
// WHOSE BYTES. A vector access moves `bytes` for every lane of the wave; a
// scalar access moves them once for the whole wave. The KernelCensus member
// says which, because the two differ by the wavefront size and nothing in this
// struct can tell them apart.
struct AccessCensus {
  struct Width {
    std::uint32_t bytes = 0;
    std::uint32_t count = 0;
    // Of `count`, how many sit inside a backward branch. Those execute an
    // unknown number of times; the rest execute at most once.
    std::uint32_t looped = 0;
  };

  // Ascending by `bytes`, one entry per width the kernel actually uses.
  std::vector<Width> widths;

  [[nodiscard]] std::uint32_t count() const noexcept;
  [[nodiscard]] std::uint32_t looped_count() const noexcept;
  // Bytes per issuing thread (vector) or per wave (scalar), over the whole
  // body. `straight_line_bytes` excludes everything under a backward branch.
  [[nodiscard]] std::uint64_t bytes() const noexcept;
  [[nodiscard]] std::uint64_t straight_line_bytes() const noexcept;
  [[nodiscard]] bool empty() const noexcept { return widths.empty(); }
  [[nodiscard]] bool loops() const noexcept { return looped_count() != 0; }

  void add(std::uint32_t width_bytes, bool in_loop);
};

// One kernel's emitted body, counted.
struct KernelCensus {
  std::string entry;

  // Per lane. Global covers every off-chip space the target names; private is
  // the per-lane scratch a spill or an alloca lands in.
  AccessCensus global_loads;
  AccessCensus global_stores;
  AccessCensus shared_loads;  // workgroup scratch
  AccessCensus shared_stores;
  AccessCensus private_loads;
  AccessCensus private_stores;
  // Per WAVE, not per lane: one issue serves every lane.
  AccessCensus scalar_loads;

  DeviceFact<std::uint32_t> instructions;
  DeviceFact<std::uint32_t> vector_alu;
  DeviceFact<std::uint32_t> scalar_alu;
  // Subsets of vector_alu, counted apart because they land on different
  // throughput roofs: a dot instruction contracts several products in one
  // issue, a matrix instruction contracts a whole fragment cooperatively.
  DeviceFact<std::uint32_t> dot_products;
  DeviceFact<std::uint32_t> fused_multiply_adds;
  DeviceFact<std::uint32_t> matrix_ops;
  // Multiply-accumulate operations the arithmetic above performs, per lane,
  // over the body. A dot instruction contributes the products it contracts and
  // an fma contributes one. It is here rather than derived by the caller
  // because how many products one issue contracts is an ISA fact and a count is
  // all the caller may need to know.
  //
  // AN UPPER BOUND ON THE CONTRACTION'S OWN WORK, deliberately. Arithmetic
  // spent placing a dequantized code or forming an address is a multiply-add
  // too, and nothing in the body separates it from the reduction — the float
  // codec here spends two fma per weight where the reduction needs one. So
  // traffic divided by this is a LOWER bound on bytes per unit of work, which
  // is the direction that makes "the object moves more than intended"
  // decidable and its converse not.
  //
  // Unknown where a matrix instruction is present: what one of those contracts
  // is a property of the wave, not of the lane, and mixing the two would be the
  // signature 32x error of this area.
  DeviceFact<std::uint32_t> multiply_accumulates;
  // Lane-to-lane exchanges. Deliberately NOT shared-memory traffic: on the
  // targets this reads, a lane permute crosses the scratch unit's network
  // without addressing scratch, and counting it as a load overstates the
  // scratch term.
  DeviceFact<std::uint32_t> lane_exchanges;

  DeviceFact<std::uint32_t> branches;
  // Backward branches. Non-zero means part of this body repeats, so the counts
  // above are per traversal of it and not per launch.
  DeviceFact<std::uint32_t> backward_branches;

  // WHERE THE WAITS SIT. A load is only useful once a wait retires it, so the
  // number of loads outstanding when a wait fires is the memory parallelism the
  // emission actually achieved.
  //
  // `memory_waits` counts waits that retire at least one load;
  // `deepest_load_batch` is the most loads any one of them retired; and
  // `serializing_waits` counts the ones that retired exactly one, which is the
  // shape of a body that waits for each load before issuing the next.
  DeviceFact<std::uint32_t> memory_waits;
  DeviceFact<std::uint32_t> deepest_load_batch;
  DeviceFact<std::uint32_t> serializing_waits;

  // Instructions the classifier did not recognise. Coverage is a fact too: a
  // census with a large unclassified share is not evidence of a small one.
  DeviceFact<std::uint32_t> unclassified;

  // Whether every access counted here executes at most once, so the static
  // counts are the whole body's traffic rather than one traversal's.
  [[nodiscard]] bool straight_line() const noexcept {
    return backward_branches.known() && backward_branches.value == 0;
  }

  [[nodiscard]] bool any() const noexcept { return instructions.known(); }

  [[nodiscard]] std::string describe() const;
};

}  // namespace lse::backend
