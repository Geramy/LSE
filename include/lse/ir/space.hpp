// What a kernel covers, as an iteration space rather than a grid.
//
// A grid is a number of workgroups. It cannot be split, so "run this op on
// half the devices" is not expressible and every distribution scheme has to be
// bolted on beside the kernel instead of falling out of it. An iteration space
// is the set of work items plus, per dimension, whether a window of it may be
// handed to somebody else and at what granularity.
//
// The load-bearing field is `kind`. Splitting a `parallel` dimension is free —
// tokens, experts, output columns. Splitting a `reduction` dimension buys a
// collective on every launch, because each window holds a partial sum.
// Splitting a `sequential` dimension does not run concurrently at all: piece
// n+1 consumes piece n, so the pieces pipeline. Without the kind a scheduler
// cannot tell those apart, and every placement decision downstream is a guess.
// One mechanism covers all four kinds of parallelism people name separately:
// splitting the token dimension is data parallel, the expert dimension is
// expert parallel, the layer dimension is pipeline parallel, and a matmul's
// rows or columns are tensor parallel.
//
// When a kernel is emitted for a window rather than the whole space, the
// window's base index arrives in the dispatch constants — `window_base` names
// the field. That base is a RUNTIME value, and it is a different thing from a
// runtime extent: a base moves the origin of an address, so it appears with a
// constant coefficient and the stride stays constant, while an extent used in
// address arithmetic would make the stride itself vary. `ir::verify` enforces
// exactly that distinction.
//
// Nothing declares a space yet. This is the vocabulary, checked by the
// verifier and tested; wiring a kernel's ThreadPlan to it is the work-item
// descriptor on the queue.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lse::ir {

// What splitting a dimension costs.
enum class DimKind : std::uint8_t {
  // Windows are independent. Splitting is free.
  kParallel,
  // Each window produces a partial result. Splitting costs a collective.
  kReduction,
  // Loop-carried: window n+1 consumes window n, so windows pipeline rather
  // than run concurrently.
  kSequential,
};

[[nodiscard]] std::string_view to_string(DimKind k) noexcept;

struct Dim {
  std::string name;
  // Work items along this dimension.
  std::int64_t extent = 0;
  DimKind kind = DimKind::kParallel;
  // Smallest number of items a window may contain. A wave-per-column GEMV
  // splits at one column; a stage that stages into LDS and barriers cannot
  // leave its workgroup, so its granularity is the whole workgroup tile.
  // Zero means this dimension may not be split at all.
  std::int64_t granularity = 0;
  // Dispatch-constants field carrying the window's base index, when this
  // kernel is emitted for a window. Empty means it covers the whole extent.
  std::string window_base;

  [[nodiscard]] bool splittable() const noexcept { return granularity > 0; }
  // How many windows this dimension can be cut into at its granularity.
  [[nodiscard]] std::int64_t max_windows() const noexcept {
    if (granularity <= 0) return 1;
    return (extent + granularity - 1) / granularity;
  }
};

class IterationSpace {
 public:
  void add(Dim d) { dims_.push_back(std::move(d)); }
  [[nodiscard]] std::span<const Dim> dims() const noexcept { return dims_; }
  [[nodiscard]] bool empty() const noexcept { return dims_.empty(); }

  // Total work items. One kernel covering the whole space runs this many.
  [[nodiscard]] std::int64_t items() const noexcept {
    std::int64_t n = dims_.empty() ? 0 : 1;
    for (const Dim& d : dims_) n *= d.extent;
    return n;
  }

  [[nodiscard]] const Dim* find(std::string_view name) const noexcept {
    for (const Dim& d : dims_) {
      if (d.name == name) return &d;
    }
    return nullptr;
  }

 private:
  std::vector<Dim> dims_;
};

}  // namespace lse::ir
