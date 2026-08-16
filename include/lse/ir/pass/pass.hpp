// The middle end.
//
// A pass rewrites a body and reports how many times it fired. The pipeline is
// not optional and has no switch: a pass that is not correct enough to be on is
// not done. What it may contain is bounded by one rule — comgr runs clang -O3
// over the source this IR lowers to, so constant folding, strength reduction,
// scheduling, register allocation and mem2reg are already done, better than we
// would do them. This middle end exists only for what LLVM CANNOT do, because
// the information is gone by the time HIP text exists:
//
//   - cross-stage structure: fused siblings that each staged the same row into
//     their own `__shared__` array. LLVM sees distinct allocations at distinct
//     addresses separated by a barrier and cannot prove they hold the same
//     bytes. We can, because we know they came from the same source region.
//   - LDS allocation and the workgroup budget it has to fit in.
//   - redundant work across what used to be separate kernel bodies.
//
// If a proposed pass would also be done by -O3, it does not belong here.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/ir/body.hpp"

namespace lse::ir {

class Pass {
 public:
  virtual ~Pass() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  // How many rewrites this made. Zero means the pass did not fire, which is
  // the number worth reporting: a pass that never fires is dead weight.
  virtual std::size_t run(Body& body) const = 0;
};

struct PassStat {
  std::string_view name;
  std::size_t fired = 0;
};

class PassPipeline {
 public:
  PassPipeline& add(std::unique_ptr<Pass> pass);

  // Runs every pass in order and verifies the body afterwards. A pass that
  // leaves the IR malformed fails here rather than in the generated source.
  Status run(Body& body, std::vector<PassStat>* stats = nullptr) const;

  [[nodiscard]] std::size_t size() const noexcept { return passes_.size(); }

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

// The pipeline every kernel body goes through, built once.
[[nodiscard]] const PassPipeline& default_pipeline();

// Cumulative counts since process start, for `--stats`. One row per pass.
[[nodiscard]] std::vector<PassStat> pass_totals();
void record_pass_totals(const std::vector<PassStat>& stats);

}  // namespace lse::ir
