// Device-agnostic kernel codegen contract.
//
// The graph layer knows that a fusion group becomes source and then a code
// object; it does not know which language or which toolchain. A backend that
// can generate kernels supplies both halves, and one that cannot supplies
// neither — see backend::IBackend::emitter / ::compiler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backend/census.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/dialect_source.hpp"
#include "lse/opt/traffic.hpp"
#include "lse/ir/spell.hpp"

namespace lse::graph {

class Node;
using NodePtr = std::shared_ptr<Node>;
struct FusionGroup;

// Spelled by the IR, which is where a literal's text belongs.
using ir::float_literal;

// Layout of the dispatch constants block. HRX separates buffer bindings from a
// flat push-constant block; the emitter builds this and bakes the matching
// signature into the generated source, so the two cannot drift.
struct ConstantsLayout {
  struct Field {
    std::string name;
    std::uint16_t offset;
    std::uint8_t size;
  };

  std::vector<Field> fields;
  std::uint32_t total_bytes = 0;

  std::uint16_t add(std::string name, std::uint8_t size);
};

struct EmittedKernel {
  std::string source;
  // What one workgroup of this launch means to move, by operand class. Unstated
  // where no stage of the run could say — never a zero, which would read as a
  // kernel that touches nothing.
  opt::TrafficModel traffic;
  // Which language `source` is in, so the text carries its dialect instead of
  // the caller remembering which emitter produced it. Must equal the emitting
  // IKernelEmitter::dialect(); the default is the dialect of the only emitter
  // that predates this field.
  Dialect dialect = Dialect::kHip;
  std::string entry_name;
  ConstantsLayout constants;
  std::vector<NodePtr> binding_order;
  backend::LaunchDims dims;
  std::uint32_t lds_bytes = 0;
  // Workspace for values the phase produces and consumes itself. Not a
  // launch argument per tensor: one buffer, offsets baked into the source.
  std::size_t scratch_bytes = 0;
  // Kernel takes `const float* const* buf` and binding_order[i] is buf[i].
  bool pointer_table = false;
  // Dependent stages run as a resident grid; last binding is the grid barrier.
  //
  // A LANDMINE for a second dialect: a resident grid is a grid-wide barrier,
  // and Loom refuses grid-wide sync by design — the persistent-grid path can
  // never be Loom, whatever else it gains. Harmless as it stands because
  // nothing sets this true, so a Loom kernel never reaches the barrier
  // binding; a dialect that sets it must first say how it synchronizes.
  bool persist_grid = false;
};

// Folding a run of nodes into one launch body. Which nodes an emitter can
// carry as a stage, and how wide each would be if it owned the launch, are
// properties of that emitter's lowering — the graph asks, it does not decide.
//
// An emitter whose lowering has no staged form declines by returning nullptr
// from IKernelEmitter::staging(). The scheduler then gives every node a group
// of its own, which is already the path a declined node takes.
class IPhaseStaging {
 public:
  virtual ~IPhaseStaging() = default;

  // Can `n` be one stage of a phase body this emitter writes?
  [[nodiscard]] virtual bool can_stage(const Node& n) const noexcept = 0;

  // Independent work items the node could spend if it owned the launch. The
  // phase splitter breaks a chain here: a stage wanting thousands of them
  // cannot share the one-workgroup fallback its dependent neighbours need.
  [[nodiscard]] virtual std::uint32_t stage_threads(
      const Node& n, const backend::DeviceInfo& device) const = 0;

  // Would thread i of this stage touch element i and nothing else? No gather,
  // no reduction, no primitive that owns its indexing, no store back into an
  // input.
  [[nodiscard]] virtual bool lane_stage(const Node& n) const noexcept = 0;

  // Does `consumer` read `producer`'s output at the index the producing thread
  // wrote, over the same element count? Both are then one flat space under one
  // map, so a read-after-write between them never leaves the thread: a
  // workgroup barrier orders it and the two can share one fat grid. Every
  // other dependence needs the grid-wide barrier a launch boundary is, which
  // is why the splitter breaks the chain there and not here.
  [[nodiscard]] virtual bool lane_aligned(
      const Node& producer, const Node& consumer) const noexcept = 0;
};

// Workgroup size and launch count for a group with no primitive of its own.
//
// ENGINE POLICY, not a backend's: it counts elements, threads and residency,
// and nothing it counts requires knowing the target's instruction set or the
// language the body will be written in. It lived in two emitters, restated
// verbatim, with a test pinning the copies together — a seam kept in step by
// hand is a seam that has already decided it should be one function.
[[nodiscard]] backend::LaunchDims choose_launch_dims(
    const FusionGroup& group, const backend::DeviceInfo& device,
    std::uint32_t lds_bytes);

class IKernelEmitter {
 public:
  virtual ~IKernelEmitter() = default;

  virtual Result<EmittedKernel> emit(const FusionGroup& group,
                                     const backend::DeviceInfo& device) const = 0;

  // JIT identity for this group on this device. Must change when generated
  // source would change without FusionGroup::signature() changing (a
  // specialized primitive). Arch is mixed in by the cache, not here.
  [[nodiscard]] virtual std::uint64_t cache_key(
      const FusionGroup& group, const backend::DeviceInfo& device) const;

  // Dialect of EmittedKernel::source, and of the source a primitive must
  // supply to land in it.
  [[nodiscard]] virtual Dialect dialect() const noexcept = 0;

  // Declarations every kernel this emitter produces may rely on: the target's
  // runtime header, the dispatch-constants struct. Kernel primitives that own
  // a whole translation unit are prefixed with it.
  [[nodiscard]] virtual std::string_view prelude() const noexcept = 0;

  // How this backend spells each built-in primitive. Primitives carry no
  // device text of their own unless they are written against an intrinsic.
  [[nodiscard]] virtual DialectSourceTable sources() const noexcept = 0;

  // nullptr when this emitter writes one node per launch and has no staged
  // phase body to fold them into.
  [[nodiscard]] virtual const IPhaseStaging* staging() const noexcept {
    return nullptr;
  }

  // What a candidate run of sibling stages would cost in workgroup scratch,
  // both ways, so the engine can compare their residency before the run is
  // formed.
  //
  // THE BACKEND REPORTS BYTES; THE ENGINE COUNTS RESIDENCY. Which arrays a
  // merged body declares, which a solo body declares, and whether a hoisted
  // panel relieves a stage of its own staging are all facts about this
  // emitter's lowering. What those bytes are worth in resident workgroups is
  // arithmetic over the device's facts, and that stays in lse::opt.
  //
  // BOTH FIGURES MUST BE THE SAME QUANTITY — the sum of the workgroup-shared
  // arrays in the body that would be printed — or the comparison means
  // nothing. Pricing the unfused arrangement as the fused one's hoisted panel
  // makes every candidate a tie, which is exactly the state this replaced.
  //
  // `fused == 0` or `threads == 0` is NO ANSWER, not a free run: an emitter
  // that would not write this run as one body has nothing to report, and the
  // engine keeps the arrangement it had.
  struct RunScratch {
    std::uint32_t threads = 0;
    std::uint32_t fused = 0;
    std::uint32_t worst_solo = 0;
    // Entry names the two arrangements would be compiled as, so a decision
    // can be scored from previous compiles instead of from these counts.
    // Empty asks nothing of the measurement table.
    std::string fused_entry;
    std::vector<std::string> solo_entries;
  };
  [[nodiscard]] virtual RunScratch run_scratch(
      std::span<const NodePtr> run,
      const backend::DeviceInfo& device) const {
    (void)run;
    (void)device;
    return {};
  }
};

// A compiled kernel and what its compiler said about it. The resources travel
// with the bytes rather than behind a second query: the point they are cheapest
// to read is the point the object was produced, and a separate call would be a
// second thing to remember and a second thing to skip on a cache hit.
struct CompiledKernel {
  std::vector<std::byte> code;
  // One entry per kernel symbol the object defines. Empty when the toolchain
  // reports nothing measurable, which is a real answer and not a failure.
  std::vector<backend::KernelResources> resources;
  // What the emitted body counts up to, one entry per kernel symbol, read off
  // the object the same way and at the same moment. A compiler that cannot
  // disassemble its own output leaves this empty.
  std::vector<backend::KernelCensus> census;

  // Resources for `entry`, or nullptr. When the object defines exactly one
  // kernel an empty name matches it, so a caller that never named its entry
  // still gets the numbers.
  [[nodiscard]] const backend::KernelResources* resources_for(
      std::string_view entry) const noexcept;
  [[nodiscard]] const backend::KernelCensus* census_for(
      std::string_view entry) const noexcept;
};

class IKernelCompiler {
 public:
  virtual ~IKernelCompiler() = default;

  // The bytes, plus whatever the toolchain reports about them. A compiler with
  // no metadata to offer returns an empty `resources` — never a row of zeros.
  virtual Result<CompiledKernel> compile(std::string_view source,
                                         std::string_view arch) const = 0;

  // What the instructions in an object this compiler produced add up to.
  //
  // Separate from compile() because the object outlives the compile: a warm
  // cache hands back bytes nobody compiled this run, and the counts have to be
  // recoverable from those bytes alone or a cached kernel would be a kernel the
  // engine knows nothing about. Empty when this compiler cannot read back what
  // it wrote.
  [[nodiscard]] virtual std::vector<backend::KernelCensus> census(
      std::span<const std::byte> object) const;

  [[nodiscard]] virtual bool available() const = 0;

  // Everything that changes the bytes this compiler produces from identical
  // source: its version, its option list, its action pipeline. It goes in the
  // JIT cache key, so an upgraded toolchain or an edited flag invalidates
  // stale objects on its own. The alternative is a hand-maintained revision
  // constant, which is one forgotten increment away from serving an object
  // built by a different compiler.
  [[nodiscard]] virtual std::string identity() const = 0;
};

}  // namespace lse::graph
