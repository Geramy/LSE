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
  // Finalized exports may share one compiled artifact across graph identities.
  bool content_addressed = false;
  std::string structural_entry_name;
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

// Rename a generated self export from the complete source body. Invocation
// bindings and launch metadata remain unchanged; JIT verifies exact source.
[[nodiscard]] bool finalize_source_identity(
    EmittedKernel& emitted, std::string_view entry_prefix = "lse_body_");

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
  // Whether `n` can be a stage of a phase body on `device`: can_stage(n),
  // and the stage body actually emits for this node's shapes on this target.
  // The planner asks this before staging a node, so a node the phase emitter
  // would decline is planned as its own group and dispatch never meets a
  // decline. Defaults to can_stage(n).
  [[nodiscard]] virtual bool can_stage_on(const Node& n,
                                          const backend::DeviceInfo& device) const {
    (void)device;
    return can_stage(n);
  }

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

  // Does thread i of this stage STORE element i, into its own binding? Asks
  // about the store ONLY, and deliberately says nothing about where the stage
  // read from -- that is a different edge, judged on its own.
  //
  // A cooperative reduction is the case that needs the distinction: a coop
  // rms_norm re-derives its whole row inside each workgroup (see the LDS fold
  // in phase_emit) precisely so it depends on no other workgroup, then stores
  // lane-for-lane like any elementwise stage. lane_stage calls it a gather --
  // true of its READ -- and seeding a chunk's lane state from that let one
  // norm poison everything after it.
  //
  // SAFE ONLY WITHOUT SLOT RECYCLING. Judging a producer by its store alone
  // admits merges across a cut boundary, and Workgroup::plan_slots recycles
  // slots at exactly those boundaries; the merged kernel would then carry an
  // intra-launch write-after-read between workgroups.
  [[nodiscard]] virtual bool lane_writes(const Node& n) const noexcept = 0;
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

  // What emit() returns minus the source text: entry, bindings, constants and
  // launch geometry. For a kernel already resident on the device, whose text
  // nothing reads again; a step launching thousands of resident kernels then
  // copies no text. An emitter that keeps no launch description apart from
  // its text writes the whole kernel.
  virtual Result<EmittedKernel> emit_launch(const FusionGroup& group,
                                            const backend::DeviceInfo& device) const {
    return emit(group, device);
  }

  // Launch descriptions kept across processes. An emitter that keeps the
  // description emit_launch() returns under its cache_key() says so here,
  // and can then be handed one the JIT read back from its launch index:
  // `launch` is that description with no source and no bindings, filed under
  // `key`. A warm start then makes a kernel resident and launches it without
  // writing its source at all. False (the default) keeps the JIT from
  // indexing this emitter's kernels, and every start writes them out.
  [[nodiscard]] virtual bool keeps_launches() const noexcept { return false; }
  // False when the description was not taken (wrong dialect, carries text or
  // bindings, or the table is full). The kernel is resident either way; an
  // untaken description is written again the first time it is launched.
  virtual bool adopt_launch(std::uint64_t key, const EmittedKernel& launch) const {
    (void)key;
    (void)launch;
    return false;
  }

  // JIT identity for this group on this device. Must change when generated
  // source would change without FusionGroup::signature() changing (a
  // specialized primitive). Arch is mixed in by the cache, not here.
  // How many variants the group's self-indexed primitive offers for this
  // device (KernelPrimitiveBase::variants); 1 when it has no such primitive.
  [[nodiscard]] virtual std::uint32_t variants(const FusionGroup&,
                                               const backend::DeviceInfo&) const {
    return 1;
  }
  // KernelPrimitiveBase::variant_reassociates for the group's primitive.
  [[nodiscard]] virtual bool variant_reassociates(const FusionGroup&,
                                                  const backend::DeviceInfo&,
                                                  std::uint32_t) const {
    return false;
  }
  [[nodiscard]] virtual std::uint64_t cache_key(
      const FusionGroup& group, const backend::DeviceInfo& device) const;
  // The shape class a variant decision for the group covers on this device
  // (KernelPrimitiveBase::variant_class): the same across a KV pool's growth
  // and a prefill's chunk lengths. The group's own identity by default.
  [[nodiscard]] virtual std::uint64_t variant_class(const FusionGroup& group,
                                                    const backend::DeviceInfo& device) const {
    return cache_key(group, device);
  }

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
    // Workgroups the merged launch would actually run, from the primitives'
    // own ThreadPlan. Not derivable from element counts: a contraction tiles
    // far wider than outputs/threads, and a caller that guessed the grid from
    // element counts read 20 workgroups where the emitted kernel launches 640.
    std::uint32_t workgroups = 0;
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

  // Whether this emitter can write `run` -- sibling nodes reading one input --
  // as one multi-output body. Asked by the planner before it joins them, so a
  // run the emitter cannot express is never planned as one group. True unless
  // the emitter knows otherwise.
  [[nodiscard]] virtual bool joins_run(std::span<const NodePtr> run,
                                       const backend::DeviceInfo& device) const {
    (void)run;
    (void)device;
    return true;
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

  // Re-read metadata from cached bytes after a resource-reader correction.
  [[nodiscard]] virtual std::uint32_t resource_metadata_version() const noexcept {
    return 0;
  }
  [[nodiscard]] virtual std::vector<backend::KernelResources> resources(
      std::span<const std::byte>) const { return {}; }

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
