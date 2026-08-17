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
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/graph/dialect_source.hpp"
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
};

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
};

class IKernelCompiler {
 public:
  virtual ~IKernelCompiler() = default;

  virtual Result<std::vector<std::byte>> compile(std::string_view source,
                                                 std::string_view arch) const = 0;

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
