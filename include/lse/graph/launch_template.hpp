// A step's launch plan with its nodes named by position: the cut into launch
// groups, the phases with their activation slots, and each launch's
// description (Scheduler, eval_step). A request rebuilds its pass graph from
// scratch, and splitting, planning and describing that graph again is host
// time, most of it before the first launch (gfx1201, a 304-token prefill:
// phases 2.8 ms, launch cut 2.9, slot plan 3.0, descriptions 8.6). A graph
// whose key equals a recorded one takes the recorded plan instead, its
// descriptions written as its launches go out; its workspace is allocated
// fresh, as a split one's is. Templates are recorded by a step that split its
// graph and by Scheduler::prepare, so a server's first request of each
// prepared shape takes one.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lse/graph/codegen.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/workgroup.hpp"

namespace lse::graph {

// Every node reachable from `roots`, inputs before their consumers, in the
// order a depth-first walk over `roots` and each node's inputs meets them:
// two graphs of the same structure list their nodes in the same order.
[[nodiscard]] std::vector<NodePtr> template_nodes(std::span<const NodePtr> roots);

// The position of each node in `nodes`.
[[nodiscard]] std::unordered_map<const Node*, std::uint32_t> node_positions(
    std::span<const NodePtr> nodes);

// The complete structural key of a step over `nodes` (template_nodes of
// `roots`), written out field by field and never hashed, so two steps share a
// key only when every field below is equal; `context` names the device, the
// toolchain and the engine build. A node contributes its op, fusion class and
// kernel primitive, shape, dtype, float and integer attributes, inputs by
// position, the flags the partitioner and the emitter read (materialized,
// owned storage, host or device dirty, proposal-only, member, and the
// consumer count of a node the step computes), its quantized planes and
// activation panels, and its buffer: whether it has one, its window and
// residency, and which earlier node's allocation it shares. What it does not contribute is any address, so a graph built over
// another request's state takes the same key as the one before it.
[[nodiscard]] std::string launch_template_key(std::span<const NodePtr> nodes,
                                              std::span<const NodePtr> roots,
                                              std::string_view context);

struct LaunchTemplate {
  struct Launch {
    std::vector<std::uint32_t> nodes, inputs, outputs;
    OpKind anchor = OpKind::kCustom;
    FusionClass anchor_class = FusionClass::kLeaf;
    std::uint32_t launches = 1;
    bool is_phase = false;
    // A kernel launch, as opposed to a group of views that launches nothing.
    bool kernel = false;
    // Its description as the emitter wrote it, without source text and with
    // the binding order held by position in `binding`.
    EmittedKernel emitted;
    std::vector<std::uint32_t> binding;
    std::uint64_t ident = 0;
    bool aliases = false;
    // The variant decision it was emitted under: the variant book's key, or 0
    // where the kernel offers one variant.
    std::uint64_t variant_key = 0;
    std::uint32_t variant = 0;
  };
  std::string key;
  std::vector<Launch> launches;
  std::vector<WorkgroupShape> phases;
};

// The group `launch` names over `nodes`; nullopt when it names a position
// past them.
[[nodiscard]] std::optional<FusionGroup> template_group(
    const LaunchTemplate::Launch& launch, std::span<const NodePtr> nodes);

// `launch`'s description over `nodes`; nullopt when its binding order names a
// position past them.
[[nodiscard]] std::optional<EmittedKernel> template_emission(
    const LaunchTemplate::Launch& launch, std::span<const NodePtr> nodes);

// Equality of two launch groups and of two launch descriptions, field by
// field and node by node: what a template applied to a graph must give
// against a fresh split and emit of that graph.
[[nodiscard]] bool same_launch_group(const FusionGroup& a, const FusionGroup& b);
[[nodiscard]] bool same_emission(const EmittedKernel& a, const EmittedKernel& b);
// The first field in which two descriptions differ, or empty when they agree.
[[nodiscard]] std::string emission_difference(const EmittedKernel& a, const EmittedKernel& b);

}  // namespace lse::graph
