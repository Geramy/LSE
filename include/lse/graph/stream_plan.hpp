// Stream placement for one scheduling step.
//
// Separate from the scheduler because it is a decision, not a mechanism: it
// reads the backend's declared StreamCapabilities and the groups it is given,
// and answers where each one runs and what it must wait for. Nothing in here
// touches a device, which is also what makes the policy testable without one.
#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>

#include "lse/backend/backend.hpp"

namespace lse::graph {

struct FusionGroup;

// Which stream each group in a step runs on, and where the ordering between
// them has to be spelled out.
//
// The whole step is planned before any of it is dispatched, because an event
// has to be recorded *between* two dispatches to be worth anything: recording
// one lazily, at the moment a consumer asks, would snapshot the producer's
// stream after whatever else has since been queued behind it, and the consumer
// would wait for work it does not depend on. Knowing the plan first is what
// lets a fork actually fork.
//
// Dependencies come from the buffers a group binds, not from the node graph
// alone: slot recycling and in-place ops put two unrelated nodes on one
// allocation, and a write-after-write there is as real as a read-after-write on
// a value. Anything whose buffers are not yet bound is treated as depending on
// everything before it — a cold first step plans as one ordered chain, and the
// steady state, where every buffer is bound, is the one that spreads.
struct StreamPlan {
  std::vector<std::uint32_t> stream;        // per group
  std::vector<std::uint8_t> record_after;   // snapshot this stream after group i
  std::vector<std::vector<std::uint32_t>> waits;  // groups i must wait on
  // Which device each group runs on, and the dependencies that sit on a
  // different one. Sharing a stream index only orders two groups when they
  // share a queue, and two members do not: the same index on each is two
  // command buffers that know nothing about each other. Those edges are held
  // apart from `waits` because they are settled differently -- see the
  // scheduler -- and because an event belongs to the device that recorded it.
  std::vector<std::size_t> member;
  std::vector<std::vector<std::uint32_t>> cross;
  std::uint32_t streams_used = 1;
  std::uint32_t waits_total = 0;
  // Longest chain of dependent groups. The step cannot finish in fewer than
  // this many sequential dispatches however many streams there are, so it is
  // the honest ceiling on what spreading can buy.
  std::uint32_t chain = 1;
};

// A group that dispatches nothing: a reshape is a window onto bytes that
// already exist, so the scheduler skips it and the planner never places it.
// One definition, because two would drift apart and the plan's indices are
// the scheduler's indices.
[[nodiscard]] bool views_only(const FusionGroup& group) noexcept;

// Groups are placed in the order given; index i of every vector is groups[i].
// `members` is which device each group was placed on, parallel to `groups`;
// an empty span means one device holds everything.
// `forced` pins a group to a stream the caller has already chosen -- what a
// device set that places by stream does, where the stream IS the member and the
// planner has no freedom to move it. Empty means the planner places.
[[nodiscard]] StreamPlan plan_streams(std::span<const FusionGroup> groups,
                                      const backend::StreamCapabilities& caps,
                                      const backend::DeviceInfo& info,
                                      std::span<const std::size_t> members = {},
                                      std::span<const std::uint32_t> forced = {});

}  // namespace lse::graph
