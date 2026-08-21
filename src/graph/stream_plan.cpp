#include "lse/graph/stream_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>

#include "lse/graph/graph.hpp"

namespace lse::graph {

// Invisible to the planner as well as to the scheduler: the buffer a reshape
// names is the producer's, so a consumer reading it depends on the producer
// directly and the view in between adds nothing to order against.
bool views_only(const FusionGroup& group) noexcept {
  for (const NodePtr& n : group.nodes) {
    if (n->kind != OpKind::kReshape && n->kind != OpKind::kConstant &&
        n->kind != OpKind::kBuffer) {
      return false;
    }
  }
  return !group.nodes.empty();
}

namespace {

struct Region {
  std::uint64_t handle = 0;
  std::size_t begin = 0;
  std::size_t end = 0;

  [[nodiscard]] bool overlaps(const Region& o) const noexcept {
    return handle == o.handle && begin < o.end && o.begin < end;
  }
};

[[nodiscard]] bool region_of(const Node& n, Region* out) noexcept {
  if (!n.buffer.valid() || n.buffer.handle == 0) return false;
  out->handle = n.buffer.handle;
  out->begin = n.buffer.offset;
  out->end = n.buffer.offset + n.buffer.size_bytes;
  return out->end > out->begin;
}

// Output elements stand in for work items: that is what the emitter turns into
// a grid. The width that saturates the machine comes from the live device, so
// the same test says something different on a 4-CU part and a 304-CU one.
[[nodiscard]] std::uint64_t work_items(const FusionGroup& g) noexcept {
  std::uint64_t elements = 0;
  for (const NodePtr& n : g.outputs) {
    if (n) elements += n->element_count();
  }
  return elements;
}

[[nodiscard]] std::uint64_t device_width(
    const backend::DeviceInfo& info) noexcept {
  return static_cast<std::uint64_t>(info.compute_units) *
         static_cast<std::uint64_t>(info.max_threads_per_workgroup);
}

}  // namespace

StreamPlan plan_streams(std::span<const FusionGroup> groups,
                        const backend::StreamCapabilities& caps,
                        const backend::DeviceInfo& info,
                        std::span<const std::size_t> members) {
  const auto n = groups.size();
  StreamPlan plan;
  plan.stream.assign(n, 0);
  plan.record_after.assign(n, 0);
  plan.waits.resize(n);
  plan.cross.resize(n);
  plan.member.assign(n, 0);
  for (std::size_t i = 0; i < n && i < members.size(); ++i) {
    plan.member[i] = members[i];
  }
  const auto member_of = [&plan](std::uint32_t g) { return plan.member[g]; };
  if (n == 0) return plan;
  plan.chain = 0;

  // One stream, a device that cannot run two at once, or a backend on which
  // moving a group could never pay: the plan is the order it is already in,
  // and no event is worth recording. Leaving early is not just the answer, it
  // is the whole answer — the dependency pass below exists to decide
  // placement, and there is no placement to decide.
  if (!caps.may_spread()) {
    plan.chain = static_cast<std::uint32_t>(n);
    return plan;
  }
  const std::uint32_t usable = std::min<std::uint32_t>(
      caps.concurrent_streams, static_cast<std::uint32_t>(n));
  const std::uint64_t width = device_width(info);
  struct Access {
    Region region;
    std::uint32_t group;
    bool write;
  };
  // Live accesses per allocation, not a log of every access in the step. Two
  // rules keep it bounded, and a prefill chunk ladder is long enough that it
  // has to be:
  //   - a write supersedes every live access it overlaps, because a group
  //     ordered after the new write is transitively ordered after whatever the
  //     write displaced;
  //   - readers collapse to one entry per stream, widened to cover them,
  //     because ordering after the latest reader on a stream already orders
  //     you after every earlier reader on it. Readers never depend on each
  //     other, so a weight every group reads stays free.
  // A handle with more distinct live writes than this collapses to their union
  // — coarser, never wrong, and only reached by a buffer being written in many
  // disjoint pieces, which is already a serial pattern.
  constexpr std::size_t kMaxLiveWrites = 32;
  struct Live {
    std::vector<Access> writes;
    std::vector<Access> reads;   // one per stream
    std::vector<std::uint32_t> reader_stream;
  };
  std::unordered_map<std::uint64_t, Live> live;
  std::vector<std::uint32_t> depth(n, 0);
  // Latest group placed on each stream; kNone while the stream is unused.
  constexpr std::uint32_t kNone = ~0u;
  std::vector<std::uint32_t> last_on(usable, kNone);

  // Index of the last group whose buffers could not be read. Everything after
  // it depends on it, which is what lets the accesses before it be dropped:
  // a later conflict with one of them is ordered transitively through it.
  std::uint32_t barrier = kNone;

  std::vector<Region> writes;
  std::vector<Region> reads;
  std::vector<std::uint32_t> deps;
  for (std::uint32_t g = 0; g < n; ++g) {
    const FusionGroup& group = groups[g];
    if (views_only(group)) continue;
    writes.clear();
    reads.clear();
    deps.clear();
    bool opaque = false;

    Region r;
    // A node with no buffer yet is one this dispatch is about to give one to,
    // or alias onto somebody else's. Either way its bytes are not known here,
    // so the group cannot be placed against them and is treated as depending
    // on everything issued so far. A constant is the exception: it is spelled
    // as a literal in the kernel and never aliases device memory.
    auto unknown = [&r](const Node& node) {
      return !region_of(node, &r) && node.kind != OpKind::kConstant;
    };
    for (const NodePtr& m : group.nodes) {
      if (!m) continue;
      if (!region_of(*m, &r)) {
        opaque = true;
        break;
      }
      writes.push_back(r);
      for (const NodePtr& in : m->inputs) {
        if (!in) continue;
        bool internal = false;
        for (const NodePtr& q : group.nodes) {
          if (q.get() == in.get()) {
            internal = true;
            break;
          }
        }
        if (internal) continue;
        if (unknown(*in)) {
          opaque = true;
          break;
        }
        if (region_of(*in, &r)) reads.push_back(r);
      }
      if (opaque) break;
    }
    for (const NodePtr& in : group.inputs) {
      if (opaque) break;
      if (!in) continue;
      if (unknown(*in)) {
        opaque = true;
        break;
      }
      if (region_of(*in, &r)) reads.push_back(r);
    }

    auto note_dep = [&deps](std::uint32_t j) {
      if (std::find(deps.begin(), deps.end(), j) == deps.end()) deps.push_back(j);
    };
    if (barrier != kNone) note_dep(barrier);
    if (opaque) {
      // Waiting for the tail of every stream orders us after everything issued
      // so far, without naming each group individually.
      for (std::uint32_t s = 0; s < usable; ++s) {
        if (last_on[s] != kNone) note_dep(last_on[s]);
      }
    } else {
      for (const Region& rd : reads) {
        auto it = live.find(rd.handle);
        if (it == live.end()) continue;
        for (const Access& a : it->second.writes) {
          if (a.region.overlaps(rd)) note_dep(a.group);
        }
      }
      for (const Region& w : writes) {
        auto it = live.find(w.handle);
        if (it == live.end()) continue;
        // WAW and WAR both land here: any live access to these bytes.
        for (const Access& a : it->second.writes) {
          if (a.region.overlaps(w)) note_dep(a.group);
        }
        for (const Access& a : it->second.reads) {
          if (a.region.overlaps(w)) note_dep(a.group);
        }
      }
    }
    std::sort(deps.begin(), deps.end());

    std::uint32_t d = 0;
    for (std::uint32_t j : deps) d = std::max(d, depth[j] + 1);
    depth[g] = d;
    plan.chain = std::max(plan.chain, d + 1);

    // Placement. Staying on the stream a dependency already sits at the end of
    // is free — the command buffer is in order — so that is always the first
    // choice, and it is what keeps a dependency chain on one stream.
    std::uint32_t target = 0;
    if (deps.empty()) {
      // Free of every event, but not free: the same cost model decides here as
      // below, because a backend whose other streams cost more per launch loses
      // on an unconstrained group exactly as it loses on a moved one. Without
      // this the first group of a step lands on an idle stream and drags its
      // whole dependency chain after it.
      target = 0;
      if (caps.worth_moving(work_items(group), width)) {
        for (std::uint32_t s = 0; s < usable; ++s) {
          if (last_on[s] == kNone) {
            target = s;
            break;
          }
        }
      }
    } else {
      const std::uint32_t latest = deps.back();
      target = plan.stream[latest];
      if (last_on[target] != latest &&
          caps.worth_moving(work_items(group), width)) {
        // Our dependency has been overtaken on its own stream, so staying
        // would queue behind work we do not need. Find somewhere the wait is
        // free, then somewhere idle, and only then give up and stay.
        std::uint32_t moved = kNone;
        for (std::uint32_t s = 0; s < usable && moved == kNone; ++s) {
          if (last_on[s] != kNone &&
              std::find(deps.begin(), deps.end(), last_on[s]) != deps.end()) {
            moved = s;
          }
        }
        for (std::uint32_t s = 0; s < usable && moved == kNone; ++s) {
          if (last_on[s] == kNone) moved = s;
        }
        if (moved != kNone) target = moved;
      }
    }

    // Everything on `target` is ordered before us already; every other stream
    // holding a dependency needs one wait, on its latest such group.
    for (std::uint32_t j : deps) {
      if (member_of(j) != member_of(g)) {
        plan.cross[g].push_back(j);
        continue;
      }
      if (plan.stream[j] == target) continue;
      bool superseded = false;
      for (std::uint32_t& queued : plan.waits[g]) {
        if (plan.stream[queued] != plan.stream[j]) continue;
        superseded = true;
        if (queued < j) queued = j;
        break;
      }
      if (!superseded) plan.waits[g].push_back(j);
    }
    for (std::uint32_t j : plan.waits[g]) plan.record_after[j] = 1;
    plan.waits_total += static_cast<std::uint32_t>(plan.waits[g].size());

    plan.stream[g] = target;
    last_on[target] = g;
    plan.streams_used = std::max(plan.streams_used, target + 1);
    if (opaque) {
      barrier = g;
      live.clear();
    } else {
      for (const Region& w : writes) {
        Live& slot = live[w.handle];
        std::erase_if(slot.writes,
                      [&w](const Access& a) { return a.region.overlaps(w); });
        for (std::size_t i = slot.reads.size(); i-- > 0;) {
          if (!slot.reads[i].region.overlaps(w)) continue;
          slot.reads.erase(slot.reads.begin() + static_cast<std::ptrdiff_t>(i));
          slot.reader_stream.erase(slot.reader_stream.begin() +
                                   static_cast<std::ptrdiff_t>(i));
        }
        slot.writes.push_back({w, g, true});
        if (slot.writes.size() > kMaxLiveWrites) {
          Access merged = slot.writes.front();
          for (const Access& a : slot.writes) {
            merged.region.begin = std::min(merged.region.begin, a.region.begin);
            merged.region.end = std::max(merged.region.end, a.region.end);
            merged.group = std::max(merged.group, a.group);
          }
          slot.writes.assign(1, merged);
        }
      }
      for (const Region& rd : reads) {
        Live& slot = live[rd.handle];
        std::size_t at = slot.reads.size();
        for (std::size_t i = 0; i < slot.reader_stream.size(); ++i) {
          if (slot.reader_stream[i] == target) {
            at = i;
            break;
          }
        }
        if (at == slot.reads.size()) {
          slot.reads.push_back({rd, g, false});
          slot.reader_stream.push_back(target);
        } else {
          Access& a = slot.reads[at];
          a.region.begin = std::min(a.region.begin, rd.begin);
          a.region.end = std::max(a.region.end, rd.end);
          a.group = g;
        }
      }
    }
  }
  return plan;
}

}  // namespace lse::graph
