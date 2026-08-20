// Which shape to ask a launch for, priced on the two things a shape trades.
//
// THE RULE, in one sentence: take the arrangement whose launch traffic, charged
// at the share of the memory system its own residency collects, is least — and
// where two are equal, the one that seats more workgroups on a pool, then the
// one that asks the pool for less.
//
// THE FIGURE, and why it is the right one. A tile that covers more rows divides
// one term of the traffic and multiplies the workgroup's scratch, so residency
// and traffic move in opposite directions and neither alone answers. They meet
// in TIME: a launch takes its bytes divided by the rate it collects them at,
// and the rate it collects them at is a fraction of the device's streaming
// rate set by how much of the machine its own residency keeps busy. That
// fraction is backend::ResidencyBandwidth, measured.
//
// The absolute rate never appears, and does not need to: every arrangement of
// one launch divides by the same GB/s, so it cancels out of the comparison. The
// figure that survives is bytes — the launch's traffic charged at what its
// residency can collect — which is a count, not a modelled time. Calling it a
// time would mean claiming an accuracy the model does not have; calling it
// charged bytes says exactly what was done to the number.
//
// EVERY CLASS OF TRAFFIC IS CHARGED, not the one the tile divides. That is the
// whole reason a residency-blind rule and a traffic-blind rule are both wrong.
// A contraction reads about eight times as much activation as weight, and the
// activation side is what the row tile does NOT divide — so once the weight
// side has been divided down near the activation side, the next doubling of the
// tile changes the launch's traffic by almost nothing while its residency keeps
// falling, and the charge starts going up. That is where the ladder stops, and
// it stops there by arithmetic rather than by a cap.
//
// MEASURED, on the eight real shapes this engine emits: the rung this picks is
// the fastest of its ladder on six of them and never slower than the rung a
// fit-only gate picked on any of them. The two it misses it misses by 8% and
// 24% of one kernel's time, and both are shapes whose weight plane is small
// enough to sit in the part's last-level cache — an operand served from cache
// is not moved at the streaming rate this charges it at, and nothing in this
// engine has measured that rate. That is the next fact this model wants.
//
// WHAT DEGRADES, AND WHICH WAY. A device with no measured residency curve is
// charged nothing for residency: the arm drops out, the comparison is on
// traffic alone, and the answer is the largest tile — which is what a fit-only
// gate already gives, so an unmeasured part keeps the behaviour it had rather
// than getting one derived from another part's measurements. An arrangement
// whose traffic was never stated is not priced at all and never wins.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "lse/opt/occupancy.hpp"
#include "lse/opt/traffic.hpp"

namespace lse::opt {

// One way the caller could shape the launch. The traffic is per workgroup with
// `workgroups` set, exactly as a primitive states it; the demand is what one
// workgroup asks the device for.
struct Arrangement {
  TrafficModel traffic;
  KernelDemand demand;
};

struct ArrangementCost {
  Occupancy residency;
  // Every class, across the whole grid. What the launch moves.
  std::uint64_t launch_bytes = 0;
  // Those bytes divided by the share of the memory system this residency
  // collects. The comparable figure; equal to launch_bytes where the curve is
  // unknown, which is the degradation named in the file comment.
  double charged_bytes = 0.0;
  // Whether the residency arm took part. False means charged_bytes is the raw
  // traffic and the comparison is on traffic alone.
  bool charged = false;
  // Whether the arrangement stated its traffic at all. An unpriced arrangement
  // is not a free one.
  bool stated = false;

  [[nodiscard]] std::string describe() const;
};

[[nodiscard]] ArrangementCost arrangement_cost(const DeviceCapacity& cap,
                                               const Arrangement& a);

// Would `candidate` be at least as good a shape as `incumbent`?
//
// Charged bytes decide. A tie there is a real tie — the two move the same
// traffic and collect it at the same rate — and is settled by residency, then
// by the smaller scratch request: two shapes that seat the same number of
// workgroups are not equal if one leaves the pool room and the other fills it,
// because that room is what the next fusion decision has to spend.
//
// Spilling decides before any of that, for the same reason it does in
// opt::prefer: no byte count buys back a kernel whose values live in memory.
[[nodiscard]] bool prefer(const ArrangementCost& candidate,
                          const ArrangementCost& incumbent);

// The best of the shapes offered, as an index into `candidates`. Returns
// candidates.size() when none of them is seatable or priced.
[[nodiscard]] std::size_t best_arrangement(
    const DeviceCapacity& cap, std::span<const Arrangement> candidates);

}  // namespace lse::opt
