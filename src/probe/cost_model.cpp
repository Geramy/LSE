#include "lse/probe/cost_model.hpp"

#include <algorithm>
#include <cmath>

namespace lse::probe {

namespace {

Cost from(const Measured& m) noexcept { return Cost{m.value, m.provenance}; }

// A term the work does not have costs nothing and claims nothing about the
// hardware, so it must not drag the combined provenance down.
constexpr Cost kFree{0.0, Provenance::kMeasured};

}  // namespace

Work Work::scaled(double fraction) const noexcept {
  const double f = fraction < 0.0 ? 0.0 : fraction;
  Work w = *this;
  w.bytes_streamed = static_cast<std::size_t>(
      static_cast<double>(bytes_streamed) * f);
  w.flops = static_cast<std::uint64_t>(static_cast<double>(flops) * f);
  w.bytes_moved = static_cast<std::size_t>(
      static_cast<double>(bytes_moved) * f);
  // Residency scales with the share for the same reason the stream does: a
  // member given a third of the columns holds a third of the weights. Truncating
  // rounds it down, which keeps a share inside a capacity it was solved against.
  w.bytes_resident = static_cast<std::size_t>(
      static_cast<double>(bytes_resident) * f);
  return w;
}

Work matmul_work(std::size_t m, std::size_t n, std::size_t k, DType storage) {
  Work w;
  w.bytes_streamed = dtype_storage_bytes(storage, n * k);
  w.flops = static_cast<std::uint64_t>(2ull * m * n * k);
  w.operand = operand_of_storage(storage);
  w.bytes_moved = w.bytes_streamed;
  // The weight block is read once per pass and held for every pass, so for a
  // matmul the streamed and resident figures are the same bytes. They stop
  // agreeing the moment a caller describes work that re-reads them.
  w.bytes_resident = w.bytes_streamed;
  w.launches = 1;
  return w;
}

bool CostModel::runs_natively(math::MatrixElem operand,
                              const DeviceId& device) const noexcept {
  const DeviceProfile* d = pool_->device(device);
  if (d == nullptr) return false;
  const ComputePath* p = d->path_for(operand);
  return p != nullptr && p->native && p->flops.positive();
}

Cost CostModel::compute_cost(const Work& work,
                             const DeviceId& device) const noexcept {
  const DeviceProfile* d = pool_->device(device);
  if (d == nullptr) return Cost::unknown();

  const Cost stream =
      work.bytes_streamed == 0 ? kFree : from(d->stream_ns(work.bytes_streamed));

  Cost flop = kFree;
  if (work.flops != 0) {
    const ComputePath* p = d->path_for(work.operand);
    if (p == nullptr || !p->flops.positive()) return Cost::unknown();
    flop = Cost::of(static_cast<double>(work.flops) * 1e9 / p->flops.value,
                    p->flops.provenance);
  }

  Cost launch = kFree;
  if (work.launches != 0) {
    if (!d->launch_overhead_ns.known()) return Cost::unknown();
    launch = Cost::of(d->launch_overhead_ns.value * work.launches,
                      d->launch_overhead_ns.provenance);
  }

  if (!stream.known() || !flop.known() || !launch.known()) {
    return Cost::unknown();
  }
  // Roofline: a kernel is bound by whichever of the two it cannot overlap away,
  // and the dispatch is paid on top of both.
  return Cost::of(std::max(stream.ns, flop.ns) + launch.ns,
                  weaker(weaker(stream.provenance, flop.provenance),
                         launch.provenance));
}

Capacity CostModel::capacity_for(const Work& work,
                                 const DeviceId& device) const noexcept {
  Capacity c;
  c.bytes_resident = work.bytes_resident;
  const DeviceProfile* d = pool_->device(device);
  // A device that is not a member is one this pool has measured nothing about,
  // so it gets the non-answer compute_cost gives it rather than a verdict. In
  // particular not the zero-byte fit below: `fits()` names a device, and a
  // device the pool has never seen cannot be the subject of a true statement
  // about its memory.
  if (d == nullptr) return c;
  // Provenance says who wrote the number down; it does not say the number is a
  // byte count. A negative or non-finite free-memory figure is unfounded
  // whatever it is labelled, and it must not be arithmetic'd on: NaN compares
  // false against every bound, so reading it as a measurement makes a device
  // look empty to one comparison and full to the other.
  const bool founded = d->free_memory.known() &&
                       std::isfinite(d->free_memory.value) &&
                       d->free_memory.value >= 0.0;
  // Work that has to hold nothing fits anywhere. That is a fact about the work,
  // not a claim about how much room the device has, so it is settled before the
  // free-memory figure is consulted and holds whether or not there is one.
  if (work.bytes_resident == 0) {
    c.fit = Fit::kFits;
    c.bytes_free = founded ? d->free_memory.value : 0.0;
    c.provenance = founded ? d->free_memory.provenance : Provenance::kMeasured;
    return c;
  }
  if (!founded) return c;
  c.bytes_free = d->free_memory.value;
  c.provenance = d->free_memory.provenance;
  c.fit = static_cast<double>(work.bytes_resident) <= c.bytes_free
              ? Fit::kFits
              : Fit::kExceeds;
  return c;
}

Cost CostModel::transfer_cost(std::size_t bytes, const DeviceId& src,
                              const DeviceId& dst) const noexcept {
  if (src == dst) return Cost::of(0.0, Provenance::kMeasured);
  const LinkProfile* l = pool_->link(src, dst);
  if (l == nullptr) return Cost::unknown();
  return from(l->cost_ns(bytes));
}

double bubble_fraction(std::size_t stages, std::uint32_t depth) noexcept {
  if (stages <= 1 || depth == 0) return 0.0;
  const double s = static_cast<double>(stages - 1);
  return s / (s + static_cast<double>(depth));
}

Throughput CostModel::pipeline_throughput(std::span<const Stage> stages,
                                          std::uint32_t depth) const {
  Throughput t;
  if (stages.empty() || depth == 0) return t;

  double sum = 0.0;
  double peak = 0.0;
  Provenance provenance = Provenance::kMeasured;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const Cost compute = compute_cost(stages[i].work, stages[i].device);
    const DeviceId& next = stages[(i + 1) % stages.size()].device;
    const Cost move =
        transfer_cost(stages[i].bytes_to_next, stages[i].device, next);
    if (!compute.known() || !move.known()) return t;
    const double stage_ns = compute.ns + move.ns;
    sum += stage_ns;
    peak = std::max(peak, stage_ns);
    provenance =
        weaker(provenance, weaker(compute.provenance, move.provenance));
  }
  // Items overlap: one is in stage k while the next is in stage k-1. At depth 1
  // the sum is the whole cost and every transfer is on the critical path; as
  // depth rises only the slowest stage still is.
  const double makespan = sum + static_cast<double>(depth - 1) * peak;
  if (makespan <= 0.0) return t;
  t.items_per_s = static_cast<double>(depth) * 1e9 / makespan;
  t.item_latency_ns = sum;
  t.bubble = bubble_fraction(stages.size(), depth);
  t.provenance = provenance;
  return t;
}

CostModel::OffloadDecision CostModel::should_offload(
    const Work& work, const DeviceId& home, const DeviceId& candidate,
    std::size_t bytes_out, std::size_t bytes_back, std::uint32_t depth) const {
  OffloadDecision d;
  const Cost there = transfer_cost(bytes_out, home, candidate);
  const Cost back = transfer_cost(bytes_back, candidate, home);
  if (there.known() && back.known()) {
    d.transfer = Cost::of(there.ns + back.ns,
                          weaker(there.provenance, back.provenance));
  }

  d.home_capacity = capacity_for(work, home);
  d.candidate_capacity = capacity_for(work, candidate);

  const Stage staying[] = {Stage{home, work, 0}};
  d.stay = pipeline_throughput(staying, depth);

  if (home == candidate) {
    d.reason = "the candidate is where the work already is";
    return d;
  }

  // Two stages: home hands the operands on, the candidate runs it and returns
  // the answer. The wrap puts the return leg on the candidate's stage, which
  // is where it is actually paid. The handoff holds nothing of its own, so its
  // residency is zero rather than the work's.
  Work idle;
  idle.launches = 0;
  const Stage moving[] = {Stage{home, idle, bytes_out},
                          Stage{candidate, work, bytes_back}};
  d.moved = pipeline_throughput(moving, depth);

  // Capacity settles first and outranks every throughput term below it. A
  // device that cannot hold the work is not a slow placement, it is not a
  // placement, so no margin and no measured advantage gets to overrule it.
  if (d.candidate_capacity.fit == Fit::kExceeds) {
    d.reason = "the candidate cannot hold what this work needs resident";
    return d;
  }
  if (d.home_capacity.fit == Fit::kExceeds && d.candidate_capacity.fits()) {
    // The capacity-bound regime, where the split is mandatory rather than an
    // optimization: staying put is not the cheaper answer, it is not an answer.
    d.relocate = true;
    d.reason = "home cannot hold this work resident and the candidate can; "
               "capacity decides this, not throughput";
    return d;
  }
  if (!d.home_capacity.known() || !d.candidate_capacity.known()) {
    d.reason = "free memory is unknown at one end, so whether the work fits "
               "there is unknown and nothing is placed on it";
    return d;
  }

  if (!d.stay.known() || !d.moved.known()) {
    d.reason = "a term is unknown; nothing moves on a number nobody measured";
    return d;
  }
  if (d.moved.items_per_s <= d.stay.items_per_s * (1.0 + kThroughputMargin)) {
    d.reason = "moving does not raise throughput at this queue depth";
    return d;
  }
  d.relocate = true;
  d.reason = "measured: throughput at this depth is higher on the candidate";
  return d;
}

CostModel::Halves CostModel::halves(const Work& work, const DeviceId& device,
                                    const DeviceId& home,
                                    double fraction) const {
  Halves h;
  const Work share = work.scaled(fraction);
  const Cost compute = compute_cost(share, device);
  const Cost carry = transfer_cost(share.bytes_moved, home, device);
  if (!compute.known() || !carry.known()) return h;
  h.compute_ns = compute.ns;
  h.transfer_ns = carry.ns;
  h.provenance = weaker(compute.provenance, carry.provenance);
  return h;
}

double CostModel::Halves::per_item_ns(std::uint32_t depth) const noexcept {
  const double sum = transfer_ns + compute_ns;
  const double peak = std::max(transfer_ns, compute_ns);
  const double makespan = sum + static_cast<double>(depth - 1) * peak;
  return makespan / static_cast<double>(depth);
}

CostModel::Split CostModel::split(const Work& work,
                                  std::span<const DeviceId> devices,
                                  const DeviceId& home,
                                  std::uint32_t depth) const {
  if (depth == 0) depth = 1;
  Split out;
  out.shares.reserve(devices.size());
  for (const DeviceId& d : devices) {
    out.shares.push_back(Share{d, 0.0, Cost::unknown(), Capacity{}});
  }

  // Per-item time is monotone in the share but not affine — the overlap of a
  // member's transfer with its own compute puts a max() in it — so the split is
  // solved rather than inverted: bisect the common per-item time, ask every
  // member for the largest share it can carry inside it, and move the target
  // until the shares add to one.
  //
  // Capacity enters as a per-member ceiling on that answer. It is downward
  // closed in the fraction, so it needs no search of its own: a member with
  // `free` bytes can hold at most free/bytes_resident of the work, and the time
  // solve simply never offers it more.
  std::vector<double> cap(devices.size(), 0.0);
  std::vector<bool> live(devices.size(), false);
  Provenance provenance = Provenance::kMeasured;
  Provenance capacity_provenance = Provenance::kMeasured;
  double ceiling = 0.0;
  // Summed in bytes rather than in fractions. Eight devices each holding an
  // eighth of the model is PLAN.md's headline case and 8*(1/8) is not 1.0 in
  // binary; feasibility must not turn on that.
  double holdable_bytes = 0.0;
  bool any_priced = false;
  bool capacity_unknown = false;
  // A member the time model could not price contributes no capacity either, so
  // it weakens a refusal for the same reason an unmeasured one does: the set
  // was not established. Tracked separately because it is a different repair —
  // price the member, rather than measure its memory.
  bool any_unpriced = false;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    const Halves whole = halves(work, devices[i], home, 1.0);
    if (!whole.known()) {
      any_unpriced = true;
      continue;
    }
    any_priced = true;
    const Capacity c = capacity_for(work, devices[i]);
    if (!c.known()) {
      // Not a refusal and not a pass: this member is left out because nobody
      // knows what it has, and the caller is told that below.
      capacity_unknown = true;
      continue;
    }
    capacity_provenance = weaker(capacity_provenance, c.provenance);
    cap[i] = work.bytes_resident == 0
                 ? 1.0
                 : std::min(1.0, c.bytes_free /
                                     static_cast<double>(work.bytes_resident));
    if (cap[i] <= 0.0) continue;
    live[i] = true;
    provenance = weaker(provenance, weaker(whole.provenance, c.provenance));
    // Any live member could take the whole of its own ceiling, so no target
    // above this is ever needed and none below the largest zero-share cost is
    // reachable.
    ceiling = std::max(ceiling, whole.per_item_ns(depth));
    holdable_bytes +=
        std::min(static_cast<double>(work.bytes_resident), c.bytes_free);
  }

  if (!any_priced) {
    out.reason = "no member could be priced; nothing is split on a number "
                 "nobody measured";
    return out;
  }
  if (holdable_bytes < static_cast<double>(work.bytes_resident)) {
    // Two different answers, and conflating them is what makes a planner run
    // work that cannot fit: one is a measured refusal, the other is an absence
    // of measurement that some unmeasured member might still have satisfied.
    // kExceeds is the strong one — no assignment exists at any price, and a
    // caller acts on it by not loading the model — so it may only be said when
    // every member of the set was established.
    if (capacity_unknown) {
      out.reason = "a member's free memory is unknown, so whether any "
                   "assignment fits is unknown";
      return out;
    }
    if (any_unpriced) {
      out.reason = "a member could not be priced, so its capacity never "
                   "entered the sum; whether any assignment fits is unknown";
      return out;
    }
    out.fit = Fit::kExceeds;
    out.provenance = capacity_provenance;
    out.reason = "no assignment fits: the set's measured free memory is less "
                 "than this work needs resident";
    return out;
  }
  if (ceiling <= 0.0) {
    // Priced, feasible, and free: there is no time here to divide, so the
    // bisection below has nothing to solve and no share is more right than
    // any other.
    out.reason = "the work costs nothing on every member that can hold it; "
                 "there is nothing to divide";
    return out;
  }

  // 40 halvings is a relative precision of 1e-12 on a fraction, which is far
  // past anything a share is used for; the loop is nested, so the count is
  // what the whole solve costs.
  constexpr int kHalvings = 40;

  const auto share_within = [&](std::size_t i, double target) {
    const Halves none = halves(work, devices[i], home, 0.0);
    if (!none.known() || none.per_item_ns(depth) > target) return 0.0;
    // Test the ceiling itself rather than approaching it from below. A member
    // pinned exactly at its capacity would otherwise come back a hair short,
    // and the normalization at the end would scale that hair straight back into
    // a share the device cannot hold.
    const Halves full = halves(work, devices[i], home, cap[i]);
    if (full.known() && full.per_item_ns(depth) <= target) return cap[i];
    double lo = 0.0;
    double hi = cap[i];
    for (int step = 0; step < kHalvings; ++step) {
      const double mid = 0.5 * (lo + hi);
      const Halves h = halves(work, devices[i], home, mid);
      if (h.known() && h.per_item_ns(depth) <= target) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return lo;
  };

  double lo = 0.0;
  double hi = ceiling;
  for (int step = 0; step < kHalvings; ++step) {
    const double target = 0.5 * (lo + hi);
    double total = 0.0;
    for (std::size_t i = 0; i < devices.size(); ++i) {
      if (live[i]) total += share_within(i, target);
    }
    if (total >= 1.0) {
      hi = target;
    } else {
      lo = target;
    }
  }

  double total = 0.0;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (!live[i]) continue;
    out.shares[i].fraction = share_within(i, hi);
    total += out.shares[i].fraction;
  }
  if (total <= 0.0) {
    out.reason = "no member can repay its own fixed cost at any share";
    return out;
  }
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (!live[i]) continue;
    // The bisection lands on the target from above, so the shares sum to at
    // least one; normalizing is what makes them a partition rather than a plan
    // that does 100.0001% of the work. It only ever scales down, which is why
    // it cannot push a share past the ceiling it was solved against.
    out.shares[i].fraction /= total;
    if (out.shares[i].fraction > 0.0) {
      out.shares[i].per_item = Cost::of(hi, provenance);
    }
  }
  bool every_share_fits = true;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    // What this member was actually given, which for one that got nothing is
    // the whole work — the verdict on that is what explains why it got nothing.
    out.shares[i].capacity = capacity_for(
        out.shares[i].fraction > 0.0 ? work.scaled(out.shares[i].fraction)
                                     : work,
        devices[i]);
    if (out.shares[i].fraction > 0.0 && !out.shares[i].capacity.fits()) {
      every_share_fits = false;
    }
  }

  // kFits is a claim about these shares, so it is checked against them rather
  // than asserted from the ceilings they were solved against. The two are the
  // same arithmetic read in opposite directions and they are only guaranteed to
  // agree while every input is a founded byte count; a verdict that outranks
  // throughput has to be the one that was verified, not the one that was
  // intended.
  if (!every_share_fits) {
    for (Share& s : out.shares) {
      s.fraction = 0.0;
      s.per_item = Cost::unknown();
    }
    out.reason = "the solved shares do not all fit the members they landed "
                 "on; this set has no partition the model will stand behind";
    return out;
  }

  out.fit = Fit::kFits;
  out.provenance = provenance;
  out.reason = capacity_unknown
                   ? "the returned shares fit; a member whose free memory is "
                     "unknown was left out of the split"
                   : "every share is resident-feasible on the member that got "
                     "it";
  return out;
}

}  // namespace lse::probe
