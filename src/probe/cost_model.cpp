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
  return w;
}

Work matmul_work(std::size_t m, std::size_t n, std::size_t k, DType storage) {
  Work w;
  w.bytes_streamed = dtype_storage_bytes(storage, n * k);
  w.flops = static_cast<std::uint64_t>(2ull * m * n * k);
  w.operand = operand_of_storage(storage);
  w.bytes_moved = w.bytes_streamed;
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

  const Stage staying[] = {Stage{home, work, 0}};
  d.stay = pipeline_throughput(staying, depth);

  if (home == candidate) {
    d.reason = "the candidate is where the work already is";
    return d;
  }

  // Two stages: home hands the operands on, the candidate runs it and returns
  // the answer. The wrap puts the return leg on the candidate's stage, which
  // is where it is actually paid.
  Work idle;
  idle.launches = 0;
  const Stage moving[] = {Stage{home, idle, bytes_out},
                          Stage{candidate, work, bytes_back}};
  d.moved = pipeline_throughput(moving, depth);

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

std::vector<CostModel::Share> CostModel::split(
    const Work& work, std::span<const DeviceId> devices, const DeviceId& home,
    std::uint32_t depth) const {
  if (depth == 0) depth = 1;
  std::vector<Share> shares;
  shares.reserve(devices.size());
  for (const DeviceId& d : devices) {
    shares.push_back(Share{d, 0.0, Cost::unknown()});
  }

  // Per-item time is monotone in the share but not affine — the overlap of a
  // member's transfer with its own compute puts a max() in it — so the split is
  // solved rather than inverted: bisect the common per-item time, ask every
  // member for the largest share it can carry inside it, and move the target
  // until the shares add to one.
  std::vector<bool> live(devices.size(), false);
  Provenance provenance = Provenance::kMeasured;
  double ceiling = 0.0;
  bool any = false;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    const Halves whole = halves(work, devices[i], home, 1.0);
    if (!whole.known()) continue;
    live[i] = true;
    any = true;
    provenance = weaker(provenance, whole.provenance);
    // Any member could take the whole thing, so no target above this is ever
    // needed and none below the largest zero-share cost is reachable.
    ceiling = std::max(ceiling, whole.per_item_ns(depth));
  }
  if (!any || ceiling <= 0.0) return shares;

  // 40 halvings is a relative precision of 1e-12 on a fraction, which is far
  // past anything a share is used for; the loop is nested, so the count is
  // what the whole solve costs.
  constexpr int kHalvings = 40;

  const auto share_within = [&](std::size_t i, double target) {
    const Halves none = halves(work, devices[i], home, 0.0);
    if (!none.known() || none.per_item_ns(depth) > target) return 0.0;
    double lo = 0.0;
    double hi = 1.0;
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
    shares[i].fraction = share_within(i, hi);
    total += shares[i].fraction;
  }
  if (total <= 0.0) return shares;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (!live[i]) continue;
    // The bisection lands on the target from above, so the shares sum to just
    // over one; normalizing is what makes them a partition rather than a plan
    // that does 100.0001% of the work.
    shares[i].fraction /= total;
    if (shares[i].fraction > 0.0) {
      shares[i].per_item = Cost::of(hi, provenance);
    }
  }
  return shares;
}

}  // namespace lse::probe
