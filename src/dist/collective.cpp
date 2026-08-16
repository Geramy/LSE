#include "lse/dist/collective.hpp"

#include <array>
#include <limits>

#include "lse/dist/collectives.hpp"
#include "lse/dist/quick_reduce.hpp"

namespace lse::dist {

namespace {

// One rank's share of the traffic, for either algorithm. Both the ring and the
// two-shot move 2*(P-1)/P of the payload per rank; they differ in how many
// bytes each element costs on the wire, not in how many elements move.
double link_ns(const Capabilities& caps, double wire_bytes, Rank world) {
  const double steps = 2.0 * static_cast<double>(world - 1);
  const double per_step = wire_bytes / static_cast<double>(world);
  // A transport with no P2P stages every transfer through a host bounce
  // buffer, so each byte crosses the link twice.
  const double bytes = caps.device_memory_direct ? per_step : per_step * 2.0;
  const double bw = caps.bandwidth_bytes_per_s != 0
                        ? static_cast<double>(caps.bandwidth_bytes_per_s)
                        : 1.0;
  return steps * (static_cast<double>(caps.latency_ns) + bytes * 1e9 / bw);
}

// Two encodes and two decodes per element: once inbound to the segment owner,
// once outbound from it.
double codec_ns(const CodecEngine& engine, DType wire, std::size_t elems) {
  const std::uint64_t rate = engine.throughput_bytes_per_s(wire);
  if (rate == 0) return std::numeric_limits<double>::infinity();
  const double bytes = 2.0 * static_cast<double>(elems) * sizeof(float);
  return bytes * 1e9 / static_cast<double>(rate);
}

// Every wire format the engine might offer. Order is irrelevant — the model
// scores them all — so a new format is one more row and no other change.
constexpr std::array<DType, 4> kWireFormats{DType::kF16, DType::kQ8,
                                            DType::kQ6, DType::kQ4};

}  // namespace

std::string_view to_string(CollectiveAlgo algo) noexcept {
  switch (algo) {
    case CollectiveAlgo::kNative: return "native";
    case CollectiveAlgo::kRing: return "ring";
    case CollectiveAlgo::kTwoShot: return "two-shot";
  }
  return "?";
}

CollectivePlan select_all_reduce(const CollectiveCost& cost,
                                 const CodecEngine* engine) noexcept {
  CollectivePlan plan;
  if (cost.world_size <= 1) {
    plan.reason = "world size 1: nothing to reduce";
    return plan;
  }
  if (cost.caps.native_collectives) {
    plan.algo = CollectiveAlgo::kNative;
    plan.reason = "transport declares native collectives";
    return plan;
  }

  const double payload = static_cast<double>(cost.elems) * sizeof(float);
  plan.algo = CollectiveAlgo::kRing;
  plan.predicted_ns = link_ns(cost.caps, payload, cost.world_size);
  plan.reason = "uncompressed ring is the cheapest model";

  if (payload <= static_cast<double>(cost.caps.latency_bound_threshold())) {
    plan.reason =
        "below the transport's bandwidth-delay product: latency dominates and "
        "a smaller payload buys nothing";
    return plan;
  }
  // A lossy wire only has a defensible error bound under a linear reduction.
  if (cost.op != ReduceOp::kSum && cost.op != ReduceOp::kAvg) {
    plan.reason = "compression bounds error for sum/avg only";
    return plan;
  }
  if (engine == nullptr) {
    plan.reason = "no codec engine on this backend";
    return plan;
  }
  if (!quick_reduce_shape_ok(cost.elems, cost.world_size)) {
    plan.reason =
        "payload does not divide into world_size whole codec blocks; two-shot "
        "will not reshape it";
    return plan;
  }

  for (DType format : kWireFormats) {
    if (!engine->supports(format)) continue;
    const double wire_bytes =
        static_cast<double>(codec_wire_bytes(format, cost.elems));
    const double total = link_ns(cost.caps, wire_bytes, cost.world_size) +
                         codec_ns(*engine, format, cost.elems);
    if (total < plan.predicted_ns) {
      plan.algo = CollectiveAlgo::kTwoShot;
      plan.wire = format;
      plan.predicted_ns = total;
      plan.reason =
          "compressed wire saves more link time than the codec costs";
    }
  }
  return plan;
}

Result<CollectivePlan> plan_all_reduce(const CollectiveContext& ctx,
                                       const CommBuffer& buf, ReduceOp op) {
  if (ctx.transport == nullptr) {
    return LSE_ERROR(kInvalidArgument, "all_reduce needs a transport");
  }
  CollectiveCost cost;
  cost.caps = ctx.transport->capabilities();
  cost.world_size = ctx.transport->world_size();
  cost.elems = buf.bytes / sizeof(float);
  cost.op = op;
  return select_all_reduce(cost, ctx.codec);
}

Status run_all_reduce(const CollectiveContext& ctx, const CollectivePlan& plan,
                      CommBuffer& buf, ReduceOp op) {
  if (ctx.transport == nullptr) {
    return LSE_ERROR(kInvalidArgument, "all_reduce needs a transport");
  }
  switch (plan.algo) {
    case CollectiveAlgo::kNative:
      return ctx.transport->all_reduce(buf, op);
    case CollectiveAlgo::kRing:
      return ring_all_reduce_over(*ctx.transport, buf, op);
    case CollectiveAlgo::kTwoShot:
      if (ctx.codec == nullptr) {
        return LSE_ERROR(kInvalidArgument,
                         "two-shot all-reduce needs a codec engine");
      }
      return two_shot_all_reduce(*ctx.transport, *ctx.codec, plan.wire, buf,
                                 op);
  }
  return LSE_ERROR(kInternal, "unknown collective algorithm");
}

Status all_reduce(const CollectiveContext& ctx, CommBuffer& buf, ReduceOp op) {
  LSE_ASSIGN_OR(const CollectivePlan plan, plan_all_reduce(ctx, buf, op));
  return run_all_reduce(ctx, plan, buf, op);
}

}  // namespace lse::dist
