#include "lse/trace/record.hpp"

#include <ctime>

#include <algorithm>

namespace lse::trace {

namespace {

std::uint64_t clock_ns(clockid_t id) noexcept {
  timespec ts{};
  if (clock_gettime(id, &ts) != 0) return 0;
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

}  // namespace

HostClockPair read_host_clocks() noexcept {
  HostClockPair pair;
  pair.monotonic_ns = clock_ns(CLOCK_MONOTONIC);
  pair.boottime_ns = clock_ns(CLOCK_BOOTTIME);
  return pair;
}

Result<double> routing_skew(const CounterSet& counts) {
  if (counts.bins.empty()) {
    return LSE_ERROR(kUnimplemented,
                     "no routing histogram: nothing has recorded expert "
                     "selections");
  }
  std::uint64_t sum = 0;
  std::uint64_t peak = 0;
  for (const std::uint64_t bin : counts.bins) {
    sum += bin;
    peak = std::max(peak, bin);
  }
  if (sum == 0) {
    return LSE_ERROR(kUnimplemented,
                     "routing histogram is all zero: no token was routed");
  }
  const double mean =
      static_cast<double>(sum) / static_cast<double>(counts.bins.size());
  return static_cast<double>(peak) / mean;
}

const GroupRecord* find_group(const TraceData& data, GroupId id) {
  if (!id.valid()) return nullptr;
  const auto it = std::find_if(
      data.groups.begin(), data.groups.end(),
      [&](const GroupRecord& g) { return g.id.cache_key == id.cache_key; });
  return it == data.groups.end() ? nullptr : &*it;
}

}  // namespace lse::trace
