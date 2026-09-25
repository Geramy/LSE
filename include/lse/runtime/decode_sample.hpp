#pragma once
#include <cstdint>

namespace lse::runtime {
struct DecodeSampleCounters {
  std::uint64_t compiles = 0;
  std::uint64_t disk_loads = 0;
  std::uint64_t partition_passes = 0;
  std::uint64_t host_groups = 0;
};
// Counter equality names actual work. A timer around the replay branch can
// advance by a few nanoseconds even though the partitioner was never invoked.
inline bool warm_decode_sample(const DecodeSampleCounters& before,
                               const DecodeSampleCounters& after) {
  return before.compiles == after.compiles && before.disk_loads == after.disk_loads &&
         before.partition_passes == after.partition_passes && before.host_groups == after.host_groups;
}
}  // namespace lse::runtime
