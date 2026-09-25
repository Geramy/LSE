#pragma once

#include "lse/runtime/generator.hpp"
#include "nlohmann/json.hpp"

namespace lse::server::detail {

// Generator snapshots the scheduler's persistent JitCache. Generation and
// session restarts do not reset these counters; they are never request costs.
struct JitTotals {
  std::uint64_t memory_hits = 0;
  std::uint64_t disk_hits = 0;
  std::uint64_t compiles = 0;
  std::uint64_t compile_ns = 0;

  static JitTotals from(const runtime::GenerationStats& stats) {
    return {stats.jit_memory_hits, stats.jit_disk_hits,
            stats.jit_compiles, stats.jit_compile_ns};
  }

  void append_to(nlohmann::json& timings) const {
    timings["jit_memory_hits_total"] = memory_hits;
    timings["jit_disk_hits_total"] = disk_hits;
    timings["jit_compiles_total"] = compiles;
    timings["jit_compile_ms_total"] = static_cast<double>(compile_ns) / 1e6;
  }
};

}  // namespace lse::server::detail
