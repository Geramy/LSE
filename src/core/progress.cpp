#include "lse/core/progress.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace lse::progress {
namespace {
std::mutex g_lock;
Snapshot g_state{"idle", "", -1.0, 0};
}  // namespace

void begin(std::string_view phase, std::string_view detail) {
  // LSE_TIME_LOAD=1: when each phase starts, from the first one, so a slow
  // load names the phase it spent its time in.
  static const bool timed = std::getenv("LSE_TIME_LOAD") != nullptr;
  if (timed) {
    static const auto t0 = std::chrono::steady_clock::now();
    const double at = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[phase] %8.3f s %.*s\n", at,
                 static_cast<int>(phase.size()), phase.data());
  }
  std::lock_guard held(g_lock);
  g_state.phase = phase;
  g_state.detail = detail;
  g_state.fraction = -1.0;
  ++g_state.sequence;
}

void advance(std::uint64_t done, std::uint64_t total, std::string_view detail) {
  std::lock_guard held(g_lock);
  g_state.fraction = total == 0 ? -1.0
                                : static_cast<double>(done > total ? total : done) /
                                      static_cast<double>(total);
  if (!detail.empty()) g_state.detail = detail;
  ++g_state.sequence;
}

Snapshot current() {
  std::lock_guard held(g_lock);
  return g_state;
}

}  // namespace lse::progress
