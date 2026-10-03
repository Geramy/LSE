#include "lse/core/progress.hpp"

#include <cstdint>
#include <mutex>

namespace lse::progress {
namespace {
std::mutex g_lock;
Snapshot g_state{"idle", "", -1.0, 0};
}  // namespace

void begin(std::string_view phase, std::string_view detail) {
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
