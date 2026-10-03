// Process-wide load progress: what the engine is doing while it opens the
// devices and loads a model, for a caller that cannot read its stderr (an
// embedding app). Long steps report through here; lse_status reads it.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace lse::progress {

struct Snapshot {
  std::string phase;   // e.g. "opening_devices", "loading_weights", "ready"
  std::string detail;  // free text: a path, a tensor name, a message
  double fraction = -1.0;  // 0..1 within the phase, negative when unknown
  std::uint64_t sequence = 0;  // increments on every report
};

// Starts a phase (fraction unknown until reported).
void begin(std::string_view phase, std::string_view detail = {});
// Progress within the current phase. `done` of `total` units; total 0 leaves
// the fraction unknown.
void advance(std::uint64_t done, std::uint64_t total, std::string_view detail = {});
[[nodiscard]] Snapshot current();

}  // namespace lse::progress
