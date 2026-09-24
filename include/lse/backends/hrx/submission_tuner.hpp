#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lse::backend {

// Measures natural, completed decode work. No extra model evaluation, token
// replay, synthetic host pacing, GPU timestamp claim, or persistent pointer key.
// Five interleaved observations per candidate reduce monotonic clock/context
// drift. The baseline survives noise, insufficient evidence, and <5% wins.
class SubmissionTuner {
 public:
  static constexpr std::array<std::uint32_t, 4> candidates{16, 64, 256, 0};
  static constexpr unsigned samples_per_candidate = 5;
  static constexpr unsigned max_attempts = 80;
  struct State {
    bool used = false, done = false, selected = false;
    std::uint64_t key = 0, signature = 0;
    unsigned attempts = 0, warm = 0, samples = 0;
    std::uint32_t interval = 16;
    std::array<std::array<double, samples_per_candidate>, 4> times{};
    std::array<double, 4> medians{};
  };
  State* find(std::uint64_t key) {
    for (auto& state : states_) if (state.used && state.key == key) return &state;
    for (auto& state : states_) if (!state.used) {
      state.used = true;
      state.key = key;
      return &state;
    }
    return nullptr;  // Bound model/shape residency rather than evict active work.
  }
  static unsigned candidate_index(unsigned sample) {
    const unsigned round = sample / candidates.size();
    const unsigned position = sample % candidates.size();
    return round % 2 == 0 ? position : candidates.size() - position - 1;
  }
  static std::uint32_t next(const State& state) {
    if (state.done) return state.interval;
    if (state.warm < 2) return 16;
    return candidates[candidate_index(state.samples)];
  }
  static void observe(State& state, std::uint64_t signature, double nanoseconds,
                      bool eligible) {
    if (state.done && signature == state.signature) return;
    if (signature != state.signature) {
      // A new actual launch shape invalidates both measurements and a cached
      // choice. Retain the attempt budget so unstable graphs cannot tune forever.
      const auto attempts = state.attempts;
      const auto key = state.key;
      state = {};
      state.used = true; state.key = key; state.attempts = attempts;
      state.signature = signature;
      eligible = false;
    }
    if (++state.attempts >= max_attempts) { state.done = true; return; }
    if (!eligible || !std::isfinite(nanoseconds) || nanoseconds <= 0) return;
    if (state.warm < 2) { ++state.warm; return; }
    const unsigned index = candidate_index(state.samples);
    state.times[index][state.samples / candidates.size()] = nanoseconds;
    if (++state.samples != candidates.size() * samples_per_candidate) return;
    state.done = true;
    std::array<bool, candidates.size()> stable{};
    for (unsigned i = 0; i < candidates.size(); ++i) {
      auto sorted = state.times[i];
      std::sort(sorted.begin(), sorted.end());
      const double median = sorted[samples_per_candidate / 2];
      state.medians[i] = median;
      for (auto& value : sorted) value = std::abs(value - median);
      std::sort(sorted.begin(), sorted.end());
      stable[i] = sorted[samples_per_candidate / 2] <= median * 0.05;
    }
    // Compare only measured stable policies. A noisy rejected candidate must
    // not veto an independently stable improvement over a stable baseline.
    if (!stable[0]) return;
    double best = state.medians[0];
    for (unsigned i = 1; i < candidates.size(); ++i)
      if (stable[i]) best = std::min(best, state.medians[i]);
    // Prefer the smallest bounded batch within 2% of the best. Deferred-only
    // (0) is last, because equivalent throughput does not justify later start.
    for (unsigned i = 0; i < candidates.size(); ++i) {
      if (stable[i] && state.medians[i] <= best * 1.02 &&
          state.medians[i] <= state.medians[0] * 0.95) {
        state.interval = candidates[i];
        state.selected = true;
        break;
      }
    }
  }
 private:
  std::array<State, 16> states_{};
};

inline bool automatic_submission_policy(const char* explicit_interval,
                                         const char* automatic_setting) {
  // Even an invalid explicit spelling opts out: the backend retains its
  // existing parsing/default behavior, never replaces an operator's override.
  return explicit_interval == nullptr &&
         (automatic_setting == nullptr || automatic_setting[0] != '0');
}

}  // namespace lse::backend
