// Perplexity of a fixed token sequence through the prompt prefill path.
//
// The sequence is cut into windows the standard sliding-window way: windows of
// `window` tokens start every `stride` tokens, and each window scores only the
// tokens the previous window did not, so every token after the first is scored
// exactly once, with at least window - stride tokens of context once the first
// window is past. stride == window gives disjoint windows, each scoring every
// token but its first.
//
// Each window runs in a fresh session through Generator::score, which is the
// prefill a request's prompt takes (the same pass plan, attention phase and KV
// storage), followed by the LM head over the scored rows.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::model { class HybridLM; }

namespace lse::runtime {

class Generator;

struct PerplexityWindow {
  // Tokens [begin, end) of the sequence are the window's input.
  std::size_t begin = 0;
  std::size_t end = 0;
  // Tokens [first_target, end) are scored, each from the tokens before it in
  // the window. begin < first_target < end.
  std::size_t first_target = 0;

  [[nodiscard]] std::size_t scored() const noexcept { return end - first_target; }
  friend bool operator==(const PerplexityWindow&, const PerplexityWindow&) = default;
};

// The windows over `tokens` tokens, in order. Fails when window < 2, when
// stride is 0 or exceeds window, or when there are fewer than two tokens.
Result<std::vector<PerplexityWindow>> perplexity_windows(std::size_t tokens,
                                                         std::size_t window,
                                                         std::size_t stride);

struct WindowScore {
  PerplexityWindow window;
  double nll_sum = 0.0;
  std::uint64_t elapsed_ns = 0;

  [[nodiscard]] double mean_nll() const noexcept {
    return window.scored() == 0 ? 0.0 : nll_sum / static_cast<double>(window.scored());
  }
};

struct PerplexityReport {
  std::size_t tokens = 0;
  std::size_t window = 0;
  std::size_t stride = 0;
  std::size_t scored = 0;
  double nll_sum = 0.0;
  std::uint64_t elapsed_ns = 0;
  std::vector<WindowScore> windows;

  [[nodiscard]] double mean_nll() const noexcept {
    return scored == 0 ? 0.0 : nll_sum / static_cast<double>(scored);
  }
  [[nodiscard]] double perplexity() const noexcept { return std::exp(mean_nll()); }
};

// Called after each window with the windows done so far and the total.
using PerplexityProgress =
    std::function<void(const WindowScore&, std::size_t done, std::size_t total)>;

// Scores `tokens` window by window. `max_windows` 0 scores every window;
// otherwise the first max_windows. `gen` must have no draft module attached.
// Every token id must be below the model's vocabulary and every window must fit
// the model's KV capacity.
Result<PerplexityReport> score_perplexity(Generator& gen, model::HybridLM& model,
                                          std::span<const std::uint32_t> tokens,
                                          std::size_t window, std::size_t stride,
                                          std::size_t max_windows = 0,
                                          const PerplexityProgress& progress = {});

}  // namespace lse::runtime
