#include "lse/runtime/perplexity.hpp"

#include <chrono>
#include <string>

#include "lse/model/hybrid_lm.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/runtime/session.hpp"

namespace lse::runtime {

namespace {

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

Result<std::vector<PerplexityWindow>> perplexity_windows(std::size_t tokens,
                                                         std::size_t window,
                                                         std::size_t stride) {
  if (window < 2) return LSE_ERROR(kInvalidArgument, "perplexity window must be at least 2 tokens");
  if (stride == 0 || stride > window)
    return LSE_ERROR(kInvalidArgument, "perplexity stride must be from 1 to the window (",
                     std::to_string(window), "), got ", std::to_string(stride));
  if (tokens < 2) return LSE_ERROR(kInvalidArgument, "perplexity needs at least 2 tokens, got ",
                                   std::to_string(tokens));
  std::vector<PerplexityWindow> out;
  std::size_t previous_end = 0;
  for (std::size_t begin = 0; begin < tokens; begin += stride) {
    const std::size_t end = std::min(begin + window, tokens);
    // The first token of a window has no context inside it, and tokens an
    // earlier window scored are not scored again.
    const std::size_t first = std::max(begin + 1, previous_end);
    if (first < end) out.push_back({begin, end, first});
    previous_end = end;
    if (end == tokens) break;
  }
  return out;
}

Result<PerplexityReport> score_perplexity(Generator& gen, model::HybridLM& model,
                                          std::span<const std::uint32_t> tokens,
                                          std::size_t window, std::size_t stride,
                                          std::size_t max_windows,
                                          const PerplexityProgress& progress) {
  LSE_ASSIGN_OR(std::vector<PerplexityWindow> plan,
                perplexity_windows(tokens.size(), window, stride));
  if (max_windows != 0 && plan.size() > max_windows) plan.resize(max_windows);

  PerplexityReport report;
  report.tokens = tokens.size();
  report.window = window;
  report.stride = stride;
  report.windows.reserve(plan.size());
  const std::uint64_t started = now_ns();
  for (const PerplexityWindow& w : plan) {
    const std::uint64_t window_started = now_ns();
    Result<std::vector<double>> nll = [&]() -> Result<std::vector<double>> {
      // A fresh session per window, as a request without a session id gets.
      Session session("", model.state_slots());
      auto scored = gen.score(session, tokens.subspan(w.begin, w.end - w.begin),
                              w.first_target - w.begin);
      // What the model retained was built against this session's state.
      const Status dropped = model.drop_retained_passes();
      if (!scored.ok()) return scored.status();
      if (!dropped.ok()) return dropped;
      return scored;
    }();
    if (!nll.ok()) {
      return LSE_ERROR(kInternal, "scoring window [", std::to_string(w.begin), ", ",
                       std::to_string(w.end), "): ", std::string(nll.status().message()));
    }
    if (nll->size() != w.scored()) {
      return LSE_ERROR(kInternal, "window [", std::to_string(w.begin), ", ",
                       std::to_string(w.end), ") scored ", std::to_string(nll->size()),
                       " tokens, expected ", std::to_string(w.scored()));
    }
    WindowScore score{w, 0.0, 0};
    for (const double v : *nll) score.nll_sum += v;
    score.elapsed_ns = now_ns() - window_started;
    report.scored += w.scored();
    report.nll_sum += score.nll_sum;
    report.windows.push_back(score);
    if (progress) progress(score, report.windows.size(), plan.size());
  }
  report.elapsed_ns = now_ns() - started;
  return report;
}

}  // namespace lse::runtime
