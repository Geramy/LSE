// Perplexity and KL divergence of a fixed token sequence through the prompt
// prefill path.
//
// Two ways to cut the sequence into scored windows:
//
//  - kChunks (the default) is llama.cpp's llama-perplexity method: the
//    sequence is split into floor(tokens / ctx) non-overlapping chunks of ctx
//    tokens, any remainder is dropped, and each chunk scores only its second
//    half. The first ctx/2 tokens are context; the tokens after position ctx/2
//    in the chunk are scored, which is ctx - ctx/2 - 1 tokens per chunk.
//  - kSliding is the sliding-window method: windows of ctx tokens start every
//    stride tokens, and each scores only the tokens no earlier window scored,
//    so every token after the first is scored once. stride == ctx gives
//    disjoint windows that score every token but their first.
//
// Each window runs in a fresh session through Generator::score, which is the
// prefill a request's prompt takes (the same pass plan, attention phase and KV
// storage), followed by the LM head over the scored rows.
//
// KL divergence follows llama.cpp's --kl-divergence-base / --kl-divergence
// flow, but records the base run's top-k token log-probabilities per scored
// token instead of every vocabulary entry (a 248k vocabulary would need about
// 0.5 MB per token). The comparison run reads its own log-probabilities at the
// base's top-k ids; the mass outside the top k is one more bucket, so the
// reported KL divergence is the divergence between the two distributions
// coarsened to k + 1 outcomes, which never exceeds the full-vocabulary value.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::model { class HybridLM; }

namespace lse::runtime {

class Generator;

enum class PerplexityMethod : std::uint32_t { kChunks = 0, kSliding = 1 };

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

// llama.cpp's chunks over `tokens` tokens. Fails when ctx < 4 or the sequence
// holds fewer than two chunks (llama.cpp's own minimum).
Result<std::vector<PerplexityWindow>> perplexity_chunks(std::size_t tokens, std::size_t ctx);

// Sliding windows over `tokens` tokens, in order. Fails when window < 2, when
// stride is 0 or exceeds window, or when there are fewer than two tokens.
Result<std::vector<PerplexityWindow>> perplexity_windows(std::size_t tokens,
                                                         std::size_t window,
                                                         std::size_t stride);

// What Generator::score returns for one window, one entry per scored token.
struct TokenScores {
  std::vector<double> nll;
  // With top_k > 0: this run's k most likely ids per scored token, best
  // first, and their log-probabilities. [scored, top_k].
  std::size_t top_k = 0;
  std::vector<std::uint32_t> top_ids;
  std::vector<float> top_logprobs;
  // With probe ids: this run's log-probability of each. [scored, probe_k].
  std::size_t probe_k = 0;
  std::vector<double> probe_logprobs;
};

// The base run of a KL-divergence comparison: per scored token, the negative
// log-likelihood and the top-k (id, log-probability) pairs. Written once with
// --perplexity-kld-base and read by every comparison run, which must score the
// same token ids with the same windows.
struct KldBase {
  static constexpr char kMagic[8] = {'L', 'S', 'E', 'K', 'L', 'D', '0', '1'};
  std::uint32_t vocab = 0;
  std::uint32_t top_k = 0;
  PerplexityMethod method = PerplexityMethod::kChunks;
  std::uint64_t ctx = 0;
  std::uint64_t stride = 0;
  std::uint64_t windows = 0;
  std::uint64_t tokens = 0;
  // Hex SHA-256 of the token ids as little-endian uint32.
  std::string token_ids_sha256;
  std::vector<double> nll;            // [scored]
  std::vector<std::uint32_t> ids;     // [scored, top_k]
  std::vector<float> logprobs;        // [scored, top_k]

  [[nodiscard]] std::size_t scored() const noexcept { return nll.size(); }
  Status write(const std::string& path) const;
  static Result<KldBase> read(const std::string& path);
};

struct KldStats {
  std::size_t count = 0;
  std::uint32_t top_k = 0;
  double base_nll_sum = 0.0;
  double mean_kld = 0.0, kld_uncertainty = 0.0;
  double kld_p99 = 0.0, kld_max = 0.0;
  double same_top = 0.0, same_top_uncertainty = 0.0;  // fractions
  double mean_delta_p = 0.0, delta_p_uncertainty = 0.0, rms_delta_p = 0.0;
  // Mean over tokens of nll - base nll, i.e. log(PPL / base PPL).
  double mean_log_ratio = 0.0, log_ratio_uncertainty = 0.0;
  // Mean base probability mass inside the recorded top k.
  double mean_base_top_mass = 0.0;

  [[nodiscard]] double base_perplexity() const noexcept {
    return count == 0 ? 0.0 : std::exp(base_nll_sum / static_cast<double>(count));
  }
};

// Per-token KL divergence of `current` from the base distribution, given the
// base's top-k log-probabilities and the current run's log-probabilities at
// the same ids (both natural log). The remaining mass is one more outcome.
double coarse_kld(std::span<const float> base_logprobs, std::span<const double> current_logprobs);

struct WindowScore {
  PerplexityWindow window;
  double nll_sum = 0.0;
  std::uint64_t elapsed_ns = 0;

  [[nodiscard]] double mean_nll() const noexcept {
    return window.scored() == 0 ? 0.0 : nll_sum / static_cast<double>(window.scored());
  }
};

struct PerplexityOptions {
  PerplexityMethod method = PerplexityMethod::kChunks;
  std::size_t ctx = 512;
  // kSliding only; 0 means ctx.
  std::size_t stride = 0;
  // Score only the first N windows; 0 scores all of them.
  std::size_t max_windows = 0;
  // Record a KL-divergence base with this many top ids per token (1..32).
  std::size_t record_top_k = 0;
  // Compare against this base.
  const KldBase* compare = nullptr;
  // The token ids' SHA-256, checked against a base and stored in a new one.
  std::string token_ids_sha256;
};

struct PerplexityReport {
  std::size_t tokens = 0;
  PerplexityMethod method = PerplexityMethod::kChunks;
  std::size_t ctx = 0;
  std::size_t stride = 0;
  std::size_t scored = 0;
  double nll_sum = 0.0;
  double nll_sq_sum = 0.0;
  std::uint64_t elapsed_ns = 0;
  std::vector<WindowScore> windows;
  // With record_top_k: the base this run wrote.
  std::optional<KldBase> recorded;
  // With compare: the comparison.
  std::optional<KldStats> kld;

  [[nodiscard]] double mean_nll() const noexcept {
    return scored == 0 ? 0.0 : nll_sum / static_cast<double>(scored);
  }
  [[nodiscard]] double perplexity() const noexcept { return std::exp(mean_nll()); }
  // llama.cpp's "+/-": the standard error of the mean NLL, propagated to PPL.
  [[nodiscard]] double perplexity_uncertainty() const noexcept {
    if (scored < 2) return 0.0;
    const double mean = mean_nll();
    const double var = nll_sq_sum / static_cast<double>(scored) - mean * mean;
    return var > 0.0 ? perplexity() * std::sqrt(var / static_cast<double>(scored - 1)) : 0.0;
  }
};

// Called after each window with the windows done so far and the total.
using PerplexityProgress =
    std::function<void(const WindowScore&, std::size_t done, std::size_t total)>;

// Scores `tokens` window by window. `gen` must have no draft module attached.
// Every token id must be below the model's vocabulary and every window must fit
// the model's KV capacity.
Result<PerplexityReport> score_perplexity(Generator& gen, model::HybridLM& model,
                                          std::span<const std::uint32_t> tokens,
                                          const PerplexityOptions& options,
                                          const PerplexityProgress& progress = {});

}  // namespace lse::runtime
