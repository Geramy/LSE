#include "lse/runtime/perplexity.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
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

const char* method_name(PerplexityMethod m) {
  return m == PerplexityMethod::kChunks ? "chunks" : "sliding";
}

}  // namespace

Result<std::vector<PerplexityWindow>> perplexity_chunks(std::size_t tokens, std::size_t ctx) {
  if (ctx < 4) return LSE_ERROR(kInvalidArgument, "perplexity context must be at least 4 tokens");
  if (tokens < 2 * ctx)
    return LSE_ERROR(kInvalidArgument, "perplexity with a context of ", std::to_string(ctx),
                     " needs at least ", std::to_string(2 * ctx), " tokens; the text has ",
                     std::to_string(tokens));
  std::vector<PerplexityWindow> out;
  // llama.cpp: logits from position ctx/2 on predict the tokens after it.
  const std::size_t first = ctx / 2 + 1;
  for (std::size_t i = 0; i < tokens / ctx; ++i)
    out.push_back({i * ctx, (i + 1) * ctx, i * ctx + first});
  return out;
}

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

double coarse_kld(std::span<const float> base_logprobs, std::span<const double> current_logprobs) {
  double kld = 0.0, base_mass = 0.0, current_mass = 0.0;
  for (std::size_t i = 0; i < base_logprobs.size(); ++i) {
    const double lp = base_logprobs[i], lq = current_logprobs[i];
    const double p = std::exp(lp);
    kld += p * (lp - lq);
    base_mass += p;
    current_mass += std::exp(lq);
  }
  const double base_rest = 1.0 - base_mass, current_rest = 1.0 - current_mass;
  // The bucket of everything outside the base's top k. Rounding can leave a
  // vanishing or slightly negative remainder; it then carries no mass.
  if (base_rest > 1e-12 && current_rest > 1e-12) kld += base_rest * std::log(base_rest / current_rest);
  return kld;
}

Status KldBase::write(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) return LSE_ERROR(kIoError, "cannot write the KL-divergence base ", path);
  const auto put = [&](const auto& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
  out.write(kMagic, sizeof(kMagic));
  put(std::uint32_t{1});
  put(vocab);
  put(top_k);
  put(static_cast<std::uint32_t>(method));
  put(ctx);
  put(stride);
  put(windows);
  put(tokens);
  char sha[64] = {};
  std::memcpy(sha, token_ids_sha256.data(), std::min<std::size_t>(64, token_ids_sha256.size()));
  out.write(sha, sizeof(sha));
  put(static_cast<std::uint64_t>(nll.size()));
  out.write(reinterpret_cast<const char*>(nll.data()), static_cast<std::streamsize>(nll.size() * 8));
  out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
  out.write(reinterpret_cast<const char*>(logprobs.data()),
            static_cast<std::streamsize>(logprobs.size() * 4));
  if (!out) return LSE_ERROR(kIoError, "writing the KL-divergence base ", path, " failed");
  return OkStatus();
}

Result<KldBase> KldBase::read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return LSE_ERROR(kIoError, "cannot read the KL-divergence base ", path);
  const auto get = [&](auto& v) { in.read(reinterpret_cast<char*>(&v), sizeof(v)); };
  char magic[8];
  in.read(magic, sizeof(magic));
  std::uint32_t version = 0, method = 0;
  KldBase b;
  get(version);
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 || version != 1)
    return LSE_ERROR(kInvalidArgument, path, " is not an LSE KL-divergence base (version 1)");
  get(b.vocab);
  get(b.top_k);
  get(method);
  get(b.ctx);
  get(b.stride);
  get(b.windows);
  get(b.tokens);
  char sha[64];
  in.read(sha, sizeof(sha));
  b.token_ids_sha256.assign(sha, strnlen(sha, sizeof(sha)));
  std::uint64_t scored = 0;
  get(scored);
  if (!in || method > 1 || b.top_k == 0 || b.top_k > 32 || scored > (std::uint64_t{1} << 32))
    return LSE_ERROR(kInvalidArgument, path, " has a malformed KL-divergence header");
  b.method = static_cast<PerplexityMethod>(method);
  b.nll.resize(scored);
  b.ids.resize(scored * b.top_k);
  b.logprobs.resize(scored * b.top_k);
  in.read(reinterpret_cast<char*>(b.nll.data()), static_cast<std::streamsize>(b.nll.size() * 8));
  in.read(reinterpret_cast<char*>(b.ids.data()), static_cast<std::streamsize>(b.ids.size() * 4));
  in.read(reinterpret_cast<char*>(b.logprobs.data()),
          static_cast<std::streamsize>(b.logprobs.size() * 4));
  if (!in) return LSE_ERROR(kInvalidArgument, path, " is truncated");
  return b;
}

Result<PerplexityReport> score_perplexity(Generator& gen, model::HybridLM& model,
                                          std::span<const std::uint32_t> tokens,
                                          const PerplexityOptions& options,
                                          const PerplexityProgress& progress) {
  const std::size_t stride = options.stride == 0 ? options.ctx : options.stride;
  std::vector<PerplexityWindow> plan;
  if (options.method == PerplexityMethod::kChunks) {
    LSE_ASSIGN_OR(plan, perplexity_chunks(tokens.size(), options.ctx));
  } else {
    LSE_ASSIGN_OR(plan, perplexity_windows(tokens.size(), options.ctx, stride));
  }
  if (options.max_windows != 0 && plan.size() > options.max_windows) plan.resize(options.max_windows);
  std::size_t total = 0;
  for (const PerplexityWindow& w : plan) total += w.scored();

  const auto vocab = static_cast<std::uint32_t>(model.config().vocab_size);
  if (options.record_top_k > 32)
    return LSE_ERROR(kInvalidArgument, "a KL-divergence base records at most 32 ids per token");
  const KldBase* base = options.compare;
  if (base != nullptr) {
    const auto mismatch = [&](const std::string& what) {
      return LSE_ERROR(kInvalidArgument, "the KL-divergence base differs from this run: ", what);
    };
    if (base->vocab != vocab) return mismatch("vocabulary " + std::to_string(base->vocab));
    if (base->method != options.method) return mismatch(std::string("method ") + method_name(base->method));
    if (base->ctx != options.ctx) return mismatch("context " + std::to_string(base->ctx));
    if (options.method == PerplexityMethod::kSliding && base->stride != stride)
      return mismatch("stride " + std::to_string(base->stride));
    if (base->tokens != tokens.size()) return mismatch(std::to_string(base->tokens) + " tokens");
    if (!options.token_ids_sha256.empty() && base->token_ids_sha256 != options.token_ids_sha256)
      return mismatch("token ids (SHA-256 " + base->token_ids_sha256 + ")");
    if (base->windows != plan.size() || base->scored() != total)
      return mismatch(std::to_string(base->windows) + " windows, " + std::to_string(base->scored()) +
                      " scored tokens; pass the same --perplexity-chunks");
  }

  PerplexityReport report;
  report.tokens = tokens.size();
  report.method = options.method;
  report.ctx = options.ctx;
  report.stride = options.method == PerplexityMethod::kSliding ? stride : options.ctx;
  report.windows.reserve(plan.size());
  if (options.record_top_k != 0) {
    KldBase& r = report.recorded.emplace();
    r.vocab = vocab;
    r.top_k = static_cast<std::uint32_t>(options.record_top_k);
    r.method = options.method;
    r.ctx = options.ctx;
    r.stride = report.stride;
    r.windows = plan.size();
    r.tokens = tokens.size();
    r.token_ids_sha256 = options.token_ids_sha256;
    r.nll.reserve(total);
    r.ids.reserve(total * r.top_k);
    r.logprobs.reserve(total * r.top_k);
  }
  const std::size_t top_k = std::max(options.record_top_k, base != nullptr ? std::size_t{1} : 0);
  const std::size_t probe_k = base != nullptr ? base->top_k : 0;

  KldStats kld;
  std::vector<double> klds;
  double kld_sq = 0.0, same = 0.0, dp = 0.0, dp_sq = 0.0, lr = 0.0, lr_sq = 0.0, mass = 0.0;
  if (base != nullptr) {
    kld.top_k = base->top_k;
    klds.reserve(total);
  }

  std::size_t offset = 0;  // scored tokens before this window
  const std::uint64_t started = now_ns();
  for (const PerplexityWindow& w : plan) {
    const std::uint64_t window_started = now_ns();
    const std::span<const std::uint32_t> probe =
        base != nullptr ? std::span<const std::uint32_t>(base->ids).subspan(offset * probe_k, w.scored() * probe_k)
                        : std::span<const std::uint32_t>{};
    Result<TokenScores> got = [&]() -> Result<TokenScores> {
      // A fresh session per window, as a request without a session id gets.
      Session session("", model.state_slots());
      auto scored = gen.score(session, tokens.subspan(w.begin, w.end - w.begin), w.first_target - w.begin,
                              top_k, probe, probe_k);
      // What the model retained was built against this session's state.
      const Status dropped = model.drop_retained_passes();
      if (!scored.ok()) return scored.status();
      if (!dropped.ok()) return dropped;
      return scored;
    }();
    if (!got.ok()) {
      return Status(got.status().code(), "scoring window [" + std::to_string(w.begin) + ", " +
                                             std::to_string(w.end) + "): " +
                                             std::string(got.status().message()));
    }
    if (got->nll.size() != w.scored()) {
      return LSE_ERROR(kInternal, "window [", std::to_string(w.begin), ", ",
                       std::to_string(w.end), ") scored ", std::to_string(got->nll.size()),
                       " tokens, expected ", std::to_string(w.scored()));
    }
    WindowScore score{w, 0.0, 0};
    for (const double v : got->nll) {
      score.nll_sum += v;
      report.nll_sq_sum += v * v;
    }
    if (report.recorded) {
      KldBase& r = *report.recorded;
      r.nll.insert(r.nll.end(), got->nll.begin(), got->nll.end());
      for (std::size_t t = 0; t < w.scored(); ++t) {
        for (std::size_t j = 0; j < r.top_k; ++j) {
          r.ids.push_back(got->top_ids[t * top_k + j]);
          r.logprobs.push_back(got->top_logprobs[t * top_k + j]);
        }
      }
    }
    if (base != nullptr) {
      for (std::size_t t = 0; t < w.scored(); ++t) {
        const std::size_t at = offset + t;
        const std::span<const float> lb(base->logprobs.data() + at * probe_k, probe_k);
        const std::span<const double> lq(got->probe_logprobs.data() + t * probe_k, probe_k);
        const double k = coarse_kld(lb, lq);
        klds.push_back(k);
        kld.mean_kld += k;
        kld_sq += k * k;
        for (const float v : lb) mass += std::exp(static_cast<double>(v));
        if (got->top_ids[t * top_k] == base->ids[at * probe_k]) same += 1.0;
        const double nll = got->nll[t], nll_base = base->nll[at];
        kld.base_nll_sum += nll_base;
        const double d = std::exp(-nll) - std::exp(-nll_base);
        dp += d;
        dp_sq += d * d;
        lr += nll - nll_base;
        lr_sq += (nll - nll_base) * (nll - nll_base);
      }
    }
    offset += w.scored();
    score.elapsed_ns = now_ns() - window_started;
    report.scored += w.scored();
    report.nll_sum += score.nll_sum;
    report.windows.push_back(score);
    if (progress) progress(score, report.windows.size(), plan.size());
  }
  report.elapsed_ns = now_ns() - started;

  if (base != nullptr && !klds.empty()) {
    const auto n = static_cast<double>(klds.size());
    const auto stderr_of = [&](double sum, double sq) {
      const double mean = sum / n, var = sq / n - mean * mean;
      return n > 1 && var > 0 ? std::sqrt(var / (n - 1)) : 0.0;
    };
    kld.count = klds.size();
    kld.kld_uncertainty = stderr_of(kld.mean_kld, kld_sq);
    kld.mean_kld /= n;
    kld.same_top = same / n;
    kld.same_top_uncertainty = n > 1 ? std::sqrt(kld.same_top * (1 - kld.same_top) / (n - 1)) : 0.0;
    kld.delta_p_uncertainty = stderr_of(dp, dp_sq);
    kld.mean_delta_p = dp / n;
    kld.rms_delta_p = std::sqrt(dp_sq / n);
    kld.log_ratio_uncertainty = stderr_of(lr, lr_sq);
    kld.mean_log_ratio = lr / n;
    kld.mean_base_top_mass = mass / n;
    std::sort(klds.begin(), klds.end());
    kld.kld_max = klds.back();
    kld.kld_p99 = klds[std::min(klds.size() - 1, static_cast<std::size_t>(0.99 * n))];
    report.kld = kld;
  }
  return report;
}

}  // namespace lse::runtime
