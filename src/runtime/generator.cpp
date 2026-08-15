#include "lse/runtime/generator.hpp"

#include <algorithm>
#include <chrono>

#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"

namespace lse::runtime {

namespace {

using graph::Array;
using lse::DType;
using lse::Shape;

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Token ids ride the graph as f32, matching how every other index does.
Result<Array> token_array(const std::vector<std::uint32_t>& ids) {
  Array a = Array::zeros(Shape{1, static_cast<std::int64_t>(ids.size())},
                         DType::kF32);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to hold token ids");
  }
  graph::Node& n = *a.node();
  LSE_RETURN_IF_ERROR(graph::interpreter::ensure_output_buffer(n, sched->backend()));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    graph::interpreter::store_element(n, i, static_cast<float>(ids[i]));
  }
  n.materialized = true;
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_to_device(n, sched->backend()));
  return a;
}

void snapshot_trace(GenerationStats* stats, std::vector<std::string>* reasons) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return;
  const graph::Scheduler::Trace& t = sched->accumulated_trace();
  stats->device_groups = t.device_groups;
  stats->host_groups = t.host_groups;
  stats->kernels_launched = t.kernels_launched;
  stats->phase_groups = t.phase_groups;
  stats->phase_ideal_launches = t.phase_ideal_launches;
  stats->views_aliased = t.views_aliased;
  stats->host_fallbacks = t.host_fallbacks;
  stats->partition_ns = t.partition_ns;
  stats->emit_ns = t.emit_ns;
  stats->launch_ns = t.launch_ns;
  stats->sync_ns = t.sync_ns;
  const graph::Scheduler::JitStats jit = sched->jit_stats();
  stats->jit_memory_hits = jit.memory_hits;
  stats->jit_disk_hits = jit.disk_hits;
  stats->jit_compiles = jit.compiles;
  stats->jit_compile_ns = jit.compile_ns;
  if (reasons != nullptr) *reasons = t.host_group_reasons;
}

}  // namespace

Result<Array> Generator::last_hidden(const Array& hidden) {
  if (!hidden.valid()) {
    return LSE_ERROR(kInvalidArgument, "last_hidden on an empty Array");
  }
  const Shape& s = hidden.shape();
  if (s.rank() < 2) {
    return LSE_ERROR(kInvalidArgument, "hidden must be [.., T, D], got rank ",
                     std::to_string(s.rank()));
  }
  const std::int64_t t = s.dim(s.rank() - 2);
  if (t <= 0) {
    return LSE_ERROR(kInvalidArgument, "hidden has no sequence axis");
  }
  Array row = graph::slice(hidden, static_cast<int>(s.rank()) - 2, t - 1, t);
  Shape flat;
  for (std::size_t i = 0; i < s.rank(); ++i) {
    if (i + 2 == s.rank()) continue;
    flat.push_back(s.dim(i));
  }
  return graph::reshape(row, flat);
}

Result<std::vector<float>> Generator::step(
    Session& session, const std::vector<std::uint32_t>& tokens) {
  LSE_ASSIGN_OR(Array ids, token_array(tokens));
  LSE_ASSIGN_OR(Array hidden, model_.hidden(ids, &session.states(), nullptr));
  LSE_ASSIGN_OR(Array last, last_hidden(hidden));
  LSE_ASSIGN_OR(Array logits, model_.lm_head(last));

  std::vector<float> out(logits.shape().elem_count());
  LSE_RETURN_IF_ERROR(
      logits.to_host(out.data(), out.size() * sizeof(float)));
  return out;
}

Result<std::vector<std::uint32_t>> Generator::generate(
    const std::vector<std::uint32_t>& prompt, const GenerationLimits& limits,
    const TokenCallback& on_token) {
  if (owned_ == nullptr) {
    owned_ = std::make_unique<Session>("", model_.num_layers());
  }
  owned_->clear();
  return generate(*owned_, prompt, limits, on_token);
}

Result<std::vector<std::uint32_t>> Generator::generate(
    Session& session, const std::vector<std::uint32_t>& prompt,
    const GenerationLimits& limits, const TokenCallback& on_token) {
  if (prompt.empty()) {
    return LSE_ERROR(kInvalidArgument, "cannot generate from an empty prompt");
  }
  stats_ = GenerationStats{};
  host_reasons_.clear();
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    sched->reset_accumulated_trace();
  }

  // Only what the cache does not already cover. A follow-up turn whose prompt
  // extends the previous one therefore costs its new tokens, not the whole
  // conversation; anything else is a cold start.
  const auto covered = static_cast<std::size_t>(session.position());
  const bool continues =
      covered > 0 && covered <= prompt.size() &&
      std::equal(session.history().begin(),
                 session.history().begin() + static_cast<std::ptrdiff_t>(covered),
                 prompt.begin());
  if (!continues) {
    session.clear();
  }
  const std::size_t start = continues ? covered : 0;
  const std::vector<std::uint32_t> fresh(prompt.begin() + static_cast<std::ptrdiff_t>(start),
                                         prompt.end());
  if (fresh.empty()) {
    return LSE_ERROR(kInvalidArgument,
                     "the prompt is already fully cached; nothing to score");
  }

  session.history() = prompt;
  stats_.prompt_tokens = static_cast<std::int32_t>(fresh.size());

  const std::uint64_t prefill_start = now_ns();
  LSE_ASSIGN_OR(std::vector<float> logits, step(session, fresh));
  session.advance(static_cast<std::int32_t>(fresh.size()));
  stats_.prefill_ns = now_ns() - prefill_start;

  std::vector<std::uint32_t> generated;
  generated.reserve(static_cast<std::size_t>(std::max(limits.max_tokens, 0)));

  const auto is_stop = [&limits](std::uint32_t id) {
    return std::find(limits.stop_tokens.begin(), limits.stop_tokens.end(), id) !=
           limits.stop_tokens.end();
  };

  const std::uint64_t decode_start = now_ns();
  for (std::int32_t n = 0; n < limits.max_tokens; ++n) {
    const std::uint32_t next = sampler_.sample(logits, session.history());
    if (is_stop(next)) break;

    generated.push_back(next);
    session.history().push_back(next);
    ++stats_.generated_tokens;

    if (on_token && !on_token(next)) break;
    if (n + 1 == limits.max_tokens) break;

    // Only the new token goes in: every block's state already holds the rest.
    LSE_ASSIGN_OR(logits, step(session, {next}));
    session.advance(1);
  }
  stats_.decode_ns = now_ns() - decode_start;
  snapshot_trace(&stats_, &host_reasons_);

  return generated;
}

}  // namespace lse::runtime
