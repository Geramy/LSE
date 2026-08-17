#include "lse/runtime/generator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

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

// Tokens per prefill pass. Extents are baked into the generated HIP, so one
// pass over the whole prompt makes every distinct prompt length its own JIT
// cold start; a ladder caps the shapes the engine can ever see.
//
// Measured on this box: a fresh cache pays ~35-48 compiles at ~54 ms for each
// new pass width, and the ladder below makes the reachable widths exactly
// {1,2,4,8,16,32,64,128} — so the whole engine costs ~290 compiles once and a
// novel prompt length costs zero. Without it every length is its own set.
//
// The row axis is bucketed the same way (model::kBatchRungs) and for the same
// reason, and the two share the ladder: a decode pass of B sequences and a
// prefill pass of B tokens present the same row count to every GEMM.
constexpr std::size_t kPrefillChunk = 128;

std::size_t prefill_chunk() { return kPrefillChunk; }

// Splits `n` tokens into consecutive passes sized from {chunk} u {powers of
// two below it}, so the whole engine only ever compiles that many prefill
// shapes. 0 is one pass, whatever the length.
//
// Ascending, which puts the ragged remainder first: the last pass is then the
// widest one, and it is the pass whose final row feeds the LM head. That keeps
// the logits row on the same linear kernel a single pass would have used,
// which is the closest a split can get to the unsplit answer.
std::vector<std::size_t> prefill_plan(std::size_t n, std::size_t chunk) {
  if (chunk == 0) return {n};
  std::vector<std::size_t> plan;
  std::size_t rest = n % chunk;
  for (std::size_t step = 1; rest != 0; step <<= 1) {
    if ((rest & step) != 0) {
      plan.push_back(step);
      rest -= step;
    }
  }
  plan.insert(plan.end(), n / chunk, chunk);
  return plan;
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
  stats->streams_used = t.streams_used;
  stats->stream_waits = t.stream_waits;
  stats->stream_chain = t.stream_chain;
  stats->streams_available =
      sched->backend().stream_capabilities().stream_count;
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

Status Generator::poke_decode_ids(std::uint32_t token) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to hold token ids");
  }
  if (!decode_ids_.valid()) {
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, 1);
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    decode_ids_ = Array::from_buffer(buf.release(), Shape{1, 1}, DType::kF32);
  }
  graph::Node& n = *decode_ids_.node();
  const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
  if (n.host_mirror.size() < bytes) n.host_mirror.resize(bytes);
  graph::interpreter::store_element(n, 0, static_cast<float>(token));
  n.materialized = true;
  // No upload here: the forward replay pokes the slot (its own 4-byte H2D)
  // and a rebuild syncs it when the embedding group binds it.
  return OkStatus();
}

Result<graph::Array> Generator::decode_head(Session& session,
                                            std::uint32_t token, bool greedy) {
  LSE_RETURN_IF_ERROR(poke_decode_ids(token));
  LSE_ASSIGN_OR(Array hidden,
                model_.hidden(decode_ids_, &session.states(), nullptr));
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the lm_head");
  }

  const bool reuse = head_.hidden.valid() && head_.logits.valid() &&
                     head_.hidden.node().get() == hidden.node().get() &&
                     head_.greedy == greedy && (!greedy || head_.pick.valid());
  if (!reuse) {
    head_ = DecodeHead{};
    head_.hidden = hidden;
    head_.greedy = greedy;
    // T == 1, so the last row is the whole tensor: a reshape view suffices
    // and the slice copy of the general path is never built.
    const Shape& s = hidden.shape();
    Shape flat;
    for (std::size_t i = 0; i < s.rank(); ++i) {
      if (i + 2 == s.rank()) continue;
      flat.push_back(s.dim(i));
    }
    Array last = graph::reshape(hidden, flat);
    LSE_ASSIGN_OR(head_.logits, model_.lm_head(last));
    head_.compute = {last.node(), head_.logits.node()};
    if (greedy) {
      head_.pick = graph::argmax(head_.logits);
      if (!head_.pick.valid()) {
        return LSE_ERROR(kInternal, "argmax over an empty logit row");
      }
      head_.compute.push_back(head_.pick.node()->inputs[0]);
      head_.compute.push_back(head_.pick.node());
    }
  }

  Array root = greedy ? head_.pick : head_.logits;
  for (const graph::NodePtr& n : head_.compute) {
    if (n) n->materialized = false;
  }
  const graph::NodePtr roots[] = {root.node()};
  LSE_RETURN_IF_ERROR(sched->eval(roots, true, &head_.program));
  return root;
}

Result<std::uint32_t> Generator::greedy_step(Session& session,
                                             std::uint32_t token) {
  LSE_ASSIGN_OR(Array pick, decode_head(session, token, true));
  const float id = graph::interpreter::load_element(*pick.node(), 0);
  return static_cast<std::uint32_t>(id);
}

Result<std::vector<float>> Generator::step(
    Session& session, const std::vector<std::uint32_t>& tokens) {
  if (tokens.size() == 1) {
    LSE_ASSIGN_OR(Array logits, decode_head(session, tokens[0], false));
    std::vector<float> out(logits.shape().elem_count());
    LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(
        *logits.node(), out.data(), out.size() * sizeof(float)));
    return out;
  }

  // Every pass carries the block state forward exactly as decode does, so the
  // split is invisible to the model: the KV write cursor, the RoPE angles and
  // the attention masks all read the shared device position slot, which counts
  // absolute tokens, not tokens within a pass.
  Array hidden;
  std::size_t at = 0;
  for (std::size_t take : prefill_plan(tokens.size(), prefill_chunk())) {
    const auto first = tokens.begin() + static_cast<std::ptrdiff_t>(at);
    LSE_ASSIGN_OR(Array ids, token_array(std::vector<std::uint32_t>(
                                 first, first + static_cast<std::ptrdiff_t>(take))));
    LSE_ASSIGN_OR(hidden, model_.hidden(ids, &session.states(), nullptr));
    at += take;
  }
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

  // Pure greedy with the repetition penalty at its no-op setting is the one
  // mode where the sampler never reads more than the argmax, so the index can
  // come back from the device instead of the whole logit row. Any other knob
  // keeps the host path. (The penalty's condition in Sampler::sample is
  // `repetition_penalty != 1.0f`.)
  const SamplingParams& sp = sampler_.params();
  const bool device_greedy =
      sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;

  const std::uint64_t decode_start = now_ns();
  std::uint32_t next = 0;
  if (limits.max_tokens > 0) next = sampler_.sample(logits, session.history());
  for (std::int32_t n = 0; n < limits.max_tokens; ++n) {
    if (is_stop(next)) break;

    generated.push_back(next);
    session.history().push_back(next);
    ++stats_.generated_tokens;

    if (on_token && !on_token(next)) break;
    if (n + 1 == limits.max_tokens) break;

    // Only the new token goes in: every block's state already holds the rest.
    if (device_greedy) {
      LSE_ASSIGN_OR(next, greedy_step(session, next));
    } else {
      LSE_ASSIGN_OR(logits, step(session, {next}));
      next = sampler_.sample(logits, session.history());
    }
    session.advance(1);
  }
  stats_.decode_ns = now_ns() - decode_start;
  snapshot_trace(&stats_, &host_reasons_);

  return generated;
}

}  // namespace lse::runtime
