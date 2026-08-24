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
// {1,2,4,8,16,32,64,128,256} — so the whole engine costs ~340 compiles once
// and a novel prompt length costs zero. Without it every length is its own
// set. 256 and not 128: every chunk pays the phase ladder's fixed launch cost
// again, and with programs retained across requests the one-time compiles for
// the extra width amortize to nothing while the halved chunk count is paid
// back on every long prompt.
//
// The row axis is bucketed the same way (model::kBatchRungs) and for the same
// reason, and the two share the ladder: a decode pass of B sequences and a
// prefill pass of B tokens present the same row count to every GEMM.
constexpr std::size_t kPrefillChunk = 256;

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
  stats->peer_migrations = t.peer_migrations;
  stats->peer_bytes = t.peer_bytes;
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

Status Generator::read_hidden(const Array& hidden, std::vector<float>* out) {
  if (!hidden.valid() || hidden.dtype() != DType::kF32) {
    return LSE_ERROR(kInternal,
                     "the MTP module reads hidden states as f32; this pass "
                     "produced another dtype");
  }
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend to read from");
  // The pass left the value on the device; nothing has asked for it on the
  // host, so the mirror is where it has to be moved to.
  LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(*hidden.node(),
                                                           sched->backend()));
  out->resize(hidden.shape().elem_count());
  return graph::interpreter::read_raw(*hidden.node(), out->data(),
                                      out->size() * sizeof(float));
}

// Fills the module's cache for one prefill chunk. Row j of the chunk sits at
// absolute position `first + j`, carries the token there, and takes the
// decoder's hidden state from the position before it — so row 0 needs the
// previous chunk's last row, which `carry` holds (zeros before the first
// chunk, which is the one position with no predecessor).
Status Generator::mtp_prefill_chunk(const Array& hidden,
                                    std::span<const std::uint32_t> tokens,
                                    std::int32_t first,
                                    std::vector<float>* carry) {
  LSE_RETURN_IF_ERROR(read_hidden(hidden, &spec_hidden_));
  const std::size_t width = spec_hidden_.size() / tokens.size();
  std::vector<float> shifted(spec_hidden_.size());
  std::copy(carry->begin(), carry->end(), shifted.begin());
  std::copy(spec_hidden_.begin(),
            spec_hidden_.end() - static_cast<std::ptrdiff_t>(width),
            shifted.begin() + static_cast<std::ptrdiff_t>(width));
  carry->assign(spec_hidden_.end() - static_cast<std::ptrdiff_t>(width),
                spec_hidden_.end());
  return mtp_->draft(shifted, tokens, first).status();
}

Result<std::vector<float>> Generator::step(
    Session& session, const std::vector<std::uint32_t>& tokens) {
  // A one-token prompt still goes the long way when the module is in play: the
  // decode head skips the hidden state the module needs for its first row.
  if (tokens.size() == 1 && mtp_ == nullptr) {
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
  const auto base = static_cast<std::int32_t>(session.position());
  std::vector<float> carry;
  if (mtp_ != nullptr) {
    carry.assign(static_cast<std::size_t>(model_.config().hidden_size), 0.0f);
  }
  Array hidden;
  std::size_t at = 0;
  for (std::size_t take : prefill_plan(tokens.size(), prefill_chunk())) {
    const auto first = tokens.begin() + static_cast<std::ptrdiff_t>(at);
    LSE_ASSIGN_OR(Array ids, token_array(std::vector<std::uint32_t>(
                                 first, first + static_cast<std::ptrdiff_t>(take))));
    LSE_ASSIGN_OR(hidden, model_.hidden(ids, &session.states(), nullptr));
    if (mtp_ != nullptr) {
      LSE_RETURN_IF_ERROR(mtp_prefill_chunk(
          hidden, std::span<const std::uint32_t>(&*first, take),
          base + static_cast<std::int32_t>(at), &carry));
    }
    at += take;
  }
  if (mtp_ != nullptr) prefill_tail_ = std::move(carry);
  LSE_ASSIGN_OR(Array last, last_hidden(hidden));
  LSE_ASSIGN_OR(Array logits, model_.lm_head(last));

  std::vector<float> out(logits.shape().elem_count());
  LSE_RETURN_IF_ERROR(
      logits.to_host(out.data(), out.size() * sizeof(float)));
  return out;
}

Result<Generator::Verified> Generator::verify(Session& session,
                                             std::uint32_t a, std::uint32_t b,
                                             bool replaces_previous) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the verify pass");
  }
  if (!spec_ids_.valid()) {
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, 2);
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    spec_ids_ = Array::from_buffer(buf.release(), Shape{1, 2}, DType::kF32);
  }
  {
    graph::Node& n = *spec_ids_.node();
    const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
    if (n.host_mirror.size() < bytes) n.host_mirror.resize(bytes);
    graph::interpreter::store_element(n, 0, static_cast<float>(a));
    graph::interpreter::store_element(n, 1, static_cast<float>(b));
    n.materialized = true;
  }

  const std::uint64_t started = now_ns();
  LSE_ASSIGN_OR(Array hidden,
                model_.hidden(spec_ids_, &session.states(), nullptr, nullptr,
                              nullptr, replaces_previous));

  const SamplingParams& sp = sampler_.params();
  const bool greedy = sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;
  const bool reuse = spec_.hidden.valid() && spec_.logits.valid() &&
                     spec_.hidden.node().get() == hidden.node().get() &&
                     spec_.greedy == greedy && (!greedy || spec_.pick.valid());
  if (!reuse) {
    spec_ = SpecHead{};
    spec_.hidden = hidden;
    spec_.greedy = greedy;
    // [1, 2, D] -> [1, 2, vocab]: both rows go through the head, because the
    // second row's logits are what a verified proposal buys.
    LSE_ASSIGN_OR(spec_.logits, model_.lm_head(hidden));
    spec_.compute = {spec_.logits.node()};
    if (greedy) {
      spec_.pick = graph::argmax(spec_.logits);
      if (!spec_.pick.valid()) {
        return LSE_ERROR(kInternal, "argmax over an empty logit row");
      }
      spec_.compute.push_back(spec_.pick.node()->inputs[0]);
      spec_.compute.push_back(spec_.pick.node());
    }
  }

  Array root = greedy ? spec_.pick : spec_.logits;
  for (const graph::NodePtr& n : spec_.compute) {
    if (n) n->materialized = false;
  }
  const graph::NodePtr roots[] = {root.node()};
  LSE_RETURN_IF_ERROR(sched->eval(roots, true, &spec_.program));

  Verified out;
  if (greedy) {
    out.first = static_cast<std::uint32_t>(
        graph::interpreter::load_element(*spec_.pick.node(), 0));
    out.second = static_cast<std::uint32_t>(
        graph::interpreter::load_element(*spec_.pick.node(), 1));
  } else {
    LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(
        *spec_.logits.node(), sched->backend()));
    spec_logits_.resize(spec_.logits.shape().elem_count());
    LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(
        *spec_.logits.node(), spec_logits_.data(),
        spec_logits_.size() * sizeof(float)));
  }
  LSE_RETURN_IF_ERROR(read_hidden(hidden, &spec_hidden_));
  stats_.spec_verify_ns += now_ns() - started;
  ++stats_.spec_verify_passes;
  return out;
}

// Two tokens per decoder pass: the module proposes the second, the decoder
// verifies it in the same pass, and a rejected proposal is undone by replaying
// the pass with the decoder's own token in its place.
//
// The undo is why this is a pass and not a snapshot. After a pass the carried
// recurrent state still sits where that pass started from — the produced state
// lives on the other node of the carry pair and is not folded across until the
// next step — so re-running the pass from the same input with the corrected
// token reproduces exactly what a non-speculating decode would have carried.
// The paged KV needs no more than its cursor put back: the redo overwrites the
// same two slots.
Result<std::vector<std::uint32_t>> Generator::speculate(
    Session& session, std::vector<float>& prefill_logits,
    const GenerationLimits& limits, const TokenCallback& on_token) {
  std::vector<std::uint32_t> generated;
  generated.reserve(static_cast<std::size_t>(std::max(limits.max_tokens, 0)));
  if (limits.max_tokens <= 0) return generated;

  const auto is_stop = [&limits](std::uint32_t id) {
    return std::find(limits.stop_tokens.begin(), limits.stop_tokens.end(), id) !=
           limits.stop_tokens.end();
  };
  // Emits one token. False means generation is over, either because this one
  // is a stop token (which is not emitted), the caller cancelled, or the limit
  // is reached.
  const auto give = [&](std::uint32_t id) {
    if (is_stop(id)) return false;
    generated.push_back(id);
    session.history().push_back(id);
    ++stats_.generated_tokens;
    if (on_token && !on_token(id)) return false;
    return static_cast<std::int32_t>(generated.size()) < limits.max_tokens;
  };

  std::uint32_t pending = sampler_.sample(prefill_logits, session.history());
  bool running = give(pending);
  // The module's first row: the last prompt position's hidden state paired
  // with the token the decoder just produced there.
  std::uint32_t draft = 0;
  if (running) {
    const std::uint64_t started = now_ns();
    LSE_ASSIGN_OR(draft,
                  mtp_->draft(prefill_tail_, std::span(&pending, 1),
                              static_cast<std::int32_t>(session.position())));
    stats_.spec_draft_ns += now_ns() - started;
  }

  while (running) {
    const auto at = static_cast<std::int32_t>(session.position());
    LSE_ASSIGN_OR(Verified v, verify(session, pending, draft, false));
    ++stats_.spec_steps;

    std::uint32_t first = v.first;
    if (!spec_.greedy) {
      first = sampler_.sample(
          std::span<float>(spec_logits_.data(), spec_logits_.size() / 2),
          session.history());
    }
    const bool accepted = first == draft;
    if (accepted) ++stats_.spec_accepted;

    running = give(first);
    if (!running) {
      session.advance(2);
      break;
    }
    if (!accepted) {
      // The proposal was wrong, so the pass that consumed it is discarded and
      // re-run with the decoder's own token. Positions go back first: the pass
      // must land on the same two KV slots.
      model_.rewind(session.states(), at);
      LSE_ASSIGN_OR(v, verify(session, pending, first, true));
    }

    std::uint32_t second = v.second;
    if (!spec_.greedy) {
      second = sampler_.sample(
          std::span<float>(spec_logits_.data() + spec_logits_.size() / 2,
                           spec_logits_.size() / 2),
          session.history());
    }
    session.advance(2);
    running = give(second);
    pending = second;
    if (!running) break;

    const std::uint32_t pair[] = {first, second};
    const std::uint64_t drafted = now_ns();
    LSE_ASSIGN_OR(draft, mtp_->draft(spec_hidden_, pair, at + 1));
    stats_.spec_draft_ns += now_ns() - drafted;
  }

  // A step that stopped between its two halves left the decoder holding one
  // token more than was emitted, and then the cache no longer describes the
  // text. Dropping it is what keeps a follow-up turn honest; it costs that turn
  // a re-prefill and nothing else.
  if (static_cast<std::size_t>(session.position()) + 1 !=
      session.history().size()) {
    if (!session.restart().ok()) session.clear();
    mtp_->reset();
  }
  return generated;
}

Result<std::vector<std::uint32_t>> Generator::generate(
    const std::vector<std::uint32_t>& prompt, const GenerationLimits& limits,
    const TokenCallback& on_token) {
  if (owned_ == nullptr) {
    owned_ = std::make_unique<Session>("", model_.state_slots());
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
    // A cold start on a session that already owns arrays keeps them: restart()
    // zeroes the recurrence and releases the KV rows in place, so a program
    // retained against those arrays can still replay. clear() would drop the
    // arrays and orphan every retained program's leaves.
    if (!session.restart().ok()) session.clear();
    if (mtp_ != nullptr) mtp_->reset();
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
  if (mtp_ != nullptr) {
    LSE_ASSIGN_OR(generated, speculate(session, logits, limits, on_token));
    stats_.decode_ns = now_ns() - decode_start;
    snapshot_trace(&stats_, &host_reasons_);
    return generated;
  }
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
