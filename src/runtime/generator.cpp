#include "lse/runtime/generator.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <chrono>

#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/core/hash.hpp"
#include "lse/runtime/decode_sample.hpp"
#include "lse/runtime/feature_prefix.hpp"

namespace lse::runtime {

namespace {
// Widest top-k the verify pass reads from the device instead of whole rows
// (topk_pairs bound).
constexpr std::size_t kSpecTopLimit = 32;

using graph::Array;
using lse::DType;
using lse::Shape;

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Ensure every early return (and unwinding) abandons an active observation.
// Cancellation changes policy bookkeeping only; it never claims GPU retirement.
struct DecodeSampleScope {
  backend::IBackend* backend = nullptr;
  ~DecodeSampleScope() { if (backend != nullptr) backend->cancel_decode_sample(); }
  Status finish(std::uint64_t elapsed, bool eligible) {
    if (backend == nullptr) return OkStatus();
    const Status status = backend->end_decode_sample(elapsed, eligible);
    if (status.ok()) backend = nullptr;
    return status;
  }
};

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

// Splits `n` tokens into consecutive passes sized from {chunk} u {powers of
// two from kRaggedRun below it} u {1 .. kRaggedRun - 1}, so the engine only
// ever compiles that many prefill shapes. 0 is one pass, whatever the length.
//
// Every pass streams the whole model's weights once, so a pass costs about
// the same at one row as at sixteen. The remainder below kRaggedRun is
// therefore one pass of its own width rather than one pass per set bit:
// 137 tokens run as 9 + 128, not 1 + 8 + 128.
//
// Ascending, which puts the ragged remainder first: the last pass is then the
// widest one, and it is the pass whose final row feeds the LM head. That keeps
// the logits row on the same linear kernel a single pass would have used,
// which is the closest a split can get to the unsplit answer.
constexpr std::size_t kRaggedRun = 16;

// The passes for n tokens: full chunks, and a remainder of at least
// kRaggedRun tokens as ONE pass padded up to a multiple of kRaggedRun, its
// padding past the last real token (HybridLM::hidden makes padded steps
// inert). A remainder below kRaggedRun runs at its own width. Every pass
// streams all the weights, so one padded pass costs what its width costs,
// where the power-of-two split paid a whole weight pass per set bit. The
// widths stay on a ladder of kRaggedRun, so the engine compiles at most
// chunk / kRaggedRun prefill shapes.
struct PassPlan { std::size_t width, valid; };
std::vector<PassPlan> padded_plan(std::size_t n, std::size_t chunk) {
  std::vector<PassPlan> plan;
  if (chunk == 0) return {{n, n}};
  const std::size_t rest = n % chunk;
  if (rest != 0) {
    const std::size_t width = rest < kRaggedRun ? rest
        : std::min(chunk, (rest + kRaggedRun - 1) / kRaggedRun * kRaggedRun);
    plan.push_back({width, rest});
  }
  for (std::size_t i = 0; i < n / chunk; ++i) plan.push_back({chunk, chunk});
  return plan;
}

std::vector<std::size_t> prefill_plan(std::size_t n, std::size_t chunk) {
  if (chunk == 0) return {n};
  std::vector<std::size_t> plan;
  std::size_t rest = n % chunk;
  if (const std::size_t ragged = rest % std::min(kRaggedRun, chunk); ragged != 0) {
    plan.push_back(ragged);
    rest -= ragged;
  }
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

Result<Array> Generator::last_hidden(const Array& hidden, std::int64_t valid) {
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
  const std::int64_t last = valid > 0 && valid <= t ? valid : t;
  Array row = graph::slice(hidden, static_cast<int>(s.rank()) - 2, last - 1, last);
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

// Shift target rows by one position, retaining the previous chunk's last row.
Status Generator::mtp_prefill_chunk(const Array& hidden,
                                    std::span<const std::uint32_t> tokens,
                                    std::int32_t first, Array* carry) {
  if (!hidden.valid() || hidden.dtype() != DType::kF32 ||
      hidden.shape().rank() != 3 || hidden.shape().dim(0) != 1 ||
      hidden.shape().dim(1) != static_cast<std::int64_t>(tokens.size())) {
    return LSE_ERROR(kInvalidArgument, "invalid target hidden rows for MTP");
  }
  const auto rows = static_cast<std::int64_t>(tokens.size());
  Array shifted = rows == 1 ? *carry
                            : graph::concat(
                                  {*carry, graph::slice(hidden, 1, 0, rows - 1)}, 1);
  LSE_RETURN_IF_ERROR(mtp_->draft(shifted, tokens, first).status());
  Array last = graph::slice(hidden, 1, rows - 1, rows);
  LSE_RETURN_IF_ERROR(last.materialize_owned());
  // Detach the retained value from a target graph that the next chunk replays.
  *carry = Array::from_buffer(last.node()->buffer, last.shape(), last.dtype());
  return OkStatus();
}

Result<std::vector<float>> Generator::step(
    Session& session, const std::vector<std::uint32_t>& tokens) {
  // A context still held from an earlier prompt lands before this one.
  LSE_RETURN_IF_ERROR(flush_draft_context());
  // A one-token prompt still goes the long way when the module is in play: the
  // decode head skips the hidden state the module needs for its first row.
  if (tokens.size() == 1 && mtp_ == nullptr && dflash2_ == nullptr) {
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
  Array carry;
  if (mtp_ != nullptr) {
    carry = base == 0
                ? Array::zeros(Shape{1, 1, model_.config().hidden_size}, DType::kF32)
                : session.mtp_tail();
    if (!carry.valid()) return LSE_ERROR(kInternal, "missing resident MTP target tail");
  }
  Array hidden;
  std::size_t at = 0;
  if (!prefill_batch_.valid())
    return LSE_ERROR(kInvalidArgument, "invalid prefill batch sizes");
  std::vector<PassPlan> passes;
  if (mtp_ == nullptr && prefill_batch_.batch_size % prefill_batch_.ubatch_size == 0) {
    passes = padded_plan(tokens.size(), prefill_batch_.ubatch_size);
  } else {
    for (const auto batch : prefill_plan(tokens.size(), prefill_batch_.batch_size)) {
      for (const auto c : prefill_plan(batch, prefill_batch_.ubatch_size))
        passes.push_back({c, c});
    }
  }
  std::size_t previous_width = 0;
  std::size_t last_valid = 0;
  for (const PassPlan& pass : passes) {
    const std::size_t take = pass.valid;
    if (previous_width != 0) {
      hidden = {};
      if (pass.width != previous_width) {
        LSE_RETURN_IF_ERROR(model_.retire_completed_passes(session.states()));
      }
    }
    previous_width = pass.width;
    last_valid = take;
    const auto first = tokens.begin() + static_cast<std::ptrdiff_t>(at);
    std::vector<std::uint32_t> pass_ids(first, first + static_cast<std::ptrdiff_t>(take));
    pass_ids.resize(pass.width, 0u);  // padding past the last real token
    LSE_ASSIGN_OR(Array ids, token_array(pass_ids));
    model::FeatureCapture capture;
    if (dflash2_ != nullptr) capture.layer_ids = dflash2_->target_layers();
    LSE_ASSIGN_OR(hidden, model_.hidden(ids, &session.states(), nullptr,
                                       nullptr, nullptr, false,
                                       dflash2_ != nullptr ? &capture : nullptr, false,
                                       ops::AttentionExecutionPhase::kPrefill,
                                       static_cast<std::int32_t>(take)));
    if (dflash2_ != nullptr && take < pass.width)
      capture.features = graph::slice(capture.features, 1, 0, static_cast<std::int64_t>(take));
    if (dflash2_ != nullptr) {
      if (&pass == &passes.back()) {
        pending_context_ = capture.features;
        pending_context_first_ = base + static_cast<std::int32_t>(at);
      } else {
        LSE_RETURN_IF_ERROR(dflash2_->append_context(
            capture.features, base + static_cast<std::int32_t>(at)));
      }
    }
    if (mtp_ != nullptr) {
      LSE_RETURN_IF_ERROR(mtp_prefill_chunk(
          hidden, std::span<const std::uint32_t>(&*first, take),
          base + static_cast<std::int32_t>(at), &carry));
    }
    at += take;
  }
  if (mtp_ != nullptr) {
    LSE_RETURN_IF_ERROR(session.retain_mtp_tail(
        carry, base + static_cast<std::int32_t>(tokens.size()), *mtp_));
    prefill_tail_ = session.mtp_tail();
  }
  LSE_ASSIGN_OR(Array last, last_hidden(hidden, static_cast<std::int64_t>(last_valid)));
  LSE_ASSIGN_OR(Array logits, model_.lm_head(last));

  std::vector<float> out(logits.shape().elem_count());
  LSE_RETURN_IF_ERROR(
      logits.to_host(out.data(), out.size() * sizeof(float)));
  LSE_RETURN_IF_ERROR(model_.retire_prefill(session.states()));
  if (dflash2_ != nullptr && !pending_context_.valid())
    LSE_RETURN_IF_ERROR(dflash2_->retire_prefill());
  return out;
}

Status Generator::flush_draft_context() {
  if (!pending_context_.valid()) return OkStatus();
  Array features = std::move(pending_context_);
  pending_context_ = {};
  if (dflash2_ == nullptr) return OkStatus();
  LSE_RETURN_IF_ERROR(dflash2_->append_context(features, pending_context_first_));
  return dflash2_->retire_prefill();
}

Status Generator::verify(Session& session,
                         std::span<const std::uint32_t> rows,
                         bool replaces_previous) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the verify pass");
  }
  const auto m = static_cast<std::int64_t>(rows.size());
  if (spec_ids_.valid() &&
      spec_ids_.shape().elem_count() != rows.size()) {
    // Park this width's head and ids and take the other width's back out;
    // each width keeps its recorded programs across switches.
    spec_by_m_[spec_ids_.shape().elem_count()] = std::move(spec_);
    spec_ids_by_m_[spec_ids_.shape().elem_count()] = std::move(spec_ids_);
    spec_ = std::move(spec_by_m_[rows.size()]);
    spec_ids_ = std::move(spec_ids_by_m_[rows.size()]);
  }
  if (!spec_ids_.valid()) {
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, rows.size());
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    spec_ids_ = Array::from_buffer(buf.release(), Shape{1, m}, DType::kF32);
  }
  {
    graph::Node& n = *spec_ids_.node();
    const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
    if (n.host_mirror.size() < bytes) n.host_mirror.resize(bytes);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      graph::interpreter::store_element(n, i, static_cast<float>(rows[i]));
    }
    n.materialized = true;
  }

  const std::uint64_t started = now_ns();
  LSE_RETURN_IF_ERROR(begin_verify_burst());
  model::FeatureCapture capture;
  if (dflash2_ != nullptr) capture.layer_ids = dflash2_->target_layers();
  LSE_ASSIGN_OR(Array hidden,
                model_.hidden(spec_ids_, &session.states(), nullptr, nullptr,
                              nullptr, replaces_previous,
                              dflash2_ != nullptr ? &capture : nullptr,
                              m > 1, ops::AttentionExecutionPhase::kSpeculative));
  spec_features_ = std::move(capture.features);

  const SamplingParams& sp = sampler_.params();
  const bool greedy = sp.greedy_argmax();
  const auto vocab = static_cast<std::size_t>(model_.config().vocab_size);
  const std::uint32_t top_k = !greedy && sampler_.decided_by_top(kSpecTopLimit, vocab)
                                  ? static_cast<std::uint32_t>(sp.top_k) : 0u;
  const bool reuse = spec_.hidden.valid() && spec_.logits.valid() &&
                     spec_.hidden.node().get() == hidden.node().get() &&
                     spec_.greedy == greedy && (!greedy || spec_.pick.valid()) &&
                     spec_.top_k == top_k;
  if (!reuse) {
    spec_ = SpecHead{};
    spec_.hidden = hidden;
    spec_.greedy = greedy;
    // [1, m, D] -> [1, m, vocab]: every row goes through the head, because
    // each verified proposal buys its own row of logits.
    LSE_ASSIGN_OR(spec_.logits, model_.lm_head(hidden));
    spec_.compute = {spec_.logits.node()};
    if (greedy) {
      spec_.pick = graph::argmax(spec_.logits);
      if (!spec_.pick.valid()) {
        return LSE_ERROR(kInternal, "argmax over an empty logit row");
      }
      spec_.compute.push_back(spec_.pick.node()->inputs[0]);
      spec_.compute.push_back(spec_.pick.node());
    } else if (top_k != 0) {
      spec_.top = graph::topk_pairs(spec_.logits, static_cast<int>(top_k));
      if (!spec_.top.valid()) {
        return LSE_ERROR(kInternal, "no device top-", std::to_string(top_k),
                         " over the verify logits");
      }
      spec_.top_k = top_k;
      std::vector<graph::NodePtr> stages;
      for (graph::NodePtr n = spec_.top.node(); n && n != spec_.logits.node();
           n = n->inputs.empty() ? nullptr : n->inputs[0]) {
        stages.push_back(n);
      }
      spec_.compute.insert(spec_.compute.end(), stages.rbegin(), stages.rend());
    }
  }

  Array root = greedy ? spec_.pick : spec_.top_k != 0 ? spec_.top : spec_.logits;
  for (const graph::NodePtr& n : spec_.compute) {
    if (n) n->materialized = false;
  }
  const graph::NodePtr roots[] = {root.node()};
  // Greedy pulls its m picks back here. Sampling leaves the logits on the
  // device: the acceptance walk reads rows as it reaches them
  // (spec_logit_rows), and it stops at the first rejected proposal.
  LSE_RETURN_IF_ERROR(sched->eval(roots, greedy, &spec_.program));
  // Greedy picks came back with the pass.
  if (greedy) end_verify_burst();

  if (!greedy) {
    if (spec_.logits.dtype() != DType::kF32 || spec_.logits.shape().rank() != 3 ||
        spec_.logits.shape().dim(1) != m) {
      return LSE_ERROR(kInternal, "verifier logits are not [1, m, vocab] f32");
    }
    if (spec_.top_k != 0) {
      spec_top_.resize(spec_.top.shape().elem_count());
      spec_top_ready_ = false;
    } else {
      spec_logits_.resize(spec_.logits.shape().elem_count());
      spec_rows_ready_ = 0;
    }
  }
  stats_.spec_verify_ns += now_ns() - started;
  ++stats_.spec_verify_passes;
  return OkStatus();
}

// Rows [0, rows) of the last verify pass's logits, on the host.
//
// The walk over a pass's rows ends at the first proposal it rejects, so most
// rows are never looked at: at the 68% acceptance measured on the pinned
// Qwen 27B + DFlash2 run a step reads about 3 of its 8 rows, and a row is
// vocab x 4 bytes (0.95 MiB at 248320), copied over the link at ~1.75 GB/s.
// Reading all eight cost 4.6 ms of every ~94 ms step with the GPU idle.
// Transfers double -- one row, one more, then two, then four -- so a walk
// that accepts everything pays four transfers instead of one, and a walk
// that stops early pays for little more than the rows it read.
Status Generator::spec_logit_rows(std::size_t rows) {
  if (rows <= spec_rows_ready_) return OkStatus();
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the verify logits");
  }
  const auto m = static_cast<std::size_t>(spec_.logits.shape().dim(1));
  if (rows > m || spec_logits_.size() != spec_.logits.shape().elem_count()) {
    return LSE_ERROR(kInternal, "verify logit row ", std::to_string(rows),
                     " past the pass's ", std::to_string(m));
  }
  const std::uint64_t started = now_ns();
  graph::Node& node = *spec_.logits.node();
  if (spec_rows_ready_ == 0) {
    // The pass was submitted, not awaited.
    LSE_RETURN_IF_ERROR(sched->drain());
  }
  const std::size_t v = spec_logits_.size() / m;
  const std::size_t row_bytes = v * sizeof(float);
  if (node.kv_fragments || node.buffer.ptr != nullptr || !node.buffer.valid() ||
      !node.device_dirty) {
    // Not a plain device buffer: the whole tensor through the interpreter,
    // which knows every other residency.
    LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(node, sched->backend()));
    LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(
        node, spec_logits_.data(), spec_logits_.size() * sizeof(float)));
    spec_rows_ready_ = m;
  } else {
    const std::size_t want = std::min(
        m, std::max(rows, spec_rows_ready_ == 0 ? std::size_t{1} : 2 * spec_rows_ready_));
    LSE_RETURN_IF_ERROR(sched->backend().copy(
        spec_logits_.data() + spec_rows_ready_ * v,
        backend::MemRef(node.buffer, spec_rows_ready_ * row_bytes),
        (want - spec_rows_ready_) * row_bytes));
    spec_rows_ready_ = want;
  }
  stats_.spec_verify_ns += now_ns() - started;
  end_verify_burst();
  return OkStatus();
}

Status Generator::begin_verify_burst() {
  end_verify_burst();
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr || sched->devices().size() != 1) return OkStatus();
  burst_ = &sched->backend();
  return burst_->begin_launch_burst();
}

void Generator::end_verify_burst() noexcept {
  if (burst_ != nullptr) burst_->end_launch_burst();
  burst_ = nullptr;
}

// Row `row` of the last verify pass's device top-k. All m rows' candidates
// are a few KiB, so the first row read brings every row in one copy.
Status Generator::spec_top_row(std::size_t row) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the verify candidates");
  }
  const auto m = static_cast<std::size_t>(spec_.logits.shape().dim(1));
  const std::size_t k = spec_.top_k;
  if (row >= m || k == 0 || spec_top_.size() != m * k * 2) {
    return LSE_ERROR(kInternal, "verify candidate row ", std::to_string(row),
                     " past the pass's ", std::to_string(m));
  }
  if (!spec_top_ready_) {
    const std::uint64_t started = now_ns();
    graph::Node& node = *spec_.top.node();
    // The pass was submitted, not awaited. A device-to-host copy waits for
    // the work it reads (or rides the stream behind it), so the plain-buffer
    // read below needs no drain of its own.
    if (node.kv_fragments || node.buffer.ptr != nullptr || !node.buffer.valid() ||
        !node.device_dirty) {
      LSE_RETURN_IF_ERROR(sched->drain());
      LSE_RETURN_IF_ERROR(graph::interpreter::sync_from_device(node, sched->backend()));
      LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(
          node, spec_top_.data(), spec_top_.size() * sizeof(float)));
    } else {
      LSE_RETURN_IF_ERROR(sched->backend().copy(
          spec_top_.data(), backend::MemRef(node.buffer, 0),
          spec_top_.size() * sizeof(float)));
    }
    spec_top_ready_ = true;
    stats_.spec_verify_ns += now_ns() - started;
    end_verify_burst();
  }
  const auto vocab = static_cast<float>(model_.config().vocab_size);
  spec_top_values_.resize(k);
  spec_top_ids_.resize(k);
  for (std::size_t j = 0; j < k; ++j) {
    const float value = spec_top_[(row * k + j) * 2];
    const float index = spec_top_[(row * k + j) * 2 + 1];
    if (!(index >= 0.0f && index < vocab) || std::floor(index) != index) {
      return LSE_ERROR(kInternal, "verify candidate index ", std::to_string(index),
                       " is not a token");
    }
    spec_top_values_[j] = value;
    spec_top_ids_[j] = static_cast<std::uint32_t>(index);
  }
  return OkStatus();
}

Status Generator::append_draft_context(std::size_t rows, std::int32_t first) {
  if (dflash2_ == nullptr) return OkStatus();
  if (!spec_features_.valid() || rows == 0 ||
      rows > static_cast<std::size_t>(spec_features_.shape().dim(1))) {
    return LSE_ERROR(kInternal, "invalid verified draft feature prefix");
  }
  const auto started = now_ns();
  LSE_ASSIGN_OR(Array prefix, materialized_feature_prefix(spec_features_, rows));
  LSE_RETURN_IF_ERROR(dflash2_->append_context(prefix, first));
  stats_.spec_draft_ns += now_ns() - started;
  return OkStatus();
}

// Commit valid verifier inputs and discard the unaccepted suffix's state.
Result<std::vector<std::uint32_t>> Generator::speculate(
    Session& session, std::vector<float>& prefill_logits,
    const GenerationLimits& limits, const TokenCallback& on_token) {
  std::vector<std::uint32_t> generated;
  if (limits.max_tokens <= 0) return generated;
  struct BurstScope {
    Generator* g;
    ~BurstScope() { g->end_verify_burst(); }
  } burst_scope{this};
  const auto capacity = static_cast<std::size_t>(model_.config().kv_capacity());

  const std::uint32_t depth = dflash2_ != nullptr
                                  ? dflash2_verify_depth(dflash2_->block_size())
                                  : limits.mtp_depth;
  std::size_t m = 0;
  const bool sampled_dflash = dflash2_ != nullptr && sampler_.params().temperature > 0;
  SpeculativeSampler speculative_sampler(sampler_.params().seed ^
      (static_cast<std::uint64_t>(session.position()) * 0x9e3779b97f4a7c15ull));
  std::vector<DiscreteDistribution> proposal_distributions;
  const auto next_width = [&] {
    return mtp_verify_rows(
        depth, static_cast<std::uint64_t>(limits.max_tokens) - generated.size(),
        static_cast<std::int64_t>(model_.config().kv_capacity()) - session.position());
  };
  const auto draft_for_width = [&](const Array& hidden,
                                    std::span<const std::uint32_t> tokens,
                                    std::int32_t first, std::uint32_t proposals)
      -> Result<std::vector<std::uint32_t>> {
    if (dflash2_ != nullptr) {
      if (proposals == 0) return std::vector<std::uint32_t>{};
      if (sampled_dflash) {
        LSE_ASSIGN_OR(auto proposal, dflash2_->draft_sampled(tokens.back(),
            first + static_cast<std::int32_t>(tokens.size()) - 1,
            proposals, sampler_.params().temperature, speculative_sampler));
        proposal_distributions = std::move(proposal.conditionals);
        return std::move(proposal.tokens);
      }
      return dflash2_->draft(tokens.back(),
                            first + static_cast<std::int32_t>(tokens.size()) - 1,
                            proposals);
    }
    if (proposals == 0) {
      // Even a one-row verifier must keep the draft cache caught up.
      LSE_RETURN_IF_ERROR(mtp_->draft(hidden, tokens, first).status());
      return std::vector<std::uint32_t>{};
    }
    return mtp_->draft_chain(hidden, tokens, first, proposals);
  };

  const auto is_stop = [&limits](std::uint32_t id) {
    return std::find(limits.stop_tokens.begin(), limits.stop_tokens.end(), id) !=
           limits.stop_tokens.end();
  };
  const auto give = [&](std::uint32_t id) {
    if (is_stop(id)) { stats_.stop_reason = StopReason::kStopToken; return false; }
    // The prompt and the generated tokens never exceed the KV length.
    if (session.history().size() >= capacity) {
      stats_.stop_reason = StopReason::kContextFull;
      return false;
    }
    generated.push_back(id);
    session.history().push_back(id);
    ++stats_.generated_tokens;
    if (on_token && !on_token(id)) { stats_.stop_reason = StopReason::kCallback; return false; }
    if (static_cast<std::int32_t>(generated.size()) >= limits.max_tokens) {
      stats_.stop_reason = StopReason::kMaxTokens;
      return false;
    }
    return true;
  };
  // No verifier row fits: the KV length is used up.
  const auto out_of_positions = [&] {
    if (stats_.stop_reason == StopReason::kNone) stats_.stop_reason = StopReason::kContextFull;
  };
  // The decoder's answer for row i of the pass just verified.
  const auto answer = [&](std::size_t i) -> Result<std::uint32_t> {
    if (spec_.greedy) {
      return static_cast<std::uint32_t>(
          graph::interpreter::load_element(*spec_.pick.node(), i));
    }
    if (spec_.top_k != 0) {
      if (sampled_dflash) {
        LSE_ASSIGN_OR(auto target, sampler_.distribution_top(spec_top_values_, spec_top_ids_));
        return speculative_sampler.sample_target(target);
      }
      return sampler_.sample_top(spec_top_values_, spec_top_ids_);
    }
    const std::size_t v = spec_logits_.size() / m;
    auto logits = std::span<float>(spec_logits_.data() + i * v, v);
    if (sampled_dflash) {
      LSE_ASSIGN_OR(auto target, sampler_.distribution(logits, session.history()));
      return speculative_sampler.sample_target(target);
    }
    return sampler_.sample(logits, session.history());
  };

  std::uint32_t pending = sampler_.sample(prefill_logits, session.history());
  bool running = give(pending);
  // The first token is out; the draft's context for the prompt's last pass
  // goes in now, before anything drafts from it.
  LSE_RETURN_IF_ERROR(flush_draft_context());
  std::uint64_t decode_start = 0;

  std::size_t retained_rows = 0;
  std::int32_t retained_first = 0;
  std::vector<std::uint32_t> row_in;   // the m tokens the next pass consumes
  if (running) {
    const std::uint32_t width = next_width();
    if (width == 0) {
      out_of_positions();
      running = false;
    } else {
      decode_start = now_ns();
      const std::uint64_t started = now_ns();
      LSE_ASSIGN_OR(std::vector<std::uint32_t> chain,
                    draft_for_width(prefill_tail_, std::span(&pending, 1),
                                    session.position(), width - 1));
      stats_.spec_draft_ns += now_ns() - started;
      row_in.assign(1, pending);
      row_in.insert(row_in.end(), chain.begin(), chain.end());
    }
  }

  while (running) {
    m = row_in.size();
    const auto at = static_cast<std::int32_t>(session.position());
    LSE_RETURN_IF_ERROR(verify(session, row_in, false));
    ++stats_.spec_steps;

    std::size_t emitted_to = 0;
    bool mismatch = false;
    std::vector<std::uint32_t> answers(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
      if (!spec_.greedy) {
        LSE_RETURN_IF_ERROR(spec_.top_k != 0 ? spec_top_row(i) : spec_logit_rows(i + 1));
      }
      bool accepted = false;
      double overlap = 0, candidate_mass = 0, deterministic_mass = 0;
      if (sampled_dflash && i + 1 < m) {
        const auto v = spec_.top_k != 0 ? 0 : spec_logits_.size() / m;
        LSE_ASSIGN_OR(auto target, spec_.top_k != 0
            ? sampler_.distribution_top(spec_top_values_, spec_top_ids_)
            : sampler_.distribution(std::span<float>(spec_logits_.data() + i * v, v),
                                    session.history()));
        const auto& proposal = proposal_distributions[i];
        const auto best = static_cast<std::size_t>(std::max_element(
            proposal.probabilities.begin(), proposal.probabilities.end()) - proposal.probabilities.begin());
        deterministic_mass = target.probability(proposal.ids[best]);
        for (std::size_t k = 0; k < proposal.ids.size(); ++k) {
          const auto p = target.probability(proposal.ids[k]);
          candidate_mass += p;
          overlap += std::min(p, proposal.probabilities[k]);
        }
        LSE_ASSIGN_OR(auto verified, speculative_sampler.verify(target, proposal, row_in[i + 1]));
        answers[i] = verified.token;
        accepted = verified.accepted;
      } else {
        LSE_ASSIGN_OR(answers[i], answer(i));
        accepted = i + 1 < m && answers[i] == row_in[i + 1];
      }
      running = give(answers[i]);
      emitted_to = i + 1;
      if (!running) break;
      if (i + 1 < m) {
        ++stats_.spec_tested;
        ++stats_.spec_tested_by_position[i];
        stats_.spec_overlap_sum[i] += overlap;
        stats_.spec_candidate_mass_sum[i] += candidate_mass;
        stats_.spec_deterministic_mass_sum[i] += deterministic_mass;
        if (accepted) {
          ++stats_.spec_accepted;
          ++stats_.spec_accepted_by_position[i];
        } else {
          mismatch = true;
          break;
        }
      }
    }
    retained_rows = emitted_to;
    retained_first = at;
    if (!running) {
      if (emitted_to < m) {
        const auto started = now_ns();
        LSE_RETURN_IF_ERROR(model_.commit_prefix(session.states(), emitted_to));
        stats_.spec_verify_ns += now_ns() - started;
      }
      LSE_RETURN_IF_ERROR(append_draft_context(emitted_to, at));
      session.advance(static_cast<std::int32_t>(emitted_to));
      break;
    }

    LSE_RETURN_IF_ERROR(append_draft_context(emitted_to, at));
    if (!mismatch) {
      // Every row agreed: the pass stands, and the module catches up on the
      // decoder's hiddens for all of it before chaining the next proposals.
      session.advance(static_cast<std::int32_t>(m));
      pending = answers[m - 1];
      std::vector<std::uint32_t> caught(answers.begin(), answers.end());
      const std::uint32_t width = next_width();
      if (width == 0) {
        out_of_positions();
        running = false;
        break;
      }
      const std::uint64_t drafted = now_ns();
      LSE_ASSIGN_OR(std::vector<std::uint32_t> chain,
                    draft_for_width(spec_.hidden, caught, at + 1, width - 1));
      stats_.spec_draft_ns += now_ns() - drafted;
      row_in.assign(1, pending);
      row_in.insert(row_in.end(), chain.begin(), chain.end());
      continue;
    }

    const std::size_t good = emitted_to;
    const auto commit_started = now_ns();
    LSE_RETURN_IF_ERROR(model_.commit_prefix(session.states(), good));
    stats_.spec_verify_ns += now_ns() - commit_started;
    session.advance(static_cast<std::int32_t>(good));
    pending = answers[good - 1];
    const std::uint32_t width = next_width();
    if (width == 0) {
      out_of_positions();
      running = false;
      break;
    }
    const auto drafted = now_ns();
    LSE_ASSIGN_OR(
        std::vector<std::uint32_t> chain,
        draft_for_width(
            graph::slice(spec_.hidden, 1, 0, static_cast<std::int64_t>(good)),
            std::span<const std::uint32_t>(answers.data(), good), at + 1,
            width - 1));
    stats_.spec_draft_ns += now_ns() - drafted;
    row_in.assign(1, pending);
    row_in.insert(row_in.end(), chain.begin(), chain.end());
  }

  if (mtp_ != nullptr && retained_rows > 0) {
    const auto started = now_ns();
    // Row zero already used the preceding verified target hidden. Replace the
    // remaining draft KV rows with their verified target-hidden/token pairs.
    if (retained_rows > 1) {
      LSE_ASSIGN_OR(Array prefix,
                    materialized_feature_prefix(spec_.hidden, retained_rows - 1));
      LSE_RETURN_IF_ERROR(mtp_->draft(
          prefix, std::span<const std::uint32_t>(row_in.data() + 1, retained_rows - 1),
          retained_first + 1).status());
    }
    LSE_RETURN_IF_ERROR(mtp_->truncate(session.position()));
    const auto last = static_cast<std::int64_t>(retained_rows - 1);
    Array tail = graph::slice(spec_.hidden, 1, last, last + 1);
    LSE_RETURN_IF_ERROR(session.retain_mtp_tail(tail, session.position(), *mtp_));
    stats_.spec_draft_ns += now_ns() - started;
  }

  const auto draft_position = mtp_ != nullptr ? mtp_->position()
                                              : dflash2_->context_position();
  const auto covered = static_cast<std::size_t>(session.position());
  const bool history_matches = covered + 1 == session.history().size() ||
                               covered == session.history().size();
  if (!history_matches || draft_position != session.position()) {
    if (!session.restart().ok()) session.clear();
    if (mtp_ != nullptr) mtp_->reset();
    if (dflash2_ != nullptr) dflash2_->reset();
  }
  stats_.decode_ns = decode_start == 0 ? 0 : now_ns() - decode_start;
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
  if (!prefill_batch_.valid())
    return LSE_ERROR(kInvalidArgument, "invalid prefill batch sizes");
  if (!valid_mtp_depth(limits.mtp_depth)) {
    return LSE_ERROR(kInvalidArgument, "mtp_depth must be an integer from 1 to 7");
  }
  if (prompt.empty()) {
    return LSE_ERROR(kInvalidArgument, "cannot generate from an empty prompt");
  }
  const std::int32_t capacity = model_.config().kv_capacity();
  if (prompt.size() > static_cast<std::size_t>(capacity)) {
    return LSE_ERROR(kOutOfRange, "context_full: the prompt is ", std::to_string(prompt.size()),
                     " tokens and the context holds ", std::to_string(capacity));
  }
  LSE_RETURN_IF_ERROR(flush_draft_context());
  stats_ = GenerationStats{};
  stats_.context_length = capacity;
  // Set on every return path below that produced tokens.
  struct ContextTally {
    GenerationStats& stats;
    const Session& session;
    ~ContextTally() { stats.context_tokens = static_cast<std::int32_t>(session.history().size()); }
  } tally{stats_, session};
  stats_.mtp_depth = mtp_ != nullptr ? limits.mtp_depth : 0;
  stats_.dflash2_depth = dflash2_ != nullptr ? dflash2_verify_depth(dflash2_->block_size()) : 0;
  host_reasons_.clear();
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    sched->reset_accumulated_trace();
  }

  // Only what the cache does not already cover. A follow-up turn whose prompt
  // extends the previous one therefore costs its new tokens, not the whole
  // conversation; anything else is a cold start.
  const auto covered = static_cast<std::size_t>(session.position());
  const bool continues =
      (mtp_ == nullptr || session.mtp_context_matches(*mtp_)) &&
      covered > 0 && covered < prompt.size() &&
      covered <= session.history().size() &&
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
    if (dflash2_ != nullptr) dflash2_->reset();
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
  LSE_RETURN_IF_ERROR(model_.retire_completed_passes(session.states()));
  LSE_ASSIGN_OR(std::vector<float> logits, step(session, fresh));
  session.advance(static_cast<std::int32_t>(fresh.size()));
  stats_.prefill_ns = now_ns() - prefill_start;

  std::vector<std::uint32_t> generated;

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
  const bool device_greedy = sp.greedy_argmax();

  std::uint64_t decode_start = 0;
  if (mtp_ != nullptr || dflash2_ != nullptr) {
    LSE_ASSIGN_OR(generated, speculate(session, logits, limits, on_token));
    snapshot_trace(&stats_, &host_reasons_);
    return generated;
  }
  std::uint32_t next = 0;
  if (limits.max_tokens > 0) next = sampler_.sample(logits, session.history());
  const auto full = static_cast<std::size_t>(capacity);
  for (std::int32_t n = 0; n < limits.max_tokens; ++n) {
    if (is_stop(next)) { stats_.stop_reason = StopReason::kStopToken; break; }
    // The prompt and the generated tokens never exceed the KV length.
    if (session.history().size() >= full) { stats_.stop_reason = StopReason::kContextFull; break; }

    generated.push_back(next);
    session.history().push_back(next);
    ++stats_.generated_tokens;

    if (on_token && !on_token(next)) { stats_.stop_reason = StopReason::kCallback; break; }
    if (n + 1 == limits.max_tokens) { stats_.stop_reason = StopReason::kMaxTokens; break; }
    // A token the context cannot hold is not worth a decode step.
    if (session.history().size() >= full) { stats_.stop_reason = StopReason::kContextFull; break; }

    // The first token came from prefill. Start timing only when its successor
    // needs a model step, after the first token has been delivered.
    if (decode_start == 0) decode_start = now_ns();
    // Measure ordinary complete decode steps, never prefill or speculative
    // verification. hidden() alone only submits work: logits/readback closes
    // the interval and includes the actual host recording/GPU overlap.
    graph::Scheduler* tuning_sched = graph::default_scheduler();
    const auto* chain = tuning_sched != nullptr && tuning_sched->devices().size() == 1
        ? tuning_sched->toolchain(0) : nullptr;
    backend::IBackend* tuning_backend = chain != nullptr && chain->dialect == graph::Dialect::kLoom
        ? &tuning_sched->backend() : nullptr;
    DecodeSampleCounters before;
    std::uint64_t partition_ns_before = 0, tuning_key = 0;
    DecodeSampleScope observation{tuning_backend};
    // Include any begin-boundary drain. The backend adds its final drain to
    // this interval before accepting a sample; neither cost is hidden.
    const auto step_start = now_ns();
    if (tuning_backend != nullptr) {
      // Cache only within this backend lifetime. Context buckets prevent a
      // short-context choice from silently covering very different KV work.
      tuning_key = hash_mix(kHashSeed, reinterpret_cast<std::uintptr_t>(&model_));
      tuning_key = hash_mix(tuning_key, device_greedy);
      tuning_key = hash_mix(tuning_key, static_cast<std::uint64_t>(session.position()) / 128);
      LSE_RETURN_IF_ERROR(tuning_backend->begin_decode_sample(tuning_key));
      const auto jit = tuning_sched->jit_stats();
      const auto& trace = tuning_sched->accumulated_trace();
      before = {jit.compiles, jit.disk_hits, trace.partition_passes, trace.host_groups};
      partition_ns_before = trace.partition_ns;
    }
    const auto advanced = [&]() -> Result<std::uint32_t> {
      // Only the new token goes in: each block already holds the prior state.
      if (device_greedy) return greedy_step(session, next);
      LSE_ASSIGN_OR(logits, step(session, {next}));
      return sampler_.sample(logits, session.history());
    }();
    if (!advanced.ok()) return advanced.status();
    const auto step_ns = now_ns() - step_start;
    if (tuning_backend != nullptr) {
      const auto jit_after = tuning_sched->jit_stats();
      const auto& trace = tuning_sched->accumulated_trace();
      const DecodeSampleCounters after{jit_after.compiles, jit_after.disk_hits,
                                       trace.partition_passes, trace.host_groups};
      const bool warm = warm_decode_sample(before, after);
      static const bool tuning_trace = [] {
        const char* value = std::getenv("LSE_AUTO_BATCH_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
      }();
      static std::atomic<unsigned> evidence_lines{0};
      if (tuning_trace && evidence_lines.fetch_add(1) < 256) {
        std::fprintf(stderr, "[batch-tune-evidence] workload=%llx position=%d eligible=%d "
                             "compiles=%llu disk-loads=%llu partition-passes=%llu "
                             "partition-ns=%llu host-groups=%llu elapsed-ns=%llu\n",
                     static_cast<unsigned long long>(tuning_key), session.position(), warm ? 1 : 0,
                     static_cast<unsigned long long>(after.compiles - before.compiles),
                     static_cast<unsigned long long>(after.disk_loads - before.disk_loads),
                     static_cast<unsigned long long>(after.partition_passes - before.partition_passes),
                     static_cast<unsigned long long>(trace.partition_ns - partition_ns_before),
                     static_cast<unsigned long long>(after.host_groups - before.host_groups),
                     static_cast<unsigned long long>(step_ns));
      }
      LSE_RETURN_IF_ERROR(observation.finish(step_ns, warm));
    }
    next = *advanced;
    session.advance(1);
  }
  stats_.decode_ns = decode_start == 0 ? 0 : now_ns() - decode_start;
  snapshot_trace(&stats_, &host_reasons_);

  return generated;
}

}  // namespace lse::runtime
