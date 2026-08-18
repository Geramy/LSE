#include "lse/runtime/batch.hpp"

#include <algorithm>
#include <chrono>

#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/attention.hpp"
#include "lse/runtime/generator.hpp"

namespace lse::runtime {

namespace {

using graph::Array;

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Tokens one pass may carry. The same ladder generator.cpp walks a prompt with,
// and for the same reason: extents are baked into the generated HIP, so the
// reachable widths have to be a small fixed set. A batched pass of B sequences
// and a prefill pass of B tokens present the same row count to every GEMM.
constexpr std::int32_t kMaxStepWidth = 128;

// Zeroes one batch row of a device tensor in place.
//
// A slot handed to a new sequence must start from a clean recurrent state:
// GatedDeltaNet carries by value replacement, so whatever the previous occupant
// left in that row would be mixed into the new sequence's first token and the
// result would be fluent and wrong. This writes the buffer the next pass reads —
// Program::fold_carries moves the produced buffer onto the input node at the top
// of the next step, so the tensor a state Array names now is the one that
// arrives as input then.
Status zero_device_row(Array& a, std::int32_t row) {
  if (!a.valid() || !a.node()) return OkStatus();
  graph::Node& n = *a.node();
  // Never written, so it is still the zeros it was allocated as.
  if (!n.buffer.valid()) return OkStatus();
  const std::int64_t rows = n.shape.rank() > 0 ? n.shape.dim(0) : 0;
  if (rows <= 0 || row < 0 || row >= rows) {
    return LSE_ERROR(kOutOfRange, "no row ", std::to_string(row), " in ",
                     n.shape.to_string());
  }
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return LSE_ERROR(kInternal, "no backend to clear a row");
  const std::size_t per = n.element_count() / static_cast<std::size_t>(rows);
  const std::size_t bytes = dtype_storage_bytes(n.dtype, per);
  const std::size_t offset =
      dtype_storage_bytes(n.dtype, per * static_cast<std::size_t>(row));
  if (bytes == 0) return OkStatus();
  const std::vector<std::byte> zeros(bytes, std::byte{0});
  LSE_RETURN_IF_ERROR(
      sched->backend().copy_h2d(zeros.data(), n.buffer, bytes, offset));
  n.materialized = true;
  n.device_dirty = true;
  n.host_dirty = false;
  return OkStatus();
}

}  // namespace

BatchScheduler::BatchScheduler(HybridLM& model, SamplingParams params,
                               BatchLimits limits)
    : model_(model),
      sampler_(params),
      limits_(limits),
      policy_(kv::kBlockSize, limits.kv_reserve, /*host_blocks=*/0) {
  // Pure greedy with the penalty at its no-op setting is the one mode where the
  // sampler never reads more than the argmax, so B indices come back from the
  // device instead of B whole logit rows. Same condition Generator uses.
  const SamplingParams& sp = sampler_.params();
  device_greedy_ = sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f;
}

BatchScheduler::~BatchScheduler() = default;

Status BatchScheduler::configure_layers() {
  if (bucket_ == 0) {
    LSE_ASSIGN_OR(bucket_, model::batch_bucket(limits_.max_batch));
    slots_.assign(static_cast<std::size_t>(bucket_), Slot{});
    states_ = model_.make_states();
    const std::int32_t per_row =
        kv::blocks_for(model_.config().kv_capacity(), kv::kBlockSize);
    ceiling_ = limits_.kv_blocks > 0 ? limits_.kv_blocks : per_row * bucket_;
    if (ceiling_ <= 0) {
      return LSE_ERROR(kInvalidArgument, "a KV block budget of ",
                       std::to_string(ceiling_), " holds no sequence");
    }
    stats_.bucket = bucket_;
  }
  for (model::MixerState& st : states_) st.paged.block_ceiling = ceiling_;
  return OkStatus();
}

Status BatchScheduler::submit(BatchRequest request) {
  LSE_RETURN_IF_ERROR(configure_layers());
  if (request.prompt.empty()) {
    return LSE_ERROR(kInvalidArgument, "sequence '", request.id,
                     "' has an empty prompt");
  }
  Waiting w;
  w.request = std::move(request);
  if (w.request.max_tokens <= 0) w.request.max_tokens = limits_.max_tokens;
  w.submitted_ns = now_ns();
  waiting_.push_back(std::move(w));
  return OkStatus();
}

std::int32_t BatchScheduler::step_width() const {
  std::int32_t least = 0;
  for (const Slot& s : slots_) {
    if (!s.occupied || s.pending.empty()) continue;
    const auto have = static_cast<std::int32_t>(s.pending.size());
    if (least == 0 || have < least) least = have;
  }
  if (least == 0) return 0;
  // Round down to a rung. Down, never up: the width is what every row consumes,
  // and a row cannot overrun the tokens it has.
  std::int32_t w = 1;
  while (w * 2 <= least && w * 2 <= kMaxStepWidth) w *= 2;
  return w;
}

Status BatchScheduler::clear_row(std::int32_t row) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched != nullptr) {
    // The previous step's dispatches may still be reading this row. Retirement
    // is rare against the step rate, so the drain costs nothing per token.
    LSE_RETURN_IF_ERROR(sched->backend().synchronize());
  }
  for (model::MixerState& st : states_) {
    if (st.paged.valid()) {
      LSE_RETURN_IF_ERROR(ops::release_row(st.paged, row));
    }
    LSE_RETURN_IF_ERROR(zero_device_row(st.gdn_state, row));
    LSE_RETURN_IF_ERROR(zero_device_row(st.gdn_conv_q, row));
    LSE_RETURN_IF_ERROR(zero_device_row(st.gdn_conv_k, row));
    LSE_RETURN_IF_ERROR(zero_device_row(st.gdn_conv_v, row));
    LSE_RETURN_IF_ERROR(zero_device_row(st.gdn_conv_qkv, row));
  }
  return OkStatus();
}

Status BatchScheduler::retire(std::int32_t row, std::vector<SequenceResult>* out) {
  Slot& s = slots_[static_cast<std::size_t>(row)];
  if (!s.occupied) return OkStatus();
  SequenceResult r;
  r.id = s.id;
  r.generated = s.generated;
  r.prompt_tokens = s.prompt_tokens;
  r.ttft_ns = s.first_token_ns > s.submitted_ns
                  ? s.first_token_ns - s.submitted_ns
                  : 0;
  r.decode_ns = s.last_token_ns > s.first_token_ns
                    ? s.last_token_ns - s.first_token_ns
                    : 0;
  r.preemptions = s.preemptions;
  out->push_back(std::move(r));
  LSE_RETURN_IF_ERROR(clear_row(row));
  s = Slot{};
  return OkStatus();
}

Status BatchScheduler::preempt(std::int32_t row) {
  Slot& s = slots_[static_cast<std::size_t>(row)];
  if (!s.occupied) return OkStatus();
  // Dropped, not swapped out: with no host swap space the policy says drop, and
  // a dropped sequence pays a re-prefill of everything it has said so far. Its
  // history is exactly that prompt, so it resumes rather than restarting.
  Waiting w;
  w.request.id = s.id;
  w.request.prompt = s.history;
  w.request.max_tokens =
      s.max_tokens - static_cast<std::int32_t>(s.generated.size());
  w.submitted_ns = s.submitted_ns;
  w.preemptions = s.preemptions + 1;
  w.generated = s.generated;
  w.prompt_tokens = s.prompt_tokens;
  w.first_token_ns = s.first_token_ns;
  LSE_RETURN_IF_ERROR(clear_row(row));
  s = Slot{};
  waiting_.push_front(std::move(w));
  ++stats_.preemptions;
  return OkStatus();
}

// Blocks row `row` holds right now, read off the first layer that has a table
// for it. Every attention layer covers the same positions, so they agree.
std::int32_t BatchScheduler::blocks_held(std::size_t row) const {
  for (const model::MixerState& st : states_) {
    if (st.paged.tables.size() > row) {
      return st.paged.tables[row].size();
    }
  }
  return 0;
}

std::int32_t BatchScheduler::kv_blocks_free() const noexcept {
  for (const model::MixerState& st : states_) {
    if (st.paged.alloc.total() > 0) return st.paged.alloc.free_count();
  }
  return 0;
}

std::int32_t BatchScheduler::kv_blocks_total() const noexcept {
  for (const model::MixerState& st : states_) {
    if (st.paged.alloc.total() > 0) return st.paged.alloc.total();
  }
  return 0;
}

Status BatchScheduler::make_room(std::int32_t width) {
  for (;;) {
    std::vector<kv::SequenceDemand> resident;
    std::int32_t held_total = 0;
    std::int32_t need = 0;
    std::int32_t worst_row = -1;
    std::int32_t worst_short = 0;
    for (std::size_t r = 0; r < slots_.size(); ++r) {
      if (!slots_[r].occupied) continue;
      const std::int32_t held = blocks_held(r);
      const std::int32_t after = slots_[r].position + width;
      held_total += held;
      need += kv::blocks_for(after, kv::kBlockSize);
      resident.push_back({slots_[r].id, held, after, slots_[r].last_used});
      const std::int32_t shortfall =
          kv::blocks_for(after, kv::kBlockSize) - held;
      if (shortfall > worst_short) {
        worst_short = shortfall;
        worst_row = static_cast<std::int32_t>(r);
      }
    }
    // What the rows will hold once this step lands still fits the budget.
    if (need <= ceiling_ || worst_row < 0) return OkStatus();

    kv::BlockAllocator budget(ceiling_);
    for (std::int32_t i = 0; i < held_total && i < ceiling_; ++i) {
      LSE_ASSIGN_OR(const kv::BlockId taken, budget.acquire());
      (void)taken;
    }
    const Slot& want_slot = slots_[static_cast<std::size_t>(worst_row)];
    const kv::SequenceDemand want{want_slot.id, blocks_held(
                                      static_cast<std::size_t>(worst_row)),
                                  want_slot.position + width,
                                  want_slot.last_used};
    const kv::Admission plan = policy_.admit(want, budget, resident);
    if (plan.verdict == kv::Verdict::kRefuse) {
      return LSE_ERROR(kOutOfMemory, "sequence '", want.id, "' cannot grow: ",
                       plan.reason, " (pool budget ", std::to_string(ceiling_),
                       " block(s) of ", std::to_string(kv::kBlockSize),
                       " tokens)");
    }
    bool freed = false;
    for (const kv::Preemption& p : plan.preempt) {
      for (std::size_t r = 0; r < slots_.size(); ++r) {
        if (slots_[r].occupied && slots_[r].id == p.id) {
          LSE_RETURN_IF_ERROR(preempt(static_cast<std::int32_t>(r)));
          freed = true;
          break;
        }
      }
    }
    if (freed) continue;
    // The policy served this one row out of the free list, but the row set as a
    // whole is still over budget. Same rule applied to the set: oldest first,
    // id to break the tie, and never the row being served.
    std::int32_t victim = -1;
    for (std::size_t r = 0; r < slots_.size(); ++r) {
      if (!slots_[r].occupied ||
          static_cast<std::int32_t>(r) == worst_row) {
        continue;
      }
      if (victim < 0 ||
          slots_[r].last_used < slots_[static_cast<std::size_t>(victim)].last_used ||
          (slots_[r].last_used ==
               slots_[static_cast<std::size_t>(victim)].last_used &&
           slots_[r].id < slots_[static_cast<std::size_t>(victim)].id)) {
        victim = static_cast<std::int32_t>(r);
      }
    }
    if (victim < 0) {
      return LSE_ERROR(kOutOfMemory, "sequence '", want.id, "' needs ",
                       std::to_string(need), " block(s) alone, past the pool "
                       "budget of ", std::to_string(ceiling_));
    }
    LSE_RETURN_IF_ERROR(preempt(victim));
  }
}

Status BatchScheduler::admit_waiting() {
  while (!waiting_.empty()) {
    std::int32_t free_row = -1;
    for (std::size_t r = 0; r < slots_.size(); ++r) {
      if (!slots_[r].occupied) {
        free_row = static_cast<std::int32_t>(r);
        break;
      }
    }
    if (free_row < 0) return OkStatus();

    const Waiting& want = waiting_.front();
    // What the pool looks like to the policy: the whole budget, minus what the
    // resident sequences are committed to. The layer's own allocator is sized to
    // the rung it has grown to so far, which is smaller than the budget and
    // would make the policy refuse against a limit nobody set.
    //
    // "Committed", not "held": a sequence admitted this step has acquired
    // nothing yet but is going to need its whole prompt, and admitting a second
    // one against blocks the first has already been promised is how a pool that
    // was scheduled to fit runs out mid-step.
    std::vector<kv::SequenceDemand> resident;
    std::int32_t held_total = 0;
    for (std::size_t r = 0; r < slots_.size(); ++r) {
      if (!slots_[r].occupied) continue;
      const std::int32_t committed = std::max(
          blocks_held(r),
          kv::blocks_for(slots_[r].position +
                             static_cast<std::int32_t>(slots_[r].pending.size()),
                         kv::kBlockSize));
      held_total += committed;
      resident.push_back({slots_[r].id, committed, slots_[r].position + 1,
                          slots_[r].last_used});
    }
    kv::BlockAllocator budget(ceiling_);
    for (std::int32_t i = 0; i < held_total && i < ceiling_; ++i) {
      LSE_ASSIGN_OR(const kv::BlockId taken, budget.acquire());
      (void)taken;
    }

    kv::SequenceDemand demand;
    demand.id = want.request.id;
    demand.held = 0;
    demand.tokens_after = static_cast<std::int32_t>(want.request.prompt.size());
    demand.last_used = clock_;
    const kv::Admission plan = policy_.admit(demand, budget, resident);

    if (plan.verdict == kv::Verdict::kRefuse) {
      return LSE_ERROR(kOutOfMemory, "sequence '", want.request.id,
                       "' cannot be admitted: ", plan.reason,
                       " (pool budget ", std::to_string(ceiling_),
                       " block(s) of ", std::to_string(kv::kBlockSize),
                       " tokens)");
    }
    // kPreempt means it fits only by evicting somebody already decoding. That
    // is a trade this makes for a row that has to grow, never for one that has
    // not started: the sequence waits instead, and comes in when a row frees.
    if (plan.verdict == kv::Verdict::kPreempt) return OkStatus();

    Waiting taken = std::move(waiting_.front());
    waiting_.pop_front();
    LSE_RETURN_IF_ERROR(clear_row(free_row));
    Slot& s = slots_[static_cast<std::size_t>(free_row)];
    s = Slot{};
    s.id = taken.request.id;
    s.pending.assign(taken.request.prompt.begin(), taken.request.prompt.end());
    s.history = taken.request.prompt;
    s.generated = taken.generated;
    s.prompt_tokens = taken.prompt_tokens > 0
                          ? taken.prompt_tokens
                          : static_cast<std::int32_t>(taken.request.prompt.size());
    s.max_tokens = taken.request.max_tokens +
                   static_cast<std::int32_t>(taken.generated.size());
    s.position = 0;
    s.submitted_ns = taken.submitted_ns;
    s.admitted_ns = now_ns();
    s.first_token_ns = taken.first_token_ns;
    s.last_used = clock_;
    s.preemptions = taken.preemptions;
    s.occupied = true;
    ++stats_.admissions;
    // Only the first admission: a re-prefill after a preemption is a cost, not
    // more prompt, and counting it as prompt would flatter the throughput.
    if (s.preemptions == 0) stats_.prompt_tokens += s.prompt_tokens;
  }
  return OkStatus();
}

Status BatchScheduler::poke_ids(std::int32_t width) {
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to hold token ids");
  }
  if (!ids_.valid() || ids_width_ != width) {
    const auto count = static_cast<std::size_t>(bucket_) *
                       static_cast<std::size_t>(width);
    const std::size_t bytes = dtype_storage_bytes(DType::kF32, count);
    auto buf = sched->backend().allocate(bytes, backend::MemoryClass::kDevice);
    if (!buf.ok()) return buf.status();
    ids_ = Array::from_buffer(buf.release(), Shape{bucket_, width},
                              DType::kF32);
    ids_width_ = width;
  }
  graph::Node& n = *ids_.node();
  const std::size_t bytes = dtype_storage_bytes(n.dtype, n.element_count());
  if (n.host_mirror.size() < bytes) n.host_mirror.resize(bytes);
  for (std::size_t r = 0; r < slots_.size(); ++r) {
    const Slot& s = slots_[r];
    for (std::int32_t t = 0; t < width; ++t) {
      // A row holding no sequence still runs every kernel, so it needs a token;
      // 0 is as good as any, and its output is discarded by the caller and by
      // the attention guards alike.
      const float id =
          s.occupied && static_cast<std::size_t>(t) < s.pending.size()
              ? static_cast<float>(s.pending[static_cast<std::size_t>(t)])
              : 0.0f;
      graph::interpreter::store_element(
          n, r * static_cast<std::size_t>(width) + static_cast<std::size_t>(t),
          id);
    }
  }
  n.materialized = true;
  return OkStatus();
}

Result<std::vector<std::uint32_t>> BatchScheduler::run_step(std::int32_t width) {
  LSE_RETURN_IF_ERROR(configure_layers());
  LSE_RETURN_IF_ERROR(poke_ids(width));

  model::StepRows plan;
  plan.first.resize(slots_.size());
  for (std::size_t r = 0; r < slots_.size(); ++r) {
    plan.first[r] = slots_[r].occupied ? slots_[r].position : -1;
  }

  LSE_ASSIGN_OR(Array hidden,
                model_.hidden(ids_, &states_, nullptr, nullptr, &plan));

  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend for the lm_head");
  }

  const bool reuse = head_.hidden.valid() && head_.logits.valid() &&
                     head_.hidden.node().get() == hidden.node().get() &&
                     head_.greedy == device_greedy_ &&
                     (!device_greedy_ || head_.pick.valid());
  if (!reuse) {
    head_ = Head{};
    head_.hidden = hidden;
    head_.greedy = device_greedy_;
    const Shape& s = hidden.shape();
    Array last;
    if (s.dim(s.rank() - 2) == 1) {
      // T == 1, so the last row is the whole tensor: a reshape view suffices
      // and the slice copy of the general path is never built.
      Shape flat;
      for (std::size_t i = 0; i < s.rank(); ++i) {
        if (i + 2 == s.rank()) continue;
        flat.push_back(s.dim(i));
      }
      last = graph::reshape(hidden, flat);
    } else {
      LSE_ASSIGN_OR(last, Generator::last_hidden(hidden));
    }
    LSE_ASSIGN_OR(head_.logits, model_.lm_head(last));
    head_.compute = {last.node(), head_.logits.node()};
    if (device_greedy_) {
      head_.pick = graph::argmax(head_.logits);
      if (!head_.pick.valid()) {
        return LSE_ERROR(kInternal, "argmax over an empty logit row");
      }
      head_.compute.push_back(head_.pick.node()->inputs[0]);
      head_.compute.push_back(head_.pick.node());
    }
  }

  Array root = device_greedy_ ? head_.pick : head_.logits;
  for (const graph::NodePtr& n : head_.compute) {
    if (n) n->materialized = false;
  }
  const graph::NodePtr roots[] = {root.node()};
  LSE_RETURN_IF_ERROR(sched->eval(roots, true, &head_.program));

  std::vector<std::uint32_t> picks(static_cast<std::size_t>(bucket_), 0);
  if (device_greedy_) {
    for (std::int32_t r = 0; r < bucket_; ++r) {
      picks[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
          graph::interpreter::load_element(*root.node(),
                                           static_cast<std::size_t>(r)));
    }
    return picks;
  }

  const auto count = static_cast<std::size_t>(root.shape().elem_count());
  std::vector<float> logits(count);
  LSE_RETURN_IF_ERROR(graph::interpreter::read_raw(
      *root.node(), logits.data(), logits.size() * sizeof(float)));
  const std::size_t vocab = count / static_cast<std::size_t>(bucket_);
  for (std::size_t r = 0; r < slots_.size(); ++r) {
    std::vector<float> row(logits.begin() + static_cast<std::ptrdiff_t>(r * vocab),
                           logits.begin() +
                               static_cast<std::ptrdiff_t>((r + 1) * vocab));
    picks[r] = sampler_.sample(row, slots_[r].history);
  }
  return picks;
}

Result<std::vector<SequenceResult>> BatchScheduler::run(
    const TokenCallback& on_token) {
  LSE_RETURN_IF_ERROR(configure_layers());
  std::vector<SequenceResult> out;
  stats_.wall_ns = 0;
  const std::uint64_t start = now_ns();

  if (graph::Scheduler* sched = graph::default_scheduler()) {
    sched->reset_accumulated_trace();
  }

  const auto is_stop = [this](std::uint32_t id) {
    return std::find(limits_.stop_tokens.begin(), limits_.stop_tokens.end(),
                     id) != limits_.stop_tokens.end();
  };

  for (;;) {
    LSE_RETURN_IF_ERROR(admit_waiting());

    std::int32_t width = step_width();
    if (width == 0) break;
    // Blocks before tokens: a row that cannot hold what this step writes has to
    // be found now, while there is still a decision to make, rather than inside
    // the allocator with the pass half recorded.
    LSE_RETURN_IF_ERROR(make_room(width));
    width = step_width();
    // Everything got preempted to make room for a row that then also went. The
    // queue holds them all; go round and admit again.
    if (width == 0) continue;

    ++clock_;
    LSE_ASSIGN_OR(const std::vector<std::uint32_t> picks, run_step(width));
    ++stats_.steps;
    stats_.total_row_steps += bucket_;

    for (std::size_t r = 0; r < slots_.size(); ++r) {
      Slot& s = slots_[r];
      if (!s.occupied) continue;
      ++stats_.occupied_row_steps;
      s.last_used = clock_;
      for (std::int32_t t = 0; t < width && !s.pending.empty(); ++t) {
        s.pending.pop_front();
      }
      s.position += width;
      if (!s.pending.empty()) continue;  // still feeding its prompt

      // The prompt is in. This pass's last row is this sequence's next token.
      const std::uint32_t next = picks[r];
      const std::uint64_t at = now_ns();
      if (is_stop(next) ||
          static_cast<std::int32_t>(s.generated.size()) >= s.max_tokens) {
        s.done = true;
        continue;
      }
      s.generated.push_back(next);
      s.history.push_back(next);
      if (s.first_token_ns == 0) s.first_token_ns = at;
      s.last_token_ns = at;
      ++stats_.generated_tokens;
      if (on_token && !on_token(s.id, next)) {
        s.done = true;
        continue;
      }
      if (static_cast<std::int32_t>(s.generated.size()) >= s.max_tokens) {
        s.done = true;
        continue;
      }
      s.pending.push_back(next);
    }

    for (std::size_t r = 0; r < slots_.size(); ++r) {
      if (slots_[r].occupied && slots_[r].done) {
        LSE_RETURN_IF_ERROR(retire(static_cast<std::int32_t>(r), &out));
      }
    }
  }

  // Anything still holding a row when the loop stops has nothing left to feed.
  for (std::size_t r = 0; r < slots_.size(); ++r) {
    if (slots_[r].occupied) {
      LSE_RETURN_IF_ERROR(retire(static_cast<std::int32_t>(r), &out));
    }
  }

  stats_.wall_ns = now_ns() - start;
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    const graph::Scheduler::Trace& t = sched->accumulated_trace();
    stats_.kernels_launched = t.kernels_launched;
    stats_.host_groups = t.host_groups;
    stats_.jit_compiles = sched->jit_stats().compiles;
  }
  return out;
}

}  // namespace lse::runtime
