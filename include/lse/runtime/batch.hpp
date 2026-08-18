// Continuous batching: several sequences decoding in one step, joining and
// leaving between steps.
//
// The engine has a fixed number of rows — a batch bucket, because the bucket is
// what the JIT keys on (model::kBatchRungs) — and a sequence occupies one of
// them for as long as it is in flight. Every step advances every occupied row by
// the same number of tokens; the rows differ in where they are, not in how far
// they move. That difference is the whole thing: rows sit at different absolute
// positions with different live KV lengths, and the step descriptor carries a
// pair per row so the causal mask, the softmax bound and the RoPE origin are all
// per row (see kv/block.hpp).
//
// Why this and not a faster link, a better kernel or a bigger cache: PLAN.md's
// argument is that a latency is only on the critical path when there is nothing
// else to do. A pipeline of s stages with n items in flight idles
// (s-1)/(s-1+n) of the time, so depth is what turns a latency problem into a
// throughput problem — and depth is exactly what a batch of sequences is. It is
// also the only thing that raises M above 1 during decode, which is the only
// way the matrix cores are reachable there at all.
//
// What a row costs, measured on this box at KV~450: eight sequences in one step
// cost 1.18x what one costs. The per-row work is real but the step is
// latency-bound, not bandwidth-bound, so the rows past the first are nearly
// free until the arithmetic changes family.
//
// Relationship to Session: a Session is a conversation's *resident* cache, held
// between turns and decoded alone. A Slot here is a sequence's *in-flight*
// state, and its KV lives in the batch's shared pool for as long as it is
// admitted. They are deliberately not the same object.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/program.hpp"
#include "lse/kv/policy.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/runtime/sampler.hpp"

namespace lse::runtime {

using model::HybridLM;

struct BatchLimits {
  // Rows the engine holds. Rounded up to a batch bucket, because the bucket is
  // the shape the JIT compiled for; the rows past the sequences in flight run
  // the same kernels and answer zero.
  //
  // 8 rather than 32 for a measured reason, not a cautious one: linear_indexed
  // has only the wave-per-column GEMV, so at 16 rows every routed expert in an
  // MoE model drops to the per-element scalar body and the block cost jumps 4.9x
  // (task #47 is the tiled form that lifts it). Raise it past 8 on a dense model
  // and the head gets 4.5x better; on an MoE model the body gets much worse.
  std::int32_t max_batch = 8;
  // Tokens each sequence may generate.
  std::int32_t max_tokens = 256;
  std::vector<std::uint32_t> stop_tokens;
  // Blocks one attention layer's pool may hold. 0 derives it from the engine KV
  // length times the row count, which is the pool that never has to preempt.
  // Setting it lower is what makes the pool a scarce resource and the policy a
  // decision rather than a formality.
  std::int32_t kv_blocks = 0;
  // Blocks held back from admission so a sequence already decoding can always
  // append its next block. See kv::BlockPolicy.
  std::int32_t kv_reserve = 2;
};

struct BatchRequest {
  std::string id;
  std::vector<std::uint32_t> prompt;
  // 0 takes BatchLimits::max_tokens.
  std::int32_t max_tokens = 0;
};

// What one sequence got and what it cost it. Latency is per sequence and
// throughput is per engine, and they move in opposite directions: a wider batch
// raises the second and lowers the first, so reporting only one of them hides
// the trade this whole mechanism makes.
struct SequenceResult {
  std::string id;
  std::vector<std::uint32_t> generated;
  std::int32_t prompt_tokens = 0;
  // Submission to the first generated token.
  std::uint64_t ttft_ns = 0;
  // First generated token to the last. Spans a preemption and the re-prefill
  // that follows it rather than restarting at the resume, because that wait is
  // part of what this sequence saw; a clock that restarted would report a
  // preempted sequence as the fastest one in the batch.
  std::uint64_t decode_ns = 0;
  // Times this sequence lost its blocks to a preemption and re-prefilled.
  std::int32_t preemptions = 0;

  // What this sequence saw, which is slower than the engine's aggregate.
  [[nodiscard]] double tokens_per_second() const noexcept {
    if (decode_ns == 0 || generated.size() < 2) return 0.0;
    return static_cast<double>(generated.size() - 1) * 1e9 /
           static_cast<double>(decode_ns);
  }
};

struct BatchStats {
  std::int32_t bucket = 0;
  std::int32_t steps = 0;
  std::int32_t generated_tokens = 0;
  std::int32_t prompt_tokens = 0;
  std::int32_t admissions = 0;
  std::int32_t preemptions = 0;
  std::uint64_t wall_ns = 0;
  // Row-steps a sequence occupied, over row-steps the engine ran. Below 1 means
  // the engine ran rows that held nothing, which is what a batch draining looks
  // like.
  std::int64_t occupied_row_steps = 0;
  std::int64_t total_row_steps = 0;
  std::uint32_t kernels_launched = 0;
  std::uint32_t host_groups = 0;
  std::uint64_t jit_compiles = 0;

  // Tokens per second across every sequence in flight. This is the number a
  // batch exists to move.
  [[nodiscard]] double aggregate_tokens_per_second() const noexcept {
    if (wall_ns == 0) return 0.0;
    return static_cast<double>(generated_tokens) * 1e9 /
           static_cast<double>(wall_ns);
  }
  [[nodiscard]] double occupancy() const noexcept {
    if (total_row_steps == 0) return 0.0;
    return static_cast<double>(occupied_row_steps) /
           static_cast<double>(total_row_steps);
  }
};

class BatchScheduler {
 public:
  BatchScheduler(HybridLM& model, SamplingParams params, BatchLimits limits);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  // Called with (id, token) as each token is produced. Returning false retires
  // that sequence at the end of the step.
  using TokenCallback = std::function<bool(const std::string&, std::uint32_t)>;

  // Queues a request. Admission happens between steps, against the block pool.
  Status submit(BatchRequest request);

  // Runs steps until every submitted sequence has finished or been refused.
  // A sequence the pool cannot hold even empty is refused by name rather than
  // left in the queue forever.
  Result<std::vector<SequenceResult>> run(const TokenCallback& on_token = {});

  [[nodiscard]] const BatchStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::int32_t bucket() const noexcept { return bucket_; }
  // Blocks one layer's pool may hold, after the limit is resolved against the
  // engine KV length.
  [[nodiscard]] std::int32_t block_ceiling() const noexcept { return ceiling_; }

  // The first attention layer's free list and pool size. Every attention layer
  // covers the same positions, so one answers for all. A run that admits and
  // retires many sequences must end with the free list back where it started:
  // a retirement that dropped a block table instead of releasing it leaks
  // silently, and the pool only runs dry much later.
  [[nodiscard]] std::int32_t kv_blocks_free() const noexcept;
  [[nodiscard]] std::int32_t kv_blocks_total() const noexcept;

 private:
  // One row of the engine. Occupied or not; a free row is not a hole to skip but
  // a row with zero live length, which is what makes it cost nothing in
  // attention and stay indistinguishable to every live row.
  struct Slot {
    std::string id;
    // Tokens still to feed: the prompt first, then one sampled token at a time.
    std::deque<std::uint32_t> pending;
    std::vector<std::uint32_t> history;
    std::vector<std::uint32_t> generated;
    std::int32_t prompt_tokens = 0;
    std::int32_t position = 0;
    std::int32_t max_tokens = 0;
    std::uint64_t admitted_ns = 0;
    std::uint64_t submitted_ns = 0;
    std::uint64_t first_token_ns = 0;
    std::uint64_t last_token_ns = 0;
    std::uint64_t last_used = 0;
    std::int32_t preemptions = 0;
    bool occupied = false;
    // The prompt is fed, so logits are this sequence's next token rather than a
    // token it already had.
    bool generating = false;
    bool done = false;
  };

  struct Waiting {
    BatchRequest request;
    std::uint64_t submitted_ns = 0;
    std::int32_t preemptions = 0;
    // A preempted sequence resumes rather than restarts: its prompt is
    // everything it has said so far, and these are the tokens of that which it
    // already handed to the caller.
    std::vector<std::uint32_t> generated;
    std::int32_t prompt_tokens = 0;
    // Carried across the preemption so the resumed slot's latency still counts
    // from the first token this sequence ever produced.
    std::uint64_t first_token_ns = 0;
  };

  Status retire(std::int32_t row, std::vector<SequenceResult>* out);
  Status admit_waiting();
  // Preempts until every row already decoding can hold what this step writes.
  // Exhaustion is a decision: the policy names victims oldest-first, they hand
  // their blocks back, and they come back with their history as their prompt.
  // Only a row that does not fit in the whole pool alone is an error.
  Status make_room(std::int32_t width);
  Status preempt(std::int32_t row);
  // Frees row `row`'s blocks on every layer and zeroes its recurrent state, so
  // the next sequence to hold it starts where a fresh one would.
  Status clear_row(std::int32_t row);
  Status configure_layers();
  // Blocks row `row` holds right now. Every attention layer covers the same
  // positions, so the first one that has a table for it answers for all.
  [[nodiscard]] std::int32_t blocks_held(std::size_t row) const;

  // Tokens every occupied row advances this step: the largest prefill rung that
  // no row has to overrun. A row that is decoding has one token pending, so a
  // batch with any decoding row steps one token at a time and a newly admitted
  // sequence feeds its prompt into the same stream.
  [[nodiscard]] std::int32_t step_width() const;
  Result<std::vector<std::uint32_t>> run_step(std::int32_t width);
  Status poke_ids(std::int32_t width);

  HybridLM& model_;
  Sampler sampler_;
  BatchLimits limits_;
  // 0 until the limits are resolved against the batch ladder, which is where a
  // max_batch that fits no bucket is refused by name.
  std::int32_t bucket_ = 0;
  std::int32_t ceiling_ = 0;
  bool device_greedy_ = false;

  std::vector<model::MixerState> states_;
  std::vector<Slot> slots_;
  std::deque<Waiting> waiting_;
  std::uint64_t clock_ = 0;
  BatchStats stats_;

  kv::BlockPolicy policy_;

  // [bucket, width] token slot, reallocated only when the width changes.
  graph::Array ids_;
  std::int32_t ids_width_ = 0;

  struct Head {
    graph::Program program;
    graph::Array hidden;
    graph::Array logits;
    graph::Array pick;
    std::vector<graph::NodePtr> compute;
    bool greedy = false;
  };
  Head head_;
};

}  // namespace lse::runtime
