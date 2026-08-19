// Sampler, session cache, and the decode loop.
//
// The load-bearing test here is cached_decode_matches_a_full_forward_pass: a
// KV cache is only correct if stepping one token at a time gives the same
// logits as running the whole sequence at once. Everything else in M7 rests on
// that.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/allocator.hpp"
#include "lse/kv/block.hpp"
#include "lse/kv/policy.hpp"
#include "lse/model/config.hpp"
#include "lse/ops/rope.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/lemonseed.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/runtime/batch.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/runtime/sampler.hpp"
#include "lse/runtime/session.hpp"

using namespace lse;
using namespace lse::runtime;

namespace {

std::string model_dir() {
  if (const char* p = std::getenv("LSE_TEST_MODEL")) return p;
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/Documents/Dev/LDE/model" : "";
}

bool have_model() {
  std::error_code ec;
  return !model_dir().empty() && std::filesystem::is_directory(model_dir(), ec);
}

}  // namespace

LSE_TEST(greedy_sampling_ignores_every_other_knob) {
  SamplingParams p;
  p.temperature = 0.0f;
  p.top_k = 1;
  p.top_p = 0.1f;
  Sampler s(p);

  std::vector<float> logits{0.1f, 9.0f, 0.2f, 3.0f};
  LSE_EXPECT_EQ(s.sample(logits, {}), 1u);
  LSE_EXPECT_EQ(argmax(logits), 1u);
}

LSE_TEST(top_k_of_one_is_deterministic_whatever_the_seed) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 1;
  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    p.seed = seed;
    Sampler s(p);
    std::vector<float> logits{1.0f, 5.0f, 2.0f, 0.0f};
    LSE_EXPECT_EQ(s.sample(logits, {}), 1u);
  }
}

LSE_TEST(the_same_seed_gives_the_same_stream) {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 12345;

  const auto run = [&p] {
    Sampler s(p);
    std::vector<std::uint32_t> out;
    for (int i = 0; i < 32; ++i) {
      std::vector<float> logits{1.0f, 1.2f, 0.9f, 1.1f};
      out.push_back(s.sample(logits, {}));
    }
    return out;
  };
  LSE_EXPECT(run() == run());

  // A different seed must not give the same 32 draws.
  const auto a = run();
  p.seed = 999;
  const auto b = run();
  LSE_EXPECT(a != b);
}

LSE_TEST(top_p_keeps_the_smallest_prefix_reaching_the_threshold) {
  // One token holds ~95% of the mass, so any p <= 0.95 must select only it.
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_p = 0.5f;
  p.seed = 7;
  Sampler s(p);
  for (int i = 0; i < 64; ++i) {
    std::vector<float> logits{10.0f, 0.0f, 0.0f, 0.0f};
    LSE_EXPECT_EQ(s.sample(logits, {}), 0u);
  }
}

LSE_TEST(repetition_penalty_pushes_a_seen_token_down) {
  SamplingParams p;
  p.temperature = 0.0f;  // greedy, so the effect is visible in the choice
  p.repetition_penalty = 2.0f;
  Sampler s(p);

  const std::vector<std::uint32_t> history{0};
  std::vector<float> logits{2.0f, 1.5f};
  // Without the penalty token 0 wins; halved to 1.0 it loses to 1.5.
  LSE_EXPECT_EQ(s.sample(logits, history), 1u);
}

LSE_TEST(repetition_penalty_does_not_promote_negative_logits) {
  // Dividing a negative logit raises it. A sign-blind implementation would
  // *encourage* the repeat it is meant to suppress.
  SamplingParams p;
  p.temperature = 0.0f;
  p.repetition_penalty = 2.0f;
  Sampler s(p);

  const std::vector<std::uint32_t> history{0};
  std::vector<float> logits{-1.0f, -1.2f};
  LSE_EXPECT_EQ(s.sample(logits, history), 1u);
}

LSE_TEST(a_session_reports_and_releases_its_cache) {
  SessionStore store(4, /*budget_bytes=*/0);
  Session& a = store.get_or_create("a");
  LSE_EXPECT(a.cache_bytes() == 0u);
  LSE_EXPECT(a.position() == 0);

  a.advance(8);
  LSE_EXPECT(a.position() == 8);
  a.clear();
  LSE_EXPECT(a.position() == 0);
  LSE_EXPECT(a.history().empty());

  LSE_EXPECT(store.find("a") != nullptr);
  LSE_EXPECT(store.find("missing") == nullptr);
  store.erase("a");
  LSE_EXPECT(store.find("a") == nullptr);
}

LSE_TEST(the_store_keeps_sessions_apart) {
  SessionStore store(2, 0);
  store.get_or_create("alice").history().push_back(1);
  store.get_or_create("bob").history().push_back(2);

  LSE_EXPECT_EQ(store.size(), 2u);
  LSE_EXPECT_EQ(store.get_or_create("alice").history().size(), 1u);
  LSE_EXPECT_EQ(store.get_or_create("alice").history()[0], 1u);
  LSE_EXPECT_EQ(store.get_or_create("bob").history()[0], 2u);
}

LSE_TEST(cached_decode_matches_a_full_forward_pass) {
  // Step token by token through a session, then run the same tokens in one
  // pass with no cache. The final logits must agree: if they do not, the cache
  // is feeding attention the wrong keys and every generation is wrong.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> tokens{1, 42, 1337, 7};

  const auto logits_of = [&](const std::vector<std::uint32_t>& ids,
                             std::vector<model::MixerState>* states) {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
    std::vector<float> out;
    if (!a.eval().ok()) return out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
    }
    auto h = lm->hidden(a, states, nullptr);
    if (!h.ok()) return out;
    auto last = Generator::last_hidden(h.release());
    if (!last.ok()) return out;
    auto lg = lm->lm_head(*last);
    if (!lg.ok()) return out;
    graph::Array logits = lg.release();
    out.resize(logits.shape().elem_count());
    if (!logits.to_host(out.data(), out.size() * sizeof(float)).ok()) {
      out.clear();
      return out;
    }
    return out;
  };

  // One shot, no cache.
  const std::vector<float> whole = logits_of(tokens, nullptr);
  LSE_EXPECT(!whole.empty());
  if (whole.empty()) return;

  // Same tokens, one at a time, through a session's cache.
  Session session("diff", lm->num_layers());
  std::vector<float> stepped;
  const auto cap = cfg->kv_capacity();
  for (std::uint32_t id : tokens) {
    stepped = logits_of({id}, &session.states());
    LSE_EXPECT(!stepped.empty());
    if (stepped.empty()) return;
    for (const auto& st : session.states()) {
      if (!st.key_cache.valid()) continue;
      // Paged: the pool is [blocks, kv_heads, block_size, head_dim], so the
      // position axis is one block wide and the pool is sized by a rung, never
      // by the engine capacity.
      LSE_EXPECT_EQ(st.key_cache.shape().rank(), 4u);
      LSE_EXPECT_EQ(st.key_cache.shape().dim(2), kv::kBlockSize);
      LSE_EXPECT_EQ(st.value_cache.shape().dim(2), kv::kBlockSize);
      LSE_EXPECT_EQ(st.key_cache.shape().dim(0), kv::kMinPoolBlocks);
      LSE_EXPECT(st.key_cache.shape().dim(0) * kv::kBlockSize < cap);
      LSE_EXPECT_EQ(st.paged.stride(), kv::blocks_for(cap, kv::kBlockSize));
      break;
    }
  }

  double max_abs = 0.0;
  double ref_absmax = 0.0;
  for (std::size_t i = 0; i < whole.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(stepped[i] - whole[i])));
    ref_absmax = std::max(ref_absmax, std::abs(static_cast<double>(whole[i])));
  }
  const double max_rel = ref_absmax > 0.0 ? max_abs / ref_absmax : max_abs;
  std::printf("       cached vs whole: max_abs=%.3e max_rel=%.3e over %zu\n",
              max_abs, max_rel, whole.size());

  // Also the decision, not just the numbers. WMMA decode is f16×f16
  // accumulated in f32; a full-seq tile and a padded M=1 tile disagree
  // around 1e-3, which is still the same argmax.
  LSE_EXPECT_EQ(argmax(stepped), argmax(whole));
  LSE_EXPECT(max_rel < 2e-3);
}

LSE_TEST(a_second_decode_step_replays_the_held_program) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  Session session("replay", lm->num_layers());
  const auto run = [&](std::uint32_t id) {
    graph::Array a =
        graph::Array::zeros(Shape{1, 1}, DType::kF32);
    if (!a.eval().ok()) return false;
    graph::interpreter::store_element(*a.node(), 0, static_cast<float>(id));
    auto h = lm->hidden(a, &session.states(), nullptr);
    return h.ok();
  };

  LSE_EXPECT(run(1));
  for (const auto& st : session.states()) {
    if (!st.key_cache.valid()) continue;
    LSE_EXPECT_EQ(st.key_cache.shape().dim(2), kv::kBlockSize);
    break;
  }

  sched->reset_accumulated_trace();
  LSE_EXPECT(run(42));
  LSE_EXPECT(sched->last_trace().replayed);
  LSE_EXPECT(sched->last_trace().partition_ns < 100000ull);

  session.clear();
  sched->reset_accumulated_trace();
  LSE_EXPECT(run(7));
  LSE_EXPECT(!sched->last_trace().replayed);
}

LSE_TEST(graph_argmax_matches_the_host_sampler_argmax) {
  // 5000 elements crosses the 4096-per-chunk partial stage, so both the
  // in-chunk reduce and the cross-chunk combine are exercised. The duplicated
  // maximum checks the tie rule: smallest index, exactly like runtime::argmax.
  const std::int64_t n = 5000;
  std::vector<float> row(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = 3.0f * std::sin(static_cast<float>(i) * 0.7f);
  }
  row[1234] = 9.5f;
  row[4321] = 9.5f;

  const auto pick_of = [](const std::vector<float>& values) -> std::uint32_t {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(values.size())}, DType::kF32);
    if (!a.eval().ok()) return 0xffffffffu;
    for (std::size_t i = 0; i < values.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, values[i]);
    }
    graph::Array pick = graph::argmax(a);
    if (!pick.valid()) return 0xffffffffu;
    auto v = pick.item();
    if (!v.ok()) return 0xffffffffu;
    return static_cast<std::uint32_t>(*v);
  };

  LSE_EXPECT_EQ(pick_of(row), argmax(row));
  LSE_EXPECT_EQ(pick_of(row), 1234u);

  // Maximum inside the trailing, partial chunk.
  row[4321] = 11.0f;
  LSE_EXPECT_EQ(pick_of(row), argmax(row));
  LSE_EXPECT_EQ(pick_of(row), 4321u);

  // All-negative row: nothing beats index ordering on the way down.
  for (float& v : row) v = -std::abs(v) - 1.0f;
  LSE_EXPECT_EQ(pick_of(row), argmax(row));
}

LSE_TEST(greedy_device_decode_matches_the_host_argmax_path) {
  // The device path reads back one f32 index; it must pick exactly the token
  // the host sampler picks from the full logit row, step for step.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> prompt{1, 42, 1337};
  constexpr std::int32_t kSteps = 4;

  const auto logits_of = [&](const std::vector<std::uint32_t>& ids,
                             std::vector<model::MixerState>* states) {
    graph::Array a = graph::Array::zeros(
        Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
    std::vector<float> out;
    if (!a.eval().ok()) return out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
    }
    auto h = lm->hidden(a, states, nullptr);
    if (!h.ok()) return out;
    auto last = Generator::last_hidden(h.release());
    if (!last.ok()) return out;
    auto lg = lm->lm_head(*last);
    if (!lg.ok()) return out;
    graph::Array logits = lg.release();
    out.resize(logits.shape().elem_count());
    if (!logits.to_host(out.data(), out.size() * sizeof(float)).ok()) out.clear();
    return out;
  };

  Session ref("greedy-ref", lm->num_layers());
  std::vector<std::uint32_t> want;
  std::vector<float> logits = logits_of(prompt, &ref.states());
  LSE_EXPECT(!logits.empty());
  if (logits.empty()) return;
  std::uint32_t next = argmax(logits);
  want.push_back(next);
  for (std::int32_t i = 1; i < kSteps; ++i) {
    logits = logits_of({next}, &ref.states());
    LSE_EXPECT(!logits.empty());
    if (logits.empty()) return;
    next = argmax(logits);
    want.push_back(next);
  }

  SamplingParams sp;
  sp.temperature = 0.0f;  // repetition_penalty stays at its 1.0 no-op
  Generator gen(*lm, sp);
  Session dev("greedy-dev", lm->num_layers());
  GenerationLimits limits;
  limits.max_tokens = kSteps;
  auto got = gen.generate(dev, prompt, limits);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;
  LSE_EXPECT(*got == want);
}

LSE_TEST(a_continued_session_only_scores_the_new_tokens) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  SamplingParams sp;
  sp.temperature = 0.0f;
  Generator gen(*lm, sp);

  Session session("chat", lm->num_layers());
  GenerationLimits limits;
  limits.max_tokens = 2;

  auto first = gen.generate(session, {1, 42, 1337}, limits);
  LSE_EXPECT(first.ok());
  if (!first.ok()) return;
  LSE_EXPECT_EQ(gen.stats().prompt_tokens, 3);

  // The next turn extends what the cache already holds, so only the tail is
  // scored — this is the whole reason the session owns the cache. The tail is
  // everything past position(), which trails history() by one: the last token
  // sampled was never fed back through the model.
  std::vector<std::uint32_t> next = session.history();
  next.push_back(99);
  const auto expected =
      static_cast<std::int32_t>(next.size()) - session.position();
  auto second = gen.generate(session, next, limits);
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;
  std::printf("       continued turn scored %d of %zu prompt token(s)\n",
              gen.stats().prompt_tokens, next.size());
  LSE_EXPECT(gen.stats().prompt_tokens == expected);
  LSE_EXPECT(gen.stats().prompt_tokens < static_cast<std::int32_t>(next.size()));
}

// --- device clock, against whatever device this box actually has ------------

LSE_TEST(the_device_clock_is_named_and_rated_or_cleanly_refused) {
  // The process-wide backend the rest of this suite already runs on, rather
  // than a fresh one: bringing the GPU up is a process-wide act, and a second
  // create+init returns "GPU accelerator already initialized" -- a failure with
  // nothing to do with clocks that would silently skip this whole case.
  //
  // The contract is the same whichever backend answers, and neither half of it
  // is optional: a clock that is claimed is fully specified, a timestamp that
  // is handed over carries a usable clock and moves forward, and anything
  // absent is kUnimplemented rather than a device error or a half-filled
  // struct. It passes today because hrx declines the timestamp, and it keeps
  // passing unchanged on the day hrx can answer -- which is the point of
  // writing it against the contract rather than against today's gap.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  backend::IBackend& be = sched->backend();
  const std::string name(be.name());

  auto clock = be.device_clock();
  if (clock.ok()) {
    LSE_EXPECT(clock->known());
    LSE_EXPECT(clock->domain != backend::ClockDomain::kUnknown);
    LSE_EXPECT(clock->ticks_per_second > 0);
    LSE_EXPECT(clock->valid_bits > 0);
  } else {
    LSE_EXPECT(clock.status().code() == StatusCode::kUnimplemented);
  }

  // hrx cannot read a tick today, but it can say which counter a tick would be
  // on and how fast that counter runs, and that answer is a live device query
  // rather than a constant. This guards the query: a wrong attribute ordinal
  // returns some neighbouring field's value, a plausible-looking number that
  // nothing else in the system would catch.
  if (name == "hrx") {
    LSE_EXPECT(clock.ok());
    if (clock.ok()) {
      LSE_EXPECT(clock->domain == backend::ClockDomain::kDeviceAgent);
      // A tick rate, which is neither the engine clock in MHz nor the host's
      // in GHz -- both values this could be confused with, and both outside
      // this decade.
      LSE_EXPECT(clock->ticks_per_second >= 10'000'000);
      LSE_EXPECT(clock->ticks_per_second <= 500'000'000);
      LSE_EXPECT_EQ(clock->valid_bits, 64u);
    }
  }
  if (clock.ok()) {
    std::printf("       %s device clock: %s, %llu Hz, %u bits\n", name.c_str(),
                std::string(backend::clock_domain_name(clock->domain)).c_str(),
                static_cast<unsigned long long>(clock->ticks_per_second),
                clock->valid_bits);
  } else {
    std::printf("       %s device clock: declined (%s)\n", name.c_str(),
                clock.status().to_string().c_str());
  }

  auto first = be.sample_device_time();
  if (!first.ok()) {
    LSE_EXPECT(first.status().code() == StatusCode::kUnimplemented);
    std::printf("       %s device timestamp: declined (%s)\n", name.c_str(),
                first.status().to_string().c_str());
    return;
  }

  // A timestamp exists, so it must behave like one: usable clock, moving
  // forward, and a known sleep coming back as that interval in the clock's own
  // units. The band is wide against scheduler slop and still an order of
  // magnitude tighter than the ~10x error a mistaken tick rate produces, which
  // is the failure it is here to catch.
  LSE_EXPECT(first->valid());
  constexpr double kSleepNs = 20'000'000.0;
  std::this_thread::sleep_for(
      std::chrono::nanoseconds(static_cast<std::int64_t>(kSleepNs)));
  auto second = be.sample_device_time();
  LSE_EXPECT(second.ok());
  if (!second.ok()) return;

  auto elapsed = backend::nanoseconds_between(*first, *second);
  LSE_EXPECT(elapsed.ok());
  if (elapsed.ok()) {
    LSE_EXPECT(*elapsed > 0.0);
    LSE_EXPECT_NEAR(*elapsed, kSleepNs, kSleepNs * 0.25);
  }
}


// --- paged KV -----------------------------------------------------------------

LSE_TEST(the_block_pool_refcounts_and_reports_exhaustion) {
  kv::BlockAllocator pool(3);
  LSE_EXPECT_EQ(pool.total(), 3);
  LSE_EXPECT_EQ(pool.free_count(), 3);
  LSE_EXPECT_EQ(pool.used(), 0);

  auto a = pool.acquire();
  LSE_EXPECT_OK(a.status());
  auto b = pool.acquire();
  LSE_EXPECT_OK(b.status());
  LSE_EXPECT_EQ(pool.used(), 2);
  LSE_EXPECT_EQ(*pool.refcount(*a), 1);

  // A second holder of the same block: this is what makes prefix sharing a
  // refcount bump rather than a copy.
  LSE_EXPECT_OK(pool.retain(*a));
  LSE_EXPECT_EQ(*pool.refcount(*a), 2);
  auto first = pool.release(*a);
  LSE_EXPECT_OK(first.status());
  LSE_EXPECT(!*first);  // still held by the other reference
  LSE_EXPECT_EQ(pool.free_count(), 1);
  auto last = pool.release(*a);
  LSE_EXPECT_OK(last.status());
  LSE_EXPECT(*last);
  LSE_EXPECT_EQ(pool.free_count(), 2);

  // Exhaustion names the pool size rather than handing out a stale block.
  LSE_EXPECT_OK(pool.acquire().status());
  LSE_EXPECT_OK(pool.acquire().status());
  auto dry = pool.acquire();
  LSE_EXPECT(!dry.ok());
  LSE_EXPECT(dry.status().code() == StatusCode::kOutOfMemory);
  LSE_EXPECT(dry.status().message().find("all 3 blocks") != std::string::npos);

  // A free block cannot be retained and a held block cannot be double-released.
  kv::BlockAllocator fresh(2);
  LSE_EXPECT(!fresh.retain(0).ok());
  auto one = fresh.acquire();
  LSE_EXPECT_OK(one.status());
  LSE_EXPECT_OK(fresh.release(*one).status());
  LSE_EXPECT(!fresh.release(*one).ok());
  LSE_EXPECT(!fresh.release(7).ok());
}

LSE_TEST(a_block_table_covers_positions_and_locates_them) {
  kv::BlockAllocator pool(8);
  kv::BlockTable table(kv::kBlockSize);
  LSE_EXPECT_OK(pool.cover(table, 1));
  LSE_EXPECT_EQ(table.size(), 1);
  LSE_EXPECT_OK(pool.cover(table, kv::kBlockSize));
  LSE_EXPECT_EQ(table.size(), 1);  // still one block: 16 tokens fit
  LSE_EXPECT_OK(pool.cover(table, kv::kBlockSize + 1));
  LSE_EXPECT_EQ(table.size(), 2);

  auto slot = table.locate(kv::kBlockSize);
  LSE_EXPECT_OK(slot.status());
  LSE_EXPECT_EQ(slot->block, table.blocks()[1]);
  LSE_EXPECT_EQ(slot->offset, 0);
  LSE_EXPECT(!table.locate(2 * kv::kBlockSize).ok());

  // Covering more than the pool holds leaves the table and the pool untouched.
  const std::int32_t held = table.size();
  const std::int32_t free_before = pool.free_count();
  LSE_EXPECT(!pool.cover(table, 4096).ok());
  LSE_EXPECT_EQ(table.size(), held);
  LSE_EXPECT_EQ(pool.free_count(), free_before);

  LSE_EXPECT_OK(pool.release_all(table));
  LSE_EXPECT_EQ(pool.used(), 0);
  LSE_EXPECT(table.empty());

  // The device image: padded rows and unreached slots take the pad block, so
  // every row runs the identical address arithmetic.
  kv::BlockTable one(kv::kBlockSize);
  LSE_EXPECT_OK(pool.cover(one, 2 * kv::kBlockSize));
  std::vector<kv::BlockTable> rows{one};
  std::vector<float> image(2 * 4, -1.0f);
  LSE_EXPECT_OK(kv::write_table_rows(rows, 4, /*pad=*/0, image));
  LSE_EXPECT_EQ(image[0], static_cast<float>(one.blocks()[0]));
  LSE_EXPECT_EQ(image[1], static_cast<float>(one.blocks()[1]));
  LSE_EXPECT_EQ(image[2], 0.0f);
  LSE_EXPECT_EQ(image[4], 0.0f);  // the padded row
  LSE_EXPECT(!kv::write_table_rows(rows, 4, kv::kNoBlock, image).ok());
}

LSE_TEST(the_pool_ladder_bounds_the_shapes_a_context_can_compile) {
  // Rungs, not exact fits: the pool block count is a tensor dimension the JIT
  // keys on.
  LSE_EXPECT_EQ(kv::pool_rung(1, 256), 8);
  LSE_EXPECT_EQ(kv::pool_rung(8, 256), 8);
  LSE_EXPECT_EQ(kv::pool_rung(9, 256), 16);
  LSE_EXPECT_EQ(kv::pool_rung(200, 256), 256);
  LSE_EXPECT_EQ(kv::pool_rung(999, 256), 256);
  // 4096 tokens at 16 per block is 256 blocks, so the whole ladder a session can
  // ever walk is six rungs.
  int rungs = 0;
  for (std::int32_t b = 1; b <= 256; ++b) {
    if (b == 1 || kv::pool_rung(b, 256) != kv::pool_rung(b - 1, 256)) ++rungs;
  }
  LSE_EXPECT_EQ(rungs, 6);
}

LSE_TEST(admission_preempts_the_oldest_and_swaps_while_host_room_lasts) {
  kv::BlockPolicy policy(kv::kBlockSize, /*reserve=*/1, /*host_blocks=*/2);
  kv::BlockAllocator pool(8);

  kv::SequenceDemand want{"new", 0, 3 * kv::kBlockSize, 10};
  LSE_EXPECT_EQ(policy.shortfall(want), 3);
  {
    const kv::Admission a = policy.admit(want, pool, {});
    LSE_EXPECT(a.verdict == kv::Verdict::kAdmit);
    LSE_EXPECT_EQ(a.blocks_needed, 3);
  }

  // Fill the pool, then ask for three more with two older sequences resident.
  for (int i = 0; i < 8; ++i) LSE_EXPECT_OK(pool.acquire().status());
  const std::vector<kv::SequenceDemand> resident{
      {"old", 4, 4 * kv::kBlockSize, 1},
      {"newer", 4, 4 * kv::kBlockSize, 5},
  };
  const kv::Admission a = policy.admit(want, pool, resident);
  LSE_EXPECT(a.verdict == kv::Verdict::kPreempt);
  LSE_EXPECT_EQ(a.preempt.size(), 1u);
  LSE_EXPECT(a.preempt[0].id == "old");  // oldest first
  // 4 blocks needed to land, only 2 of host room: it drops rather than pretending
  // to swap.
  LSE_EXPECT(a.preempt[0].how == kv::Eviction::kDrop);

  kv::BlockPolicy roomy(kv::kBlockSize, 1, /*host_blocks=*/16);
  const kv::Admission b = roomy.admit(want, pool, resident);
  LSE_EXPECT(b.preempt[0].how == kv::Eviction::kSwapOut);

  // Nothing resident and nothing free: not transient, so it refuses.
  const kv::Admission c = policy.admit(want, pool, {});
  LSE_EXPECT(c.verdict == kv::Verdict::kRefuse);
  LSE_EXPECT(c.reason.find("needs 3") != std::string::npos);
}

LSE_TEST(the_batch_ladder_selects_a_bucket_and_refuses_what_fits_none) {
  LSE_EXPECT_EQ(*model::batch_bucket(1), 1);
  LSE_EXPECT_EQ(*model::batch_bucket(2), 2);
  LSE_EXPECT_EQ(*model::batch_bucket(3), 4);
  LSE_EXPECT_EQ(*model::batch_bucket(9), 16);
  LSE_EXPECT_EQ(*model::batch_bucket(16), 16);
  LSE_EXPECT_EQ(*model::batch_bucket(17), 32);
  LSE_EXPECT_EQ(*model::batch_bucket(32), 32);

  auto none = model::batch_bucket(33);
  LSE_EXPECT(!none.ok());
  LSE_EXPECT(none.status().code() == StatusCode::kOutOfRange);
  // The error names the size that did not fit and the top of the ladder.
  LSE_EXPECT(none.status().message().find("batch of 33") != std::string::npos);
  LSE_EXPECT(none.status().message().find("32") != std::string::npos);
  LSE_EXPECT(!model::batch_bucket(0).ok());
}

namespace {

// A device Array holding exactly `v`.
graph::Array filled(Shape shape, const std::vector<float>& v) {
  graph::Array a = graph::Array::zeros(shape, DType::kF32);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return {};
  graph::Node& n = *a.node();
  if (!graph::interpreter::ensure_output_buffer(n, sched->backend()).ok()) {
    return {};
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    graph::interpreter::store_element(n, i, v[i]);
  }
  n.materialized = true;
  if (!graph::interpreter::sync_to_device(n, sched->backend()).ok()) return {};
  return a;
}

std::vector<float> read_all(graph::Array& a) {
  std::vector<float> out(static_cast<std::size_t>(a.shape().elem_count()));
  if (!a.to_host(out.data(), out.size() * sizeof(float)).ok()) out.clear();
  return out;
}

// The step descriptor kv/block.hpp specifies, from one {first query position,
// live KV length} pair per row. A length of 0 is a row holding no sequence.
graph::Array step_meta(
    const std::vector<std::pair<std::int32_t, std::int32_t>>& rows) {
  const auto n = static_cast<std::int32_t>(rows.size());
  std::vector<float> m(
      static_cast<std::size_t>(kv::step_meta_elems(n)), 0.0f);
  std::int32_t pos = 0;
  std::int32_t len = 0;
  std::int32_t live = 0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (rows[r].second == 0) continue;
    const auto at = static_cast<std::size_t>(kv::kStepMetaHeader) +
                    r * static_cast<std::size_t>(kv::kStepMetaPerRow);
    m[at] = static_cast<float>(rows[r].first);
    m[at + 1] = static_cast<float>(rows[r].second);
    pos = std::max(pos, rows[r].first);
    len = std::max(len, rows[r].second);
    live = static_cast<std::int32_t>(r) + 1;
  }
  m[0] = static_cast<float>(pos);
  m[1] = static_cast<float>(len);
  m[2] = static_cast<float>(live);
  return filled(Shape{static_cast<std::int64_t>(m.size())}, m);
}

// Deterministic and not symmetric, so an index mistake shows up as a value
// mistake rather than cancelling out.
float noise(std::size_t i) {
  return static_cast<float>(static_cast<double>((i * 2654435761u) % 2003) /
                                1000.0 -
                            1.0);
}

}  // namespace

LSE_TEST(a_paged_read_matches_a_contiguous_read_exactly) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kQh = 4;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kHd = 8;
  constexpr std::int64_t kTq = 1;
  const std::int32_t bs = kv::kBlockSize;

  // Two lengths: a whole number of blocks, and one that leaves the last block
  // part-filled so the guard on the live length is the thing under test.
  for (std::int32_t live : {3 * kv::kBlockSize, 3 * kv::kBlockSize - 3}) {
    const std::int32_t nblk = kv::blocks_for(live, bs);
    const std::int32_t pool_blocks = kv::kMinPoolBlocks;
    const std::int32_t offset = live - static_cast<std::int32_t>(kTq);

    std::vector<float> q(static_cast<std::size_t>(kQh * kTq * kHd));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 7);

    // Contiguous [1, kvh, live, hd] and a pool [pool_blocks, kvh, bs, hd] with
    // the same logical content, plus junk in the slots past `live` and in the
    // blocks the table does not name.
    std::vector<float> flat(static_cast<std::size_t>(kKvh * live * kHd));
    std::vector<float> flatv(flat.size());
    for (std::size_t i = 0; i < flat.size(); ++i) {
      flat[i] = noise(i + 101);
      flatv[i] = noise(i + 9001);
    }
    std::vector<float> pool(
        static_cast<std::size_t>(pool_blocks * kKvh * bs * kHd), 1e30f);
    std::vector<float> poolv(pool.size(), -1e30f);
    // Logical block b lives in physical block `phys(b)`, which is deliberately
    // not b: an identity table is satisfied by a kernel that ignores the table
    // and walks the pool linearly, which is the whole thing under test. The
    // three values below are distinct for pool_blocks == kMinPoolBlocks.
    const auto phys = [pool_blocks](std::int32_t b) {
      return (b * 5 + 3) % pool_blocks;
    };
    for (std::int32_t blk = 0; blk < nblk; ++blk) {
      for (std::int64_t h = 0; h < kKvh; ++h) {
        for (std::int32_t sl = 0; sl < bs; ++sl) {
          const std::int32_t j = blk * bs + sl;
          if (j >= live) continue;
          for (std::int64_t d = 0; d < kHd; ++d) {
            const auto dst = static_cast<std::size_t>(
                ((phys(blk) * kKvh + h) * bs + sl) * kHd + d);
            const auto src = static_cast<std::size_t>((h * live + j) * kHd + d);
            pool[dst] = flat[src];
            poolv[dst] = flatv[src];
          }
        }
      }
    }

    graph::Array qa = filled(Shape{1, kQh, kTq, kHd}, q);
    graph::Array ka = filled(Shape{1, kKvh, live, kHd}, flat);
    graph::Array va = filled(Shape{1, kKvh, live, kHd}, flatv);
    graph::Array kp = filled(Shape{pool_blocks, kKvh, bs, kHd}, pool);
    graph::Array vp = filled(Shape{pool_blocks, kKvh, bs, kHd}, poolv);

    const std::int32_t stride = 16;
    std::vector<float> table(static_cast<std::size_t>(stride), 0.0f);
    for (std::int32_t i = 0; i < nblk; ++i) table[static_cast<std::size_t>(i)] =
        static_cast<float>(phys(i));
    graph::Array ta = filled(Shape{1, stride}, table);
    graph::Array meta = step_meta({{offset, live}});

    const float scale = 0.125f;
    graph::Array contig = graph::sdpa(qa, ka, va, scale,
                                      graph::MaskKind::kCausal, 0, offset);
    graph::Array pagedo =
        graph::sdpa_paged(qa, kp, vp, scale, graph::MaskKind::kCausal, 0, meta,
                          ta, bs);
    LSE_EXPECT_OK(contig.eval());
    LSE_EXPECT_OK(pagedo.eval());
    const std::vector<float> a = read_all(contig);
    const std::vector<float> b = read_all(pagedo);
    LSE_EXPECT(!a.empty() && a.size() == b.size());
    if (a.size() != b.size()) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++differ;
    }
    std::printf("       paged vs contiguous at live=%d: %zu of %zu differ\n",
                live, differ, a.size());
    LSE_EXPECT_EQ(differ, 0u);
  }
}

LSE_TEST(a_padded_batch_row_reads_its_own_blocks_and_pad_rows_answer_zero) {
  // The width-invariance case for the batch axis: two rows whose block tables
  // point at different blocks must give different answers, and each must equal
  // what it gives alone. Rows past the live count must answer zero without
  // touching another row's blocks.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kQh = 2;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kHd = 8;
  const std::int32_t bs = kv::kBlockSize;
  const std::int32_t live = bs;  // one block per row
  const std::int32_t pool_blocks = kv::kMinPoolBlocks;
  const std::int32_t stride = 8;
  const std::int32_t bucket = 4;

  std::vector<float> pool(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kHd));
  std::vector<float> poolv(pool.size());
  for (std::size_t i = 0; i < pool.size(); ++i) {
    pool[i] = noise(i + 31);
    poolv[i] = noise(i + 555);
  }
  graph::Array kp = filled(Shape{pool_blocks, kKvh, bs, kHd}, pool);
  graph::Array vp = filled(Shape{pool_blocks, kKvh, bs, kHd}, poolv);

  // Row 0 -> block 2, row 1 -> block 5. Rows 2 and 3 are padding and point at
  // block 0, which is a real block neither live row uses.
  std::vector<float> table(static_cast<std::size_t>(bucket * stride), 0.0f);
  table[0] = 2.0f;
  table[static_cast<std::size_t>(stride)] = 5.0f;
  graph::Array ta = filled(Shape{bucket, stride}, table);
  graph::Array meta =
      step_meta({{0, live}, {0, live}, {0, 0}, {0, 0}});

  std::vector<float> q(static_cast<std::size_t>(bucket * kQh * 1 * kHd));
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 77);
  graph::Array qa = filled(Shape{bucket, kQh, 1, kHd}, q);

  graph::Array out = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                      graph::MaskKind::kCausal, 0, meta, ta, bs);
  LSE_EXPECT_OK(out.eval());
  const std::vector<float> got = read_all(out);
  const auto per_row = static_cast<std::size_t>(kQh * kHd);
  LSE_EXPECT_EQ(got.size(), per_row * static_cast<std::size_t>(bucket));
  if (got.size() != per_row * static_cast<std::size_t>(bucket)) return;

  // Row 1 is not row 0. This is the assertion the quant_linear row-offset defect
  // would have failed while still producing fluent text.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (got[i] == got[per_row + i]) ++same;
  }
  std::printf("       row0 vs row1: %zu of %zu elements identical\n", same,
              per_row);
  LSE_EXPECT(same < per_row);

  // Padded rows answered zero and did not read a live row's block.
  for (std::size_t r = 2; r < static_cast<std::size_t>(bucket); ++r) {
    for (std::size_t i = 0; i < per_row; ++i) {
      LSE_EXPECT_EQ(got[r * per_row + i], 0.0f);
    }
  }

  // Each live row alone gives what it gave in the batch: the same token cannot
  // depend on how many rows shared the pass.
  for (std::int32_t r = 0; r < 2; ++r) {
    std::vector<float> one_q(per_row);
    for (std::size_t i = 0; i < per_row; ++i) {
      one_q[i] = q[static_cast<std::size_t>(r) * per_row + i];
    }
    std::vector<float> one_table(static_cast<std::size_t>(stride), 0.0f);
    one_table[0] = table[static_cast<std::size_t>(r) * stride];
    graph::Array q1 = filled(Shape{1, kQh, 1, kHd}, one_q);
    graph::Array t1 = filled(Shape{1, stride}, one_table);
    graph::Array m1 = step_meta({{0, live}});
    graph::Array o1 = graph::sdpa_paged(q1, kp, vp, 0.25f,
                                        graph::MaskKind::kCausal, 0, m1, t1, bs);
    LSE_EXPECT_OK(o1.eval());
    const std::vector<float> alone = read_all(o1);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < per_row; ++i) {
      if (std::memcmp(&alone[i], &got[static_cast<std::size_t>(r) * per_row + i],
                      sizeof(float)) != 0) {
        ++differ;
      }
    }
    std::printf("       row %d batched vs alone: %zu of %zu differ\n", r, differ,
                per_row);
    LSE_EXPECT_EQ(differ, 0u);
  }
}

namespace {

// One ragged attention pass: `rows` gives each row's {first query position,
// live KV length} and `blocks` its block list. Returns the whole [bucket, Hq,
// 1, Hd] output.
struct RaggedRow {
  std::int32_t first = 0;
  std::int32_t len = 0;
  std::vector<std::int32_t> blocks;
};

constexpr std::int64_t kRagQh = 2;
constexpr std::int64_t kRagKvh = 2;
constexpr std::int64_t kRagHd = 8;
constexpr std::int32_t kRagStride = 8;
constexpr std::int32_t kRagPool = 16;

std::vector<float> ragged_pass(const std::vector<RaggedRow>& rows,
                               const std::vector<float>& q,
                               const graph::Array& kp, const graph::Array& vp) {
  const auto bucket = static_cast<std::int64_t>(rows.size());
  std::vector<float> table(
      static_cast<std::size_t>(bucket * kRagStride), 0.0f);
  std::vector<std::pair<std::int32_t, std::int32_t>> meta;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t i = 0; i < rows[r].blocks.size(); ++i) {
      table[r * static_cast<std::size_t>(kRagStride) + i] =
          static_cast<float>(rows[r].blocks[i]);
    }
    meta.push_back({rows[r].first, rows[r].len});
  }
  graph::Array ta = filled(Shape{bucket, kRagStride}, table);
  graph::Array ma = step_meta(meta);
  graph::Array qa = filled(Shape{bucket, kRagQh, 1, kRagHd}, q);
  graph::Array out = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                       graph::MaskKind::kCausal, 0, ma, ta,
                                       kv::kBlockSize);
  if (!out.eval().ok()) return {};
  return read_all(out);
}

}  // namespace

LSE_TEST(a_row_gets_the_same_bits_whoever_shares_its_step) {
  // The acceptance gate for continuous batching, and the shape of the defect
  // that has bitten this codebase three times: a token's answer must not depend
  // on how many rows shared its pass, on where those rows sit, or on how long
  // they are. Here every row is at a different absolute position with a
  // different live length and its own blocks — the case a shared meta[0]/meta[1]
  // gets wrong three separate ways (causal mask, softmax bound, RoPE origin)
  // while still producing fluent text.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  std::vector<float> pool(static_cast<std::size_t>(
      kRagPool * kRagKvh * kv::kBlockSize * kRagHd));
  std::vector<float> poolv(pool.size());
  for (std::size_t i = 0; i < pool.size(); ++i) {
    pool[i] = noise(i + 17);
    poolv[i] = noise(i + 4211);
  }
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, pool);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, poolv);

  // Deliberately ragged: one row inside its first block, one three blocks deep,
  // one exactly on a block boundary, one holding no sequence at all.
  std::vector<RaggedRow> rows{
      {4, 5, {2}},
      {39, 40, {5, 1, 7}},
      {15, 16, {11}},
      {0, 0, {}},
  };
  const auto per_row = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per_row * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 88);

  const std::vector<float> batched = ragged_pass(rows, q, kp, vp);
  LSE_EXPECT_EQ(batched.size(), per_row * rows.size());
  if (batched.size() != per_row * rows.size()) return;

  // 1. Each live row alone, at its own position and length, gives the same bits.
  for (std::size_t r = 0; r + 1 < rows.size(); ++r) {
    const std::vector<RaggedRow> solo{rows[r]};
    const std::vector<float> one_q(q.begin() + static_cast<std::ptrdiff_t>(r * per_row),
                                   q.begin() + static_cast<std::ptrdiff_t>((r + 1) * per_row));
    const std::vector<float> alone = ragged_pass(solo, one_q, kp, vp);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < per_row; ++i) {
      if (std::memcmp(&alone[i], &batched[r * per_row + i], sizeof(float)) != 0) {
        ++differ;
      }
    }
    std::printf("       row %zu (pos %d, len %d) batched vs alone: %zu of %zu differ\n",
                r, rows[r].first, rows[r].len, differ, per_row);
    LSE_EXPECT_EQ(differ, 0u);
  }

  // 2. A row holding no sequence answers zero and reads nobody's blocks.
  for (std::size_t i = 0; i < per_row; ++i) {
    LSE_EXPECT_EQ(batched[3 * per_row + i], 0.0f);
  }

  // 3. Row 0 does not move when the rest of the batch is rewritten under it.
  // This is the assertion an index that lost its row term fails while every
  // single-row test still passes.
  std::vector<RaggedRow> other{
      rows[0],
      {7, 8, {3}},
      {0, 0, {}},
      {60, 61, {9, 12, 6, 14}},
  };
  std::vector<float> q2 = q;
  for (std::size_t i = per_row; i < q2.size(); ++i) q2[i] = noise(i + 9999);
  const std::vector<float> shuffled = ragged_pass(other, q2, kp, vp);
  LSE_EXPECT_EQ(shuffled.size(), batched.size());
  if (shuffled.size() != batched.size()) return;
  std::size_t moved = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (std::memcmp(&shuffled[i], &batched[i], sizeof(float)) != 0) ++moved;
  }
  std::printf("       row 0 under a different batch: %zu of %zu moved\n", moved,
              per_row);
  LSE_EXPECT_EQ(moved, 0u);

  // 4. The rows are not each other: a kernel that read one row's descriptor for
  // every row would pass 1-3 above if the rows happened to agree, so say it.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (batched[i] == batched[per_row + i]) ++same;
  }
  LSE_EXPECT(same < per_row);
}

LSE_TEST(a_ragged_write_puts_each_row_at_its_own_position) {
  // The write side of the same rule. Two rows at different absolute positions
  // must land in their own blocks at their own slots; a shared position would
  // have one of them overwrite the other's KV, which reads as a model that
  // slowly forgets under load.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  const std::int32_t pool_blocks = 6;
  const std::int32_t stride = 4;

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);

  std::vector<float> src(static_cast<std::size_t>(3 * kKvh * 1 * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 61) + 3.0f;
  graph::Array sa = filled(Shape{3, kKvh, 1, kW}, src);

  // Row 0 writes position 2 of block 1; row 1 writes position 17, which is slot
  // 1 of its *second* block, block 4; row 2 holds no sequence.
  std::vector<float> table(static_cast<std::size_t>(3 * stride), 0.0f);
  table[0] = 1.0f;
  table[static_cast<std::size_t>(stride)] = 5.0f;
  table[static_cast<std::size_t>(stride) + 1] = 4.0f;
  table[static_cast<std::size_t>(2 * stride)] = 3.0f;
  graph::Array ta = filled(Shape{3, stride}, table);
  graph::Array meta = step_meta({{2, 3}, {17, 18}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  const auto at = [&](std::int32_t blk, std::int64_t h, std::int32_t slot,
                      std::int64_t w) {
    return static_cast<std::size_t>(((blk * kKvh + h) * bs + slot) * kW + w);
  };
  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t w = 0; w < kW; ++w) {
      LSE_EXPECT_EQ(got[at(1, h, 2, w)],
                    src[static_cast<std::size_t>((0 * kKvh + h) * kW + w)]);
      LSE_EXPECT_EQ(got[at(4, h, 1, w)],
                    src[static_cast<std::size_t>((1 * kKvh + h) * kW + w)]);
    }
  }
  // Block 3 belongs to the row holding no sequence, and blocks 0/2/5 were never
  // a write target: all of them stay zero.
  for (std::int32_t blk : {0, 2, 3, 5}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(the_paged_write_lands_where_the_block_table_says) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  const std::int32_t pool_blocks = 4;
  const std::int32_t stride = 4;
  const std::int64_t t = 3;
  const std::int32_t pos = bs - 1;  // straddles the block boundary

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);

  std::vector<float> src(static_cast<std::size_t>(2 * kKvh * t * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 13) + 2.0f;
  graph::Array sa = filled(Shape{2, kKvh, t, kW}, src);

  // Row 0 -> blocks 1 then 3; row 1 is padding and must write nothing.
  std::vector<float> table(static_cast<std::size_t>(2 * stride), 0.0f);
  table[0] = 1.0f;
  table[1] = 3.0f;
  table[static_cast<std::size_t>(stride)] = 2.0f;
  graph::Array ta = filled(Shape{2, stride}, table);
  graph::Array meta = step_meta(
      {{pos, pos + static_cast<std::int32_t>(t)}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t j = 0; j < t; ++j) {
      const std::int32_t abs = pos + static_cast<std::int32_t>(j);
      const std::int32_t blk = abs < bs ? 1 : 3;
      const std::int32_t slot = abs % bs;
      for (std::int64_t w = 0; w < kW; ++w) {
        const auto dst = static_cast<std::size_t>(
            ((blk * kKvh + h) * bs + slot) * kW + w);
        const auto s_i = static_cast<std::size_t>((h * t + j) * kW + w);
        LSE_EXPECT_EQ(got[dst], src[s_i]);
      }
    }
  }
  // Block 2 is the padded row's target and block 0 was never named: both stay
  // zero, so a pad row cannot corrupt the pool.
  for (std::int32_t blk : {0, 2}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(the_same_token_gets_the_same_logits_at_every_pass_width) {
  // Width invariance across the row axis: prefill the same prompt through
  // different pass plans and the final logits must agree. Extents are baked, so
  // each plan is a different set of kernels reading the same paged KV.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<std::uint32_t> prompt{2, 3, 5, 7, 11, 13, 17, 19};

  const auto run = [&](const std::vector<std::size_t>& plan) {
    Session session("width", lm->num_layers());
    std::vector<float> out;
    std::size_t at = 0;
    for (std::size_t take : plan) {
      std::vector<float> ids(take);
      for (std::size_t i = 0; i < take; ++i) {
        ids[i] = static_cast<float>(prompt[at + i]);
      }
      graph::Array tokens =
          filled(Shape{1, static_cast<std::int64_t>(take)}, ids);
      auto h = lm->hidden(tokens, &session.states(), nullptr);
      if (!h.ok()) return out;
      at += take;
      if (at != prompt.size()) continue;
      const Shape& hs = h->shape();
      graph::Array row = graph::slice(*h, static_cast<int>(hs.rank()) - 2,
                                     hs.dim(hs.rank() - 2) - 1,
                                     hs.dim(hs.rank() - 2));
      auto lg = lm->lm_head(row);
      if (!lg.ok()) return out;
      out = read_all(*lg);
    }
    return out;
  };

  const std::vector<float> ones = run({1, 1, 1, 1, 1, 1, 1, 1});
  LSE_EXPECT(!ones.empty());
  if (ones.empty()) return;
  for (const std::vector<std::size_t>& plan :
       std::vector<std::vector<std::size_t>>{{8}, {4, 4}, {1, 2, 4, 1}, {2, 2, 4}}) {
    const std::vector<float> got = run(plan);
    LSE_EXPECT_EQ(got.size(), ones.size());
    if (got.size() != ones.size()) continue;
    double worst = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(got[i] - ones[i])));
      ref = std::max(ref, std::abs(static_cast<double>(ones[i])));
    }
    const double rel = ref > 0.0 ? worst / ref : worst;
    std::printf("       plan of %zu pass(es): max_rel=%.3e\n", plan.size(), rel);
    LSE_EXPECT_EQ(argmax(got), argmax(ones));
    LSE_EXPECT(rel < 2e-3);
  }
}

LSE_TEST(a_paged_session_holds_only_the_blocks_it_reached) {
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  Session session("budget", lm->num_layers());
  std::vector<float> ids{2.0f, 3.0f, 5.0f, 7.0f, 11.0f};
  graph::Array tokens = filled(Shape{1, 5}, ids);
  auto h = lm->hidden(tokens, &session.states(), nullptr);
  LSE_EXPECT_OK(h.status());
  if (!h.ok()) return;

  // 5 tokens is one block per attention layer, in a pool at the smallest rung.
  std::size_t attn_layers = 0;
  for (const auto& st : session.states()) {
    if (!st.paged.valid()) continue;
    ++attn_layers;
    LSE_EXPECT_EQ(st.paged.tables.size(), 1u);
    LSE_EXPECT_EQ(st.paged.tables[0].size(), 1);
    LSE_EXPECT_EQ(st.key_cache.shape().dim(0), kv::kMinPoolBlocks);
  }
  LSE_EXPECT(attn_layers > 0u);
  LSE_EXPECT_EQ(session.kv_blocks(), attn_layers);

  // What a contiguous cache at the engine length would have cost, against what
  // the pools actually hold.
  const auto cap = static_cast<std::size_t>(cfg->kv_capacity());
  std::size_t paged_bytes = 0;
  std::size_t contiguous_bytes = 0;
  for (const auto& st : session.states()) {
    if (!st.paged.valid()) continue;
    paged_bytes += st.paged.pool_bytes();
    const Shape& p = st.key_cache.shape();
    const auto per_token = static_cast<std::size_t>(p.dim(1) * p.dim(3)) *
                           sizeof(float) * 2;
    contiguous_bytes += per_token * cap;
  }
  std::printf("       KV pools: %.3f MiB paged vs %.3f MiB contiguous (%.1fx)\n",
              static_cast<double>(paged_bytes) / 1048576.0,
              static_cast<double>(contiguous_bytes) / 1048576.0,
              static_cast<double>(contiguous_bytes) /
                  static_cast<double>(paged_bytes));
  LSE_EXPECT(paged_bytes * 8 <= contiguous_bytes);
}

namespace {

// The lemonseed fixture: the model plus the memory-mapped checkpoint its
// weights are views into, which therefore has to outlive it. Empty when the
// checkpoint is not on this box.
struct Lemonseed {
  std::unique_ptr<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  explicit operator bool() const noexcept { return lm != nullptr; }
};

Lemonseed load_lemonseed() {
  Lemonseed out;
  if (!have_model()) return out;
  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return out;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return out;
  out.weights = std::make_unique<model::SafeTensors>(ckpt.release());
  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*out.weights);
  if (!lm->load(binder).ok()) return out;
  out.lm = std::move(lm);
  return out;
}

SamplingParams greedy_params() {
  SamplingParams p;
  p.temperature = 0.0f;
  p.repetition_penalty = 1.0f;
  return p;
}

std::string ids_to_string(const std::vector<std::uint32_t>& v) {
  std::string s;
  for (std::uint32_t id : v) {
    if (!s.empty()) s += ' ';
    s += std::to_string(id);
  }
  return s;
}

}  // namespace

LSE_TEST(a_batch_of_one_says_exactly_what_the_single_session_path_says) {
  // The batch driver is not a second engine. One sequence through it must give
  // the same tokens as Generator gives the same sequence, because that path is
  // where every baseline in WORK.md was measured: a batch of one that differed
  // would mean the baselines no longer describe the engine.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::uint32_t> prompt{11u, 907u, 40u, 5u, 82u};
  constexpr std::int32_t kWant = 24;

  Generator gen(lm, greedy_params());
  Session session("solo", lm.num_layers());
  GenerationLimits glimits;
  glimits.max_tokens = kWant;
  auto want = gen.generate(session, prompt, glimits);
  LSE_EXPECT_OK(want.status());
  if (!want.ok()) return;

  BatchLimits limits;
  limits.max_batch = 1;
  limits.max_tokens = kWant;
  BatchScheduler one(lm, greedy_params(), limits);
  LSE_EXPECT_OK(one.submit({"solo", prompt, kWant}));
  auto got = one.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok() || got->empty()) return;

  LSE_EXPECT_EQ(one.bucket(), 1);
  std::printf("       generator [%s]\n       batch-of-1 [%s] | %.1f tok/s "
              "aggregate, %.1f tok/s per session\n",
              ids_to_string(*want).c_str(),
              ids_to_string((*got)[0].generated).c_str(),
              one.stats().aggregate_tokens_per_second(),
              (*got)[0].tokens_per_second());
  LSE_EXPECT((*got)[0].generated == *want);
}

LSE_TEST(a_sequence_decodes_the_same_whoever_shares_its_batch) {
  // The acceptance gate for the driver, through the whole model rather than one
  // kernel: run three sequences of different prompt lengths together, then run
  // each of them alone in an engine of the same width, and the token streams
  // must be identical.
  //
  // Three sequences into two rows on purpose. Rows admitted in the same step
  // advance in lockstep, so they share an absolute position however different
  // their prompts are, and a shared-scalar kernel is accidentally right about
  // them. The third sequence joins when a row frees, at a position the other
  // live row is nowhere near — which is the case a single meta[0] gets wrong
  // while still producing fluent text.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u, 40u, 5u, 82u},
      {3u, 19u},
      {77u, 4u, 913u},
  };
  constexpr std::int32_t kWant = 6;

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = kWant;

  BatchScheduler together(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(together.submit({"s" + std::to_string(i), prompts[i], kWant}));
  }
  auto batched = together.run();
  LSE_EXPECT_OK(batched.status());
  if (!batched.ok()) return;
  LSE_EXPECT_EQ(batched->size(), prompts.size());
  if (batched->size() != prompts.size()) return;

  std::printf("       batch of %zu: %.2f tok/s aggregate over %d step(s), "
              "occupancy %.2f\n",
              prompts.size(), together.stats().aggregate_tokens_per_second(),
              together.stats().steps, together.stats().occupancy());

  for (const SequenceResult& r : *batched) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s batched [%s] vs alone [%s] | %.1f tok/s per session, "
                "ttft %.1f ms\n",
                r.id.c_str(), ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str(),
                r.tokens_per_second(),
                static_cast<double>(r.ttft_ns) / 1e6);
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(a_sequence_that_joins_a_running_batch_gets_its_own_answer) {
  // Sessions join and leave between steps. Six sequences through four rows, with
  // different token budgets so they retire at different steps and the ones still
  // waiting are admitted into the rows that free up. A slot that keeps the
  // previous occupant's recurrent state would answer this wrong, fluently.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u}, {3u},        {77u, 4u, 913u},
      {5u, 6u},    {820u, 12u}, {41u},
  };
  const std::vector<std::int32_t> budget{2, 5, 3, 6, 4, 5};

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = 8;

  BatchScheduler mixed(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(mixed.submit(
        {"j" + std::to_string(i), prompts[i], budget[i]}));
  }
  auto got = mixed.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), prompts.size());
  if (got->size() != prompts.size()) return;
  // More sequences than rows, so at least one was admitted into a row somebody
  // else had already used.
  LSE_EXPECT(mixed.stats().admissions >
             static_cast<std::int32_t>(mixed.bucket()));

  for (const SequenceResult& r : *got) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], budget[i]}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s joined mid-flight [%s] vs alone [%s]\n",
                r.id.c_str(), ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str());
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(budget[i]));
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(churning_sessions_hand_every_block_back) {
  // ops::ensure_paged used to reassign a layer's block tables without releasing
  // the old ones, so every row that went away leaked its blocks. At a batch of
  // one that path never ran; with sessions retiring and being replaced it runs
  // constantly, and a leak shows up only much later as a pool that will not
  // admit anything. So watch the free list directly: many sequences through few
  // rows, twice, and the pool must come back to full both times without having
  // had to grow the second time.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::vector<std::uint32_t>> prompts;
  std::vector<std::int32_t> budget;
  for (std::uint32_t i = 0; i < 10u; ++i) {
    prompts.push_back({11u + i, 40u + i * 7u, 900u - i * 13u});
    budget.push_back(2 + static_cast<std::int32_t>(i % 5u));
  }

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = 8;
  limits.kv_blocks = 8;
  limits.kv_reserve = 0;

  BatchScheduler churn(lm, greedy_params(), limits);
  std::int32_t low_water = std::numeric_limits<std::int32_t>::max();
  auto wave = [&](const char* tag) {
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      LSE_EXPECT_OK(churn.submit(
          {std::string(tag) + std::to_string(i), prompts[i], budget[i]}));
    }
    auto got = churn.run([&](const std::string&, std::uint32_t) {
      low_water = std::min(low_water, churn.kv_blocks_free());
      return true;
    });
    LSE_EXPECT_OK(got.status());
    if (got.ok()) LSE_EXPECT_EQ(got->size(), prompts.size());
    std::printf("       %s: %zu sequence(s), %d admission(s), %d preemption(s),"
                " pool %d block(s), free %d, low water %d\n",
                tag, prompts.size(), churn.stats().admissions,
                churn.stats().preemptions, churn.kv_blocks_total(),
                churn.kv_blocks_free(), low_water);
    // Every sequence has left, so every block it held is back.
    LSE_EXPECT_EQ(churn.kv_blocks_free(), churn.kv_blocks_total());
  };

  wave("w0");
  const std::int32_t pool_after_first = churn.kv_blocks_total();
  const std::int32_t admissions_first = churn.stats().admissions;
  // Blocks were genuinely held while the batch ran, so a full free list at the
  // end is a return rather than a pool that was never used.
  LSE_EXPECT(low_water < pool_after_first);

  wave("w1");
  // The second wave found the pool exactly as the first left it: a leak would
  // have forced it to a bigger rung or to more preemptions to fit.
  LSE_EXPECT_EQ(churn.kv_blocks_total(), pool_after_first);
  LSE_EXPECT(churn.stats().admissions >= 2 * admissions_first);
}

LSE_TEST(an_exhausted_block_pool_preempts_instead_of_failing) {
  // Exhaustion is a decision. With a budget too small for every sequence at
  // once, kv::BlockPolicy names victims oldest-first, they hand their blocks
  // back, and they are re-admitted later with their history as the prompt — so
  // every sequence still finishes, and finishes with the answer it would have
  // given alone.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {11u, 907u, 40u}, {3u, 19u}, {77u, 4u}, {820u, 12u, 6u},
  };
  constexpr std::int32_t kWant = 30;

  BatchLimits limits;
  limits.max_batch = 4;
  limits.max_tokens = kWant;
  // A block covers 16 tokens, so each of these reaches three of them. Four rows
  // want twelve and the pool holds six: they fit while they are short and stop
  // fitting as they grow, which is when a running sequence — not a waiting one —
  // is the thing that cannot get a block.
  limits.kv_blocks = 6;
  limits.kv_reserve = 0;

  BatchScheduler tight(lm, greedy_params(), limits);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    LSE_EXPECT_OK(tight.submit({"p" + std::to_string(i), prompts[i], kWant}));
  }
  auto got = tight.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  std::printf("       %d block(s) for %zu sequence(s): %d preemption(s), "
              "%d admission(s)\n",
              tight.block_ceiling(), prompts.size(),
              tight.stats().preemptions, tight.stats().admissions);
  LSE_EXPECT_EQ(got->size(), prompts.size());
  if (got->size() != prompts.size()) return;
  LSE_EXPECT(tight.stats().preemptions > 0);

  BatchLimits roomy = limits;
  roomy.kv_blocks = 0;
  for (const SequenceResult& r : *got) {
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    BatchScheduler alone(lm, greedy_params(), roomy);
    LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    std::printf("       %s preempted %d time(s) [%s] vs untouched [%s]\n",
                r.id.c_str(), r.preemptions,
                ids_to_string(r.generated).c_str(),
                ids_to_string((*solo)[0].generated).c_str());
    LSE_EXPECT(r.generated == (*solo)[0].generated);
  }
}

LSE_TEST(a_pool_that_cannot_hold_one_sequence_refuses_it_by_name) {
  // The other half of the same decision: a request that does not fit even with
  // the pool empty is not a transient condition, so it is refused with the size
  // rather than queued forever or allowed to fail mid-step.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = 2;
  limits.kv_blocks = 1;  // 16 tokens
  limits.kv_reserve = 0;

  BatchScheduler tiny(lm, greedy_params(), limits);
  std::vector<std::uint32_t> long_prompt(40u, 7u);
  LSE_EXPECT_OK(tiny.submit({"big", long_prompt, 2}));
  auto got = tiny.run();
  LSE_EXPECT(!got.ok());
  std::printf("       refusal: %s\n", got.status().message().c_str());
  LSE_EXPECT(got.status().message().find("big") != std::string::npos);
  LSE_EXPECT(got.status().message().find("block") != std::string::npos);
}

LSE_TEST(a_batch_that_fits_no_bucket_is_refused_by_size) {
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;
  BatchLimits limits;
  limits.max_batch = 64;
  BatchScheduler over(lm, greedy_params(), limits);
  const Status s = over.submit({"x", {1u}, 1});
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.message().find("64") != std::string::npos);
  LSE_EXPECT(s.message().find("fits no bucket") != std::string::npos);
}

LSE_TEST(a_two_row_pass_gives_each_row_what_it_gets_alone) {
  // Width invariance for the batch axis through the whole model, not just the
  // attention kernel: MatmulKernel::specialize() picks GEMV vs WMMA by row count
  // and MoE picks per expert, so a row's answer must not depend on how many rows
  // shared the pass. Row 1 must also differ from row 0 — the quant_linear defect
  // that read row 0 for every token produced fluent text and passed everything
  // that did not check this.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  const std::vector<float> ids{11.0f, 907.0f};

  Session pair("pair", lm->num_layers());
  graph::Array tokens = filled(Shape{2, 1}, ids);
  const model::StepRows plan{{0, 0}};
  auto both = lm->hidden(tokens, &pair.states(), nullptr, nullptr, &plan);
  LSE_EXPECT_OK(both.status());
  if (!both.ok()) return;
  const std::vector<float> got = read_all(*both);
  const std::size_t width = got.size() / 2;
  LSE_EXPECT(width > 0u);
  if (width == 0u) return;

  std::size_t same = 0;
  for (std::size_t i = 0; i < width; ++i) {
    if (got[i] == got[width + i]) ++same;
  }
  std::printf("       hidden row0 vs row1: %zu of %zu identical\n", same, width);
  LSE_EXPECT(same < width / 2);

  for (std::size_t r = 0; r < 2; ++r) {
    Session solo("solo", lm->num_layers());
    graph::Array one = filled(Shape{1, 1}, {ids[r]});
    auto alone = lm->hidden(one, &solo.states(), nullptr);
    LSE_EXPECT_OK(alone.status());
    if (!alone.ok()) return;
    const std::vector<float> ref = read_all(*alone);
    LSE_EXPECT_EQ(ref.size(), width);
    if (ref.size() != width) return;
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      worst = std::max(worst,
                       std::abs(static_cast<double>(got[r * width + i] - ref[i])));
      scale = std::max(scale, std::abs(static_cast<double>(ref[i])));
    }
    const double rel = scale > 0.0 ? worst / scale : worst;
    std::printf("       row %zu in a pair vs alone: max_rel=%.3e\n", r, rel);
    LSE_EXPECT(rel < 2e-3);
  }

  // An off-ladder batch is refused by name, not silently widened.
  Session odd("odd", lm->num_layers());
  std::vector<float> three(3, 5.0f);
  graph::Array wide = filled(Shape{3, 1}, three);
  auto bad = lm->hidden(wide, &odd.states(), nullptr);
  LSE_EXPECT(!bad.ok());
  LSE_EXPECT(bad.status().message().find("not a batch bucket") !=
             std::string::npos);
}

LSE_TEST(a_context_that_outgrows_its_pool_keeps_the_keys_it_wrote) {
  // Crossing a pool rung reallocates the block pool and copies the used prefix.
  // Two pass plans over the same 132 tokens cross it at different points; if the
  // copy or the block table were wrong the two would disagree, and a prompt long
  // enough to grow would quietly read the wrong keys.
  //
  // 132 tokens is 9 blocks, one past the smallest rung of 8.
  if (!have_model()) return;

  auto paths = model::resolve_model(model_dir());
  if (!paths.ok()) return;
  auto ckpt = model::SafeTensors::open(paths->weights);
  auto cfg = model::Config::from_json_file(paths->config);
  if (!ckpt.ok() || !cfg.ok()) return;

  auto lm = model::make_lemonseed(*cfg);
  model::WeightBinder binder(*ckpt);
  if (!lm->load(binder).ok()) return;

  constexpr std::size_t kTokens = 132;
  std::vector<float> prompt(kTokens);
  for (std::size_t i = 0; i < kTokens; ++i) {
    prompt[i] = static_cast<float>(3 + (i * 37) % 900);
  }

  // `widths` cycles pass by pass, so a plan can mix widths and still keep every
  // pass on the ladder the engine already compiled.
  const auto run = [&](const std::vector<std::size_t>& widths,
                       std::int32_t* blocks_out) {
    Session session("grow", lm->num_layers());
    std::vector<float> out;
    std::size_t at = 0;
    for (std::size_t pass = 0; at < kTokens; ++pass) {
      const std::size_t n = widths[pass % widths.size()];
      if (at + n > kTokens) break;
      std::vector<float> ids(prompt.begin() + static_cast<std::ptrdiff_t>(at),
                             prompt.begin() + static_cast<std::ptrdiff_t>(at + n));
      graph::Array tokens =
          filled(Shape{1, static_cast<std::int64_t>(n)}, ids);
      auto h = lm->hidden(tokens, &session.states(), nullptr);
      if (!h.ok()) return out;
      at += n;
      if (at != kTokens) continue;
      const Shape& hs = h->shape();
      graph::Array row = graph::slice(*h, static_cast<int>(hs.rank()) - 2,
                                     hs.dim(hs.rank() - 2) - 1,
                                     hs.dim(hs.rank() - 2));
      auto lg = lm->lm_head(row);
      if (!lg.ok()) return out;
      out = read_all(*lg);
    }
    if (blocks_out != nullptr) {
      *blocks_out = 0;
      for (const auto& st : session.states()) {
        if (!st.paged.valid()) continue;
        *blocks_out = st.paged.tables[0].size();
        LSE_EXPECT(st.key_cache.shape().dim(0) > kv::kMinPoolBlocks);
        break;
      }
    }
    return out;
  };

  std::int32_t blocks = 0;
  const std::vector<float> by_four = run({4}, &blocks);
  LSE_EXPECT(!by_four.empty());
  if (by_four.empty()) return;
  std::printf("       132 tokens held in %d block(s) after growth\n", blocks);
  LSE_EXPECT_EQ(blocks, kv::blocks_for(132, kv::kBlockSize));

  for (const std::vector<std::size_t>& widths :
       std::vector<std::vector<std::size_t>>{{1}, {2}, {1, 2, 1}, {2, 1, 1}}) {
    const std::vector<float> got = run(widths, nullptr);
    LSE_EXPECT_EQ(got.size(), by_four.size());
    if (got.size() != by_four.size()) continue;
    double worst = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(got[i] - by_four[i])));
      ref = std::max(ref, std::abs(static_cast<double>(by_four[i])));
    }
    const double rel = ref > 0.0 ? worst / ref : worst;
    std::printf("       across a pool rung, %zu width(s) cycling: max_rel=%.3e\n",
                widths.size(), rel);
    LSE_EXPECT_EQ(argmax(got), argmax(by_four));
    LSE_EXPECT(rel < 2e-3);
  }
}

LSE_TEST_MAIN()

// The SwiGLU chain, at lemonseed's expert width, through the real scheduler.
// gate and up join one wide-linear group, silu and mul are one lane-fused grid
// launch, down is its own: three launches, not four. The pair used to split
// because stage_threads(2176) marks each stage fat and a fat stage that read the
// chunk forced a flush — 120 launches per token on lemonseed, 32% of them, for a
// barrier the chain never needed. Counted here rather than in the emitter
// because the splitter and the emitter have to agree: if only one of them
// changes, the pair still merges and then lands on ONE workgroup, which measured
// slower than the two launches it replaced.
LSE_TEST(the_swiglu_chain_costs_three_launches_not_four) {
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;
  if (sched->backend().emitter() == nullptr) return;

  constexpr std::int64_t kHidden = 1024;
  constexpr std::int64_t kInter = 2176;
  graph::Array h = graph::Array::zeros(Shape{1, 1, kHidden}, DType::kF32);
  graph::Array wg = graph::Array::zeros(Shape{kInter, kHidden}, DType::kF32);
  graph::Array wu = graph::Array::zeros(Shape{kInter, kHidden}, DType::kF32);
  graph::Array wd = graph::Array::zeros(Shape{kHidden, kInter}, DType::kF32);
  // Resident before the step, the way loaded weights are: an unmaterialized
  // weight is still a node in the topological order, and a sibling whose weight
  // has not been reached yet is not admitted to the run — so without this the
  // two projections never share a launch and the shape under test is gone.
  for (graph::Array* w : {&h, &wg, &wu, &wd}) {
    LSE_EXPECT(w->eval().ok());
  }
  // Built in the model's order: both projections, then the pair over them.
  // The phase splitter walks the nodes as they were recorded, so building the
  // gate's silu before the up projection is a different graph to the one
  // lemonseed records and would not exercise this.
  graph::Array gate = graph::linear(h, wg);
  graph::Array up = graph::linear(h, wu);
  graph::Array hid = graph::silu(gate) * up;
  graph::Array out = graph::linear(hid, wd);

  sched->reset_accumulated_trace();
  const Status ev = out.eval();
  LSE_EXPECT(ev.ok());
  if (!ev.ok()) return;
  const auto& t = sched->last_trace();
  LSE_EXPECT_EQ(t.host_groups, 0u);
  LSE_EXPECT_EQ(t.kernels_launched, 3u);
}

namespace {

// Adversarial ragged-attention checks. `ragged_pass` above is the batched
// entry; these need the pools rewritten between passes and the host reference
// run on the identical graph, so they build the pass themselves.
struct RagPass {
  std::vector<float> out;
  std::size_t per_row = 0;
};

RagPass rag_run(const std::vector<RaggedRow>& rows, const std::vector<float>& q,
                const std::vector<float>& kbuf, const std::vector<float>& vbuf,
                graph::MaskKind mask, int window, bool host_only) {
  RagPass r;
  r.per_row = static_cast<std::size_t>(kRagQh * kRagHd);
  const auto bucket = static_cast<std::int64_t>(rows.size());
  std::vector<float> table(static_cast<std::size_t>(bucket * kRagStride), 0.0f);
  std::vector<std::pair<std::int32_t, std::int32_t>> meta;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t b = 0; b < rows[i].blocks.size(); ++b) {
      table[i * static_cast<std::size_t>(kRagStride) + b] =
          static_cast<float>(rows[i].blocks[b]);
    }
    meta.push_back({rows[i].first, rows[i].len});
  }
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, kbuf);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, vbuf);
  graph::Array ta = filled(Shape{bucket, kRagStride}, table);
  graph::Array ma = step_meta(meta);
  graph::Array qa = filled(Shape{bucket, kRagQh, 1, kRagHd}, q);
  graph::Array out =
      graph::sdpa_paged(qa, kp, vp, 0.25f, mask, window, ma, ta, kv::kBlockSize);
  graph::Scheduler* sched = graph::default_scheduler();
  const auto saved = sched->mode();
  if (host_only) sched->set_mode(graph::Scheduler::Mode::kHostOnly);
  const bool ok = out.eval().ok();
  sched->set_mode(saved);
  if (!ok) return r;
  r.out = read_all(out);
  return r;
}

std::size_t bitdiff(const std::vector<float>& a, std::size_t ao,
                    const std::vector<float>& b, std::size_t bo, std::size_t n) {
  std::size_t d = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[ao + i], &b[bo + i], sizeof(float)) != 0) ++d;
  }
  return d;
}

std::vector<float> rag_noise(std::size_t seed) {
  std::vector<float> v(static_cast<std::size_t>(kRagPool * kRagKvh *
                                                kv::kBlockSize * kRagHd));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = noise(i + seed);
  return v;
}

// Overwrite one block of a [blocks, Hkv, block_size, Hd] pool.
void poison_block(std::vector<float>& pool, std::int32_t blk, float base) {
  const auto per = static_cast<std::size_t>(kRagKvh * kv::kBlockSize * kRagHd);
  const std::size_t at = static_cast<std::size_t>(blk) * per;
  for (std::size_t i = 0; i < per; ++i) {
    pool[at + i] = base + static_cast<float>(i % 13) * 0.75f;
  }
}

}  // namespace

LSE_TEST(verify_a_pad_row_anywhere_in_the_bucket_changes_nothing) {
  // A pad row must be inert, and not only when it sits after every live row:
  // a session that retires mid-batch leaves a hole below the last live row, so
  // the case that matters is a pad *between* two live rows and a pad *before*
  // them. Both must leave every live row bit-identical to the tight batch.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const RaggedRow a{4, 5, {2}};
  const RaggedRow b{39, 40, {5, 1, 7}};
  const RaggedRow pad{0, 0, {}};

  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q2(per * 2);
  for (std::size_t i = 0; i < q2.size(); ++i) q2[i] = noise(i + 88);
  const auto qrow = [&](std::size_t r) {
    return std::vector<float>(q2.begin() + static_cast<std::ptrdiff_t>(r * per),
                              q2.begin() +
                                  static_cast<std::ptrdiff_t>((r + 1) * per));
  };

  const RagPass tight = rag_run({a, b}, q2, kb, vb, graph::MaskKind::kCausal, 0,
                                false);
  LSE_EXPECT_EQ(tight.out.size(), per * 2);
  if (tight.out.size() != per * 2) return;

  struct Layout {
    const char* name;
    std::vector<RaggedRow> rows;
    std::vector<int> live;  // row index in this layout for a, then for b
  };
  std::vector<Layout> cases{
      {"trailing pads", {a, b, pad, pad}, {0, 1}},
      {"a hole between them", {a, pad, b}, {0, 2}},
      {"pads before and between", {pad, a, pad, b, pad}, {1, 3}},
  };
  for (const Layout& c : cases) {
    std::vector<float> q(per * c.rows.size(), 0.0f);
    for (std::size_t i = 0; i < c.rows.size(); ++i) {
      // Pad rows carry live-looking queries: inertness must come from the
      // descriptor, not from the row happening to be zero.
      const std::vector<float> src =
          static_cast<int>(i) == c.live[0]   ? qrow(0)
          : static_cast<int>(i) == c.live[1] ? qrow(1)
                                             : qrow(i % 2);
      std::copy(src.begin(), src.end(), q.begin() + static_cast<std::ptrdiff_t>(i * per));
    }
    const RagPass got =
        rag_run(c.rows, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    LSE_EXPECT_EQ(got.out.size(), per * c.rows.size());
    if (got.out.size() != per * c.rows.size()) return;
    for (std::size_t k = 0; k < 2; ++k) {
      const std::size_t d =
          bitdiff(got.out, static_cast<std::size_t>(c.live[k]) * per, tight.out,
                  k * per, per);
      std::printf("       %-24s live row %zu: %zu of %zu differ\n", c.name, k, d,
                  per);
      LSE_EXPECT_EQ(d, 0u);
    }
    for (std::size_t i = 0; i < c.rows.size(); ++i) {
      if (static_cast<int>(i) == c.live[0] || static_cast<int>(i) == c.live[1]) {
        continue;
      }
      for (std::size_t e = 0; e < per; ++e) {
        LSE_EXPECT_EQ(got.out[i * per + e], 0.0f);
      }
    }
  }
}

LSE_TEST(verify_a_short_row_cannot_see_a_long_rows_keys) {
  // Leakage, stated as a dependency: rewrite every block the short row does not
  // own — the long row's three blocks and block 0, which is what the block
  // table's pad resolves to — and the short row's answer must not move a bit.
  // The long row must move, or the poison never reached the kernel.
  if (graph::default_scheduler() == nullptr) return;
  std::vector<float> kb = rag_noise(17);
  std::vector<float> vb = rag_noise(4211);
  const RaggedRow shortr{2, 3, {2}};
  const RaggedRow longr{39, 40, {5, 1, 7}};

  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * 2);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 505);

  const RagPass before =
      rag_run({shortr, longr}, q, kb, vb, graph::MaskKind::kCausal, 0, false);
  LSE_EXPECT_EQ(before.out.size(), per * 2);
  if (before.out.size() != per * 2) return;

  for (std::int32_t blk : {0, 1, 5, 7}) {
    poison_block(kb, blk, 40.0f + static_cast<float>(blk));
    poison_block(vb, blk, -70.0f - static_cast<float>(blk));
  }
  const RagPass after =
      rag_run({shortr, longr}, q, kb, vb, graph::MaskKind::kCausal, 0, false);
  LSE_EXPECT_EQ(after.out.size(), per * 2);
  if (after.out.size() != per * 2) return;

  const std::size_t moved_short = bitdiff(before.out, 0, after.out, 0, per);
  const std::size_t moved_long =
      bitdiff(before.out, per, after.out, per, per);
  std::printf("       poisoned blocks 0/1/5/7: short row moved %zu of %zu, "
              "long row moved %zu of %zu\n",
              moved_short, per, moved_long, per);
  LSE_EXPECT_EQ(moved_short, 0u);
  LSE_EXPECT(moved_long > 0);
}

LSE_TEST(verify_the_causal_mask_bounds_every_row_at_its_own_position) {
  // The mask is the whole correctness argument for raggedness: a row at
  // position p must see nothing past p whatever its live length says. Stated
  // exactly: giving a row 40 live keys when it sits at position 5 must be
  // bit-identical to giving it 6, because keys 6..39 are masked away and a
  // masked term adds 0.0f to the denominator and fma(0, v, acc) to the sum.
  //
  // Checked with the row under test in slot 1 and again in slot 0, at two
  // different positions, so a kernel right about row 0 alone fails it.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * 2);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 313);

  struct Case {
    const char* name;
    std::int32_t under;  // which row is the one being bounded
    std::vector<RaggedRow> wide;
    std::vector<RaggedRow> tight;
    std::vector<RaggedRow> cut;  // one key short of the diagonal
  };
  const RaggedRow other{30, 31, {4, 6}};
  std::vector<Case> cases{
      {"row 1 at position 5",
       1,
       {other, {5, 40, {5, 1, 7}}},
       {other, {5, 6, {5, 1, 7}}},
       {other, {5, 5, {5, 1, 7}}}},
      {"row 0 at position 20",
       0,
       {{20, 40, {5, 1, 7}}, other},
       {{20, 21, {5, 1, 7}}, other},
       {{20, 20, {5, 1, 7}}, other}},
  };
  for (const Case& c : cases) {
    const RagPass wide =
        rag_run(c.wide, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    const RagPass tight =
        rag_run(c.tight, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    const RagPass cut =
        rag_run(c.cut, q, kb, vb, graph::MaskKind::kCausal, 0, false);
    if (wide.out.size() != per * 2 || tight.out.size() != per * 2 ||
        cut.out.size() != per * 2) {
      LSE_EXPECT(false);
      return;
    }
    const std::size_t at = static_cast<std::size_t>(c.under) * per;
    const std::size_t past = bitdiff(wide.out, at, tight.out, at, per);
    const std::size_t diag = bitdiff(wide.out, at, cut.out, at, per);
    std::printf("       %-20s keys past the diagonal: %zu of %zu differ; "
                "dropping the diagonal: %zu of %zu differ\n",
                c.name, past, per, diag, per);
    LSE_EXPECT_EQ(past, 0u);
    // The other row shares the pass and must not have moved either.
    const std::size_t sibling =
        static_cast<std::size_t>(c.under == 0 ? 1 : 0) * per;
    LSE_EXPECT_EQ(bitdiff(wide.out, sibling, tight.out, sibling, per), 0u);
    // Negative control: the identity above is not vacuous.
    LSE_EXPECT(diag > 0);
  }
}

LSE_TEST(verify_the_ragged_kernel_agrees_with_the_host_reference) {
  // The JIT kernel against interpreter.cpp, which walks the block table in
  // plain C++ and shares no code with the generator. Rows at four different
  // positions and lengths, causal and sliding-window.
  if (graph::default_scheduler() == nullptr) return;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  const std::vector<RaggedRow> rows{
      {4, 5, {2}}, {39, 40, {5, 1, 7}}, {15, 16, {11}}, {0, 0, {}},
  };
  const auto per = static_cast<std::size_t>(kRagQh * kRagHd);
  std::vector<float> q(per * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 88);

  struct Mode {
    const char* name;
    graph::MaskKind mask;
    int window;
  };
  for (const Mode& m : {Mode{"causal", graph::MaskKind::kCausal, 0},
                        Mode{"window 8", graph::MaskKind::kSlidingWindow, 8}}) {
    const RagPass dev = rag_run(rows, q, kb, vb, m.mask, m.window, false);
    const RagPass ref = rag_run(rows, q, kb, vb, m.mask, m.window, true);
    LSE_EXPECT_EQ(dev.out.size(), per * rows.size());
    LSE_EXPECT_EQ(ref.out.size(), dev.out.size());
    if (dev.out.size() != ref.out.size() || dev.out.empty()) return;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      double max_abs = 0.0;
      double max_rel = 0.0;
      for (std::size_t i = 0; i < per; ++i) {
        const double a = dev.out[r * per + i];
        const double b = ref.out[r * per + i];
        const double e = std::fabs(a - b);
        max_abs = std::max(max_abs, e);
        const double mag = std::max(std::fabs(a), std::fabs(b));
        if (mag > 1e-6) max_rel = std::max(max_rel, e / mag);
      }
      std::printf("       %-9s row %zu (pos %d, len %2d): max_abs %.3e "
                  "max_rel %.3e\n",
                  m.name, r, rows[r].first, rows[r].len, max_abs, max_rel);
      LSE_EXPECT(max_abs < 1e-5);
      LSE_EXPECT(max_rel < 1e-5);
    }
  }
}

LSE_TEST(verify_a_one_token_prompt_decodes_the_same_beside_a_long_one) {
  // The widest length spread the fixture allows, end to end: a one-token prompt
  // and a forty-token one in the same batch, generating past a block boundary
  // so both rows cross into a second block at different steps. Each must give
  // the tokens it gives alone.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::uint32_t> longp;
  for (std::uint32_t i = 0; i < 40u; ++i) longp.push_back(100u + i * 7u);
  const std::vector<std::vector<std::uint32_t>> prompts{
      {41u}, longp, {7u, 8u}, {500u, 21u, 3u, 90u, 12u, 6u, 77u},
  };
  constexpr std::int32_t kWant = 24;  // > kv::kBlockSize, so blocks are added

  for (std::int32_t width : {2, 4}) {
    BatchLimits limits;
    limits.max_batch = width;
    limits.max_tokens = kWant;

    BatchScheduler together(lm, greedy_params(), limits);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      LSE_EXPECT_OK(
          together.submit({"v" + std::to_string(i), prompts[i], kWant}));
    }
    auto batched = together.run();
    LSE_EXPECT_OK(batched.status());
    if (!batched.ok()) return;
    LSE_EXPECT_EQ(batched->size(), prompts.size());
    if (batched->size() != prompts.size()) return;

    for (const SequenceResult& r : *batched) {
      const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
      BatchScheduler alone(lm, greedy_params(), limits);
      LSE_EXPECT_OK(alone.submit({r.id, prompts[i], kWant}));
      auto solo = alone.run();
      LSE_EXPECT_OK(solo.status());
      if (!solo.ok() || solo->empty()) return;
      const bool same = r.generated == (*solo)[0].generated;
      std::printf("       width %d  %s (prompt %zu): %s\n", width, r.id.c_str(),
                  prompts[i].size(), same ? "identical" : "DIFFERS");
      if (!same) {
        std::printf("         batched [%s]\n         alone   [%s]\n",
                    ids_to_string(r.generated).c_str(),
                    ids_to_string((*solo)[0].generated).c_str());
      }
      LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
      LSE_EXPECT(same);
    }
  }
}

LSE_TEST(verify_a_long_session_survives_short_ones_churning_beside_it) {
  // The point of the feature, as a diff. One session decoding 40 tokens holds a
  // row while eight two-token sessions take the other row in turn: each retires
  // mid-flight, hands its blocks back, and the next one is admitted into the
  // slot at a position the long row is nowhere near. The long session's tokens
  // must be exactly the ones it produces alone.
  //
  // Two rows and a pool small enough that the short sessions' blocks are
  // recycled, so the long row is decoding against a free list that is being
  // handed back and re-acquired under it every few steps.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  const std::vector<std::uint32_t> longp{11u, 907u, 40u, 5u, 82u, 313u, 7u};
  constexpr std::int32_t kLong = 40;  // crosses three block boundaries

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = kLong;
  limits.kv_blocks = 8;
  limits.kv_reserve = 0;

  BatchScheduler mixed(lm, greedy_params(), limits);
  LSE_EXPECT_OK(mixed.submit({"long", longp, kLong}));
  for (std::uint32_t i = 0; i < 8u; ++i) {
    LSE_EXPECT_OK(mixed.submit(
        {"s" + std::to_string(i), {200u + i * 31u, 5u + i}, 2}));
  }
  auto got = mixed.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), 9u);
  if (got->size() != 9u) return;

  BatchScheduler solo(lm, greedy_params(), limits);
  LSE_EXPECT_OK(solo.submit({"long", longp, kLong}));
  auto alone = solo.run();
  LSE_EXPECT_OK(alone.status());
  if (!alone.ok() || alone->empty()) return;

  const SequenceResult* batched = nullptr;
  for (const SequenceResult& r : *got) {
    if (r.id == "long") batched = &r;
  }
  LSE_EXPECT(batched != nullptr);
  if (batched == nullptr) return;

  // Both directions. The long row is always the furthest along, so a kernel
  // that shared one position across the batch would take *its* position and be
  // accidentally right about it; the rows that join behind it are the ones such
  // a kernel gets wrong. Checking only the long row is not a gate.
  for (const SequenceResult& r : *got) {
    if (r.id == "long") continue;
    const std::size_t i = static_cast<std::size_t>(r.id[1] - '0');
    BatchScheduler one(lm, greedy_params(), limits);
    LSE_EXPECT_OK(one.submit(
        {r.id, {200u + static_cast<std::uint32_t>(i) * 31u,
                5u + static_cast<std::uint32_t>(i)}, 2}));
    auto solo_short = one.run();
    LSE_EXPECT_OK(solo_short.status());
    if (!solo_short.ok() || solo_short->empty()) return;
    const bool same = r.generated == (*solo_short)[0].generated;
    if (!same) {
      std::printf("       short %s batched [%s] vs alone [%s]\n", r.id.c_str(),
                  ids_to_string(r.generated).c_str(),
                  ids_to_string((*solo_short)[0].generated).c_str());
    }
    LSE_EXPECT(same);
  }

  std::size_t first_diff = kLong;
  for (std::size_t i = 0; i < batched->generated.size() &&
                          i < (*alone)[0].generated.size();
       ++i) {
    if (batched->generated[i] != (*alone)[0].generated[i]) {
      first_diff = i;
      break;
    }
  }
  std::printf("       long row: %d admission(s), %d preemption(s), free %d of "
              "%d block(s) at the end; first differing token: %s\n",
              mixed.stats().admissions, mixed.stats().preemptions,
              mixed.kv_blocks_free(), mixed.kv_blocks_total(),
              first_diff == kLong ? "none" : std::to_string(first_diff).c_str());
  if (first_diff != static_cast<std::size_t>(kLong)) {
    std::printf("         batched [%s]\n         alone   [%s]\n",
                ids_to_string(batched->generated).c_str(),
                ids_to_string((*alone)[0].generated).c_str());
  }
  LSE_EXPECT_EQ(batched->generated.size(), static_cast<std::size_t>(kLong));
  LSE_EXPECT(batched->generated == (*alone)[0].generated);
  // The short rows genuinely came and went through the slot beside it.
  LSE_EXPECT(mixed.stats().admissions >= 9);
  LSE_EXPECT_EQ(mixed.kv_blocks_free(), mixed.kv_blocks_total());
}

LSE_TEST(verify_two_rows_prefill_together_from_different_positions) {
  // The multi-token ragged pass. Every other case here has T == 1, because a
  // row that is decoding has one pending token and the step width is the batch
  // minimum. Two rows both mid-prompt at different absolute positions is the
  // one arrangement that gives T > 1 *and* raggedness, and it is the only case
  // that exercises the per-row origin of RoPE and of the paged write across a
  // span of positions rather than a single one.
  //
  // Arranged, not hoped for: a one-token sequence takes a row for the first
  // step and leaves, so the sequence admitted into the freed row starts its
  // prompt while the other row is already two tokens deep.
  Lemonseed fx = load_lemonseed();
  if (!fx) return;
  model::HybridLM& lm = *fx.lm;

  std::vector<std::uint32_t> pa;
  for (std::uint32_t i = 0; i < 48u; ++i) pa.push_back(60u + i * 5u);
  std::vector<std::uint32_t> pb;
  for (std::uint32_t i = 0; i < 32u; ++i) pb.push_back(900u - i * 11u);
  // Sixteen tokens, so the row that joins after it starts a full block behind
  // the row already running. A shift of a token or two is invisible: RoPE and
  // the causal mask are both relative, so a batch whose rows are uniformly
  // displaced still answers correctly. It is the block the displaced row then
  // over-reads that does the damage, and that needs the gap to be a block.
  const std::vector<std::uint32_t> hog{4u,  9u,  21u, 33u, 44u, 51u, 62u, 70u,
                                       81u, 93u, 14u, 25u, 36u, 47u, 58u, 69u};
  constexpr std::int32_t kWant = 12;
  // Integration coverage, not the gate. Synthetic prompts drive this model into
  // a repeating attractor whose argmax survives quite large perturbations, so a
  // token diff here is evidence of a fault but agreement is not evidence of
  // correctness. The gate for T > 1 is
  // verify_a_multi_token_pass_is_ragged_too, which compares bits.

  BatchLimits limits;
  limits.max_batch = 2;
  limits.max_tokens = 64;

  BatchScheduler together(lm, greedy_params(), limits);
  LSE_EXPECT_OK(together.submit({"hog", hog, 1}));
  LSE_EXPECT_OK(together.submit({"a", pa, kWant}));
  LSE_EXPECT_OK(together.submit({"b", pb, kWant}));
  auto got = together.run();
  LSE_EXPECT_OK(got.status());
  if (!got.ok()) return;
  LSE_EXPECT_EQ(got->size(), 3u);
  if (got->size() != 3u) return;

  // Every token of both prompts, plus what they generated, in far fewer steps
  // than there are tokens: the prompts went in several at a time.
  const std::int32_t steps = together.stats().steps;
  std::printf("       %d step(s) for %zu prompt token(s) + %d generated\n",
              steps, pa.size() + pb.size() + hog.size(), 2 * kWant + 1);
  LSE_EXPECT(steps < static_cast<std::int32_t>(pa.size()));

  for (const SequenceResult& r : *got) {
    if (r.id == "hog") continue;
    const std::vector<std::uint32_t>& prompt = r.id == "a" ? pa : pb;
    BatchScheduler alone(lm, greedy_params(), limits);
    LSE_EXPECT_OK(alone.submit({r.id, prompt, kWant}));
    auto solo = alone.run();
    LSE_EXPECT_OK(solo.status());
    if (!solo.ok() || solo->empty()) return;
    const bool same = r.generated == (*solo)[0].generated;
    std::printf("       %s (prompt %zu): %s\n", r.id.c_str(), prompt.size(),
                same ? "identical" : "DIFFERS");
    if (!same) {
      std::printf("         batched [%s]\n         alone   [%s]\n",
                  ids_to_string(r.generated).c_str(),
                  ids_to_string((*solo)[0].generated).c_str());
    }
    LSE_EXPECT_EQ(r.generated.size(), static_cast<std::size_t>(kWant));
    LSE_EXPECT(same);
  }
}

LSE_TEST(verify_a_multi_token_pass_is_ragged_too) {
  // Every other ragged check here has T == 1, because a decoding row has one
  // pending token and the step width is the batch minimum. A row still feeding
  // its prompt beside a row that started earlier gives T > 1 at two different
  // origins, and that is the only arrangement in which the per-query term of
  // the causal mask, the per-row origin of RoPE and the span of positions the
  // paged write covers are all exercised at once.
  //
  // Bits, at the kernel, because the end-to-end form of this cannot be a gate:
  // a uniform displacement of one row is invisible to both RoPE and the causal
  // mask, which are relative, so the tokens can agree while the row is reading
  // the wrong slots.
  graph::Scheduler* sched = graph::default_scheduler();
  LSE_EXPECT(sched != nullptr);
  if (sched == nullptr) return;

  constexpr std::int64_t kT = 32;
  const std::vector<float> kb = rag_noise(17);
  const std::vector<float> vb = rag_noise(4211);
  graph::Array kp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, kb);
  graph::Array vp =
      filled(Shape{kRagPool, kRagKvh, kv::kBlockSize, kRagHd}, vb);

  // Row 0 is starting from nothing; row 1 is a full block further on. Row 2
  // holds no sequence.
  struct Row {
    std::int32_t first;
    std::int32_t len;
    std::vector<std::int32_t> blocks;
  };
  const std::vector<Row> rows{
      {0, 32, {3, 9}}, {16, 48, {5, 1, 7}}, {0, 0, {}},
  };
  const auto per_row = static_cast<std::size_t>(kRagQh * kT * kRagHd);
  std::vector<float> q(per_row * rows.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = noise(i + 1201);

  const auto run = [&](const std::vector<Row>& rs,
                       const std::vector<float>& qv) {
    const auto n = static_cast<std::int64_t>(rs.size());
    std::vector<float> table(static_cast<std::size_t>(n * kRagStride), 0.0f);
    std::vector<std::pair<std::int32_t, std::int32_t>> meta;
    for (std::size_t r = 0; r < rs.size(); ++r) {
      for (std::size_t i = 0; i < rs[r].blocks.size(); ++i) {
        table[r * static_cast<std::size_t>(kRagStride) + i] =
            static_cast<float>(rs[r].blocks[i]);
      }
      meta.push_back({rs[r].first, rs[r].len});
    }
    graph::Array ta = filled(Shape{n, kRagStride}, table);
    graph::Array ma = step_meta(meta);
    graph::Array qa = filled(Shape{n, kRagQh, kT, kRagHd}, qv);
    graph::Array o = graph::sdpa_paged(qa, kp, vp, 0.25f,
                                       graph::MaskKind::kCausal, 0, ma, ta,
                                       kv::kBlockSize);
    std::vector<float> out;
    if (o.eval().ok()) out = read_all(o);
    return out;
  };

  const std::vector<float> batched = run(rows, q);
  LSE_EXPECT_EQ(batched.size(), per_row * rows.size());
  if (batched.size() != per_row * rows.size()) return;

  for (std::size_t r = 0; r + 1 < rows.size(); ++r) {
    const std::vector<float> one(
        q.begin() + static_cast<std::ptrdiff_t>(r * per_row),
        q.begin() + static_cast<std::ptrdiff_t>((r + 1) * per_row));
    const std::vector<float> alone = run({rows[r]}, one);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    const std::size_t d = bitdiff(alone, 0, batched, r * per_row, per_row);
    std::printf("       T=%lld row %zu (pos %2d, len %2d): %zu of %zu differ\n",
                static_cast<long long>(kT), r, rows[r].first, rows[r].len, d,
                per_row);
    LSE_EXPECT_EQ(d, 0u);
  }
  for (std::size_t i = 0; i < per_row; ++i) {
    LSE_EXPECT_EQ(batched[2 * per_row + i], 0.0f);
  }
  // The two rows are not each other, so the comparison above had something to
  // catch.
  std::size_t same = 0;
  for (std::size_t i = 0; i < per_row; ++i) {
    if (batched[i] == batched[per_row + i]) ++same;
  }
  LSE_EXPECT(same < per_row);
}

LSE_TEST(verify_a_multi_token_write_covers_each_rows_own_span) {
  // The write side at T > 1: a row's span of positions crosses a block boundary
  // at its own offset, not at the batch's. Row 0 writes 14,15,16 — two blocks —
  // and row 1 writes 5,6,7 inside one.
  if (graph::default_scheduler() == nullptr) return;
  const std::int32_t bs = kv::kBlockSize;
  constexpr std::int64_t kKvh = 2;
  constexpr std::int64_t kW = 4;
  constexpr std::int64_t kT = 3;
  const std::int32_t pool_blocks = 6;
  const std::int32_t stride = 4;

  std::vector<float> zero(
      static_cast<std::size_t>(pool_blocks * kKvh * bs * kW), 0.0f);
  graph::Array pool = filled(Shape{pool_blocks, kKvh, bs, kW}, zero);
  std::vector<float> src(static_cast<std::size_t>(3 * kKvh * kT * kW));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = noise(i + 909) + 5.0f;
  graph::Array sa = filled(Shape{3, kKvh, kT, kW}, src);

  std::vector<float> table(static_cast<std::size_t>(3 * stride), 0.0f);
  table[0] = 1.0f;  // row 0: positions 14,15 -> block 1
  table[1] = 4.0f;  //         position  16   -> block 4
  table[static_cast<std::size_t>(stride)] = 2.0f;      // row 1: 5,6,7 -> block 2
  table[static_cast<std::size_t>(2 * stride)] = 3.0f;  // row 2 holds nothing
  graph::Array ta = filled(Shape{3, stride}, table);
  graph::Array meta = step_meta({{14, 17}, {5, 8}, {0, 0}});

  graph::Array written = graph::kv_page_write(pool, sa, meta, ta, bs);
  LSE_EXPECT_OK(written.eval());
  const std::vector<float> got = read_all(written);
  LSE_EXPECT_EQ(got.size(), zero.size());
  if (got.size() != zero.size()) return;

  const auto at = [&](std::int32_t blk, std::int64_t h, std::int32_t slot,
                      std::int64_t w) {
    return static_cast<std::size_t>(((blk * kKvh + h) * bs + slot) * kW + w);
  };
  const auto from = [&](std::int64_t r, std::int64_t h, std::int64_t t,
                        std::int64_t w) {
    return src[static_cast<std::size_t>(((r * kKvh + h) * kT + t) * kW + w)];
  };
  for (std::int64_t h = 0; h < kKvh; ++h) {
    for (std::int64_t w = 0; w < kW; ++w) {
      LSE_EXPECT_EQ(got[at(1, h, 14, w)], from(0, h, 0, w));
      LSE_EXPECT_EQ(got[at(1, h, 15, w)], from(0, h, 1, w));
      LSE_EXPECT_EQ(got[at(4, h, 0, w)], from(0, h, 2, w));
      for (std::int32_t t = 0; t < 3; ++t) {
        LSE_EXPECT_EQ(got[at(2, h, 5 + t, w)], from(1, h, t, w));
      }
    }
  }
  // Block 3 is the row that holds no sequence; block 0 is the table's pad and
  // the row 0 slot the span never reaches. Neither was written.
  for (std::int32_t blk : {0, 3, 5}) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(kKvh * bs * kW); ++i) {
      LSE_EXPECT_EQ(got[static_cast<std::size_t>(blk * kKvh * bs * kW) + i],
                    0.0f);
    }
  }
}

LSE_TEST(verify_rope_rotates_each_row_to_its_own_position) {
  // RoPE reads the same descriptor. With T > 1 a row's angles run from its own
  // origin across its span; one shared origin rotates every row but the
  // furthest-along one to somebody else's position.
  if (graph::default_scheduler() == nullptr) return;
  constexpr std::int64_t kH = 2;
  constexpr std::int64_t kT = 4;
  constexpr std::int64_t kD = 8;
  constexpr std::int64_t kMaxT = 64;

  auto tables = ops::build_rope(static_cast<std::int32_t>(kD), kMaxT, 10000.0f);
  LSE_EXPECT_OK(tables.status());
  if (!tables.ok()) return;

  const auto per_row = static_cast<std::size_t>(kH * kT * kD);
  std::vector<float> x(per_row * 2);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = noise(i + 4004);

  graph::Array xa = filled(Shape{2, kH, kT, kD}, x);
  graph::Array off = step_meta({{0, 4}, {23, 27}});
  graph::Array both = graph::rope(xa, tables->cos, tables->sin, off);
  LSE_EXPECT_OK(both.eval());
  const std::vector<float> got = read_all(both);
  LSE_EXPECT_EQ(got.size(), per_row * 2);
  if (got.size() != per_row * 2) return;

  for (std::int32_t r = 0; r < 2; ++r) {
    const std::int32_t origin = r == 0 ? 0 : 23;
    std::vector<float> one(
        x.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * per_row),
        x.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(r) + 1) * per_row));
    graph::Array x1 = filled(Shape{1, kH, kT, kD}, one);
    // The single-sequence form: a baked offset, which is what one sequence has.
    graph::Array o1 = graph::rope(x1, tables->cos, tables->sin, origin);
    LSE_EXPECT_OK(o1.eval());
    const std::vector<float> alone = read_all(o1);
    LSE_EXPECT_EQ(alone.size(), per_row);
    if (alone.size() != per_row) return;
    const std::size_t d =
        bitdiff(alone, 0, got, static_cast<std::size_t>(r) * per_row, per_row);
    std::printf("       rope row %d at position %d: %zu of %zu differ\n", r,
                origin, d, per_row);
    LSE_EXPECT_EQ(d, 0u);
  }
}


// --- multi-token prediction --------------------------------------------------
//
// The module ships beside a checkpoint and is not one: no embedding table, no
// head, one layer. These run against a synthetic Qwen3.5 pair, so they cover
// the wiring — load, rollback, accept, and token identity — on a bare machine.
// Acceptance rate is a property of trained weights and is not testable here.

namespace {

struct NamedTensor {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::size_t tensor_elems(const NamedTensor& t) {
  std::size_t n = 1;
  for (std::int64_t d : t.dims) n *= static_cast<std::size_t>(d);
  return n;
}

std::string safetensors_header(const std::vector<NamedTensor>& tensors,
                               std::size_t* total) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedTensor& t : tensors) {
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
    }
    const std::size_t bytes = tensor_elems(t) * 4;
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  *total = offset;
  return header;
}

// `value(name, index)` fills each tensor, so a fixture can hand one tensor a
// structure and let the rest take filler.
template <typename Fill>
void write_shaped(const std::filesystem::path& path,
                  const std::vector<NamedTensor>& tensors, const Fill& value) {
  std::size_t total = 0;
  const std::string header = safetensors_header(tensors, &total);
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<float> data;
  for (const NamedTensor& t : tensors) {
    data.resize(tensor_elems(t));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = value(t.name, i);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * 4));
  }
}

// Deterministic and not symmetric: an all-zero module drafts token 0 every
// step, which would make a rejection test pass for the wrong reason.
float filler(std::size_t i) {
  return 0.03f * static_cast<float>(static_cast<int>((i * 7) % 23) - 11);
}

// `gdn_head_dim` is a knob because the device Gated DeltaNet kernel only emits
// at 16, 32, 64 and 128: any other width puts the scan on the host, which is
// the one shape of pass a speculative rollback cannot replace.
model::Config mtp_test_config(std::int32_t gdn_head_dim = 16) {
  model::Config c;
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.num_layers = 4;
  c.full_attention_interval = 4;
  c.global_attention_layers.clear();
  c.sliding_window = 0;
  c.attn_q_heads = 2;
  c.attn_kv_heads = 1;
  c.attn_head_dim = 16;
  c.rope_dim = 4;
  c.rope_theta = 10000000.0f;
  c.gdn_qk_heads = 2;
  c.gdn_v_heads = 2;
  c.gdn_head_dim = gdn_head_dim;
  c.gdn_conv_kernel = 4;
  c.mlp_intermediate = 64;
  c.num_experts = 0;
  c.num_active_experts = 0;
  c.num_shared_experts = 0;
  c.expert_intermediate = 0;
  c.tie_word_embeddings = true;
  c.dtype = "float32";
  c.kv_length = 64;
  c.mtp_layers = 1;
  return c;
}

std::vector<NamedTensor> qwen_dense_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  const std::int64_t kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim;
  const std::int64_t conv_dim = 2 * kh * ghd + vh * ghd;

  std::vector<NamedTensor> t;
  t.push_back({"language_model.model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"language_model.model.norm.weight", {h}});
  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p =
        "language_model.model.layers." + std::to_string(i) + ".";
    t.push_back({p + "input_layernorm.weight", {h}});
    t.push_back({p + "post_attention_layernorm.weight", {h}});
    if (c.is_attention_layer(i)) {
      const std::string a = p + "self_attn.";
      t.push_back({a + "q_proj.weight", {2 * qh * ahd, h}});
      t.push_back({a + "k_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "v_proj.weight", {kvh * ahd, h}});
      t.push_back({a + "o_proj.weight", {h, qh * ahd}});
      t.push_back({a + "q_norm.weight", {ahd}});
      t.push_back({a + "k_norm.weight", {ahd}});
    } else {
      const std::string g = p + "linear_attn.";
      t.push_back({g + "in_proj_qkv.weight", {conv_dim, h}});
      t.push_back({g + "in_proj_z.weight", {vh * ghd, h}});
      t.push_back({g + "in_proj_a.weight", {vh, h}});
      t.push_back({g + "in_proj_b.weight", {vh, h}});
      t.push_back({g + "conv1d.weight", {conv_dim, c.gdn_conv_kernel, 1}});
      t.push_back({g + "A_log", {vh}});
      t.push_back({g + "dt_bias", {vh}});
      t.push_back({g + "norm.weight", {ghd}});
      t.push_back({g + "out_proj.weight", {h, vh * ghd}});
    }
    const std::string m = p + "mlp.";
    t.push_back({m + "gate_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "up_proj.weight", {c.mlp_intermediate, h}});
    t.push_back({m + "down_proj.weight", {h, c.mlp_intermediate}});
  }
  return t;
}

// The module's own file, named exactly as mlx-community/Qwen3.8-27B-MTP-4bit
// names it: no `language_model.model` prefix and no layer index above zero.
std::vector<NamedTensor> mtp_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  return {
      {"fc.weight", {h, 2 * h}},
      {"pre_fc_norm_hidden.weight", {h}},
      {"pre_fc_norm_embedding.weight", {h}},
      {"norm.weight", {h}},
      {"layers.0.input_layernorm.weight", {h}},
      {"layers.0.post_attention_layernorm.weight", {h}},
      {"layers.0.self_attn.q_proj.weight", {2 * qh * ahd, h}},
      {"layers.0.self_attn.k_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.v_proj.weight", {kvh * ahd, h}},
      {"layers.0.self_attn.o_proj.weight", {h, qh * ahd}},
      {"layers.0.self_attn.q_norm.weight", {ahd}},
      {"layers.0.self_attn.k_norm.weight", {ahd}},
      {"layers.0.mlp.gate_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.up_proj.weight", {c.mlp_intermediate, h}},
      {"layers.0.mlp.down_proj.weight", {h, c.mlp_intermediate}},
  };
}

std::string mtp_config_json(const model::Config& c) {
  return std::string("{\"tie_word_embeddings\": true, \"text_config\": {") +
         "\"vocab_size\": " + std::to_string(c.vocab_size) +
         ", \"hidden_size\": " + std::to_string(c.hidden_size) +
         ", \"num_hidden_layers\": " + std::to_string(c.num_layers) +
         ", \"rms_norm_eps\": 1e-06" +
         ", \"full_attention_interval\": " +
         std::to_string(c.full_attention_interval) +
         ", \"num_attention_heads\": " + std::to_string(c.attn_q_heads) +
         ", \"num_key_value_heads\": " + std::to_string(c.attn_kv_heads) +
         ", \"head_dim\": " + std::to_string(c.attn_head_dim) +
         ", \"linear_num_key_heads\": " + std::to_string(c.gdn_qk_heads) +
         ", \"linear_num_value_heads\": " + std::to_string(c.gdn_v_heads) +
         ", \"linear_conv_kernel_dim\": " +
         std::to_string(c.gdn_conv_kernel) +
         ", \"linear_key_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"linear_value_head_dim\": " + std::to_string(c.gdn_head_dim) +
         ", \"intermediate_size\": " + std::to_string(c.mlp_intermediate) +
         ", \"rope_parameters\": {\"rope_theta\": 10000000.0,"
         " \"partial_rotary_factor\": 0.25}" +
         ", \"max_position_embeddings\": 128, \"dtype\": \"float32\"" +
         ", \"mtp_num_hidden_layers\": 1" +
         ", \"mtp_use_dedicated_embeddings\": false}}";
}

struct MtpFixture {
  model::Config config;
  model::SafeTensors weights;
  std::unique_ptr<model::HybridLM> lm;
  std::unique_ptr<model::MtpModule> mtp;
  std::string module_dir;
  bool ok = false;
};

// `passthrough` writes a module that drafts the decoder's own next token
// rather than the one after it: fc keeps the hidden half and drops the
// embedding half, and the layer's projections are zero so the block is exactly
// its residual. On a sequence that has reached a fixed point those two are the
// same token, which is how the accept path gets exercised without trained
// weights.
MtpFixture build_mtp_fixture(bool passthrough = false,
                             std::int32_t gdn_head_dim = 16) {
  MtpFixture fx;
  fx.config = mtp_test_config(gdn_head_dim);
  const std::int64_t h = fx.config.hidden_size;
  std::error_code ec;
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      (passthrough ? "lse-mtp-fixture-pt"
                   : "lse-mtp-fixture-" + std::to_string(gdn_head_dim));
  std::filesystem::create_directories(base / "mtp", ec);
  fx.module_dir = (base / "mtp").string();

  // An untrained head puts the top two logits within a few ULP of each other
  // and the greedy choice then follows the last bit of an f32 sum, which two
  // differently shaped kernels are not required to agree on. Spacing the tied
  // embedding's rows makes most of the fixture's greedy steps a property of
  // the model rather than of the rounding.
  const auto parent = [&](const std::string& name, std::size_t i) {
    const std::size_t width = static_cast<std::size_t>(h);
    float v = filler(i);
    if (name == "language_model.model.embed_tokens.weight" &&
        i % width == (i / width) % width) {
      v += 2.0f;
    }
    return v;
  };
  const auto module = [&](const std::string& name, std::size_t i) {
    if (!passthrough) return filler(i);
    if (name == "fc.weight") {
      const std::size_t width = 2 * static_cast<std::size_t>(h);
      const std::size_t row = i / width;
      const std::size_t col = i % width;
      return col == static_cast<std::size_t>(h) + row ? 1.0f : 0.0f;
    }
    if (name.find(".weight") != std::string::npos &&
        name.find("norm") != std::string::npos) {
      return 1.0f;
    }
    return 0.0f;
  };

  write_shaped(base / "model.safetensors", qwen_dense_tensors(fx.config),
               parent);
  write_shaped(base / "mtp" / "model.safetensors", mtp_tensors(fx.config),
               module);
  {
    std::ofstream out(base / "mtp" / "config.json");
    out << mtp_config_json(fx.config);
  }

  auto st = model::SafeTensors::open((base / "model.safetensors").string());
  if (!st.ok()) return fx;
  fx.weights = st.release();
  auto built = model::build_model(fx.config, fx.weights);
  if (!built.ok()) return fx;
  fx.lm = built.release();
  model::WeightBinder binder(fx.weights);
  if (!fx.lm->load(binder).ok()) return fx;

  auto mod = model::MtpModule::open(fx.module_dir, fx.config, *fx.lm);
  if (!mod.ok()) {
    std::printf("       MTP open failed: %s\n", mod.status().to_string().c_str());
    return fx;
  }
  fx.mtp = mod.release();
  fx.ok = true;
  return fx;
}

std::vector<float> array_to_host(const graph::Array& a) {
  std::vector<float> v;
  if (!a.valid()) return v;
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return v;
  // The pass left it on the device and nothing asked for it on the host, so
  // the mirror is stale until this moves it.
  if (!graph::interpreter::sync_from_device(*a.node(), sched->backend()).ok()) {
    return v;
  }
  v.resize(a.shape().elem_count());
  if (!graph::interpreter::read_raw(*a.node(), v.data(),
                                    v.size() * sizeof(float))
           .ok()) {
    v.clear();
  }
  return v;
}

// Every buffer a pass can leave behind: each Gated DeltaNet layer's recurrent
// state and conv tail, and each attention layer's paged key/value pools.
std::vector<std::vector<float>> states_image(
    std::vector<model::MixerState>& states) {
  std::vector<std::vector<float>> out;
  for (model::MixerState& st : states) {
    out.push_back(array_to_host(st.gdn_state));
    out.push_back(array_to_host(st.gdn_conv_qkv));
    out.push_back(array_to_host(st.key_cache));
    out.push_back(array_to_host(st.value_cache));
  }
  return out;
}

std::size_t images_differ(const std::vector<std::vector<float>>& a,
                          const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) {
      ++n;
      continue;
    }
    for (std::size_t j = 0; j < a[i].size(); ++j) {
      if (a[i][j] != b[i][j]) {
        ++n;
        break;
      }
    }
  }
  return n;
}

graph::Array ids_array(const std::vector<std::uint32_t>& ids) {
  graph::Array a = graph::Array::zeros(
      Shape{1, static_cast<std::int64_t>(ids.size())}, DType::kF32);
  if (!a.eval().ok()) return {};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    graph::interpreter::store_element(*a.node(), i, static_cast<float>(ids[i]));
  }
  return a;
}

}  // namespace

LSE_TEST(the_mtp_module_loads_beside_a_checkpoint_and_is_not_one_itself) {
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  LSE_EXPECT_EQ(fx.mtp->position(), 0);
  LSE_EXPECT_EQ(fx.mtp->config().num_layers, 1);
  // ...and its one layer attends, which is what its self_attn tensors describe.
  // A layer index that answered otherwise would build a Gated DeltaNet against
  // them and fail on a tensor name rather than on the shape that is wrong.
  LSE_EXPECT(fx.mtp->config().is_attention_layer(0));

  // The registry is right to refuse the module standalone, and that refusal is
  // why it needs its own load path rather than an architecture entry.
  auto st = model::SafeTensors::open(fx.module_dir + "/model.safetensors");
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  auto arch = model::detect_architecture(fx.config, *st);
  LSE_EXPECT(!arch.ok());
}

LSE_TEST(a_rejected_draft_leaves_the_caches_where_a_clean_pass_would) {
  // The paged KV rolls back by cursor — the redo overwrites the same slots —
  // but the Gated DeltaNet state does not: it is a value the pass replaces.
  // What makes that safe is that a `replaces_previous` pass starts from the
  // same carried input the discarded one did, so the two arms below must agree
  // bit for bit. Both end on the same two-row pass, so nothing here depends on
  // how a two-row kernel rounds against a one-row one.
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  const std::vector<std::uint32_t> prompt{2, 11, 33};
  const auto at = static_cast<std::int32_t>(prompt.size());

  const auto run = [&](bool reject_first, std::vector<float>* hidden,
                       std::vector<std::vector<float>>* image) {
    std::vector<model::MixerState> st = fx.lm->make_states();
    auto pre = fx.lm->hidden(ids_array(prompt), &st, nullptr);
    if (!pre.ok()) return false;
    if (reject_first) {
      // The rejected pass: a second token the decoder did not choose.
      auto bad = fx.lm->hidden(ids_array({19, 7}), &st, nullptr);
      if (!bad.ok()) return false;
      fx.lm->rewind(st, at);
    }
    auto got = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr, nullptr,
                             reject_first);
    if (!got.ok()) return false;
    *hidden = array_to_host(*got);
    *image = states_image(st);
    return true;
  };

  std::vector<float> clean_hidden, redo_hidden;
  std::vector<std::vector<float>> clean_image, redo_image;
  LSE_EXPECT(run(false, &clean_hidden, &clean_image));
  LSE_EXPECT(run(true, &redo_hidden, &redo_image));
  if (clean_hidden.empty() || redo_hidden.empty()) return;

  LSE_EXPECT_EQ(redo_hidden.size(), clean_hidden.size());
  std::size_t hidden_diff = 0;
  for (std::size_t i = 0; i < clean_hidden.size() && i < redo_hidden.size(); ++i) {
    if (clean_hidden[i] != redo_hidden[i]) ++hidden_diff;
  }
  LSE_EXPECT_EQ(hidden_diff, 0u);
  LSE_EXPECT_EQ(images_differ(clean_image, redo_image), 0u);
}

LSE_TEST(a_rebuild_cannot_pretend_to_replace_the_pass_before_it) {
  // The rollback is a replay of a held program. A rebuild would record the
  // graph from the state the discarded pass produced, which is the one thing
  // it must not start from, so it is refused rather than silently continued.
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  std::vector<model::MixerState> st = fx.lm->make_states();
  auto pre = fx.lm->hidden(ids_array({2, 11, 33}), &st, nullptr);
  LSE_EXPECT(pre.ok());
  if (!pre.ok()) return;
  // A width the cache has never held, so no program can be replayed for it.
  auto refused = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr,
                               nullptr, /*replaces_previous=*/true);
  LSE_EXPECT(!refused.ok());
}

LSE_TEST(a_pass_that_ran_on_the_host_cannot_be_replaced) {
  // The rollback rests on the decoder's carried state still being where the
  // discarded pass found it, and that is a property of the device path: a
  // group the host ran leaves it somewhere else. A head dim the Gated DeltaNet
  // kernel does not emit at is the cheapest way to produce one, and the point
  // is that the answer is a refusal rather than a continuation from the state
  // the pass was meant to discard.
  MtpFixture fx = build_mtp_fixture(/*passthrough=*/false, /*gdn_head_dim=*/8);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  std::vector<model::MixerState> st = fx.lm->make_states();
  LSE_EXPECT(fx.lm->hidden(ids_array({2, 11, 33}), &st, nullptr).ok());
  LSE_EXPECT(fx.lm->hidden(ids_array({19, 18}), &st, nullptr).ok());
  fx.lm->rewind(st, 3);
  auto refused = fx.lm->hidden(ids_array({19, 18}), &st, nullptr, nullptr,
                               nullptr, /*replaces_previous=*/true);
  LSE_EXPECT(!refused.ok());
  if (refused.ok()) return;
  LSE_EXPECT(refused.status().message().find("host") != std::string::npos);
}

LSE_TEST(speculating_gives_the_tokens_a_plain_decode_gives) {
  // The exhaustive form of this is a diff of a full continuation on a trained
  // checkpoint. What a synthetic fixture can say is narrower, because its
  // untrained logits decide some greedy steps by the last bit of an f32 sum
  // and the one-row and two-row kernels round differently there. So each
  // prompt is first asked whether the plain path itself is stable — the same
  // prompt prefilled in one piece and in two — and only a prompt the decoder
  // answers the same way twice is one whose tokens the speculative path is
  // required to reproduce.
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;

  GenerationLimits limits;
  // Odd, so the last speculative step lands on its second half and the session
  // ends holding exactly the text it emitted.
  limits.max_tokens = 7;
  GenerationLimits prefill_only;
  prefill_only.max_tokens = 0;

  const std::vector<std::vector<std::uint32_t>> prompts{
      {2, 11, 33},    {1, 5, 9, 17},  {7, 7, 7},        {40, 3},
      {60, 1, 2, 3, 4}, {12, 44, 5, 6}, {33, 2, 19}, {8, 8, 9, 10, 11}};

  std::size_t compared = 0;
  std::uint32_t rejections = 0;
  for (const std::vector<std::uint32_t>& prompt : prompts) {
    Generator plain(*fx.lm, greedy_params());
    Session ps("plain", fx.lm->num_layers());
    auto want = plain.generate(ps, prompt, limits);
    LSE_EXPECT(want.ok());
    if (!want.ok()) return;

    Generator split(*fx.lm, greedy_params());
    Session ss("split", fx.lm->num_layers());
    const std::vector<std::uint32_t> head(prompt.begin(), prompt.end() - 1);
    LSE_EXPECT(split.generate(ss, head, prefill_only).ok());
    auto again = split.generate(ss, prompt, limits);
    LSE_EXPECT(again.ok());
    if (!again.ok()) return;
    if (*again != *want) continue;  // the fixture, not the speculation

    Generator spec(*fx.lm, greedy_params());
    spec.use_mtp(*fx.mtp);
    Session sp("spec", fx.lm->num_layers());
    auto got = spec.generate(sp, prompt, limits);
    LSE_EXPECT(got.ok());
    if (!got.ok()) return;
    LSE_EXPECT(spec.stats().spec_steps > 0);
    rejections += spec.stats().spec_steps - spec.stats().spec_accepted;
    ++compared;
    LSE_EXPECT(*got == *want);
    if (*got != *want) {
      std::printf("       [%s] plain %s  spec %s\n",
                  ids_to_string(prompt).c_str(), ids_to_string(*want).c_str(),
                  ids_to_string(*got).c_str());
    }
  }
  LSE_EXPECT(compared >= prompts.size() - 1);
  // A run in which nothing was ever rejected would not have exercised the redo.
  LSE_EXPECT(rejections > 0);
}

LSE_TEST(an_accepted_draft_still_gives_the_decoders_own_tokens) {
  // The other half of the loop: a proposal the decoder agrees with is taken
  // without a second pass, and the token that rides along with it is the one
  // the decoder's own second row produced.
  MtpFixture fx = build_mtp_fixture(/*passthrough=*/true);
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;

  GenerationLimits limits;
  limits.max_tokens = 7;
  const std::vector<std::uint32_t> prompt{2, 11, 33};

  Generator plain(*fx.lm, greedy_params());
  Session ps("plain", fx.lm->num_layers());
  auto want = plain.generate(ps, prompt, limits);
  LSE_EXPECT(want.ok());
  if (!want.ok()) return;

  Generator spec(*fx.lm, greedy_params());
  spec.use_mtp(*fx.mtp);
  Session sp("spec", fx.lm->num_layers());
  auto got = spec.generate(sp, prompt, limits);
  LSE_EXPECT(got.ok());
  if (!got.ok()) return;

  LSE_EXPECT(spec.stats().spec_accepted > 0);
  LSE_EXPECT(*got == *want);
  if (*got != *want) {
    std::printf("       accepted %u/%u  plain %s  spec %s\n",
                spec.stats().spec_accepted, spec.stats().spec_steps,
                ids_to_string(*want).c_str(), ids_to_string(*got).c_str());
  }
}
