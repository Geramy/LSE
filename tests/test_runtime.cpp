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
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/lemonseed.hpp"
#include "lse/model/weights.hpp"
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
    for (std::int32_t blk = 0; blk < nblk; ++blk) {
      for (std::int64_t h = 0; h < kKvh; ++h) {
        for (std::int32_t sl = 0; sl < bs; ++sl) {
          const std::int32_t j = blk * bs + sl;
          if (j >= live) continue;
          for (std::int64_t d = 0; d < kHd; ++d) {
            const auto dst = static_cast<std::size_t>(
                ((blk * kKvh + h) * bs + sl) * kHd + d);
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
        static_cast<float>(i);
    graph::Array ta = filled(Shape{1, stride}, table);
    graph::Array meta =
        filled(Shape{3}, {static_cast<float>(offset), static_cast<float>(live),
                          1.0f});

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
  graph::Array meta = filled(
      Shape{3}, {0.0f, static_cast<float>(live), 2.0f});

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
    graph::Array m1 =
        filled(Shape{3}, {0.0f, static_cast<float>(live), 1.0f});
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
  graph::Array meta =
      filled(Shape{3}, {static_cast<float>(pos), static_cast<float>(pos + t),
                        1.0f});

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
  auto both = lm->hidden(tokens, &pair.states(), nullptr, nullptr, /*rows=*/2);
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
