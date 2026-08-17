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
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
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
  std::int64_t seen_t = -1;
  for (std::uint32_t id : tokens) {
    stepped = logits_of({id}, &session.states());
    LSE_EXPECT(!stepped.empty());
    if (stepped.empty()) return;
    for (const auto& st : session.states()) {
      if (!st.key_cache.valid()) continue;
      LSE_EXPECT_EQ(st.key_cache.shape().dim(2), cap);
      LSE_EXPECT_EQ(st.value_cache.shape().dim(2), cap);
      if (seen_t < 0) seen_t = st.key_cache.shape().dim(2);
      LSE_EXPECT_EQ(st.key_cache.shape().dim(2), seen_t);
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
  const auto cap = cfg->kv_capacity();
  for (const auto& st : session.states()) {
    if (!st.key_cache.valid()) continue;
    LSE_EXPECT_EQ(st.key_cache.shape().dim(2), cap);
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
