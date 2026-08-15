// Sampler, session cache, and the decode loop.
//
// The load-bearing test here is cached_decode_matches_a_full_forward_pass: a
// KV cache is only correct if stepping one token at a time gives the same
// logits as running the whole sequence at once. Everything else in M7 rests on
// that.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
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

  // Also the decision, not just the numbers.
  LSE_EXPECT_EQ(argmax(stepped), argmax(whole));
  LSE_EXPECT(max_rel < 1e-5);
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

LSE_TEST_MAIN()
