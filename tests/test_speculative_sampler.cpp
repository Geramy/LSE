#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "harness.hpp"
#include "lse/runtime/sampler.hpp"

using namespace lse;
using namespace lse::runtime;

LSE_TEST(target_distribution_preserves_filtered_seeded_sampling) {
  SamplingParams p;
  p.temperature = 0.85f; p.top_k = 20; p.top_p = 0.95f; p.seed = 12345;
  Sampler s(p), untouched(p);
  std::vector<float> logits(32);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = 0.1f * static_cast<float>((i * 13) % 32) - 2.0f;
  constexpr std::uint32_t expected[] = {
      27, 7, 22, 19, 27, 17, 9, 14, 16, 29, 7, 26, 11, 9, 27, 24,
      14, 26, 27, 4, 27, 29, 4, 14, 17, 12, 27, 19, 22, 21, 17, 29};
  for (const auto id : expected) {
    auto copy = logits;
    auto d = s.distribution(copy, {}); LSE_EXPECT_OK(d.status());
    LSE_EXPECT_EQ(s.sample(logits, {}), id);
    LSE_EXPECT_EQ(untouched.sample(logits, {}), id);
    if (d.ok()) {
      double sum = 0; for (double v : d->probabilities) sum += v;
      LSE_EXPECT_NEAR(sum, 1.0, 1e-12);
      LSE_EXPECT(d->probability(id) > 0);
    }
  }
}

LSE_TEST(target_distribution_preserves_unfiltered_rng_endpoints_and_reseed) {
  SamplingParams p; p.temperature = 1;
  for (const auto seed : {14258097010372255221ull, 2295574122455614247ull}) {
    p.seed = seed; Sampler s(p);
    std::vector<float> values{-1000, 0, -1000};
    auto d = s.distribution(values, {}); LSE_EXPECT_OK(d.status());
    LSE_EXPECT_EQ(s.sample(values, {}), 1u);
    s.reseed(seed); LSE_EXPECT_EQ(s.sample(values, {}), 1u);
    if (d.ok()) {
      LSE_EXPECT_EQ(sample_distribution(*d, 0.0).release(), 1u);
      LSE_EXPECT_EQ(sample_distribution(*d, std::nextafter(1.0, 0.0)).release(), 1u);
    }
  }
}

LSE_TEST(target_distribution_matches_duplicate_penalties_and_filter_order) {
  SamplingParams p; p.temperature = 1; p.repetition_penalty = 2;
  p.repetition_window = 3; p.top_k = 3; p.top_p = 0.8f;
  Sampler s(p);
  std::vector<float> logits{4, 1, -1, -2};
  const std::vector<std::uint32_t> history{3, 0, 0, 2};
  auto d = s.distribution(logits, history); LSE_EXPECT_OK(d.status());
  LSE_EXPECT_EQ(logits[0], 1.0f); LSE_EXPECT_EQ(logits[2], -2.0f);
  if (d.ok()) {
    LSE_EXPECT_EQ(d->ids.size(), 2u);
    LSE_EXPECT_NEAR(d->probability(0), 0.5, 1e-12);
    LSE_EXPECT_NEAR(d->probability(1), 0.5, 1e-12);
    LSE_EXPECT_EQ(d->probability(2), 0.0);
  }
  p.temperature = 0; Sampler greedy(p);
  logits = {4, 1.1f, -1, -2};
  auto g = greedy.distribution(logits, history); LSE_EXPECT_OK(g.status());
  if (g.ok()) { LSE_EXPECT_EQ(g->ids[0], 1u); LSE_EXPECT_EQ(g->probabilities[0], 1.0); }
}

LSE_TEST(target_distribution_uses_original_fp32_softmax_weights) {
  SamplingParams p; p.temperature = 0.73f;
  Sampler s(p);
  std::vector<float> logits{-3.1f, 0.27f, 1.2f, -0.8f};
  auto d = s.distribution(logits, {}); LSE_EXPECT_OK(d.status());
  const float inverse = 1.0f / p.temperature;
  std::array<float, 4> weights{}; double sum = 0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = std::exp((logits[i] - logits[2]) * inverse); sum += static_cast<double>(weights[i]);
  }
  if (d.ok()) for (std::size_t i = 0; i < weights.size(); ++i) {
    LSE_EXPECT_EQ(d->ids[i], i);
    LSE_EXPECT_EQ(d->probabilities[i], static_cast<double>(weights[i]) / sum);
  }
}

LSE_TEST(sparse_rejection_exact_enumeration_recovers_target_distribution) {
  const DiscreteDistribution p{{0, 1, 2}, {0.25, 0.25, 0.5}};
  const DiscreteDistribution q{{0, 1, 2}, {0.5, 0.25, 0.25}};
  std::array<int, 3> counts{}; int accepted = 0;
  for (int draft = 0; draft < 4; ++draft)
    for (int accept = 0; accept < 4; ++accept)
      for (int residual = 0; residual < 4; ++residual) {
        auto token = sample_distribution(q, (draft + 0.5) / 4.0);
        LSE_EXPECT_OK(token.status());
        auto checked = verify_proposal(p, q, *token,
            (accept + 0.5) / 4.0, (residual + 0.5) / 4.0);
        LSE_EXPECT_OK(checked.status());
        if (checked.ok()) { ++counts[checked->token]; accepted += checked->accepted; }
      }
  LSE_EXPECT_EQ(counts[0], 16); LSE_EXPECT_EQ(counts[1], 16);
  LSE_EXPECT_EQ(counts[2], 32); LSE_EXPECT_EQ(accepted, 48);
}

LSE_TEST(sparse_rejection_handles_identical_disjoint_and_filtered_support) {
  const DiscreteDistribution p{{1, 2}, {0.4, 0.6}};
  for (const auto token : {1u, 2u}) {
    auto r = verify_proposal(p, p, token, std::nextafter(1.0, 0.0), 0);
    LSE_EXPECT_OK(r.status()); LSE_EXPECT(r->accepted); LSE_EXPECT_EQ(r->token, token);
  }
  const DiscreteDistribution q{{7}, {1}};
  auto first = verify_proposal(p, q, 7, 0, 0);
  auto last = verify_proposal(p, q, 7, 0.5, std::nextafter(1.0, 0.0));
  LSE_EXPECT_OK(first.status()); LSE_EXPECT_OK(last.status());
  LSE_EXPECT(!first->accepted); LSE_EXPECT_EQ(first->token, 1u);
  LSE_EXPECT(!last->accepted); LSE_EXPECT_EQ(last->token, 2u);
  const DiscreteDistribution partial{{1, 7}, {0.5, 0.5}};
  auto filtered = verify_proposal(p, partial, 7, 0, 0.999);
  LSE_EXPECT_OK(filtered.status()); LSE_EXPECT(!filtered->accepted);
  LSE_EXPECT_EQ(filtered->token, 2u);
}

LSE_TEST(sparse_rejection_rejects_invalid_distributions_and_uniforms) {
  const DiscreteDistribution valid{{0, 1}, {0.5, 0.5}};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const auto& bad : std::vector<DiscreteDistribution>{
      {{}, {}}, {{0}, {}}, {{0, 0}, {0.5, 0.5}}, {{0}, {0}},
      {{0, 1}, {-0.1, 1.1}}, {{0}, {nan}}, {{0}, {inf}}, {{0}, {0.5}}}) {
    LSE_EXPECT(!sample_distribution(bad, 0.5).ok());
    LSE_EXPECT(!verify_proposal(valid, bad, 0, 0.5, 0.5).ok());
    LSE_EXPECT(!verify_proposal(bad, valid, 0, 0.5, 0.5).ok());
  }
  for (double u : {-0.1, 1.0, inf, nan}) {
    LSE_EXPECT(!sample_distribution(valid, u).ok());
    LSE_EXPECT(!verify_proposal(valid, valid, 0, u, 0.5).ok());
    LSE_EXPECT(!verify_proposal(valid, valid, 0, 0.5, u).ok());
  }
  LSE_EXPECT(!verify_proposal(valid, valid, 7, 0.5, 0.5).ok());
  const DiscreteDistribution zero{{0, 1}, {0, 1}};
  LSE_EXPECT(!verify_proposal(valid, zero, 0, 0.5, 0.5).ok());
}

LSE_TEST(target_distribution_rejects_invalid_logits_and_settings) {
  SamplingParams p; p.temperature = 1; Sampler s(p);
  const float inf = std::numeric_limits<float>::infinity();
  for (auto values : std::vector<std::vector<float>>{
      {}, {-inf, -inf}, {inf, 0}, {std::numeric_limits<float>::quiet_NaN(), 0}})
    LSE_EXPECT(!s.distribution(values, {}).ok());
  std::vector<float> masked{-inf, 0};
  LSE_EXPECT_OK(s.distribution(masked, {}).status());
  for (const float bad : {inf, std::numeric_limits<float>::quiet_NaN()}) {
    p.temperature = bad; Sampler invalid(p);
    LSE_EXPECT(!invalid.distribution(masked, {}).ok());
  }
  p.temperature = 0; p.repetition_penalty = std::numeric_limits<float>::min();
  Sampler overflow(p); std::vector<float> values{1, 0};
  const std::vector<std::uint32_t> repeats{0, 0};
  LSE_EXPECT(!overflow.distribution(values, repeats).ok());
}

LSE_TEST(speculative_rng_reseed_replays_and_proposals_do_not_move_target_stream) {
  const DiscreteDistribution p{{0, 1, 2}, {0.25, 0.25, 0.5}};
  const DiscreteDistribution q{{0, 1, 2}, {0.5, 0.25, 0.25}};
  SpeculativeSampler a(42), b(42);
  std::vector<ProposalVerification> results;
  for (int i = 0; i < 128; ++i) {
    auto token = a.sample_proposal(q); LSE_EXPECT_OK(token.status());
    auto checked = a.verify(p, q, *token); LSE_EXPECT_OK(checked.status());
    results.push_back(*checked);
  }
  a.reseed(42);
  for (const auto expected : results) {
    auto token = a.sample_proposal(q);
    auto checked = a.verify(p, q, *token);
    LSE_EXPECT_EQ(checked->token, expected.token);
    LSE_EXPECT_EQ(checked->accepted, expected.accepted);
  }
  a.reseed(99); b.reseed(99);
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 5; ++j) LSE_EXPECT_OK(a.sample_proposal(q).status());
    LSE_EXPECT_EQ(a.sample_target(p).release(), b.sample_target(p).release());
  }
}

LSE_TEST_MAIN()
