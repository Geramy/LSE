// Perplexity scoring: the sliding-window plan, the device log-softmax
// reduction (logits.lse_pick.v1), and Generator::score on a tiny synthetic
// Qwen3.5 model against a whole-sequence forward pass scored on the host.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/runtime/generator.hpp"
#include "lse/runtime/perplexity.hpp"
#include "lse/runtime/session.hpp"

using namespace lse;
using namespace lse::runtime;

namespace {

graph::Array filled(const Shape& shape, const std::vector<float>& values) {
  graph::Array a = graph::Array::zeros(shape, DType::kF32);
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) return a;
  graph::Node& n = *a.node();
  if (!graph::interpreter::ensure_output_buffer(n, sched->backend()).ok()) return a;
  for (std::size_t i = 0; i < values.size(); ++i) graph::interpreter::store_element(n, i, values[i]);
  n.materialized = true;
  (void)graph::interpreter::sync_to_device(n, sched->backend());
  return a;
}

std::vector<float> read_all(graph::Array a) {
  std::vector<float> out(a.shape().elem_count());
  if (!a.to_host(out.data(), out.size() * sizeof(float)).ok()) out.clear();
  return out;
}

// log p(target | row) in double, the reference every scoring path must meet.
double reference_nll(const float* row, std::size_t vocab, std::uint32_t target) {
  double top = row[0];
  for (std::size_t v = 1; v < vocab; ++v) top = std::max(top, static_cast<double>(row[v]));
  double sum = 0.0;
  for (std::size_t v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(row[v]) - top);
  return top + std::log(sum) - static_cast<double>(row[target]);
}

// --- a tiny Qwen3.5 checkpoint ------------------------------------------------

struct NamedTensor {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::size_t elems(const NamedTensor& t) {
  std::size_t n = 1;
  for (const std::int64_t d : t.dims) n *= static_cast<std::size_t>(d);
  return n;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<NamedTensor>& tensors) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedTensor& t : tensors) {
    std::string dims;
    for (const std::int64_t d : t.dims) dims += (dims.empty() ? "" : ",") + std::to_string(d);
    const std::size_t bytes = elems(t) * 4;
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims + "],\"data_offsets\":[" +
              std::to_string(offset) + "," + std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<float> data;
  std::size_t salt = 0;
  for (const NamedTensor& t : tensors) {
    data.resize(elems(t));
    const bool norm = t.name.find("norm") != std::string::npos;
    for (std::size_t i = 0; i < data.size(); ++i) {
      // Deterministic, asymmetric filler; norms near one so activations stay
      // in range through the stack.
      const float f = 0.05f * static_cast<float>(static_cast<int>(((i + salt) * 7) % 23) - 11);
      data[i] = norm ? 1.0f + 0.1f * f : f;
    }
    salt += 3;
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
  }
}

model::Config tiny_config() {
  model::Config c;
  c.vocab_size = 96;
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
  c.gdn_head_dim = 16;
  c.gdn_conv_kernel = 4;
  c.mlp_intermediate = 64;
  c.num_experts = 0;
  c.num_active_experts = 0;
  c.num_shared_experts = 0;
  c.expert_intermediate = 0;
  c.tie_word_embeddings = true;
  c.dtype = "float32";
  c.kv_length = 512;
  c.mtp_layers = 0;
  return c;
}

std::vector<NamedTensor> tiny_tensors(const model::Config& c) {
  const std::int64_t h = c.hidden_size, qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim, kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim, conv = 2 * kh * ghd + vh * ghd;
  std::vector<NamedTensor> t;
  t.push_back({"language_model.model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"language_model.model.norm.weight", {h}});
  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p = "language_model.model.layers." + std::to_string(i) + ".";
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
      t.push_back({g + "in_proj_qkv.weight", {conv, h}});
      t.push_back({g + "in_proj_z.weight", {vh * ghd, h}});
      t.push_back({g + "in_proj_a.weight", {vh, h}});
      t.push_back({g + "in_proj_b.weight", {vh, h}});
      t.push_back({g + "conv1d.weight", {conv, c.gdn_conv_kernel, 1}});
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

struct Tiny {
  model::Config config;
  model::SafeTensors weights;
  std::unique_ptr<model::HybridLM> lm;
  bool ok = false;
};

Tiny build_tiny() {
  Tiny t;
  t.config = tiny_config();
  const auto dir = std::filesystem::temp_directory_path() / "lse-perplexity-fixture";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  write_safetensors(dir / "model.safetensors", tiny_tensors(t.config));
  auto st = model::SafeTensors::open((dir / "model.safetensors").string());
  if (!st.ok()) return t;
  t.weights = st.release();
  auto built = model::build_model(t.config, t.weights, "");
  if (!built.ok()) {
    std::printf("       build failed: %s\n", built.status().to_string().c_str());
    return t;
  }
  t.lm = built.release();
  model::WeightBinder binder(t.weights);
  if (!t.lm->load(binder).ok()) return t;
  t.ok = true;
  return t;
}

std::vector<std::uint32_t> sequence(std::size_t n, std::uint32_t vocab) {
  std::vector<std::uint32_t> ids(n);
  for (std::size_t i = 0; i < n; ++i)
    ids[i] = static_cast<std::uint32_t>((i * 37 + (i * i) % 11 + 5) % vocab);
  return ids;
}

// Whole-sequence forward pass with no cache, every row through the LM head,
// log-softmax on the host: the definition Generator::score must reproduce.
std::vector<double> reference_scores(model::HybridLM& lm, const std::vector<std::uint32_t>& ids,
                                     std::size_t first_target) {
  std::vector<float> f(ids.begin(), ids.end());
  graph::Array tokens = filled(Shape{1, static_cast<std::int64_t>(ids.size())}, f);
  std::vector<double> out;
  auto h = lm.hidden(tokens, nullptr, nullptr);
  if (!h.ok()) return out;
  auto logits = lm.lm_head(*h);
  if (!logits.ok()) return out;
  const std::vector<float> all = read_all(*logits);
  const auto vocab = static_cast<std::size_t>(lm.config().vocab_size);
  if (all.size() != ids.size() * vocab) return out;
  for (std::size_t j = first_target - 1; j + 1 < ids.size(); ++j)
    out.push_back(reference_nll(all.data() + j * vocab, vocab, ids[j + 1]));
  return out;
}

}  // namespace

LSE_TEST(disjoint_windows_score_every_token_but_each_windows_first) {
  auto plan = perplexity_windows(10, 4, 4);
  LSE_EXPECT(plan.ok());
  if (!plan.ok()) return;
  const std::vector<PerplexityWindow> want{{0, 4, 1}, {4, 8, 5}, {8, 10, 9}};
  LSE_EXPECT(*plan == want);
}

LSE_TEST(overlapping_windows_score_each_token_once_with_context) {
  auto plan = perplexity_windows(10, 4, 2);
  LSE_EXPECT(plan.ok());
  if (!plan.ok()) return;
  // Window 2 starts at 2 and scores only tokens 4..5, which window 1 did not.
  const std::vector<PerplexityWindow> want{{0, 4, 1}, {2, 6, 4}, {4, 8, 6}, {6, 10, 8}};
  LSE_EXPECT(*plan == want);
  std::size_t scored = 0;
  for (const PerplexityWindow& w : *plan) scored += w.scored();
  LSE_EXPECT_EQ(scored, std::size_t{9});
}

LSE_TEST(a_one_token_tail_window_is_dropped_not_scored_empty) {
  auto plan = perplexity_windows(9, 4, 4);
  LSE_EXPECT(plan.ok());
  if (!plan.ok()) return;
  // [8, 9) has no token with context inside it.
  const std::vector<PerplexityWindow> want{{0, 4, 1}, {4, 8, 5}};
  LSE_EXPECT(*plan == want);
}

LSE_TEST(invalid_windows_are_refused) {
  LSE_EXPECT(!perplexity_windows(10, 1, 1).ok());
  LSE_EXPECT(!perplexity_windows(10, 4, 0).ok());
  LSE_EXPECT(!perplexity_windows(10, 4, 5).ok());
  LSE_EXPECT(!perplexity_windows(1, 4, 4).ok());
}

LSE_TEST(lse_pick_matches_a_double_precision_log_softmax) {
  // A vocabulary that is not a multiple of the workgroup, and rows whose
  // scale spans the range trained logits reach.
  constexpr std::size_t rows = 5, vocab = 1000;
  std::vector<float> logits(rows * vocab);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = 0.037f * static_cast<float>(static_cast<int>((i * 7919) % 2003) - 1001) *
                static_cast<float>(1 + i / vocab);
  const std::vector<float> targets{0.0f, 999.0f, 17.0f, 500.0f, 3.0f};
  graph::Array x = filled(Shape{1, rows, vocab}, logits);
  graph::Array t = filled(Shape{1, rows}, targets);
  auto pick = graph::custom("logits.lse_pick.v1", {x, t});
  LSE_EXPECT(pick.ok());
  if (!pick.ok()) return;
  LSE_EXPECT(pick->shape() == (Shape{1, rows, 3}));
  const std::vector<float> got = read_all(*pick);
  LSE_EXPECT_EQ(got.size(), rows * 3);
  if (got.size() != rows * 3) return;
  for (std::size_t r = 0; r < rows; ++r) {
    const double nll = got[3 * r] + std::log(static_cast<double>(got[3 * r + 1])) - got[3 * r + 2];
    const double want = reference_nll(logits.data() + r * vocab, vocab, static_cast<std::uint32_t>(targets[r]));
    std::printf("       row %zu: nll %.7f reference %.7f\n", r, nll, want);
    LSE_EXPECT_NEAR(nll, want, 1e-4 * std::max(1.0, std::fabs(want)));
  }
}

LSE_TEST(scoring_matches_a_whole_sequence_forward_pass) {
  Tiny tiny = build_tiny();
  LSE_EXPECT(tiny.ok);
  if (!tiny.ok) return;
  SamplingParams sampling;
  // 150 tokens at ubatch 128 run as a padded 32-row pass then a 128-row pass:
  // padding, a pass-width change and more than one head slice.
  PrefillBatch prefill;
  prefill.batch_size = 128;
  prefill.ubatch_size = 128;
  Generator gen(*tiny.lm, sampling, prefill);
  for (const std::size_t n : {std::size_t{40}, std::size_t{150}}) {
    const std::vector<std::uint32_t> ids = sequence(n, static_cast<std::uint32_t>(tiny.config.vocab_size));
    for (const std::size_t first : {std::size_t{1}, n / 2}) {
      Session session("", tiny.lm->state_slots());
      auto got = gen.score(session, ids, first);
      LSE_EXPECT(got.ok());
      if (!got.ok()) {
        std::printf("       score failed: %s\n", got.status().to_string().c_str());
        return;
      }
      LSE_EXPECT_OK(tiny.lm->drop_retained_passes());
      const std::vector<double> want = reference_scores(*tiny.lm, ids, first);
      LSE_EXPECT_EQ(got->size(), n - first);
      LSE_EXPECT_EQ(want.size(), n - first);
      if (got->size() != want.size()) return;
      // Relative, as the other prefill-versus-whole-pass checks are: a device
      // runs the paged prefill and the stateless pass on different kernels,
      // which agree to about 1e-3 of the value.
      double worst = 0.0;
      for (std::size_t i = 0; i < want.size(); ++i)
        worst = std::max(worst, std::fabs((*got)[i] - want[i]) / std::max(1.0, std::fabs(want[i])));
      std::printf("       n=%zu first=%zu: max relative |nll - reference| %.3e\n", n, first, worst);
      LSE_EXPECT(worst < 2e-3);
    }
  }
}

LSE_TEST(sliding_windows_add_up_to_the_per_window_scores) {
  Tiny tiny = build_tiny();
  LSE_EXPECT(tiny.ok);
  if (!tiny.ok) return;
  SamplingParams sampling;
  PrefillBatch prefill;
  prefill.batch_size = 128;
  prefill.ubatch_size = 128;
  Generator gen(*tiny.lm, sampling, prefill);
  const std::vector<std::uint32_t> ids = sequence(200, static_cast<std::uint32_t>(tiny.config.vocab_size));
  for (const std::size_t stride : {std::size_t{64}, std::size_t{32}}) {
    std::size_t calls = 0;
    auto report = score_perplexity(gen, *tiny.lm, ids, 64, stride, 0,
                                   [&](const WindowScore&, std::size_t done, std::size_t total) {
                                     ++calls;
                                     LSE_EXPECT(done <= total);
                                   });
    LSE_EXPECT(report.ok());
    if (!report.ok()) {
      std::printf("       report failed: %s\n", report.status().to_string().c_str());
      return;
    }
    LSE_EXPECT_EQ(calls, report->windows.size());
    std::size_t scored = 0;
    double sum = 0.0;
    for (const WindowScore& w : report->windows) {
      scored += w.window.scored();
      sum += w.nll_sum;
      // Each window on its own, as the reference scores it.
      const std::vector<std::uint32_t> part(ids.begin() + static_cast<std::ptrdiff_t>(w.window.begin),
                                            ids.begin() + static_cast<std::ptrdiff_t>(w.window.end));
      const std::vector<double> want =
          reference_scores(*tiny.lm, part, w.window.first_target - w.window.begin);
      double want_sum = 0.0;
      for (const double v : want) want_sum += v;
      LSE_EXPECT_NEAR(w.nll_sum, want_sum, 2e-3 * std::fabs(want_sum));
    }
    // Disjoint windows lose one token per window; overlapping ones lose only
    // the first token of the sequence.
    LSE_EXPECT_EQ(report->scored, scored);
    LSE_EXPECT_EQ(scored, stride == 64 ? std::size_t{200 - 4} : std::size_t{199});
    LSE_EXPECT_NEAR(report->nll_sum, sum, 1e-9);
    LSE_EXPECT(std::isfinite(report->perplexity()) && report->perplexity() > 1.0);
    std::printf("       stride %zu: %zu windows, %zu tokens, perplexity %.4f\n", stride,
                report->windows.size(), report->scored, report->perplexity());
  }
}

LSE_TEST(scoring_refuses_out_of_range_input) {
  Tiny tiny = build_tiny();
  LSE_EXPECT(tiny.ok);
  if (!tiny.ok) return;
  Generator gen(*tiny.lm, SamplingParams{});
  const std::vector<std::uint32_t> ok_ids{1, 2, 3, 4};
  {
    Session session("", tiny.lm->state_slots());
    LSE_EXPECT(!gen.score(session, ok_ids, 0).ok());
    LSE_EXPECT(!gen.score(session, ok_ids, 4).ok());
  }
  {
    Session session("", tiny.lm->state_slots());
    const std::vector<std::uint32_t> bad{1, 2, 96};
    LSE_EXPECT(!gen.score(session, bad, 1).ok());
  }
  {
    Session session("", tiny.lm->state_slots());
    const std::vector<std::uint32_t> long_ids(tiny.config.kv_capacity() + 1, 1u);
    LSE_EXPECT(!gen.score(session, long_ids, 1).ok());
  }
}

LSE_TEST_MAIN()
