// Loads the real lemonseed b1.5 checkpoint when it is present; the pure-config
// cases run everywhere.
#include "lse/model/config.hpp"

#include <cstdlib>
#include <filesystem>
#include <vector>

#include "harness.hpp"
#include "lse/model/weights.hpp"

using namespace lse;
using namespace lse::model;

namespace {

// Override with LSE_TEST_MODEL to point at another checkpoint.
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

LSE_TEST(b1_5_preset_matches_the_shipped_config) {
  const Config c = preset_b1_5();
  LSE_EXPECT_EQ(c.vocab_size, 248320);
  LSE_EXPECT_EQ(c.hidden_size, 1024);
  LSE_EXPECT_EQ(c.num_layers, 20);
  LSE_EXPECT_EQ(c.attn_q_heads, 16);
  LSE_EXPECT_EQ(c.attn_kv_heads, 2);
  LSE_EXPECT_EQ(c.gdn_chunk_size, 32);
  LSE_EXPECT_EQ(c.expert_intermediate, 2176);
}

LSE_TEST(layer_layout_follows_the_interval_rule) {
  const Config c = preset_b1_5();
  // interval 4 -> layers 3, 7, 11, 15, 19 are attention, the rest GDN.
  const int attn[] = {3, 7, 11, 15, 19};
  for (int i = 0; i < c.num_layers; ++i) {
    bool want = false;
    for (int a : attn) want = want || (a == i);
    LSE_EXPECT_EQ(c.is_attention_layer(i), want);
  }
  LSE_EXPECT(c.is_global_attention(19));
  LSE_EXPECT(!c.is_global_attention(2));
}

LSE_TEST(config_round_trips_through_json) {
  const Config a = preset_b1_5();
  auto b = Config::from_json_string(a.to_json());
  LSE_EXPECT(b.ok());
  LSE_EXPECT_EQ(b->vocab_size, a.vocab_size);
  LSE_EXPECT_EQ(b->num_layers, a.num_layers);
  LSE_EXPECT_EQ(b->global_attention_layers.size(), a.global_attention_layers.size());
  LSE_EXPECT_NEAR(b->rope_theta, a.rope_theta, 1e-3);
  LSE_EXPECT(b->dtype == a.dtype);
}

LSE_TEST(validate_rejects_incoherent_gqa) {
  Config c = preset_b1_5();
  c.attn_kv_heads = 5;  // 16 is not a multiple of 5
  LSE_EXPECT(!c.validate().ok());
}

LSE_TEST(validate_rejects_out_of_range_global_layers) {
  Config c = preset_b1_5();
  c.global_attention_layers = {99};
  auto s = c.validate();
  LSE_EXPECT(!s.ok());
  LSE_EXPECT(s.code() == StatusCode::kOutOfRange);
}

LSE_TEST(legacy_expert_names_migrate_to_stacked) {
  auto g = migrate_legacy_expert_name("blocks.3.moe.experts.5.w1.weight");
  LSE_EXPECT(g.ok());
  LSE_EXPECT(g->stacked_name == "blocks.3.moe.w_gate");
  LSE_EXPECT_EQ(g->expert_index, 5);

  auto u = migrate_legacy_expert_name("blocks.0.moe.experts.0.w3.weight");
  LSE_EXPECT(u.ok());
  LSE_EXPECT(u->stacked_name == "blocks.0.moe.w_up");

  auto d = migrate_legacy_expert_name("blocks.0.moe.experts.7.w2.weight");
  LSE_EXPECT(d.ok());
  LSE_EXPECT(d->stacked_name == "blocks.0.moe.w_down");

  // Already-current names are not legacy.
  LSE_EXPECT(!migrate_legacy_expert_name("blocks.0.moe.w_gate").ok());
}

LSE_TEST(hf_cache_root_honours_env) {
  LSE_EXPECT(hf_cache_root().find("huggingface") != std::string::npos ||
             hf_cache_root().find("hub") != std::string::npos);
}

LSE_TEST(resolve_reports_missing_models_clearly) {
  auto r = resolve_model("definitely/not-a-real-model");
  LSE_EXPECT(!r.ok());
  LSE_EXPECT(r.status().code() == StatusCode::kNotFound);
}

LSE_TEST(loads_the_real_b1_5_checkpoint) {
  if (!have_model()) {
    std::printf("       (skipped: no checkpoint at %s)\n", model_dir().c_str());
    return;
  }

  auto paths = resolve_model(model_dir());
  LSE_EXPECT(paths.ok());
  if (!paths.ok()) return;

  auto cfg = Config::from_json_file(paths->config);
  LSE_EXPECT(cfg.ok());
  if (!cfg.ok()) return;
  LSE_EXPECT_EQ(cfg->vocab_size, 248320);
  LSE_EXPECT_EQ(cfg->num_layers, 20);

  auto st = SafeTensors::open(paths->weights);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;

  LSE_EXPECT_EQ(st->tensors().size(), 517u);

  const double params = static_cast<double>(st->total_parameters()) / 1e9;
  LSE_EXPECT_NEAR(params, 1.5145, 0.001);

  // The stacked-MoE layout this checkpoint uses.
  const TensorView* w_gate = st->find("blocks.0.moe.w_gate");
  LSE_EXPECT(w_gate != nullptr);
  if (w_gate) {
    LSE_EXPECT_EQ(w_gate->shape.rank(), 3u);
    LSE_EXPECT_EQ(w_gate->shape[0], cfg->num_experts);
    LSE_EXPECT_EQ(w_gate->shape[1], cfg->expert_intermediate);
    LSE_EXPECT_EQ(w_gate->shape[2], cfg->hidden_size);
    LSE_EXPECT(w_gate->dtype == DType::kBF16);
  }

  const TensorView* embed = st->find("embed.weight");
  LSE_EXPECT(embed != nullptr);
  if (embed) {
    LSE_EXPECT_EQ(embed->shape[0], cfg->vocab_size);
    LSE_EXPECT_EQ(embed->shape[1], cfg->hidden_size);
  }

  // Every layer must have the mixer its index implies.
  for (int i = 0; i < cfg->num_layers; ++i) {
    const std::string prefix = "blocks." + std::to_string(i) + ".mixer.";
    const bool is_attn = cfg->is_attention_layer(i);
    const TensorView* gdn_marker = st->find(prefix + "A_log");
    LSE_EXPECT_EQ(gdn_marker == nullptr, is_attn);
  }

  // Widening a real bf16 tensor must produce finite values.
  const TensorView* norm = st->find("blocks.0.norm1.weight");
  LSE_EXPECT(norm != nullptr);
  if (norm) {
    std::vector<float> v(static_cast<std::size_t>(norm->element_count()));
    LSE_EXPECT_OK(norm->read_f32(v.data(), v.size()));
    bool all_finite = true;
    for (float x : v) all_finite = all_finite && std::isfinite(x);
    LSE_EXPECT(all_finite);
  }
}

LSE_TEST(param_count_estimate_is_in_the_right_ballpark) {
  if (!have_model()) return;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto st = SafeTensors::open(paths->weights);
  if (!st.ok()) return;

  const double actual = static_cast<double>(st->total_parameters());
  const double estimate = static_cast<double>(preset_b1_5().param_count());
  // The analytic estimate is a budgeting aid, not a checksum: 5% is enough to
  // catch a structural mistake without pinning every bias term.
  LSE_EXPECT(std::fabs(estimate - actual) / actual < 0.05);
}

LSE_TEST_MAIN()
