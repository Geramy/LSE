// Loads the real lemonseed b1.5 checkpoint when it is present; the pure-config
// cases run everywhere.
#include "lse/model/config.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/ops/rope.hpp"

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

// --- architecture detection -------------------------------------------------
//
// The predicates only read tensor names, so a handful of empty tensors with the
// right names is a faithful stand-in for a checkpoint and the whole matrix runs
// without one. What is being tested is that each architecture claims its own
// layout and refuses every other, because a wrong claim loads a checkpoint into
// the wrong model kernel and produces plausible garbage rather than an error.

namespace {

std::string write_fixture(const std::string& stem,
                          const std::vector<std::string>& names) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const std::string& n : names) {
    if (header.size() > 1) header += ",";
    header += "\"" + n + "\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
              std::to_string(offset) + "," + std::to_string(offset + 4) + "]}";
    offset += 4;
  }
  header += "}";
  // The header is padded to 8 bytes because safetensors requires the data
  // section to start aligned.
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const std::vector<char> zeros(offset, 0);
  out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  return path;
}

const std::vector<std::string>& lemonseed_names() {
  static const std::vector<std::string> v{"blocks.0.mod.router.weight",
                                          "blocks.0.moe.w_gate",
                                          "blocks.0.mixer.in_proj_q.weight"};
  return v;
}

const std::vector<std::string>& qwen_dense_names() {
  static const std::vector<std::string> v{
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
      "model.language_model.layers.0.mlp.gate_proj.weight"};
  return v;
}

const std::vector<std::string>& qwen_moe_names() {
  static const std::vector<std::string> v{
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
      "model.language_model.layers.0.mlp.experts.gate_up_proj",
      "model.language_model.layers.0.mlp.gate.weight"};
  return v;
}

// Detected name, or "" when nothing claims the fixture.
std::string detect(const std::string& stem, const std::vector<std::string>& names) {
  auto st = SafeTensors::open(write_fixture(stem, names));
  if (!st.ok()) return "<open failed>";
  const Config cfg;
  auto arch = detect_architecture(cfg, *st);
  if (!arch.ok()) return "";
  return std::string((*arch)->name);
}

}  // namespace

LSE_TEST(every_architecture_claims_only_its_own_checkpoint) {
  LSE_EXPECT(detect("lemonseed", lemonseed_names()) == "lemonseed");
  LSE_EXPECT(detect("q35", qwen_dense_names()) == "qwen3.5");
  LSE_EXPECT(detect("q35moe", qwen_moe_names()) == "qwen3.5-moe");
}

LSE_TEST(qwen_predicates_cannot_both_claim_a_checkpoint) {
  // The dense test requires the stacked-expert tensor absent and the MoE test
  // requires it present, so no checkpoint satisfies both. A checkpoint that
  // carries the MoE marker plus the dense MLP names is still MoE only.
  std::vector<std::string> both = qwen_moe_names();
  for (const std::string& n : qwen_dense_names()) both.push_back(n);
  LSE_EXPECT(detect("q35both", both) == "qwen3.5-moe");
}

LSE_TEST(qwen_predicates_refuse_a_lemonseed_checkpoint) {
  // Both are GDN + MoE hybrids, so this is the collision that matters: what
  // separates them is the tensor prefix and the routed-expert layout, never a
  // config name field.
  LSE_EXPECT(detect("lemonseed2", lemonseed_names()) == "lemonseed");

  // ...and lemonseed refuses theirs: it needs the Mixture-of-Depths router,
  // which no Qwen checkpoint has.
  std::vector<std::string> mixed = qwen_moe_names();
  mixed.push_back("blocks.0.moe.w_gate");
  LSE_EXPECT(detect("q35moe2", mixed) == "qwen3.5-moe");
}

LSE_TEST(an_unrecognized_checkpoint_names_what_was_tried) {
  auto st = SafeTensors::open(write_fixture("stranger", {"encoder.layer.0.weight"}));
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  const Config cfg;
  auto arch = detect_architecture(cfg, *st);
  LSE_EXPECT(!arch.ok());
  if (arch.ok()) return;
  LSE_EXPECT(arch.status().code() == StatusCode::kNotFound);
  const std::string msg = arch.status().message();
  LSE_EXPECT(msg.find("lemonseed") != std::string::npos);
  LSE_EXPECT(msg.find("qwen3.5") != std::string::npos);
  LSE_EXPECT(msg.find("qwen3.5-moe") != std::string::npos);
}

// --- the two weight reorderings the Qwen kernels rely on ---------------------

namespace {

// Same mapping qwen3_5_common.cpp uses to reconcile HF's rotate_half RoPE with
// graph::rope's adjacent-pair rotation.
std::int64_t rope_source(std::int64_t d, std::int64_t rope_dim) {
  if (d >= rope_dim) return d;
  const std::int64_t j = d / 2;
  return d % 2 == 0 ? j : j + rope_dim / 2;
}

std::string write_f32_fixture(const std::string& stem, const std::string& name,
                              const std::vector<std::int64_t>& shape,
                              const std::vector<float>& data) {
  std::string dims;
  for (std::int64_t d : shape) {
    if (!dims.empty()) dims += ",";
    dims += std::to_string(d);
  }
  std::string header = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
                       "],\"data_offsets\":[0," +
                       std::to_string(data.size() * 4) + "]}}";
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
  return path;
}

}  // namespace

LSE_TEST(rope_reordering_turns_rotate_half_into_adjacent_pairs) {
  // Qwen's weights are reordered at load so graph::rope reproduces HF's
  // rotate_half. Everything else in the attention path depends on that being
  // exactly right, and it is invisible in the tensor names, so check the
  // identity directly: interleaved-rope(reorder(x)) == reorder(rotate_half(x)).
  constexpr std::int64_t kRopeDim = 8;
  constexpr std::int64_t kHeadDim = 12;  // 4 channels past the rotation
  constexpr float kTheta = 10000.0f;
  constexpr std::int32_t kPos = 3;

  std::vector<float> x(kHeadDim);
  for (std::int64_t i = 0; i < kHeadDim; ++i) {
    x[static_cast<std::size_t>(i)] = 0.5f + 0.25f * static_cast<float>(i);
  }

  // HF: pairs (i, i + rope_dim/2) share the angle for i in [0, rope_dim/2).
  std::vector<float> want(kHeadDim);
  const std::int64_t half = kRopeDim / 2;
  for (std::int64_t i = 0; i < half; ++i) {
    const auto freq = static_cast<float>(
        1.0 / std::pow(static_cast<double>(kTheta),
                       static_cast<double>(i) / static_cast<double>(half)));
    const auto ang = static_cast<double>(kPos) * static_cast<double>(freq);
    const auto c = static_cast<float>(std::cos(ang));
    const auto s = static_cast<float>(std::sin(ang));
    const float a = x[static_cast<std::size_t>(i)];
    const float b = x[static_cast<std::size_t>(i + half)];
    want[static_cast<std::size_t>(i)] = a * c - b * s;
    want[static_cast<std::size_t>(i + half)] = b * c + a * s;
  }
  for (std::int64_t d = kRopeDim; d < kHeadDim; ++d) {
    want[static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(d)];
  }

  auto tables = ops::build_rope(kRopeDim, 16, kTheta);
  LSE_EXPECT(tables.ok());
  if (!tables.ok()) return;

  graph::Array v = graph::Array::zeros(Shape{1, 1, 1, kHeadDim}, DType::kF32);
  LSE_EXPECT_OK(v.eval());
  for (std::int64_t d = 0; d < kHeadDim; ++d) {
    graph::interpreter::store_element(
        *v.node(), static_cast<std::size_t>(d),
        x[static_cast<std::size_t>(rope_source(d, kRopeDim))]);
  }
  auto rotated = ops::apply_rope(v, *tables, kPos);
  LSE_EXPECT(rotated.ok());
  if (!rotated.ok()) return;

  std::vector<float> got(kHeadDim);
  LSE_EXPECT_OK(rotated->to_host(got.data(), got.size() * sizeof(float)));
  for (std::int64_t d = 0; d < kHeadDim; ++d) {
    LSE_EXPECT_NEAR(got[static_cast<std::size_t>(d)],
                    want[static_cast<std::size_t>(rope_source(d, kRopeDim))],
                    1e-5);
  }
}

LSE_TEST(require_rows_splits_a_fused_gate_up_expert_tensor) {
  // Qwen3.5 MoE stores gate and up as one [E, 2*inter, hidden] tensor whose
  // per-expert rows are gate then up. Getting the halves the wrong way round
  // swaps SiLU's argument with its multiplicand, which stays finite and stays
  // plausible, so it is worth pinning.
  constexpr std::int64_t kE = 2, kInter = 2, kHidden = 3;
  std::vector<float> data;
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < 2 * kInter; ++j) {
      for (std::int64_t h = 0; h < kHidden; ++h) {
        data.push_back(static_cast<float>(e * 100 + j * 10 + h));
      }
    }
  }
  const std::string path = write_f32_fixture(
      "fused_experts", "gu", {kE, 2 * kInter, kHidden}, data);
  auto st = SafeTensors::open(path);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;

  std::vector<std::int64_t> gate_rows, up_rows;
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < kInter; ++j) {
      gate_rows.push_back(e * 2 * kInter + j);
      up_rows.push_back(e * 2 * kInter + kInter + j);
    }
  }

  WeightBinder binder(*st);
  auto gate = binder.require_rows("gu", gate_rows, Shape{kE, kInter, kHidden});
  auto up = binder.require_rows("gu", up_rows, Shape{kE, kInter, kHidden});
  LSE_EXPECT(gate.ok());
  LSE_EXPECT(up.ok());
  if (!gate.ok() || !up.ok()) return;

  std::vector<float> g(static_cast<std::size_t>(kE * kInter * kHidden));
  std::vector<float> u(g.size());
  LSE_EXPECT_OK(gate->to_host(g.data(), g.size() * sizeof(float)));
  LSE_EXPECT_OK(up->to_host(u.data(), u.size() * sizeof(float)));
  for (std::int64_t e = 0; e < kE; ++e) {
    for (std::int64_t j = 0; j < kInter; ++j) {
      for (std::int64_t h = 0; h < kHidden; ++h) {
        const auto at = static_cast<std::size_t>((e * kInter + j) * kHidden + h);
        LSE_EXPECT_NEAR(g[at], static_cast<float>(e * 100 + j * 10 + h), 1e-6);
        LSE_EXPECT_NEAR(u[at],
                        static_cast<float>(e * 100 + (kInter + j) * 10 + h), 1e-6);
      }
    }
  }
}

LSE_TEST(require_rows_rejects_an_order_that_cannot_fill_the_shape) {
  const std::string path =
      write_f32_fixture("small_rows", "w", {4, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto st = SafeTensors::open(path);
  LSE_EXPECT(st.ok());
  if (!st.ok()) return;
  WeightBinder binder(*st);
  LSE_EXPECT(!binder.require_rows("w", {0, 1}, Shape{4, 2}).ok());
  LSE_EXPECT(!binder.require_rows("w", {0, 9}, Shape{2, 2}).ok());
}

// --- HuggingFace config parsing ---------------------------------------------

namespace {

// The fields Qwen3.5-4B's config.json carries, trimmed to what the engine reads.
std::string hf_config(const std::string& extra = {}) {
  return R"({"tie_word_embeddings": true, "text_config": {
    "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 32,
    "rms_norm_eps": 1e-06, "full_attention_interval": 4,
    "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "intermediate_size": 9216,
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
    )" + extra + "}}";
}

}  // namespace

LSE_TEST(hf_config_is_read_with_hf_field_names) {
  auto c = Config::from_json_string(hf_config());
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  LSE_EXPECT_EQ(c->vocab_size, 248320);
  LSE_EXPECT_EQ(c->hidden_size, 2560);
  LSE_EXPECT_EQ(c->num_layers, 32);
  LSE_EXPECT_EQ(c->attn_q_heads, 16);
  LSE_EXPECT_EQ(c->attn_kv_heads, 4);
  LSE_EXPECT_EQ(c->attn_head_dim, 256);
  // 0.25 of a 256-wide head.
  LSE_EXPECT_EQ(c->rope_dim, 64);
  LSE_EXPECT_EQ(c->gdn_qk_heads, 16);
  LSE_EXPECT_EQ(c->gdn_v_heads, 32);
  LSE_EXPECT_EQ(c->gdn_head_dim, 128);
  LSE_EXPECT_EQ(c->mlp_intermediate, 9216);
  // No sliding window, so every attention layer sees the whole prefix.
  LSE_EXPECT_EQ(c->sliding_window, 0);
  LSE_EXPECT(c->tie_word_embeddings);
  LSE_EXPECT_EQ(c->num_experts, 0);
}

LSE_TEST(hf_moe_config_carries_the_routing_shape) {
  auto c = Config::from_json_string(hf_config(
      R"(, "num_experts": 256, "num_experts_per_tok": 8,
          "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512)"));
  LSE_EXPECT(c.ok());
  if (!c.ok()) return;
  LSE_EXPECT_EQ(c->num_experts, 256);
  LSE_EXPECT_EQ(c->num_active_experts, 8);
  LSE_EXPECT_EQ(c->expert_intermediate, 512);
  LSE_EXPECT_EQ(c->shared_expert_intermediate, 512);
  LSE_EXPECT_EQ(c->num_shared_experts, 1);
  // Plain top-k, not lemonseed's band.
  LSE_EXPECT_NEAR(c->expert_score_band, 1.0, 1e-9);
}

LSE_TEST(a_missing_hf_field_is_an_error_not_a_default) {
  // Dropping head_dim must not leave the 64 the Config defaults to: that would
  // build a model of the wrong shape that still passes validate().
  std::string text = hf_config();
  const std::string key = R"("head_dim": 256,)";
  const std::size_t at = text.find(key);
  LSE_EXPECT(at != std::string::npos);
  if (at == std::string::npos) return;
  text.erase(at, key.size());

  auto c = Config::from_json_string(text);
  LSE_EXPECT(!c.ok());
  if (c.ok()) return;
  LSE_EXPECT(c.status().code() == StatusCode::kNotFound);
  LSE_EXPECT(c.status().message().find("head_dim") != std::string::npos);
}

LSE_TEST(layer_types_must_agree_with_the_interval_rule) {
  // The engine derives the mixer schedule from full_attention_interval. If the
  // checkpoint's own layer_types disagrees, every layer past the divergence
  // would be built as the wrong mixer.
  std::string types = R"(, "num_hidden_layers": 4, "layer_types":
      ["linear_attention", "full_attention", "linear_attention", "full_attention"])";
  auto bad = Config::from_json_string(hf_config(types));
  LSE_EXPECT(!bad.ok());
  if (!bad.ok()) {
    LSE_EXPECT(bad.status().message().find("layer_types") != std::string::npos);
  }

  std::string good = R"(, "num_hidden_layers": 4, "layer_types":
      ["linear_attention", "linear_attention", "linear_attention", "full_attention"])";
  LSE_EXPECT(Config::from_json_string(hf_config(good)).ok());
}

LSE_TEST(a_lemonseed_config_still_takes_the_native_path) {
  // No text_config, so none of the HF requirements apply.
  const Config a = preset_b1_5();
  auto b = Config::from_json_string(a.to_json());
  LSE_EXPECT(b.ok());
  if (!b.ok()) return;
  LSE_EXPECT_EQ(b->gdn_qk_heads, 8);
  LSE_EXPECT_EQ(b->gdn_v_heads, 4);
  LSE_EXPECT_EQ(b->sliding_window, 256);
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

// --- the builder must account for every tensor -------------------------------
//
// layer.hpp names an unclaimed tensor as the likeliest silent failure once a
// second architecture exists, so the guarantee is checked from both ends on a
// checkpoint small enough to write by hand: a tensor the builder never reads
// must fail the load, and a tensor the builder needs but the file lacks must
// fail it too. A whole checkpoint is generated from the Config so the two
// negative cases differ from the positive one by exactly one tensor.

namespace {

struct NamedShape {
  std::string name;
  std::vector<std::int64_t> dims;
};

std::string write_shaped_fixture(const std::string& stem,
                                 const std::vector<NamedShape>& tensors) {
  std::string header = "{";
  std::size_t offset = 0;
  for (const NamedShape& t : tensors) {
    std::size_t count = 1;
    std::string dims;
    for (std::int64_t d : t.dims) {
      if (!dims.empty()) dims += ",";
      dims += std::to_string(d);
      count *= static_cast<std::size_t>(d);
    }
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + count * 4) + "]}";
    offset += count * 4;
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";

  const std::string path =
      (std::filesystem::temp_directory_path() / (stem + ".safetensors")).string();
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t n = header.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  // Small non-zero values: a load must survive real numbers, and an all-zero
  // file would hide a shape error that only shows up as a NaN.
  std::vector<float> data(offset / 4);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(offset));
  return path;
}

Config tiny_qwen_config(bool moe) {
  Config c;
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
  c.gdn_head_dim = 8;
  c.gdn_conv_kernel = 4;
  c.dtype = "float32";
  if (moe) {
    c.mlp_intermediate = 0;
    c.num_experts = 4;
    c.num_active_experts = 2;
    c.num_shared_experts = 1;
    c.expert_intermediate = 16;
    c.shared_expert_intermediate = 16;
    c.expert_score_band = 1.0f;
    c.tie_word_embeddings = false;
  } else {
    c.mlp_intermediate = 64;
    c.num_experts = 0;
    c.num_active_experts = 0;
    c.num_shared_experts = 0;
    c.expert_intermediate = 0;
    c.tie_word_embeddings = true;
  }
  return c;
}

// Every tensor Qwen3.5 puts in the file, derived from the config the same way
// the builder derives what it asks for.
std::vector<NamedShape> qwen_checkpoint(const Config& c) {
  const std::int64_t h = c.hidden_size;
  const std::int64_t qh = c.attn_q_heads, kvh = c.attn_kv_heads;
  const std::int64_t ahd = c.attn_head_dim;
  const std::int64_t kh = c.gdn_qk_heads, vh = c.gdn_v_heads;
  const std::int64_t ghd = c.gdn_head_dim;
  const std::int64_t conv_dim = 2 * kh * ghd + vh * ghd;

  std::vector<NamedShape> t;
  t.push_back({"model.language_model.embed_tokens.weight", {c.vocab_size, h}});
  t.push_back({"model.language_model.norm.weight", {h}});
  if (!c.tie_word_embeddings) t.push_back({"lm_head.weight", {c.vocab_size, h}});

  for (std::int32_t i = 0; i < c.num_layers; ++i) {
    const std::string p =
        "model.language_model.layers." + std::to_string(i) + ".";
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
      t.push_back({g + "conv1d.weight", {conv_dim, 1, c.gdn_conv_kernel}});
      t.push_back({g + "A_log", {vh}});
      t.push_back({g + "dt_bias", {vh}});
      t.push_back({g + "norm.weight", {ghd}});
      t.push_back({g + "out_proj.weight", {h, vh * ghd}});
    }

    const std::string m = p + "mlp.";
    if (c.num_experts > 0) {
      const std::int64_t e = c.num_experts, in = c.expert_intermediate;
      const std::int64_t si = c.shared_expert_intermediate;
      t.push_back({m + "gate.weight", {e, h}});
      t.push_back({m + "experts.gate_up_proj", {e, 2 * in, h}});
      t.push_back({m + "experts.down_proj", {e, h, in}});
      t.push_back({m + "shared_expert.gate_proj.weight", {si, h}});
      t.push_back({m + "shared_expert.up_proj.weight", {si, h}});
      t.push_back({m + "shared_expert.down_proj.weight", {h, si}});
      t.push_back({m + "shared_expert_gate.weight", {1, h}});
    } else {
      const std::int64_t in = c.mlp_intermediate;
      t.push_back({m + "gate_proj.weight", {in, h}});
      t.push_back({m + "up_proj.weight", {in, h}});
      t.push_back({m + "down_proj.weight", {h, in}});
    }
  }
  return t;
}

// Builds and loads, returning the load status so a caller can assert on the
// message rather than just on failure.
Status load_fixture(const std::string& stem, const Config& c,
                    const std::vector<NamedShape>& tensors) {
  auto st = SafeTensors::open(write_shaped_fixture(stem, tensors));
  if (!st.ok()) return st.status();
  auto model = build_model(c, *st);
  if (!model.ok()) return model.status();
  WeightBinder binder(*st);
  return (*model)->load(binder);
}

}  // namespace

LSE_TEST(qwen_dense_load_claims_every_tensor_in_the_checkpoint) {
  const Config c = tiny_qwen_config(/*moe=*/false);
  const std::vector<NamedShape> full = qwen_checkpoint(c);
  LSE_EXPECT_OK(load_fixture("q35_exact", c, full));

  // One tensor no layer reads. Nothing in the forward pass would notice, which
  // is exactly why the audit has to.
  std::vector<NamedShape> extra = full;
  extra.push_back(
      {"model.language_model.layers.0.linear_attn.in_proj_g.weight", {8, 32}});
  const Status stray = load_fixture("q35_extra", c, extra);
  LSE_EXPECT(!stray.ok());
  if (!stray.ok()) {
    LSE_EXPECT(stray.message().find("in_proj_g") != std::string::npos);
  }

  // One tensor a layer needs and the file lacks.
  std::vector<NamedShape> absent;
  for (const NamedShape& t : full) {
    if (t.name != "model.language_model.layers.1.mlp.up_proj.weight") {
      absent.push_back(t);
    }
  }
  LSE_EXPECT_EQ(absent.size() + 1, full.size());
  const Status missing = load_fixture("q35_absent", c, absent);
  LSE_EXPECT(!missing.ok());
  if (!missing.ok()) {
    LSE_EXPECT(missing.code() == StatusCode::kNotFound);
    LSE_EXPECT(missing.message().find("layers.1.mlp.up_proj") !=
               std::string::npos);
  }
}

LSE_TEST(qwen_moe_load_claims_every_tensor_in_the_checkpoint) {
  // The MoE builder has no end-to-end run behind it, so the tensor contract is
  // the only thing pinning it: expert stack, shared expert, shared gate and an
  // untied head all have to be claimed, and nothing else may be.
  const Config c = tiny_qwen_config(/*moe=*/true);
  const std::vector<NamedShape> full = qwen_checkpoint(c);
  LSE_EXPECT_OK(load_fixture("q35moe_exact", c, full));

  std::vector<NamedShape> extra = full;
  extra.push_back({"model.language_model.layers.2.mlp.experts.bias", {4, 16}});
  const Status stray = load_fixture("q35moe_extra", c, extra);
  LSE_EXPECT(!stray.ok());
  if (!stray.ok()) {
    LSE_EXPECT(stray.message().find("experts.bias") != std::string::npos);
  }

  // The untied head is part of the contract: dropping it must fail, not
  // silently fall back to the embedding table.
  std::vector<NamedShape> headless;
  for (const NamedShape& t : full) {
    if (t.name != "lm_head.weight") headless.push_back(t);
  }
  const Status no_head = load_fixture("q35moe_headless", c, headless);
  LSE_EXPECT(!no_head.ok());
}

LSE_TEST(qwen_ignores_only_the_vision_tower_and_the_mtp_head) {
  // Two prefixes are declared not-part-of-the-decoder. They have to actually be
  // skipped, and the declaration must not widen into "ignore what we missed".
  const Config c = tiny_qwen_config(/*moe=*/false);
  std::vector<NamedShape> t = qwen_checkpoint(c);
  t.push_back({"model.visual.blocks.0.attn.qkv.weight", {8, 8}});
  t.push_back({"mtp.fc.weight", {8, 8}});
  LSE_EXPECT_OK(load_fixture("q35_ignored", c, t));

  // A near-miss on an ignored prefix is still an unread tensor.
  t.push_back({"model.visual_extra.weight", {4}});
  LSE_EXPECT(!load_fixture("q35_nearmiss", c, t).ok());
}

LSE_TEST(a_tiny_qwen_of_each_kind_runs_a_forward_pass) {
  // Claiming the right tensors is not the same as computing with them. The MoE
  // path in particular has no checkpoint small enough to run end to end, so the
  // routed experts, the shared expert and its sigmoid gate are exercised here:
  // a wrong shape or a router that indexes past the stack fails or NaNs, and
  // neither shows up in a load-only test.
  for (const bool moe : {false, true}) {
    const Config c = tiny_qwen_config(moe);
    const std::vector<NamedShape> t = qwen_checkpoint(c);
    const std::string stem = moe ? "q35moe_fwd" : "q35_fwd";
    auto st = SafeTensors::open(write_shaped_fixture(stem, t));
    LSE_EXPECT(st.ok());
    if (!st.ok()) continue;
    auto model = build_model(c, *st);
    LSE_EXPECT(model.ok());
    if (!model.ok()) continue;
    WeightBinder binder(*st);
    LSE_EXPECT_OK((*model)->load(binder));

    const std::int64_t T = 3;
    graph::Array tokens = graph::Array::zeros(Shape{1, T}, DType::kF32);
    LSE_EXPECT_OK(tokens.eval());
    for (std::int64_t i = 0; i < T; ++i) {
      graph::interpreter::store_element(*tokens.node(), (std::size_t)i,
                                        (float)(i + 1));
    }
    auto hid = (*model)->hidden(tokens, nullptr, nullptr);
    LSE_EXPECT(hid.ok());
    if (!hid.ok()) continue;
    LSE_EXPECT_EQ(hid->shape().dim(2), c.hidden_size);

    std::vector<float> v((std::size_t)(T * c.hidden_size));
    LSE_EXPECT_OK(hid->to_host(v.data(), v.size() * sizeof(float)));
    bool finite = true;
    for (float x : v) finite = finite && std::isfinite(x);
    LSE_EXPECT(finite);

    auto logits = (*model)->lm_head(*hid);
    LSE_EXPECT(logits.ok());
    if (logits.ok()) LSE_EXPECT_EQ(logits->shape().dim(2), c.vocab_size);
  }
}

// --- detection, from the other side -----------------------------------------

LSE_TEST(no_qwen_predicate_claims_a_lemonseed_shaped_checkpoint) {
  // Asserting that lemonseed wins on a lemonseed file proves nothing about the
  // Qwen predicates: on a tie detect_architecture keeps the first match it
  // finds, so a second claimant would be invisible. Make lemonseed abstain by
  // dropping the one tensor its own test requires, and the detection must then
  // fail outright — if either Qwen predicate could claim this file, it would
  // answer here.
  std::vector<std::string> names;
  for (const std::string& n : lemonseed_names()) {
    if (n != "blocks.0.mod.router.weight") names.push_back(n);
  }
  LSE_EXPECT(detect("lemonseed_no_mod", names).empty());

  // Same argument for the real checkpoint's full tensor list when it is here.
  if (!have_model()) return;
  auto paths = resolve_model(model_dir());
  if (!paths.ok()) return;
  auto st = SafeTensors::open(paths->weights);
  if (!st.ok()) return;
  std::vector<std::string> real;
  for (const auto& [name, _] : st->tensors()) {
    if (name != "blocks.0.mod.router.weight") real.push_back(name);
  }
  LSE_EXPECT(detect("lemonseed_real_no_mod", real).empty());
}

LSE_TEST_MAIN()
