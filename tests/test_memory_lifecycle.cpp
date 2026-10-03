// Device memory across requests and sessions: what a session holds goes back
// to the device when the session is closed, evicted or (for a request without
// a session_id) when its request ends, so the engine returns to what it held
// after load. Runs a small Qwen3.5-shaped model (Gated DeltaNet and paged
// attention layers) through the same Router lse_request and the HTTP server
// use, on whichever backend comes up first (the host backend on a bare
// machine, where the KV pools and recurrent state are what a session holds).
#include "harness.hpp"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/model/config.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/registry.hpp"
#include "lse/model/weights.hpp"
#include "lse/server/router.hpp"
#include "lse/tokenizer/tokenizer.hpp"

using namespace lse;
using json = nlohmann::json;

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

std::uint64_t live() { return backend::allocation_totals(backend::MemoryClass::kDevice).live; }

// A character-level BPE vocabulary: every lowercase letter and a space, plus
// the ChatML specials, so any prompt below tokenizes one token per character.
struct Engine {
  std::filesystem::path dir;
  model::Config config;
  std::optional<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  std::unique_ptr<model::MtpModule> mtp;
  std::optional<tokenizer::Tokenizer> tok;
  std::unique_ptr<server::Router> router;
  std::uint64_t baseline = 0;

  ~Engine() {
    router.reset();
    mtp.reset();
    lm.reset();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  Status open(std::size_t max_sessions, std::size_t budget, bool with_mtp = false) {
    dir = std::filesystem::temp_directory_path() / ("lse-memory-lifecycle-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    json vocab = json::object();
    std::uint32_t next = 0;
    for (char c = 'a'; c <= 'z'; ++c) vocab[std::string(1, c)] = next++;
    vocab[" "] = next++;
    json added = json::array();
    for (const char* special : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
      added.push_back({{"id", next++}, {"content", special}, {"special", true},
                       {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                       {"normalized", false}});
    {
      std::ofstream out(dir / "tokenizer.json");
      out << json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}}},
                  {"added_tokens", added}}.dump();
    }
    auto t = tokenizer::Tokenizer::from_file((dir / "tokenizer.json").string());
    if (!t.ok()) return t.status();
    tok.emplace(t.release());

    config = mtp_test_config(16);
    config.vocab_size = 64;
    config.kv_length = 4096;
    config.mtp_layers = with_mtp ? 1 : 0;
    config.sampling_defaults.temperature = 0.0f;
    write_shaped(dir / "model.safetensors", qwen_dense_tensors(config),
                 [](const std::string&, std::size_t i) { return filler(i); });
    auto st = model::SafeTensors::open((dir / "model.safetensors").string());
    if (!st.ok()) return st.status();
    weights.emplace(st.release());
    auto built = model::build_model(config, *weights, "");
    if (!built.ok()) return built.status();
    lm = built.release();
    model::WeightBinder binder(*weights);
    LSE_RETURN_IF_ERROR(lm->load(binder));
    if (with_mtp) {
      std::filesystem::create_directories(dir / "mtp");
      write_shaped(dir / "mtp" / "model.safetensors", mtp_tensors(config),
                   [](const std::string&, std::size_t i) { return filler(i); });
      std::ofstream(dir / "mtp" / "config.json") << mtp_config_json(config);
      auto opened = model::MtpModule::open((dir / "mtp").string(), config, *lm);
      if (!opened.ok()) return opened.status();
      mtp = opened.release();
    }
    server::ServerOptions options;
    options.model_id = "memory-fixture";
    options.max_sessions = max_sessions;
    options.session_memory_budget = budget;
    router = std::make_unique<server::Router>(*lm, *tok, options);
    if (mtp) router->use_mtp(*mtp);
    // What the engine holds once loaded. Caches the scheduler keeps for the
    // process (interned constants) may hold a little more from an earlier
    // engine; closing the last session lets those go too, so "back to
    // baseline" is checked as "no more than baseline".
    baseline = live();
    return OkStatus();
  }

  // One completion; returns the HTTP status.
  int complete(const std::string& session, const std::string& prompt, int max_tokens = 8) {
    json body{{"model", "memory-fixture"}, {"prompt", prompt}, {"max_tokens", max_tokens},
              {"temperature", 0}};
    if (!session.empty()) body["session_id"] = session;
    const auto reply = router->handle("POST", "/v1/completions", body.dump());
    if (reply.status != 200) std::printf("       %d: %s\n", reply.status, reply.body.c_str());
    return reply.status;
  }

  json sessions() {
    return json::parse(router->handle("GET", "/v1/lse/sessions", "").body)["data"];
  }

  int close(const std::string& session) {
    return router->handle("DELETE", "/v1/lse/sessions/" + session, "").status;
  }
};

// A prompt of `n` characters (one token each), distinct per seed.
std::string prompt(std::size_t n, unsigned seed) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += static_cast<char>('a' + (i * 7 + seed * 13) % 26);
  return s;
}

}  // namespace

LSE_TEST(one_shot_requests_return_the_device_to_its_post_load_baseline) {
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  for (unsigned r = 0; r < 4; ++r) {
    LSE_EXPECT_EQ(e.complete("", prompt(300 + 700 * r, r)), 200);
    std::printf("       one-shot %u: live %llu (baseline %llu)\n", r,
                static_cast<unsigned long long>(live()), static_cast<unsigned long long>(e.baseline));
    LSE_EXPECT(live() <= e.baseline);
  }
  LSE_EXPECT(e.sessions().empty());
}

LSE_TEST(sessions_hold_their_kv_until_closed_and_closing_all_returns_to_baseline) {
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  // Alternate three sessions with long, growing conversations.
  std::vector<std::string> history(3);
  for (unsigned turn = 0; turn < 3; ++turn) {
    for (unsigned s = 0; s < 3; ++s) {
      history[s] += prompt(400 + 100 * s, turn * 3 + s);
      LSE_EXPECT_EQ(e.complete("s" + std::to_string(s), history[s]), 200);
    }
  }
  const json listed = e.sessions();
  LSE_EXPECT_EQ(listed.size(), 3u);
  std::uint64_t held = 0;
  for (const json& s : listed) held += s["bytes"].get<std::uint64_t>();
  std::printf("       3 sessions hold %llu bytes; live %llu over baseline %llu\n",
              static_cast<unsigned long long>(held), static_cast<unsigned long long>(live()),
              static_cast<unsigned long long>(e.baseline));
  LSE_EXPECT(held > 0);
  LSE_EXPECT(live() > e.baseline);
  LSE_EXPECT_EQ(e.close("s1"), 200);
  LSE_EXPECT_EQ(e.close("s1"), 404);
  LSE_EXPECT_EQ(e.sessions().size(), 2u);
  LSE_EXPECT_EQ(e.close("s0"), 200);
  LSE_EXPECT_EQ(e.close("s2"), 200);
  LSE_EXPECT(e.sessions().empty());
  std::printf("       after closing all: live %llu (baseline %llu)\n",
              static_cast<unsigned long long>(live()), static_cast<unsigned long long>(e.baseline));
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST(a_continued_session_scores_only_its_new_tokens) {
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0));
  if (!e.router) return;
  const std::string first = prompt(500, 1);
  json body{{"model", "memory-fixture"}, {"prompt", first}, {"max_tokens", 4},
            {"temperature", 0}, {"session_id", "chat"}};
  auto reply = e.router->handle("POST", "/v1/completions", body.dump());
  LSE_EXPECT_EQ(reply.status, 200);
  const json a = json::parse(reply.body);
  // The next turn extends the first prompt and its answer.
  body["prompt"] = first + a["choices"][0]["text"].get<std::string>() + prompt(50, 2);
  reply = e.router->handle("POST", "/v1/completions", body.dump());
  LSE_EXPECT_EQ(reply.status, 200);
  const json b = json::parse(reply.body);
  const auto cached = b["usage"]["prompt_tokens_details"]["cached_tokens"].get<int>();
  std::printf("       second turn: %d of %d prompt tokens cached\n", cached,
              b["usage"]["prompt_tokens"].get<int>());
  LSE_EXPECT(cached >= 500);
  // Another session in between does not reuse or merge this one's KV ...
  LSE_EXPECT_EQ(e.complete("other", first), 200);
  reply = e.router->handle("POST", "/v1/completions",
                           json{{"model", "memory-fixture"}, {"prompt", first}, {"max_tokens", 4},
                                {"temperature", 0}, {"session_id", "fresh"}}.dump());
  LSE_EXPECT_EQ(reply.status, 200);
  LSE_EXPECT_EQ(json::parse(reply.body)["usage"]["prompt_tokens_details"]["cached_tokens"].get<int>(), 0);
  LSE_EXPECT_EQ(e.close("chat"), 200);
  LSE_EXPECT_EQ(e.close("other"), 200);
  LSE_EXPECT_EQ(e.close("fresh"), 200);
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST(idle_sessions_beyond_the_cap_or_budget_are_evicted_and_freed) {
  {
    Engine e;
    LSE_EXPECT_OK(e.open(2, 0));
    if (!e.router) return;
    for (unsigned s = 0; s < 5; ++s) LSE_EXPECT_EQ(e.complete("s" + std::to_string(s), prompt(600, s)), 200);
    const json listed = e.sessions();
    LSE_EXPECT_EQ(listed.size(), 2u);
    for (const json& s : listed) LSE_EXPECT(s["id"] == "s3" || s["id"] == "s4");
    LSE_EXPECT_EQ(e.close("s3"), 200);
    LSE_EXPECT_EQ(e.close("s4"), 200);
    LSE_EXPECT(live() <= e.baseline);
  }
  {
    // A budget of about two sessions' KV: never more than that idle, plus the
    // one generating.
    Engine probe;
    LSE_EXPECT_OK(probe.open(8, 0));
    if (!probe.router) return;
    LSE_EXPECT_EQ(probe.complete("one", prompt(600, 0)), 200);
    const auto one = probe.sessions()[0]["bytes"].get<std::uint64_t>();
    probe.router.reset();
    Engine e;
    LSE_EXPECT_OK(e.open(0, 2 * one + one / 2));
    if (!e.router) return;
    for (unsigned s = 0; s < 6; ++s) {
      LSE_EXPECT_EQ(e.complete("b" + std::to_string(s), prompt(600, s)), 200);
      std::uint64_t held = 0;
      for (const json& x : e.sessions()) held += x["bytes"].get<std::uint64_t>();
      LSE_EXPECT(held <= 3 * one);
    }
    LSE_EXPECT(e.sessions().size() <= 3u);
    for (const json& x : e.sessions()) LSE_EXPECT_EQ(e.close(x["id"].get<std::string>()), 200);
    LSE_EXPECT(live() <= e.baseline);
  }
}

LSE_TEST(a_draft_module_follows_its_session_and_is_released_with_it) {
  Engine e;
  LSE_EXPECT_OK(e.open(8, 0, /*with_mtp=*/true));
  if (!e.router) return;
  std::vector<std::string> history(2);
  for (unsigned turn = 0; turn < 3; ++turn) {
    for (unsigned s = 0; s < 2; ++s) {
      history[s] += prompt(300, turn * 2 + s);
      LSE_EXPECT_EQ(e.complete("m" + std::to_string(s), history[s], 12), 200);
    }
    LSE_EXPECT_EQ(e.complete("", prompt(200, 40 + turn), 12), 200);
  }
  LSE_EXPECT_EQ(e.close("m0"), 200);
  LSE_EXPECT_EQ(e.close("m1"), 200);
  std::printf("       with MTP after closing all: live %llu (baseline %llu)\n",
              static_cast<unsigned long long>(live()), static_cast<unsigned long long>(e.baseline));
  LSE_EXPECT(live() <= e.baseline);
}

LSE_TEST_MAIN()
