// The in-process request path (lse_request) and the HTTP server answer every
// request with the same JSON, because both are adapters over one Router. This
// drives a tiny host-only model through both and compares what comes back:
// responses, streamed chunks, errors and the route table. Request ids and
// creation times are unique per response and timings are measurements, so
// those are the only fields set aside.
#include "harness.hpp"

#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/graph/graph.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/weights.hpp"
#include "lse/models/thinking_controls.hpp"
#include "lse/server/http_server.hpp"
#include "lse/server/in_process.hpp"
#include "lse/server/router.hpp"
#include "lse/tokenizer/tokenizer.hpp"

#define CPPHTTPLIB_USE_POLL
#include "httplib.h"

using namespace lse;

#define LSE_EXPECT_STR(a, b)                                                   \
  do {                                                                         \
    const std::string _sa = (a), _sb = (b);                                   \
    if (_sa != _sb) ::lse::test::fail(__FILE__, __LINE__, _sa + " vs " + _sb); \
  } while (0)
using json = nlohmann::json;

namespace {

class ZeroMixer final : public model::IMixer {
 public:
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override {
    return OkStatus();
  }
  Result<graph::Array> forward(const graph::Array& x, model::MixerState*,
                               const model::LayerContext&) override {
    return graph::Array::zeros(x.shape(), DType::kF32);
  }
  std::string_view name() const noexcept override { return "fixture.zero_mixer"; }
};

class ZeroFfn final : public model::IFeedForward {
 public:
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override {
    return OkStatus();
  }
  Result<graph::Array> forward(const graph::Array& x, graph::Array*,
                               const model::LayerContext&) override {
    return graph::Array::zeros(x.shape(), DType::kF32);
  }
  std::string_view name() const noexcept override { return "fixture.zero_ffn"; }
};

// A character-level BPE vocabulary wide enough for a ChatML prompt.
std::vector<std::string> vocabulary() {
  std::vector<std::string> v;
  for (char c = 'a'; c <= 'z'; ++c) v.emplace_back(1, c);
  for (char c = 'A'; c <= 'Z'; ++c) v.emplace_back(1, c);
  for (const char* p : {" ", "\n", "<", ">", "/", "|", "_", ".", ",", "!", "?", ":", "-"})
    v.emplace_back(p);
  return v;
}

struct Fixture {
  // Set before build(): the context (0: the config default) and the model's
  // generation defaults (greedy unless a test says otherwise).
  std::int32_t kv_length = 0;
  models::SamplingDefaults sampling = [] {
    models::SamplingDefaults d;
    d.temperature = 0.0f;
    return d;
  }();
  std::string tag = "a";
  // Make <|endoftext|>'s embedding row three times 't''s, so wherever the
  // greedy answer was 't' (as after "hello there") the model ends its turn.
  bool eos_after_t = false;
  std::filesystem::path dir;
  std::optional<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  std::optional<tokenizer::Tokenizer> tok;

  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  Status build() {
    dir = std::filesystem::temp_directory_path() /
          ("lse-api-parity-" + tag + "-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);

    // Tokenizer: every character above, plus the ChatML specials.
    std::vector<std::string> vocab = vocabulary();
    json model_vocab = json::object();
    for (std::size_t i = 0; i < vocab.size(); ++i) model_vocab[vocab[i]] = i;
    json added = json::array();
    std::uint32_t next = static_cast<std::uint32_t>(vocab.size());
    for (const char* special : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
      added.push_back({{"id", next++}, {"content", special}, {"special", true},
                       {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                       {"normalized", false}});
    }
    const std::uint32_t vocab_size = 96;
    LSE_EXPECT(next <= vocab_size);
    {
      std::ofstream out(dir / "tokenizer.json");
      out << json{{"model", {{"type", "BPE"}, {"vocab", model_vocab}, {"merges", json::array()}}},
                  {"added_tokens", added}}
                 .dump();
    }
    auto t = tokenizer::Tokenizer::from_file((dir / "tokenizer.json").string());
    if (!t.ok()) return t.status();
    tok.emplace(t.release());

    // Model: an embedding (tied head), norms, and one block that adds nothing.
    json header;
    std::vector<float> data;
    const auto add = [&](const std::string& name, std::vector<std::int64_t> shape, bool embed) {
      const auto start = data.size() * 4;
      std::size_t count = 1;
      for (auto d : shape) count *= static_cast<std::size_t>(d);
      for (std::size_t i = 0; i < count; ++i)
        data.push_back(embed ? static_cast<float>(((i * 37) % 23) - 11) / 7.0f : 1.0f);
      header[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {start, data.size() * 4}}};
    };
    add("embed.weight", {vocab_size, 8}, true);
    if (eos_after_t) {
      const std::size_t eos = vocab.size() + 2, t = static_cast<std::size_t>('t' - 'a');
      for (std::size_t k = 0; k < 8; ++k) data[eos * 8 + k] = 3.0f * data[t * 8 + k];
    }
    add("final_norm.weight", {8}, false);
    add("blocks.0.norm1.weight", {8}, false);
    add("blocks.0.norm2.weight", {8}, false);
    auto text = header.dump();
    text.append((8 - text.size() % 8) % 8, ' ');
    {
      std::ofstream out(dir / "model.safetensors", std::ios::binary);
      const std::uint64_t bytes = text.size();
      out.write(reinterpret_cast<const char*>(&bytes), 8);
      out.write(text.data(), static_cast<std::streamsize>(text.size()));
      out.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * 4));
    }
    auto opened = model::SafeTensors::open((dir / "model.safetensors").string());
    if (!opened.ok()) return opened.status();
    weights.emplace(opened.release());

    model::Config config;
    config.hidden_size = 8;
    config.vocab_size = vocab_size;
    config.num_layers = 1;
    config.full_attention_interval = 1;
    config.sampling_defaults = sampling;
    config.kv_length = kv_length;
    model::HybridLMSpec spec;
    spec.zero_centered_norm = false;
    lm = std::make_unique<model::HybridLM>(
        config, spec, [](std::int32_t) -> Result<std::unique_ptr<model::HybridBlock>> {
          return std::make_unique<model::HybridBlock>(std::make_unique<ZeroMixer>(),
                                                      std::make_unique<ZeroFfn>(), false);
        });
    model::WeightBinder binder(*weights);
    return lm->load(binder);
  }
};

// What one request produced, as either transport delivers it.
struct Answer {
  int status = 0;
  std::vector<json> events;  // one body, or each streamed chunk
  bool done = false;         // the stream ended with [DONE]
};

// Sets aside what legitimately differs between two answers to one request.
json normalize(json j) {
  if (!j.is_object()) return j;
  j.erase("id");
  j.erase("created");
  j.erase("timings");
  if (j.contains("usage") && j["usage"].is_object()) j["usage"].erase("prompt_tokens_details");
  if (j.contains("choices") && j["choices"].is_array()) {
    for (auto& choice : j["choices"]) {
      if (choice.contains("message") && choice["message"].is_object() &&
          choice["message"].contains("tool_calls")) {
        for (auto& call : choice["message"]["tool_calls"]) call.erase("id");
      }
    }
  }
  return j;
}

Answer over_http(httplib::Client& client, const std::string& method, const std::string& path,
                 const std::string& body) {
  Answer a;
  httplib::Result r = method == "GET" ? client.Get(path)
                                      : client.Post(path, body, "application/json");
  LSE_EXPECT(static_cast<bool>(r));
  if (!r) return a;
  a.status = r->status;
  if (r->get_header_value("Content-Type").starts_with("text/event-stream")) {
    std::size_t at = 0;
    const std::string& s = r->body;
    while ((at = s.find("data: ", at)) != std::string::npos) {
      const std::size_t end = s.find("\n\n", at);
      const std::string payload = s.substr(at + 6, end - at - 6);
      if (payload == "[DONE]") a.done = true;
      else a.events.push_back(normalize(json::parse(payload)));
      at = end == std::string::npos ? s.size() : end + 2;
    }
  } else if (!r->body.empty()) {
    a.events.push_back(normalize(json::parse(r->body)));
  }
  return a;
}

Answer in_process(server::InProcess& requests, const std::string& method,
                  const std::string& path, const std::string& body) {
  Answer a;
  std::mutex m;
  std::condition_variable cv;
  bool finished = false;
  requests.submit(method, path, body,
                  [&](std::uint64_t, server::InProcess::Event event, int status,
                      const std::string* data) {
                    std::lock_guard held(m);
                    switch (event) {
                      case server::InProcess::Event::kChunk:
                        a.status = status;
                        a.events.push_back(normalize(json::parse(*data)));
                        return;
                      case server::InProcess::Event::kDone:
                        a.status = status;
                        a.done = true;
                        break;
                      case server::InProcess::Event::kResponse:
                      case server::InProcess::Event::kError:
                        a.status = status;
                        a.events.push_back(normalize(json::parse(*data)));
                        break;
                    }
                    finished = true;
                    cv.notify_all();
                  });
  std::unique_lock held(m);
  const bool ok = cv.wait_for(held, std::chrono::seconds(60), [&] { return finished; });
  LSE_EXPECT(ok);
  return a;
}

bool same(const Answer& http, const Answer& local, const std::string& what) {
  const bool equal = http.status == local.status && http.done == local.done &&
                     http.events == local.events;
  if (!equal) {
    std::fprintf(stderr, "%s differs\n  http  (%d, done=%d): %s\n  local (%d, done=%d): %s\n",
                 what.c_str(), http.status, http.done, json(http.events).dump().c_str(),
                 local.status, local.done, json(local.events).dump().c_str());
  }
  return equal;
}

}  // namespace

LSE_TEST(lse_request_and_http_answer_every_request_identically) {
  auto* scheduler = graph::default_scheduler();
  LSE_EXPECT(scheduler);
  if (!scheduler) return;
  scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);

  Fixture fx;
  const Status built = fx.build();
  LSE_EXPECT_OK(built);
  if (!built.ok()) return;

  server::ServerOptions options;
  options.model_id = "parity-fixture";
  options.max_tokens_cap = 64;
  // The fixture's directory, so the model-info routes answer about it. It is
  // no architecture this build loads: model_info says so, estimate refuses.
  options.model_path = fx.dir.string();
  {
    std::ofstream config(fx.dir / "config.json");
    config << R"({"vocab_size": 96, "hidden_size": 8, "num_layers": 1,
                 "full_attention_interval": 1, "global_attention_layers": []})";
  }
  server::Router router(*fx.lm, *fx.tok, options);
  server::InProcess requests(router);
  server::HttpServer http(router, "127.0.0.1", 0);
  LSE_EXPECT_OK(http.bind());
  std::thread listener([&] { (void)http.listen(); });
  httplib::Client client("127.0.0.1", http.port());
  client.set_read_timeout(60, 0);

  struct Case {
    const char* method;
    const char* path;
    std::string body;
  };
  const json greedy{{"temperature", 0}, {"max_tokens", 6}};
  auto with = [&](json extra) {
    json b = greedy;
    if (extra.is_object()) b.update(extra);
    return b.dump();
  };
  const std::vector<Case> cases = {
      {"GET", "/health", ""},
      {"GET", "/v1/models", ""},
      {"GET", "/v1/models/parity-fixture", ""},
      {"GET", "/v1/lse/model_info", ""},
      {"GET", "/v1/lse/estimate", ""},
      {"POST", "/v1/lse/estimate", R"({"kv_len": 256})"},
      {"POST", "/v1/lse/estimate", R"({"kv_cache_dtype": "int3"})"},
      {"POST", "/v1/completions", with({{"prompt", "hello there"}})},
      {"POST", "/v1/completions", with({{"prompt", "hello there"}, {"stream", true},
                                        {"stream_options", {{"include_usage", true}}}})},
      {"POST", "/v1/chat/completions",
       with({{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
             {"reasoning_effort", "none"}})},
      {"POST", "/v1/chat/completions",
       with({{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
             {"reasoning_effort", "none"}, {"stream", true}})},
      // Errors take the same envelope and status either way.
      {"POST", "/v1/completions", "{not json"},
      {"POST", "/v1/completions", with({{"max_tokens", 100000}, {"prompt", "x"}})},
      {"POST", "/v1/completions", with({})},
      {"POST", "/v1/embeddings", "{}"},
  };
  for (const Case& c : cases) {
    const std::string what = std::string(c.method) + " " + c.path + " " + c.body;
    const Answer h = over_http(client, c.method, c.path, c.body);
    const Answer l = in_process(requests, c.method, c.path, c.body);
    LSE_EXPECT(same(h, l, what));
    LSE_EXPECT(!h.events.empty() || h.done);
  }

  // /v1/models reports the settings the engine was opened with.
  {
    const Answer m = in_process(requests, "GET", "/v1/models", "");
    LSE_EXPECT_EQ(m.status, 200);
    if (!m.events.empty()) {
      const json& entry = m.events.front()["data"][0];
      LSE_EXPECT_EQ(entry["context_length"].get<int>(), fx.lm->config().kv_capacity());
      LSE_EXPECT_EQ(entry["kv_len"].get<int>(), fx.lm->config().kv_capacity());
      LSE_EXPECT_STR(entry["kv_cache_dtype"].get<std::string>(), std::string(kv::to_string(fx.lm->config().kv_cache_dtype)));
      LSE_EXPECT(entry["draft"].is_null());
    }
    const Answer info = in_process(requests, "GET", "/v1/lse/model_info", "");
    LSE_EXPECT_EQ(info.status, 200);
    if (!info.events.empty()) {
      LSE_EXPECT(!info.events.front()["loadable"].get<bool>());
      LSE_EXPECT(info.events.front()["served"].contains("context_length"));
    }
    const Answer bad = in_process(requests, "POST", "/v1/lse/estimate", R"({"kv_len": "x"})");
    LSE_EXPECT_EQ(bad.status, 400);
  }

  // A cancelled request ends with the cancellation error and nothing else.
  {
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> statuses;
    bool finished = false;
    const auto id = requests.submit(
        "POST", "/v1/completions", with({{"prompt", "hello there"}, {"max_tokens", 64}}),
        [&](std::uint64_t, server::InProcess::Event event, int status, const std::string*) {
          std::lock_guard held(m);
          statuses.push_back(status);
          if (event != server::InProcess::Event::kChunk) {
            finished = true;
            cv.notify_all();
          }
        });
    requests.cancel(id);
    std::unique_lock held(m);
    LSE_EXPECT(cv.wait_for(held, std::chrono::seconds(60), [&] { return finished; }));
    // Either it was cancelled in time or it had already completed.
    LSE_EXPECT(!statuses.empty() && (statuses.back() == 499 || statuses.back() == 200));
  }

  http.stop();
  listener.join();
  requests.shutdown();
}

namespace {

// One Router over a fixture, answering in process (the same JSON as HTTP).
struct Served {
  Fixture fx;
  std::unique_ptr<server::Router> router;
  std::unique_ptr<server::InProcess> requests;
  bool ok = false;

  explicit Served(std::int32_t kv_length, models::SamplingDefaults sampling, const char* tag,
                  models::ThinkingControls thinking = {}, bool eos_after_t = false) {
    auto* scheduler = graph::default_scheduler();
    if (scheduler == nullptr) return;
    scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);
    fx.kv_length = kv_length;
    fx.sampling = sampling;
    fx.tag = tag;
    fx.eos_after_t = eos_after_t;
    if (const Status built = fx.build(); !built.ok()) {
      std::fprintf(stderr, "fixture: %s\n", built.message().c_str());
      return;
    }
    server::ServerOptions options;
    options.model_id = "limits-fixture";
    options.thinking = std::move(thinking);
    router = std::make_unique<server::Router>(*fx.lm, *fx.tok, options);
    requests = std::make_unique<server::InProcess>(*router);
    ok = true;
  }
  ~Served() {
    if (requests) requests->shutdown();
  }
  Answer post(const std::string& path, const json& body) {
    return in_process(*requests, "POST", path, body.dump());
  }
};

models::SamplingDefaults greedy_defaults() {
  models::SamplingDefaults d;
  d.temperature = 0.0f;
  return d;
}

}  // namespace

LSE_TEST(no_output_cap_generation_runs_to_a_full_context_and_says_so) {
  Served s(64, greedy_defaults(), "ctx");
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  // No max_tokens anywhere: the zero model never ends its turn, so only the
  // context stops it.
  const Answer a = s.post("/v1/completions", json{{"prompt", "hello there"}});
  LSE_EXPECT_EQ(a.status, 200);
  if (a.events.empty()) return;
  const json& r = a.events.front();
  const int prompt = r["usage"]["prompt_tokens"].get<int>();
  LSE_EXPECT_STR(r["choices"][0]["finish_reason"].get<std::string>(), "length");
  LSE_EXPECT_STR(r["choices"][0]["stop_reason"].get<std::string>(), "context_full");
  LSE_EXPECT_EQ(r["usage"]["completion_tokens"].get<int>(), 64 - prompt);
  LSE_EXPECT_EQ(r["lse_context"]["tokens_used"].get<int>(), 64);
  LSE_EXPECT_EQ(r["lse_context"]["context_length"].get<int>(), 64);
  LSE_EXPECT_EQ(r["lse_context"]["tokens_remaining"].get<int>(), 0);

  // Streamed: the final chunk carries the same.
  const Answer st = s.post("/v1/chat/completions",
                           json{{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
                                {"stream", true}});
  LSE_EXPECT_EQ(st.status, 200);
  LSE_EXPECT(st.done);
  bool seen = false;
  for (const json& chunk : st.events) {
    if (!chunk.contains("lse_context")) continue;
    seen = true;
    LSE_EXPECT_STR(chunk["choices"][0]["finish_reason"].get<std::string>(), "length");
    LSE_EXPECT_STR(chunk["choices"][0]["stop_reason"].get<std::string>(), "context_full");
    LSE_EXPECT_EQ(chunk["lse_context"]["tokens_used"].get<int>(), 64);
  }
  LSE_EXPECT(seen);

  // An explicit max_tokens is honoured.
  const Answer capped = s.post("/v1/completions", json{{"prompt", "hello there"}, {"max_tokens", 5}});
  LSE_EXPECT_EQ(capped.status, 200);
  if (!capped.events.empty()) {
    const json& c = capped.events.front();
    LSE_EXPECT_EQ(c["usage"]["completion_tokens"].get<int>(), 5);
    LSE_EXPECT_STR(c["choices"][0]["finish_reason"].get<std::string>(), "length");
    LSE_EXPECT_STR(c["choices"][0]["stop_reason"].get<std::string>(), "max_tokens");
    LSE_EXPECT_EQ(c["lse_context"]["tokens_used"].get<int>(), prompt + 5);
  }
  // max_completion_tokens is the same field; nonsense is refused.
  const Answer mct = s.post("/v1/completions", json{{"prompt", "hello there"}, {"max_completion_tokens", 3}});
  LSE_EXPECT(!mct.events.empty() && mct.events.front()["usage"]["completion_tokens"] == 3);
  for (const json& bad : {json(0), json(-4), json(2.5), json("12")}) {
    const Answer b = s.post("/v1/completions", json{{"prompt", "x"}, {"max_tokens", bad}});
    LSE_EXPECT_EQ(b.status, 400);
  }

  // /v1/models: no cap, the defaults and where they came from, no thinking.
  const Answer m = in_process(*s.requests, "GET", "/v1/models", "");
  if (!m.events.empty()) {
    const json& e = m.events.front()["data"][0];
    LSE_EXPECT(e["max_tokens"].is_null());
    LSE_EXPECT_EQ(e["context_length"].get<int>(), 64);
    LSE_EXPECT(e["generation_defaults"]["max_new_tokens"].is_null());
    LSE_EXPECT(e["generation_defaults"]["sources"]["top_k"] == "lse_default");
    LSE_EXPECT(!e["thinking"]["supported"].get<bool>());
    LSE_EXPECT(e["thinking"]["levels"].empty());
  }
}

LSE_TEST(generation_ends_cleanly_at_a_stop_token_without_a_cap) {
  Served s(4096, greedy_defaults(), "eos", {}, /*eos_after_t=*/true);
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  // After "hello there" this model's greedy answer is <|endoftext|>.
  const Answer a = s.post("/v1/completions", json{{"prompt", "hello there"}});
  LSE_EXPECT_EQ(a.status, 200);
  if (a.events.empty()) return;
  const json& r = a.events.front();
  LSE_EXPECT_STR(r["choices"][0]["finish_reason"].get<std::string>(), "stop");
  LSE_EXPECT_STR(r["choices"][0]["stop_reason"].get<std::string>(), "stop_token");
  LSE_EXPECT(r["lse_context"]["tokens_used"].get<int>() < 4096);
  // A stop sequence is reported as one.
  // Without that row the model never ends its turn; a stop sequence taken
  // from its own greedy text ends it there.
  Served plain(4096, greedy_defaults(), "seq");
  const Answer probe = plain.post("/v1/completions", json{{"prompt", "hello there"}, {"max_tokens", 8}});
  const std::string greedy =
      probe.events.empty() ? std::string() : probe.events.front()["choices"][0]["text"].get<std::string>();
  LSE_EXPECT(greedy.size() >= 4);
  const Answer seq = plain.post("/v1/completions",
                                json{{"prompt", "hello there"}, {"stop", greedy.substr(2, 2)}});
  if (!seq.events.empty()) {
    LSE_EXPECT_STR(seq.events.front()["choices"][0]["finish_reason"].get<std::string>(), "stop");
    LSE_EXPECT_STR(seq.events.front()["choices"][0]["stop_reason"].get<std::string>(), "stop_sequence");
  }
}

LSE_TEST(a_prompt_that_fills_the_context_is_refused_with_context_full) {
  Served s(16, greedy_defaults(), "over");
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  for (const char* prompt : {"abcdefghijklmnop", "abcdefghijklmnopqrstuvwxyz"}) {
    const Answer a = s.post("/v1/completions", json{{"prompt", prompt}});
    LSE_EXPECT_EQ(a.status, 400);
    if (a.events.empty()) continue;
    const json& e = a.events.front()["error"];
    LSE_EXPECT_STR(e["code"].get<std::string>(), "context_full");
    LSE_EXPECT_STR(e["type"].get<std::string>(), "invalid_request_error");
    LSE_EXPECT_EQ(e["lse_context"]["context_length"].get<int>(), 16);
    LSE_EXPECT_EQ(e["lse_context"]["tokens_used"].get<int>(), static_cast<int>(std::strlen(prompt)));
  }
  const Answer chat = s.post("/v1/chat/completions",
                             json{{"messages", json::array({{{"role", "user"}, {"content", "a long message"}}})}});
  LSE_EXPECT_EQ(chat.status, 400);
  if (!chat.events.empty()) {
    LSE_EXPECT_STR(chat.events.front()["error"]["code"].get<std::string>(), "context_full");
    LSE_EXPECT_STR(chat.events.front()["error"]["param"].get<std::string>(), "messages");
  }
}

LSE_TEST(top_k_and_model_defaults_reach_the_sampler) {
  // The model's defaults: hot sampling held to the single best token.
  models::SamplingDefaults d;
  d.temperature = 1.5f;
  d.top_k = 1;
  d.sources["temperature"] = "generation_config.json";
  d.sources["top_k"] = "generation_config.json";
  Served s(256, d, "topk");
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  const auto text = [&](json body) {
    body["prompt"] = "hello there";
    body["max_tokens"] = 24;
    const Answer a = s.post("/v1/completions", body);
    LSE_EXPECT_EQ(a.status, 200);
    return a.events.empty() ? std::string() : a.events.front()["choices"][0]["text"].get<std::string>();
  };
  const std::string greedy = text(json{{"temperature", 0}});
  // Defaults applied: top_k 1 from the model makes every seed greedy.
  for (int seed = 1; seed <= 4; ++seed) LSE_EXPECT_STR(text(json{{"seed", seed}}), greedy);
  // top_k in the request is honoured the same way...
  for (int seed = 1; seed <= 4; ++seed)
    LSE_EXPECT_STR(text(json{{"seed", seed}, {"top_k", 1}, {"temperature", 1.5}}), greedy);
  // ...and overriding the model's default (top_k 0: off) lets the hot
  // distribution through, so seeds disagree.
  std::set<std::string> outputs;
  for (int seed = 1; seed <= 6; ++seed) outputs.insert(text(json{{"seed", seed}, {"top_k", 0}}));
  LSE_EXPECT(outputs.size() > 1);
  // The chat route takes top_k too.
  const Answer chat = s.post("/v1/chat/completions",
                             json{{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
                                  {"top_k", 1}, {"max_tokens", 4}});
  LSE_EXPECT_EQ(chat.status, 200);
  // And the model's defaults are reported with their source.
  const Answer m = in_process(*s.requests, "GET", "/v1/models", "");
  if (!m.events.empty()) {
    const json& g = m.events.front()["data"][0]["generation_defaults"];
    LSE_EXPECT_EQ(g["top_k"].get<int>(), 1);
    LSE_EXPECT(g["sources"]["top_k"] == "generation_config.json");
    LSE_EXPECT(g["sources"]["top_p"] == "lse_default");
  }
}

LSE_TEST(a_model_output_limit_applies_only_when_its_files_set_one) {
  models::SamplingDefaults d = greedy_defaults();
  d.max_new_tokens = 7;
  d.sources["max_new_tokens"] = "generation_config.json";
  Served s(128, d, "mnt");
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  const Answer a = s.post("/v1/completions", json{{"prompt", "hello"}});
  LSE_EXPECT(!a.events.empty() && a.events.front()["usage"]["completion_tokens"] == 7);
  LSE_EXPECT(!a.events.empty() && a.events.front()["choices"][0]["stop_reason"] == "max_tokens");
  // The client's own max_tokens outranks it.
  const Answer more = s.post("/v1/completions", json{{"prompt", "hello"}, {"max_tokens", 20}});
  LSE_EXPECT(!more.events.empty() && more.events.front()["usage"]["completion_tokens"] == 20);
}

LSE_TEST(thinking_levels_come_from_the_template_and_undefined_ones_are_refused) {
  auto controls = models::load_thinking_controls("tests/fixtures/chat_templates/qwen3.8");
  LSE_EXPECT(controls.ok());
  if (!controls.ok()) return;
  Served s(512, greedy_defaults(), "think", controls.release());
  LSE_EXPECT(s.ok);
  if (!s.ok) return;
  const json messages = json::array({{{"role", "user"}, {"content", "Hi"}}});
  for (const char* level : {"none", "low", "medium", "xhigh"}) {
    const Answer a = s.post("/v1/chat/completions",
                            json{{"messages", messages}, {"reasoning_effort", level}, {"max_tokens", 3}});
    LSE_EXPECT_EQ(a.status, 200);
  }
  const Answer high = s.post("/v1/chat/completions",
                             json{{"messages", messages}, {"reasoning_effort", "high"}, {"max_tokens", 3}});
  LSE_EXPECT_EQ(high.status, 400);
  if (!high.events.empty()) {
    const json& e = high.events.front()["error"];
    LSE_EXPECT_STR(e["code"].get<std::string>(), "unsupported_reasoning_effort");
    LSE_EXPECT_STR(e["param"].get<std::string>(), "reasoning_effort");
    LSE_EXPECT(e["levels"] == json::array({"none", "xhigh", "medium", "low"}));
  }
  const Answer m = in_process(*s.requests, "GET", "/v1/models", "");
  if (!m.events.empty()) {
    const json& t = m.events.front()["data"][0]["thinking"];
    LSE_EXPECT(t["supported"].get<bool>());
    LSE_EXPECT_STR(t["default_level"].get<std::string>(), "xhigh");
    LSE_EXPECT_STR(t["toggle"].get<std::string>(), "enable_thinking");
    LSE_EXPECT_EQ(t["levels"].size(), std::size_t{4});
  }
}

LSE_TEST_MAIN()
