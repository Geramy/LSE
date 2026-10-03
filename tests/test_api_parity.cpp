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
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/graph/graph.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/weights.hpp"
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
  std::filesystem::path dir;
  std::optional<model::SafeTensors> weights;
  std::unique_ptr<model::HybridLM> lm;
  std::optional<tokenizer::Tokenizer> tok;

  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  Status build() {
    dir = std::filesystem::temp_directory_path() / ("lse-api-parity-" + std::to_string(getpid()));
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
    config.sampling_defaults.temperature = 0.0f;
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

LSE_TEST_MAIN()
