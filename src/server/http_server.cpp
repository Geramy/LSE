#include "lse/server/http_server.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

// httplib falls back to select() without this, and select() refuses any
// socket whose descriptor is >= FD_SETSIZE (1024). A loaded model holds
// well over a thousand descriptors, so the listening socket and every
// connection land above that line and are closed without a reply. poll()
// has no such ceiling.
#define CPPHTTPLIB_USE_POLL
#include "httplib.h"
#include "lse/runtime/generator.hpp"
#include "lse/server/chat.hpp"
#include "nlohmann/json.hpp"

namespace lse::server {
namespace {

using json = nlohmann::json;

std::int64_t now_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// OpenAI ids are opaque; these only have to be unique within a process.
std::string make_id(const char* prefix) {
  static std::atomic<std::uint64_t> counter{0};
  std::ostringstream s;
  s << prefix << '-' << now_seconds() << std::hex
    << counter.fetch_add(1, std::memory_order_relaxed);
  return s.str();
}

// The error envelope clients parse. Anything that leaves this server as a
// non-2xx wears it, so a caller never has to guess between our shape and
// theirs.
void send_error(httplib::Response& res, int status, const std::string& message,
                const std::string& type = "invalid_request_error",
                const std::string& param = "") {
  json body{{"error",
             {{"message", message},
              {"type", type},
              {"code", nullptr},
              {"param", param.empty() ? json(nullptr) : json(param)}}}};
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

// A JSON member that may be absent or null, which a client is entitled to send
// for any optional field.
template <class T>
T get_or(const json& j, const char* key, T fallback) {
  if (!j.contains(key) || j.at(key).is_null()) return fallback;
  try {
    return j.at(key).get<T>();
  } catch (const json::exception&) {
    return fallback;
  }
}

// "stop" is a string or an array of them.
std::vector<std::string> get_stop_strings(const json& j) {
  std::vector<std::string> out;
  if (!j.contains("stop") || j.at("stop").is_null()) return out;
  const json& s = j.at("stop");
  if (s.is_string()) {
    out.push_back(s.get<std::string>());
  } else if (s.is_array()) {
    for (const json& e : s) {
      if (e.is_string()) out.push_back(e.get<std::string>());
    }
  }
  return out;
}

struct Request {
  std::vector<std::uint32_t> prompt;
  runtime::SamplingParams sampling;
  runtime::GenerationLimits limits;
  std::vector<std::string> stop_strings;
  bool stream = false;
  std::string model;
};

// Where a completion stopped, in OpenAI's vocabulary.
const char* finish_reason(bool hit_limit) { return hit_limit ? "length" : "stop"; }

}  // namespace

struct HttpServer::Impl {
  model::HybridLM& model;
  tokenizer::Tokenizer& tok;
  model::MtpModule* mtp = nullptr;
  ServerOptions opt;
  httplib::Server http;
  // One model on one device. Two decodes at once would interleave on the same
  // KV pool, so requests queue here instead.
  std::mutex generate_lock;
  std::vector<std::uint32_t> stop_ids;

  Impl(model::HybridLM& m, tokenizer::Tokenizer& t, ServerOptions o)
      : model(m), tok(t), opt(std::move(o)) {
    stop_ids = chat_stop_tokens(tok);
  }

  [[nodiscard]] bool authorized(const httplib::Request& req) const {
    if (opt.api_key.empty()) return true;
    auto it = req.headers.find("Authorization");
    return it != req.headers.end() &&
           it->second == "Bearer " + opt.api_key;
  }

  // Parses the parts /v1/chat/completions and /v1/completions share.
  Result<Request> parse_common(const json& body, httplib::Response& res) {
    Request r;
    r.model = get_or<std::string>(body, "model", opt.model_id);
    r.stream = get_or<bool>(body, "stream", false);
    r.stop_strings = get_stop_strings(body);

    // OpenAI's default temperature is 1.0, not this engine's 0.8: a client
    // that sends nothing must get what the API promises, not what the CLI does.
    r.sampling.temperature = get_or<float>(body, "temperature", 1.0f);
    r.sampling.top_p = get_or<float>(body, "top_p", 1.0f);
    r.sampling.top_k = get_or<std::int32_t>(body, "top_k", 0);
    r.sampling.seed = get_or<std::uint64_t>(body, "seed", 0);
    // frequency_penalty is additive in OpenAI and multiplicative here, so it
    // is mapped rather than passed: 0 means off on both sides.
    const float freq = get_or<float>(body, "frequency_penalty", 0.0f);
    r.sampling.repetition_penalty = freq > 0.0f ? 1.0f + freq : 1.0f;

    std::int32_t want = get_or<std::int32_t>(body, "max_tokens", 0);
    if (want == 0) want = get_or<std::int32_t>(body, "max_completion_tokens", 256);
    if (want <= 0) {
      send_error(res, 400, "max_tokens must be positive", "invalid_request_error",
                 "max_tokens");
      return LSE_ERROR(kInvalidArgument, "max_tokens");
    }
    if (want > opt.max_tokens_cap) {
      send_error(res, 400,
                 "max_tokens " + std::to_string(want) + " exceeds this server's cap of " +
                     std::to_string(opt.max_tokens_cap),
                 "invalid_request_error", "max_tokens");
      return LSE_ERROR(kInvalidArgument, "max_tokens cap");
    }
    r.limits.max_tokens = want;
    r.limits.stop_tokens = stop_ids;

    const std::int32_t n = get_or<std::int32_t>(body, "n", 1);
    if (n != 1) {
      send_error(res, 400, "n must be 1; this server returns one choice", "invalid_request_error", "n");
      return LSE_ERROR(kInvalidArgument, "n");
    }
    return r;
  }
};

namespace {

// One generation, streamed or not. `emit` receives each delta as it resolves
// and returns false to abandon the stream, which is what a disconnected client
// looks like from in here.
struct Outcome {
  std::string text;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  bool hit_limit = false;
};

}  // namespace

struct HttpServer::Run {
  // Runs `r` under the generate lock. `emit` is called on the calling thread,
  // so a streaming handler writes to its own socket and nothing is shared.
  static Result<Outcome> generate(
      HttpServer::Impl& impl, const Request& r,
      const std::function<bool(const std::string&)>& emit) {
    std::lock_guard<std::mutex> held(impl.generate_lock);

    runtime::Generator gen(impl.model, r.sampling);
    if (impl.mtp != nullptr) gen.use_mtp(*impl.mtp);

    tokenizer::DecodeStream stream(impl.tok);
    Outcome out;
    out.prompt_tokens = static_cast<int>(r.prompt.size());
    bool stopped_by_string = false;

    auto on_token = [&](std::uint32_t id) -> bool {
      auto piece = stream.push(id);
      if (!piece.ok()) return false;
      if (piece->empty()) return true;   // mid-character, not a boundary yet
      out.text += *piece;
      // A stop string is honoured on the decoded text, since it need not fall
      // on a token boundary. The text up to it is kept, the rest is not.
      for (const std::string& stop : r.stop_strings) {
        const std::size_t at = out.text.find(stop);
        if (at != std::string::npos) {
          const std::string keep = out.text.substr(0, at);
          if (emit && keep.size() > (out.text.size() - piece->size())) {
            if (!emit(keep.substr(out.text.size() - piece->size()))) return false;
          }
          out.text = keep;
          stopped_by_string = true;
          return false;
        }
      }
      if (emit && !emit(*piece)) return false;
      return true;
    };

    auto ids = gen.generate(r.prompt, r.limits, on_token);
    if (!ids.ok()) return ids.status();
    out.completion_tokens = static_cast<int>(ids->size());
    out.hit_limit = !stopped_by_string &&
                    out.completion_tokens >= r.limits.max_tokens;
    return out;
  }
};

namespace {

json usage_of(const Outcome& o) {
  return json{{"prompt_tokens", o.prompt_tokens},
              {"completion_tokens", o.completion_tokens},
              {"total_tokens", o.prompt_tokens + o.completion_tokens}};
}

// text_completion and chat.completion differ only in the shape of a choice.
json chat_choice(const std::string& text, bool hit_limit) {
  return json{{"index", 0},
              {"message", {{"role", "assistant"}, {"content", text}}},
              {"logprobs", nullptr},
              {"finish_reason", finish_reason(hit_limit)}};
}

json text_choice(const std::string& text, bool hit_limit) {
  return json{{"index", 0},
              {"text", text},
              {"logprobs", nullptr},
              {"finish_reason", finish_reason(hit_limit)}};
}

}  // namespace

HttpServer::HttpServer(model::HybridLM& model, tokenizer::Tokenizer& tok,
                       ServerOptions options)
    : impl_(std::make_unique<Impl>(model, tok, std::move(options))) {}

HttpServer::~HttpServer() = default;

void HttpServer::use_mtp(model::MtpModule& mtp) noexcept { impl_->mtp = &mtp; }

void HttpServer::stop() { impl_->http.stop(); }

Status HttpServer::listen() {
  Impl& impl = *impl_;

  impl.http.set_exception_handler(
      [](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
        send_error(res, 500, "internal error", "server_error");
      });

  // Any client the user points at this is entitled to ask from a browser.
  impl.http.set_pre_routing_handler(
      [&impl](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        if (req.method == "OPTIONS") {
          res.status = 204;
          return httplib::Server::HandlerResponse::Handled;
        }
        if (!impl.authorized(req)) {
          send_error(res, 401, "missing or invalid Authorization header",
                     "invalid_request_error");
          return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
      });

  impl.http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}}.dump(), "application/json");
  });

  auto list_models = [&impl](const httplib::Request&, httplib::Response& res) {
    json m{{"id", impl.opt.model_id},
           {"object", "model"},
           {"created", now_seconds()},
           {"owned_by", "lse"}};
    res.set_content(
        json{{"object", "list"}, {"data", json::array({m})}}.dump(),
        "application/json");
  };
  impl.http.Get("/v1/models", list_models);
  impl.http.Get("/v1/models/:id", [&impl](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"id", impl.opt.model_id},
                         {"object", "model"},
                         {"created", now_seconds()},
                         {"owned_by", "lse"}}
                        .dump(),
                    "application/json");
  });

  // The two completion routes differ in how the prompt is built and how a
  // choice is shaped; everything after that is shared.
  auto completion_route = [&impl](bool chat) {
    return [&impl, chat](const httplib::Request& req, httplib::Response& res) {
      json body;
      try {
        body = json::parse(req.body);
      } catch (const json::exception& e) {
        send_error(res, 400, std::string("malformed JSON: ") + e.what());
        return;
      }

      auto parsed = impl.parse_common(body, res);
      if (!parsed.ok()) return;  // parse_common already answered
      Request r = *parsed;

      std::string prompt_text;
      if (chat) {
        if (!body.contains("messages") || !body.at("messages").is_array() ||
            body.at("messages").empty()) {
          send_error(res, 400, "messages must be a non-empty array",
                     "invalid_request_error", "messages");
          return;
        }
        std::vector<ChatMessage> messages;
        for (const json& m : body.at("messages")) {
          ChatMessage cm;
          cm.role = get_or<std::string>(m, "role", "user");
          // content is a string, or the array-of-parts form; the text parts
          // are joined and anything else is refused rather than dropped.
          if (m.contains("content") && m.at("content").is_array()) {
            for (const json& part : m.at("content")) {
              const std::string type = get_or<std::string>(part, "type", "");
              if (type != "text") {
                send_error(res, 400,
                           "content part of type '" + type +
                               "' is not supported; this build decodes text only",
                           "invalid_request_error", "messages");
                return;
              }
              cm.content += get_or<std::string>(part, "text", "");
            }
          } else {
            cm.content = get_or<std::string>(m, "content", "");
          }
          messages.push_back(std::move(cm));
        }
        prompt_text = render_chatml(messages);
      } else {
        if (!body.contains("prompt")) {
          send_error(res, 400, "prompt is required", "invalid_request_error", "prompt");
          return;
        }
        const json& p = body.at("prompt");
        if (p.is_string()) {
          prompt_text = p.get<std::string>();
        } else if (p.is_array() && p.size() == 1 && p[0].is_string()) {
          prompt_text = p[0].get<std::string>();
        } else {
          send_error(res, 400,
                     "prompt must be a string or a one-element array of strings",
                     "invalid_request_error", "prompt");
          return;
        }
      }

      auto encoded = impl.tok.encode(prompt_text);
      if (!encoded.ok()) {
        send_error(res, 400, "the prompt could not be tokenized", "invalid_request_error", "prompt");
        return;
      }
      if (encoded->empty()) {
        send_error(res, 400, "the prompt encoded to no tokens", "invalid_request_error", "prompt");
        return;
      }
      r.prompt = *encoded;

      const std::string id = make_id(chat ? "chatcmpl" : "cmpl");
      const std::int64_t created = now_seconds();
      const char* object = chat ? "chat.completion" : "text_completion";

      if (!r.stream) {
        auto out = HttpServer::Run::generate(impl, r, {});
        if (!out.ok()) {
          send_error(res, 500, std::string(out.status().message()), "server_error");
          return;
        }
        json resp{{"id", id},
                  {"object", object},
                  {"created", created},
                  {"model", impl.opt.model_id},
                  {"choices", json::array({chat ? chat_choice(out->text, out->hit_limit)
                                                : text_choice(out->text, out->hit_limit)})},
                  {"usage", usage_of(*out)}};
        res.set_content(resp.dump(), "application/json");
        return;
      }

      // Server-sent events. The generation runs inside the provider so a
      // delta reaches the socket as it resolves rather than at the end.
      const std::string chunk_object =
          chat ? "chat.completion.chunk" : "text_completion";
      res.set_chunked_content_provider(
          "text/event-stream",
          [&impl, r, id, created, chat, chunk_object](std::size_t,
                                                      httplib::DataSink& sink) {
            auto send = [&sink](const json& j) {
              const std::string frame = "data: " + j.dump() + "\n\n";
              return sink.write(frame.data(), frame.size());
            };

            if (chat) {
              // The role arrives in its own first chunk, which is what clients
              // key on to open an assistant message.
              send(json{{"id", id},
                        {"object", chunk_object},
                        {"created", created},
                        {"model", impl.opt.model_id},
                        {"choices", json::array({{{"index", 0},
                                                  {"delta", {{"role", "assistant"}}},
                                                  {"finish_reason", nullptr}}})}});
            }

            auto emit = [&](const std::string& piece) {
              json choice = chat
                  ? json{{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}
                  : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
              return send(json{{"id", id},
                               {"object", chunk_object},
                               {"created", created},
                               {"model", impl.opt.model_id},
                               {"choices", json::array({choice})}});
            };

            auto out = HttpServer::Run::generate(impl, r, emit);
            if (!out.ok()) {
              // The status line is long gone, so the error rides the stream.
              send(json{{"error",
                         {{"message", std::string(out.status().message())},
                          {"type", "server_error"}}}});
              sink.done();
              return true;
            }

            json last = chat
                ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason(out->hit_limit)}}
                : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason(out->hit_limit)}};
            send(json{{"id", id},
                      {"object", chunk_object},
                      {"created", created},
                      {"model", impl.opt.model_id},
                      {"choices", json::array({last})},
                      {"usage", usage_of(*out)}});
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
          });
    };
  };

  impl.http.Post("/v1/chat/completions", completion_route(true));
  impl.http.Post("/v1/completions", completion_route(false));

  // Named so a client gets a straight answer instead of a 404 it has to guess at.
  for (const char* path : {"/v1/embeddings", "/v1/images/generations",
                           "/v1/audio/speech", "/v1/audio/transcriptions",
                           "/v1/moderations", "/v1/responses"}) {
    impl.http.Post(path, [path](const httplib::Request&, httplib::Response& res) {
      send_error(res, 501, std::string(path) + " is not implemented by this server",
                 "not_implemented");
    });
  }

  if (!impl.http.listen(impl.opt.host, impl.opt.port)) {
    return LSE_ERROR(kIoError, "could not listen on ", impl.opt.host, ":",
                     std::to_string(impl.opt.port));
  }
  return OkStatus();
}

}  // namespace lse::server
