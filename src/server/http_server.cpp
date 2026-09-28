#include "lse/server/http_server.hpp"
#include "lse/server/shutdown.hpp"
#include "jit_timings.hpp"
#include "chat_protocol.hpp"
#include "request_sampling.hpp"

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
  detail::ChatRequest chat;
  bool include_usage = false;
  bool thinking_enabled = true;
  std::string thinking_level;   // "" | low | medium | high | xhigh
};

// Where a completion stopped, in OpenAI's vocabulary.
const char* finish_reason(bool hit_limit) { return hit_limit ? "length" : "stop"; }

}  // namespace

struct HttpServer::Impl {
  model::HybridLM& model;
  tokenizer::Tokenizer& tok;
  model::MtpModule* mtp = nullptr;
  model::DFlash2Module* dflash2 = nullptr;
  ServerOptions opt;
  httplib::Server http;
  // One model on one device. Two decodes at once would interleave on the same
  // KV pool, so requests queue here instead.
  std::mutex generate_lock;
  std::vector<std::uint32_t> stop_ids;
  // Set once the process is shutting down. A decode in flight does not watch
  // the listening socket, so without this it holds the worker pool open for as
  // long as its completion takes and the server outlives the ^C that asked it
  // to stop.
  std::atomic<bool> stopping{false};
  std::atomic<bool> listener_stop_issued{false};
  // One resident session, restarted between requests instead of rebuilt: the
  // state arrays keep their nodes, so the model's retained program replays
  // and a warm request skips the per-request partition and emit entirely.
  // Guarded by generate_lock like everything else that touches the device.
  runtime::Session session{"resident", 0};
  bool session_live = false;

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
    if (!body.is_object()) {
      send_error(res, 400, "request must be a JSON object");
      return LSE_ERROR(kInvalidArgument, "request");
    }
    if (body.contains("stream_options") && body["stream_options"].is_object())
      r.include_usage = get_or<bool>(body["stream_options"], "include_usage", false);

    auto sampling = detail::request_sampling(body, model.config().sampling_defaults);
    if (!sampling.ok()) {
      send_error(res, 400, std::string(sampling.status().message()),
                 "invalid_request_error", "sampling");
      return sampling.status();
    }
    r.sampling = *sampling;

    // Thinking control. Accepts the OpenAI-style `reasoning_effort` ("none",
    // "low", "medium", "high", "xhigh") and a direct `thinking` (bool or
    // {type}); a client that sends neither keeps the model's default
    // (reasoning on at its own level). The level steers how hard the model
    // thinks via a system prompt (see reasoning_effort_instructions).
    r.thinking_enabled = true;
    for (const char* field : {"enable_thinking", "thinking"}) {
      if (body.contains(field) && body[field].is_boolean())
        r.thinking_enabled = body[field].get<bool>();
    }
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
      r.thinking_enabled = get_or<bool>(body["chat_template_kwargs"], "enable_thinking", r.thinking_enabled);
    if (body.contains("thinking")) {
      const json& t = body.at("thinking");
      if (t.is_boolean()) {
        r.thinking_enabled = t.get<bool>();
      } else if (t.is_object()) {
        r.thinking_enabled = get_or<std::string>(t, "type", "enabled") != "disabled";
      }
    }
    if (body.contains("reasoning_effort")) {
      const std::string effort = get_or<std::string>(body, "reasoning_effort", "");
      if (effort == "none") {
        r.thinking_enabled = false;
      } else if (!effort.empty()) {
        if (effort != "minimal" && effort != "low" && effort != "medium" &&
            effort != "high" && effort != "xhigh") {
          send_error(res, 400, "unsupported reasoning_effort", "invalid_request_error", "reasoning_effort");
          return LSE_ERROR(kInvalidArgument, "reasoning_effort");
        }
        r.thinking_enabled = true;
        r.thinking_level = effort == "minimal" ? "low" : effort;
      }
    }
    if (body.contains("thinking_level")) {
      const std::string lvl = get_or<std::string>(body, "thinking_level", "");
      if (lvl != "none" && !lvl.empty()) r.thinking_level = lvl;
    }

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
    r.limits.mtp_depth = opt.mtp_depth;
    if (body.contains("mtp_depth")) {
      const auto& depth = body.at("mtp_depth");
      if (!depth.is_number_integer() || depth < 1 || depth > runtime::kMaxMtpDepth) {
        send_error(res, 400, "mtp_depth must be an integer from 1 to 7",
                   "invalid_request_error", "mtp_depth");
        return LSE_ERROR(kInvalidArgument, "mtp_depth");
      }
      r.limits.mtp_depth = depth.get<std::uint32_t>();
    }
    if (!runtime::valid_mtp_depth(r.limits.mtp_depth)) {
      send_error(res, 400, "mtp_depth must be an integer from 1 to 7",
                 "invalid_request_error", "mtp_depth");
      return LSE_ERROR(kInvalidArgument, "mtp_depth");
    }

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
  int prefill_tokens = 0;
  int decode_tokens = 0;
  bool hit_limit = false;
  // Prefill and decode are different rates and a single figure hides which one
  // is the problem, so both are reported. Speculation moves the decode rate
  // without moving the pass count, which is why acceptance rides along.
  double prompt_per_second = 0.0;
  double decode_per_second = 0.0;
  std::uint64_t prefill_ns = 0;
  std::uint64_t decode_ns = 0;
  detail::JitTotals jit;
  std::uint32_t device_groups = 0, host_groups = 0, host_fallbacks = 0;
  std::uint32_t mtp_depth = 0;
  std::uint32_t dflash2_depth = 0;
  std::uint32_t spec_steps = 0, spec_tested = 0, spec_accepted = 0;
  std::uint64_t spec_draft_ns = 0, spec_verify_ns = 0;
  double acceptance = -1.0;  // negative when nothing was speculated
};

}  // namespace

struct HttpServer::Run {
  // Runs `r` under the generate lock. `emit` is called on the calling thread,
  // so a streaming handler writes to its own socket and nothing is shared.
  static Result<Outcome> generate(
      HttpServer::Impl& impl, const Request& r,
      const std::function<bool(const std::string&)>& emit) {
    std::lock_guard<std::mutex> held(impl.generate_lock);
    // A request queued behind another generation must not start new device
    // work after shutdown has asked the active generation to stop.
    if (impl.stopping.load()) return LSE_ERROR(kCancelled, "server is stopping");

    runtime::Generator gen(impl.model, r.sampling);
    if (impl.mtp != nullptr) gen.use_mtp(*impl.mtp);
    if (impl.dflash2 != nullptr) gen.use_dflash2(*impl.dflash2);

    // Generator checks the exact token prefix before reusing resident state.
    bool resident = true;
    if (!impl.session_live) {
      impl.session = runtime::Session{"resident", impl.model.state_slots()};
      impl.session_live = true;
    } else if (impl.mtp != nullptr) {
      resident = impl.session.restart().ok();
    }

    tokenizer::DecodeStream stream(impl.tok);
    Outcome out;
    out.prompt_tokens = static_cast<int>(r.prompt.size());
    bool stopped_by_string = false;

    auto on_token = [&](std::uint32_t id) -> bool {
      if (impl.stopping.load(std::memory_order_relaxed)) return false;
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

    auto ids = resident
                   ? gen.generate(impl.session, r.prompt, r.limits, on_token)
                   : gen.generate(r.prompt, r.limits, on_token);
    if (!ids.ok()) {
      impl.session.clear();
      impl.session_live = false;
      if (impl.mtp != nullptr) impl.mtp->reset();
      if (impl.dflash2 != nullptr) impl.dflash2->reset();
      return ids.status();
    }
    out.completion_tokens = static_cast<int>(ids->size());
    out.hit_limit = !stopped_by_string &&
                    out.completion_tokens >= r.limits.max_tokens;

    const runtime::GenerationStats& st = gen.stats();
    out.jit = detail::JitTotals::from(st);
    out.device_groups = st.device_groups;
    out.host_groups = st.host_groups;
    out.host_fallbacks = st.host_fallbacks;
    out.mtp_depth = st.mtp_depth;
    out.dflash2_depth = st.dflash2_depth;
    out.spec_steps = st.spec_steps;
    out.spec_tested = st.spec_tested;
    out.spec_accepted = st.spec_accepted;
    out.spec_draft_ns = st.spec_draft_ns;
    out.spec_verify_ns = st.spec_verify_ns;
    out.prefill_tokens = st.prompt_tokens;
    out.decode_tokens = st.decoded_tokens();
    out.prefill_ns = st.prefill_ns;
    out.decode_ns = st.decode_ns;
    out.decode_per_second = st.decode_tokens_per_second();
    out.prompt_per_second = st.prompt_tokens_per_second();
    if (st.spec_steps != 0) out.acceptance = st.acceptance_rate();

    std::fprintf(stderr,
                 "lse-server: prompt %d in %.2fs (%.1f tok/s) | decode %d in "
                 "%.2fs (%.1f tok/s)%s\n",
                 out.prefill_tokens, static_cast<double>(out.prefill_ns) / 1e9, out.prompt_per_second,
                 out.decode_tokens, static_cast<double>(out.decode_ns) / 1e9,
                 out.decode_per_second,
                 out.acceptance >= 0.0
                     ? (" | accepted " + std::to_string(
                            static_cast<int>(out.acceptance * 100.0)) + "%").c_str()
                     : "");
    return out;
  }
};

namespace {

json usage_of(const Outcome& o) {
  return json{{"prompt_tokens", o.prompt_tokens},
              {"prompt_tokens_details", {{"cached_tokens", o.prompt_tokens - o.prefill_tokens}}},
              {"completion_tokens", o.completion_tokens},
              {"total_tokens", o.prompt_tokens + o.completion_tokens}};
}

// Not part of the OpenAI schema. It rides beside `usage` under its own key so
// a client that does not know it ignores it, which is what llama.cpp does with
// the same information.
json timings_of(const Outcome& o) {
  json t{{"prompt_n", o.prefill_tokens},
         {"prompt_cached_n", o.prompt_tokens - o.prefill_tokens},
         {"prompt_ms", o.prefill_ns / 1e6},
         {"prompt_per_second", o.prompt_per_second},
         {"generated_n", o.completion_tokens},
         {"decode_n", o.decode_tokens},
         {"decode_ms", o.decode_ns / 1e6},
         {"decode_per_second", o.decode_per_second},
         {"predicted_n", o.decode_tokens},
         {"predicted_ms", o.decode_ns / 1e6},
         {"predicted_per_second", o.decode_per_second}};
  t["mtp_depth"] = o.mtp_depth;
  t["dflash2_depth"] = o.dflash2_depth;
  t["spec_method"] = o.dflash2_depth != 0 ? "dflash2"
                       : o.mtp_depth != 0 ? "mtp" : "none";
  o.jit.append_to(t);
  t["device_groups"] = o.device_groups;
  t["host_groups"] = o.host_groups;
  t["host_fallbacks"] = o.host_fallbacks;
  if (o.mtp_depth != 0 || o.dflash2_depth != 0) {
    t["spec_steps"] = o.spec_steps;
    t["spec_tested"] = o.spec_tested;
    t["spec_accepted"] = o.spec_accepted;
    t["spec_draft_ms"] = static_cast<double>(o.spec_draft_ns) / 1e6;
    t["spec_verify_ms"] = static_cast<double>(o.spec_verify_ns) / 1e6;
  }
  if (o.acceptance >= 0.0) t["acceptance_rate"] = o.acceptance;
  return t;
}

// text_completion and chat.completion differ only in the shape of a choice.
json chat_choice(const std::string& text, bool hit_limit, const Request& r,
                 const std::string& id) {
  detail::ChatResponseParser parser(r.thinking_enabled, r.chat, id);
  parser.push(text);
  parser.finish(hit_limit);
  return json{{"index", 0}, {"message", parser.message()}, {"logprobs", nullptr},
              {"finish_reason", parser.finish_reason(hit_limit)}};
}

json text_choice(const std::string& text, bool hit_limit, bool expose_thinking) {
  std::string reasoning;
  std::string out = text;
  if (expose_thinking) {
    auto [answer, r] = detail::split_thinking(text);
    out = answer;
    reasoning = r;
  }
  json choice{{"index", 0},
              {"text", std::move(out)},
              {"logprobs", nullptr},
              {"finish_reason", finish_reason(hit_limit)}};
  if (!reasoning.empty()) choice["reasoning"] = std::move(reasoning);
  return choice;
}

}  // namespace

HttpServer::HttpServer(model::HybridLM& model, tokenizer::Tokenizer& tok,
                       ServerOptions options)
    : impl_(std::make_unique<Impl>(model, tok, std::move(options))) {}

HttpServer::~HttpServer() = default;

void HttpServer::use_mtp(model::MtpModule& mtp) noexcept {
  impl_->mtp = &mtp; impl_->dflash2 = nullptr;
}
void HttpServer::use_dflash2(model::DFlash2Module& draft) noexcept {
  impl_->dflash2 = &draft; impl_->mtp = nullptr;
}

void HttpServer::stop() {
  impl_->stopping.store(true, std::memory_order_relaxed);
  detail::stop_listener_once(impl_->http, impl_->listener_stop_issued);
}

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

  impl.http.Get("/health", [&impl](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"},
                          {"mtp_enabled", impl.mtp != nullptr},
                          {"mtp_depth", impl.mtp != nullptr ? impl.opt.mtp_depth : 0},
                          {"dflash2_enabled", impl.dflash2 != nullptr},
                          {"dflash2_depth", impl.dflash2 != nullptr
                                               ? runtime::dflash2_verify_depth(impl.dflash2->block_size()) : 0}}
                        .dump(), "application/json");
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
      if (impl.stopping.load()) {
        send_error(res, 503, "server is stopping", "server_error");
        return;
      }
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
        try {
          r.chat = detail::prepare_chat(body, r.thinking_enabled, r.thinking_level);
          prompt_text = r.chat.prompt;
        } catch (const std::exception& e) {
          send_error(res, 400, e.what(), "invalid_request_error", "messages/tools");
          return;
        }
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
        // /v1/completions is a raw passthrough: the prompt is sent to the
        // model verbatim, exactly as llama.cpp and MLX treat it. The chat
        // template (and with it the reasoning gate and effort instruction) is
        // applied only on /v1/chat/completions, so nothing is injected here.
        // A client that wants the reasoning framing uses the chat route or
        // supplies the framing tokens itself in the prompt.
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
          send_error(res, out.status().code() == StatusCode::kCancelled ? 503 : 500,
                     std::string(out.status().message()), "server_error");
          return;
        }
        try {
          json resp{{"id", id},
                    {"object", object},
                    {"created", created},
                    {"model", impl.opt.model_id},
                    {"choices", json::array({chat ? chat_choice(out->text, out->hit_limit, r, id)
                                                  : text_choice(out->text, out->hit_limit, r.thinking_enabled)})},
                    {"usage", usage_of(*out)},
                    {"timings", timings_of(*out)}};
          res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
          send_error(res, 500, e.what(), "model_output_error");
        }
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

            detail::ChatResponseParser parser(r.thinking_enabled, r.chat, id);
            std::string parser_error;
            auto send_chat_deltas = [&](const std::vector<json>& deltas) {
              for (const auto& delta : deltas) {
                json choice{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}};
                if (!send(json{{"id", id},
                               {"object", chunk_object},
                               {"created", created},
                               {"model", impl.opt.model_id},
                               {"choices", json::array({choice})}}))
                  return false;
              }
              return true;
            };

            auto emit = [&](const std::string& piece) -> bool {
              if (!chat) {
                json choice{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                return send(json{{"id", id},
                                 {"object", chunk_object},
                                 {"created", created},
                                 {"model", impl.opt.model_id},
                                 {"choices", json::array({choice})}});
              }
              try { return send_chat_deltas(parser.push(piece)); }
              catch (const std::exception& e) { parser_error = e.what(); return false; }
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

            try {
              if (!parser_error.empty()) throw std::invalid_argument(parser_error);
              if (chat && !send_chat_deltas(parser.finish(out->hit_limit))) {
                sink.done(); return true;
              }
            } catch (const std::exception& e) {
              send(json{{"error", {{"message", e.what()}, {"type", "model_output_error"}}}});
              sink.done(); return true;
            }

            json last = chat
                ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", parser.finish_reason(out->hit_limit)}}
                : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason(out->hit_limit)}};
            send(json{{"id", id},
                      {"object", chunk_object},
                      {"created", created},
                      {"model", impl.opt.model_id},
                      {"choices", json::array({last})},
                      {"timings", timings_of(*out)}});
            if (r.include_usage)
              send(json{{"id", id}, {"object", chunk_object}, {"created", created},
                        {"model", impl.opt.model_id}, {"choices", json::array()},
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

  if (impl.stopping.load()) return OkStatus();
  if (!impl.http.listen(impl.opt.host, impl.opt.port)) {
    return LSE_ERROR(kIoError, "could not listen on ", impl.opt.host, ":",
                     std::to_string(impl.opt.port));
  }
  return OkStatus();
}

}  // namespace lse::server
