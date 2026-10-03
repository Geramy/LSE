#include "lse/server/router.hpp"
#include "jit_timings.hpp"
#include "chat_protocol.hpp"
#include "request_sampling.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/model/inspect.hpp"
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


// The error envelope clients parse. Anything that leaves this surface as a
// non-2xx wears it, so a caller never has to guess between our shape and
// theirs.
json error_json(const std::string& message, const std::string& type,
                const std::string& param) {
  return json{{"error",
               {{"message", message},
                {"type", type},
                {"code", nullptr},
                {"param", param.empty() ? json(nullptr) : json(param)}}}};
}

void send_error(RouteReply& res, int status, const std::string& message,
                const std::string& type = "invalid_request_error",
                const std::string& param = "") {
  res.status = status;
  res.body = error_json(message, type, param).dump();
  res.stream = nullptr;
}

void set_content(RouteReply& res, const json& body) {
  res.status = 200;
  res.body = body.dump();
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

std::string error_body(const std::string& message, const std::string& type,
                       const std::string& param) {
  return error_json(message, type, param).dump();
}

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
  std::array<std::uint32_t, 7> spec_tested_by_position{}, spec_accepted_by_position{};
  std::array<double, 7> spec_overlap_sum{}, spec_candidate_mass_sum{}, spec_deterministic_mass_sum{};
  std::uint64_t spec_draft_ns = 0, spec_verify_ns = 0;
  double acceptance = -1.0;  // negative when nothing was speculated
};

json timings_of(const Outcome& o);

}  // namespace

struct Router::Impl {
  model::HybridLM& model;
  tokenizer::Tokenizer& tok;
  model::MtpModule* mtp = nullptr;
  model::DFlash2Module* dflash2 = nullptr;
  ServerOptions opt;
  // One model on one device. Two decodes at once would interleave on the same
  // KV pool, so requests queue here instead.
  std::mutex generate_lock;
  std::vector<std::uint32_t> stop_ids;
  // One resident session, restarted between requests instead of rebuilt: the
  // state arrays keep their nodes, so the model's retained program replays
  // and a warm request skips the per-request partition and emit entirely.
  // Guarded by generate_lock like everything else that touches the device.
  runtime::Session session{"resident", 0};
  bool session_live = false;
  // Watched by requests that brought no stop flag of their own.
  std::atomic<bool> never_stopping{false};

  // What lse_status reports. Guarded by metrics_lock.
  mutable std::mutex metrics_lock;
  std::uint64_t generations = 0;
  json last_timings;

  Impl(model::HybridLM& m, tokenizer::Tokenizer& t, ServerOptions o)
      : model(m), tok(t), opt(std::move(o)) {
    stop_ids = chat_stop_tokens(tok);
  }

  // What /v1/models says about the one model: the OpenAI fields, the context
  // this engine enforces, and the KV settings it was opened with.
  json model_entry() const {
    json entry{{"id", opt.model_id},
               {"object", "model"},
               {"created", now_seconds()},
               {"owned_by", "lse"}};
    entry.update(served());
    return entry;
  }

  // The settings this engine was opened with, which a client cannot ask for
  // per request: the context it enforces and how its KV cache is stored.
  json served() const {
    const model::Config& c = model.config();
    json draft = nullptr;
    if (mtp != nullptr) draft = {{"kind", "mtp"}, {"depth", opt.mtp_depth}};
    if (dflash2 != nullptr)
      draft = {{"kind", "dflash2"},
               {"block_size", dflash2->block_size()},
               {"depth", runtime::dflash2_verify_depth(dflash2->block_size())}};
    return json{{"context_length", c.kv_capacity()},
                {"max_position_embeddings", c.train_seq_len},
                {"kv_len", c.kv_capacity()},
                {"kv_cache_dtype", std::string(kv::to_string(c.kv_cache_dtype))},
                {"max_tokens", opt.max_tokens_cap},
                {"draft", std::move(draft)}};
  }

  // Bytes the engine holds through its backends right now.
  static json allocated() {
    const auto device = backend::allocation_totals(backend::MemoryClass::kDevice);
    const auto staging = backend::allocation_totals(backend::MemoryClass::kStaging);
    return json{{"device_bytes", device.live},
                {"device_peak_bytes", device.peak},
                {"device_allocations", device.allocations},
                {"staging_bytes", staging.live}};
  }

  void model_info(RouteReply& res) const {
    if (opt.model_path.empty()) {
      send_error(res, 404, "this server does not know where its model was loaded from",
                 "invalid_request_error");
      return;
    }
    auto info = model::model_info(opt.model_path);
    if (!info.ok()) {
      send_error(res, 500, std::string(info.status().message()), "server_error");
      return;
    }
    json out = info.release();
    out["served"] = served();
    if (!opt.draft_path.empty()) {
      auto d = model::model_info(opt.draft_path);
      out["draft"] = d.ok() ? d.release() : json{{"error", std::string(d.status().message())}};
    }
    set_content(res, out);
  }

  // The memory plan for the loaded model, as opened, with any of the request's
  // settings changed: what a restart with those settings would allocate.
  void estimate(std::string_view body, RouteReply& res) const {
    if (opt.model_path.empty()) {
      send_error(res, 404, "this server does not know where its model was loaded from",
                 "invalid_request_error");
      return;
    }
    json b = json::object();
    if (!body.empty()) {
      b = json::parse(body.begin(), body.end(), nullptr, false);
      if (b.is_discarded() || !b.is_object()) {
        send_error(res, 400, "the request body must be a JSON object");
        return;
      }
    }
    const model::Config& c = model.config();
    model::MemoryPlanRequest req;
    req.model = opt.model_path;
    req.draft = mtp != nullptr ? model::DraftKind::kMtp
                : dflash2 != nullptr ? model::DraftKind::kDFlash2 : model::DraftKind::kNone;
    req.draft_path = opt.draft_path;
    req.mtp_depth = opt.mtp_depth;
    req.kv_cache_dtype = c.kv_cache_dtype;
    req.kv_len = c.kv_capacity();
    req.batch_size = opt.prefill.batch_size;
    req.ubatch_size = opt.prefill.ubatch_size;
    req.fragmented_kv = opt.fragmented_kv;
    req.device_arch = opt.device_arch;
    try {
      if (b.contains("kv_cache_dtype")) {
        auto f = kv::cache_dtype_from_string(b["kv_cache_dtype"].get<std::string>());
        if (!f.ok()) {
          send_error(res, 400, std::string(f.status().message()), "invalid_request_error",
                     "kv_cache_dtype");
          return;
        }
        req.kv_cache_dtype = *f;
      }
      if (b.contains("kv_len")) req.kv_len = b["kv_len"].get<std::int32_t>();
      if (b.contains("context_tokens")) req.context_tokens = b["context_tokens"].get<std::int32_t>();
      if (b.contains("batch_size")) req.batch_size = b["batch_size"].get<std::uint32_t>();
      if (b.contains("ubatch_size")) req.ubatch_size = b["ubatch_size"].get<std::uint32_t>();
      if (b.contains("sequences")) req.sequences = b["sequences"].get<std::int32_t>();
      if (b.contains("mtp_depth")) req.mtp_depth = b["mtp_depth"].get<std::uint32_t>();
      if (b.contains("device_memory_bytes"))
        req.device_memory_bytes = b["device_memory_bytes"].get<std::uint64_t>();
    } catch (const json::exception&) {
      send_error(res, 400, "estimate settings must be numbers, and kv_cache_dtype a string");
      return;
    }
    auto plan = model::estimate_memory(req);
    if (!plan.ok()) {
      send_error(res, 400, std::string(plan.status().message()));
      return;
    }
    json out = plan.release();
    out["allocated"] = allocated();
    set_content(res, out);
  }

  void record(const Outcome& out) {
    const json timings = timings_of(out);
    std::lock_guard held(metrics_lock);
    ++generations;
    last_timings = timings;
  }

  // Parses the parts /v1/chat/completions and /v1/completions share.
  Result<Request> parse_common(const json& body, RouteReply& res) {
    if (!body.is_object()) {
      send_error(res, 400, "request must be a JSON object");
      return LSE_ERROR(kInvalidArgument, "request");
    }
    Request r;
    r.model = get_or<std::string>(body, "model", opt.model_id);
    r.stream = get_or<bool>(body, "stream", false);
    r.stop_strings = get_stop_strings(body);
    if (body.contains("kv_cache_dtype") && !body["kv_cache_dtype"].is_null()) {
      const auto reject = [&](std::string message) -> Status {
        send_error(res, 400, message, "invalid_request_error", "kv_cache_dtype");
        return LSE_ERROR(kInvalidArgument, message);
      };
      if (!body["kv_cache_dtype"].is_string())
        return reject("kv_cache_dtype must be a string");
      const auto requested =
          kv::cache_dtype_from_string(body["kv_cache_dtype"].get<std::string>());
      if (!requested.ok()) return reject(std::string(requested.status().message()));
      if (*requested != model.config().kv_cache_dtype)
        return reject("kv_cache_dtype is fixed by the server's --kv-cache-dtype option");
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

  void completion(bool chat, std::string_view req_body, const std::atomic<bool>& stopping,
                  RouteReply& res);

  static Result<Outcome> generate(Impl& impl, const Request& r,
                                  const std::function<bool(const std::string&)>& emit,
                                  const std::atomic<bool>& stopping);
};


// Runs `r` under the generate lock. `emit` is called on the calling thread,
// so a streaming caller writes to its own transport and nothing is shared.
Result<Outcome> Router::Impl::generate(
    Router::Impl& impl, const Request& r,
    const std::function<bool(const std::string&)>& emit,
    const std::atomic<bool>& stopping) {
  std::lock_guard<std::mutex> held(impl.generate_lock);
  // A request queued behind another generation must not start new device
  // work after shutdown has asked the active generation to stop.
  if (stopping.load()) return LSE_ERROR(kCancelled, "server is stopping");

  runtime::Generator gen(impl.model, r.sampling, impl.opt.prefill);
  if (impl.mtp != nullptr) gen.use_mtp(*impl.mtp);
  if (impl.dflash2 != nullptr) gen.use_dflash2(*impl.dflash2);

  // Generator checks the exact token prefix before reusing resident state.
  if (!impl.session_live) {
    impl.session = runtime::Session{"resident", impl.model.state_slots()};
    impl.session_live = true;
  }

  tokenizer::DecodeStream stream(impl.tok);
  Outcome out;
  out.prompt_tokens = static_cast<int>(r.prompt.size());
  bool stopped_by_string = false;

  auto on_token = [&](std::uint32_t id) -> bool {
    if (stopping.load(std::memory_order_relaxed)) return false;
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

  auto ids = gen.generate(impl.session, r.prompt, r.limits, on_token);
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
  out.spec_tested_by_position = st.spec_tested_by_position;
  out.spec_accepted_by_position = st.spec_accepted_by_position;
  out.spec_overlap_sum = st.spec_overlap_sum;
  out.spec_candidate_mass_sum = st.spec_candidate_mass_sum;
  out.spec_deterministic_mass_sum = st.spec_deterministic_mass_sum;
  out.spec_draft_ns = st.spec_draft_ns;
  out.spec_verify_ns = st.spec_verify_ns;
  out.prefill_tokens = st.prompt_tokens;
  out.decode_tokens = st.decoded_tokens();
  out.prefill_ns = st.prefill_ns;
  out.decode_ns = st.decode_ns;
  out.decode_per_second = st.decode_tokens_per_second();
  out.prompt_per_second = st.prompt_tokens_per_second();
  if (st.spec_steps != 0) out.acceptance = st.acceptance_rate();
  impl.record(out);

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
    t["spec_tested_by_position"] = o.spec_tested_by_position;
    t["spec_accepted_by_position"] = o.spec_accepted_by_position;
    t["spec_overlap_sum"] = o.spec_overlap_sum;
    t["spec_candidate_mass_sum"] = o.spec_candidate_mass_sum;
    t["spec_deterministic_mass_sum"] = o.spec_deterministic_mass_sum;
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

// The two completion routes differ in how the prompt is built and how a
// choice is shaped; everything after that is shared.
void Router::Impl::completion(bool chat, std::string_view req_body,
                              const std::atomic<bool>& stopping, RouteReply& res) {
  Impl& impl = *this;
  if (stopping.load()) {
    send_error(res, 503, "server is stopping", "server_error");
    return;
  }
  json body;
  try {
    body = json::parse(req_body);
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
    auto out = Impl::generate(impl, r, {}, stopping);
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
      set_content(res, resp);
    } catch (const std::exception& e) {
      send_error(res, 500, e.what(), "model_output_error");
    }
    return;
  }

  // Server-sent events. The generation runs inside the provider so a
  // delta reaches the socket as it resolves rather than at the end.
  const std::string chunk_object =
      chat ? "chat.completion.chunk" : "text_completion";
  res.status = 200;
  res.body.clear();
  res.stream =
      [&impl, r, id, created, chat, chunk_object, &stopping](const RouteReply::Send& sink) -> bool {
        auto send = [&sink](const json& j) {
          return sink(RouteReply::Event::kChunk, j.dump());
        };
        auto send_failure = [&sink](const json& j) {
          return sink(RouteReply::Event::kError, j.dump());
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

        auto out = Impl::generate(impl, r, emit, stopping);
        if (!out.ok()) {
          // The status line is long gone, so the error rides the stream.
          send_failure(json{{"error",
                     {{"message", std::string(out.status().message())},
                      {"type", "server_error"}}}});
          return false;
        }

        try {
          if (!parser_error.empty()) throw std::invalid_argument(parser_error);
          if (chat && !send_chat_deltas(parser.finish(out->hit_limit))) {
            return false;
          }
        } catch (const std::exception& e) {
          send_failure(json{{"error", {{"message", e.what()}, {"type", "model_output_error"}}}});
          return false;
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
        return true;
      };
}

Router::Router(model::HybridLM& model, tokenizer::Tokenizer& tok,
               ServerOptions options)
    : impl_(std::make_unique<Impl>(model, tok, std::move(options))) {}

Router::~Router() = default;

void Router::use_mtp(model::MtpModule& mtp) noexcept {
  impl_->mtp = &mtp; impl_->dflash2 = nullptr;
}
void Router::use_dflash2(model::DFlash2Module& draft) noexcept {
  impl_->dflash2 = &draft; impl_->mtp = nullptr;
}

const ServerOptions& Router::options() const noexcept { return impl_->opt; }

namespace {

// Named so a client gets a straight answer instead of a 404 it has to guess at.
constexpr const char* kNotImplemented[] = {
    "/v1/embeddings", "/v1/images/generations", "/v1/audio/speech",
    "/v1/audio/transcriptions", "/v1/moderations", "/v1/responses"};

// "/v1/models/:id" matches one nonempty segment where the pattern says ":id".
bool matches(std::string_view pattern, std::string_view path) {
  const auto colon = pattern.find(':');
  if (colon == std::string_view::npos) return pattern == path;
  if (!path.starts_with(pattern.substr(0, colon))) return false;
  const auto segment = path.substr(colon);
  return !segment.empty() && segment.find('/') == std::string_view::npos;
}

}  // namespace

const std::vector<Route>& Router::routes() {
  static const std::vector<Route> table = [] {
    std::vector<Route> t{{"GET", "/health"},
                         {"GET", "/v1/models"},
                         {"GET", "/v1/models/:id"},
                         {"GET", "/v1/lse/model_info"},
                         {"GET", "/v1/lse/estimate"},
                         {"POST", "/v1/lse/estimate"},
                         {"POST", "/v1/chat/completions"},
                         {"POST", "/v1/completions"}};
    for (const char* path : kNotImplemented) t.push_back({"POST", path});
    return t;
  }();
  return table;
}

RouteReply Router::handle(std::string_view method, std::string_view path,
                          std::string_view body, const std::atomic<bool>* stopping_flag) {
  Impl& impl = *impl_;
  const std::atomic<bool>& stopping = stopping_flag ? *stopping_flag : impl.never_stopping;
  RouteReply res;
  try {
    if (method == "GET" && path == "/health") {
      set_content(res, json{{"status", "ok"},
                            {"mtp_enabled", impl.mtp != nullptr},
                            {"mtp_depth", impl.mtp != nullptr ? impl.opt.mtp_depth : 0},
                            {"dflash2_enabled", impl.dflash2 != nullptr},
                            {"dflash2_depth", impl.dflash2 != nullptr
                                                 ? runtime::dflash2_verify_depth(impl.dflash2->block_size()) : 0}});
      return res;
    }
    if (method == "GET" && path == "/v1/models") {
      set_content(res, json{{"object", "list"}, {"data", json::array({impl.model_entry()})}});
      return res;
    }
    if (method == "GET" && matches("/v1/models/:id", path)) {
      set_content(res, impl.model_entry());
      return res;
    }
    if (method == "GET" && path == "/v1/lse/model_info") {
      impl.model_info(res);
      return res;
    }
    if (path == "/v1/lse/estimate" && (method == "GET" || method == "POST")) {
      impl.estimate(body, res);
      return res;
    }
    if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions")) {
      impl.completion(path == "/v1/chat/completions", body, stopping, res);
      return res;
    }
    if (method == "POST") {
      for (const char* unimplemented : kNotImplemented) {
        if (path == unimplemented) {
          send_error(res, 501, std::string(unimplemented) + " is not implemented by this server",
                     "not_implemented");
          return res;
        }
      }
    }
    send_error(res, 404, std::string(method) + " " + std::string(path) + " is not a route of this server",
               "invalid_request_error");
  } catch (const std::exception&) {
    send_error(res, 500, "internal error", "server_error");
  }
  return res;
}

std::string Router::metrics_json() const {
  std::lock_guard held(impl_->metrics_lock);
  json m{{"model", impl_->opt.model_id},
         {"generations", impl_->generations},
         {"mtp_enabled", impl_->mtp != nullptr},
         {"dflash2_enabled", impl_->dflash2 != nullptr}};
  m["last_timings"] = impl_->last_timings.is_null() ? json(nullptr) : impl_->last_timings;
  return m.dump();
}

}  // namespace lse::server
