#include "lse/server/router.hpp"
#include "jit_timings.hpp"
#include "chat_protocol.hpp"
#include "request_sampling.hpp"
#include "thinking_request.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
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
  // The client's max_tokens / max_completion_tokens, when it sent one.
  std::optional<std::int32_t> max_tokens;
  std::vector<std::string> stop_strings;
  bool stream = false;
  std::string model;
  detail::ChatRequest chat;
  bool include_usage = false;
  // The thinking fields as sent, and (chat only) the level they resolve to.
  detail::ThinkingRequest thinking;
  detail::ChatFraming framing;
  // Raw completions: whether reasoning markers in the output are split out.
  bool thinking_enabled = true;
  // The session whose KV this request continues; empty for a one-shot
  // request, whose state is released when it finishes.
  std::string session_id;
};

// Where a completion stopped, in OpenAI's vocabulary.
const char* finish_reason(bool hit_limit) { return hit_limit ? "length" : "stop"; }

// Why it stopped, in LSE's: the choice's "stop_reason".
//   stop_token    the model ended its turn (an end-of-turn token)
//   stop_sequence one of the request's stop strings
//   max_tokens    the request's max_tokens, the server's cap, or the model's
//                 generation_config limit
//   context_full  the prompt and the completion fill the context; see
//                 "lse_context"
//   cancelled     the client went away
const char* stop_reason_name(runtime::StopReason r, bool by_string) {
  if (by_string) return "stop_sequence";
  switch (r) {
    case runtime::StopReason::kStopToken: return "stop_token";
    case runtime::StopReason::kMaxTokens: return "max_tokens";
    case runtime::StopReason::kContextFull: return "context_full";
    case runtime::StopReason::kCallback: return "cancelled";
    case runtime::StopReason::kNone: return "max_tokens";
  }
  return "stop_token";
}

// A 400 whose error carries a machine-readable code (and extra fields).
void send_coded_error(RouteReply& res, const std::string& message, const std::string& code,
                      const std::string& param, json extra = json::object()) {
  json e = error_json(message, "invalid_request_error", param);
  e["error"]["code"] = code;
  for (auto& [k, v] : extra.items()) e["error"][k] = v;
  res.status = 400;
  res.body = e.dump();
  res.stream = nullptr;
}

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
  bool stopped_by_string = false;
  runtime::StopReason stop = runtime::StopReason::kNone;
  // The context at the end: whole prompt plus completion, and its capacity.
  int context_tokens = 0;
  int context_length = 0;
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
  std::uint32_t spec_proposed = 0, spec_plain_steps = 0;
  std::uint32_t tree_steps = 0, tree_rows = 0;
  double spec_mean_width = 0.0;
  bool spec_adaptive = false;
  std::array<std::uint32_t, 7> spec_tested_by_position{}, spec_accepted_by_position{};
  std::array<double, 7> spec_overlap_sum{}, spec_candidate_mass_sum{}, spec_deterministic_mass_sum{};
  std::uint64_t spec_draft_ns = 0, spec_verify_ns = 0;
  double acceptance = -1.0;  // negative when nothing was speculated
  // CPU fallbacks this request caused, by cause.
  std::vector<graph::CpuFallbackEvent> cpu_fallbacks;
};

// What the client is told beside the answer: every CPU fallback the request
// caused. Absent when there were none.
json warnings_of(const Outcome& o) {
  json w = json::array();
  for (const auto& e : o.cpu_fallbacks)
    w.push_back({{"type", "cpu_fallback"}, {"cause", e.cause}, {"count", e.count}});
  return w;
}

json timings_of(const Outcome& o);

}  // namespace

struct Router::Impl {
  model::HybridLM& model;
  tokenizer::Tokenizer& tok;
  model::MtpModule* mtp = nullptr;
  model::DFlash2Module* dflash2 = nullptr;
  ServerOptions opt;
  // DFlash2's verify-width policy: what each width costs on this device and
  // how the draft's confidence maps to acceptance, kept across requests.
  runtime::DraftWidthPolicy widths;
  // MTP's chain-depth and verify-width policy, likewise.
  runtime::DraftWidthPolicy mtp_widths;
  // One model on one device. Two decodes at once would interleave on the same
  // KV pool, so requests queue here instead.
  std::mutex generate_lock;
  std::vector<std::uint32_t> stop_ids;
  // Sessions, keyed by the request's session_id. Each owns its KV, recurrent
  // state and history; a request continues its session's prefix when the
  // prompt extends it. The map's structure is guarded by sessions_lock, a
  // session's contents by generate_lock (only the generating request touches
  // them). A request without a session_id runs in a session of its own that
  // is released, with everything the model built for it, when it ends.
  struct SessionEntry {
    explicit SessionEntry(const std::string& id, std::size_t slots) : session(id, slots) {}
    runtime::Session session;
    std::uint64_t last_used = 0;
    std::uint64_t requests = 0;
    // As of the end of its last request; read without generate_lock.
    std::size_t bytes = 0;
    std::int32_t tokens = 0;
    void snapshot() {
      bytes = session.cache_bytes();
      tokens = session.position();
    }
  };
  mutable std::mutex sessions_lock;
  std::map<std::string, std::unique_ptr<SessionEntry>> sessions;
  std::uint64_t session_clock = 0;
  // The session the model's retained passes and the draft modules were last
  // built for (empty: none). Switching away lets go of them; with a draft
  // module, the session switched to starts cold, since the draft's context
  // belongs to the session before it. Guarded by generate_lock.
  std::string bound;
  bool bound_live = false;
  // The session generating now, which memory pressure must not evict.
  std::string active;
  bool active_live = false;
  std::uint64_t evictions = 0;
  std::uint64_t pressure_trimmer = 0;
  // Watched by requests that brought no stop flag of their own.
  std::atomic<bool> never_stopping{false};

  // What lse_status reports. Guarded by metrics_lock.
  mutable std::mutex metrics_lock;
  std::uint64_t generations = 0;
  json last_timings;

  Impl(model::HybridLM& m, tokenizer::Tokenizer& t, ServerOptions o)
      : model(m), tok(t), opt(std::move(o)) {
    stop_ids = chat_stop_tokens(tok);
    // Under memory pressure every idle session is evicted: its next request
    // prefills again instead of the engine running out of memory. (Keep 0:
    // the active and the bound session are never candidates.) A trim that
    // is housekeeping -- a failed request, the last session closing -- keeps
    // them.
    pressure_trimmer = backend::register_memory_trimmer(
        [this] { return backend::device_memory_pressure() ? evict_idle(0, 0) : 0; },
        [this] { return idle_bytes(); });
  }
  ~Impl() { backend::unregister_memory_trimmer(pressure_trimmer); }

  // What the sessions evict_idle could release hold.
  std::size_t idle_bytes() const {
    std::lock_guard held(sessions_lock);
    std::size_t total = 0;
    for (const auto& [id, e] : sessions) {
      if ((active_live && id == active) || (bound_live && id == bound)) continue;
      total += e->bytes;
    }
    return total;
  }

  // Evicts least recently used idle sessions until at most `keep` remain and,
  // with a nonzero budget, the sessions together hold at most `budget` bytes.
  // Never the active session or the one the model is bound to (that one is
  // released through release_bindings). Returns the bytes released.
  std::size_t evict_idle(std::size_t keep, std::size_t budget) {
    std::vector<std::unique_ptr<SessionEntry>> evicted;
    std::size_t released = 0;
    {
      std::lock_guard held(sessions_lock);
      for (;;) {
        std::size_t total = 0;
        for (const auto& [id, e] : sessions) total += e->bytes;
        const bool over = sessions.size() > keep || (budget != 0 && total > budget);
        if (!over) break;
        auto victim = sessions.end();
        for (auto it = sessions.begin(); it != sessions.end(); ++it) {
          if ((active_live && it->first == active) || (bound_live && it->first == bound)) continue;
          if (victim == sessions.end() || it->second->last_used < victim->second->last_used) victim = it;
        }
        if (victim == sessions.end()) break;
        released += victim->second->bytes;
        evicted.push_back(std::move(victim->second));
        sessions.erase(victim);
        ++evictions;
      }
    }
    // Released outside the lock: freeing device memory takes no session lock.
    for (auto& e : evicted) {
      std::fprintf(stderr, "lse-server: evicted session '%s' (%zu bytes)\n",
                   e->session.id().c_str(), e->bytes);
    }
    evicted.clear();
    return released;
  }

  // Lets go of what the model and the draft modules built for the bound
  // session. `survivor`, when that session lives on, keeps its KV: its state
  // is detached from the passes first. With a draft module the survivor is
  // cleared instead, since the draft's context is lost. Caller holds
  // generate_lock.
  void release_bindings(runtime::Session* survivor) {
    if (survivor != nullptr) {
      if ((mtp != nullptr || dflash2 != nullptr) ||
          !model.retire_completed_passes(survivor->states()).ok())
        survivor->clear();
      std::lock_guard held(sessions_lock);
      for (auto& [id, e] : sessions)
        if (&e->session == survivor) e->snapshot();
    }
    (void)model.drop_retained_passes();
    if (mtp != nullptr) mtp->reset();
    if (dflash2 != nullptr) dflash2->release_programs();
    bound.clear();
    bound_live = false;
  }

  // Releases the session `id`: its KV and state, and what the model holds for
  // it. False when there is no such session.
  bool close_session(const std::string& id) {
    std::lock_guard<std::mutex> device(generate_lock);
    std::unique_ptr<SessionEntry> closing;
    bool empty = false;
    {
      std::lock_guard held(sessions_lock);
      auto it = sessions.find(id);
      if (it == sessions.end()) return false;
      closing = std::move(it->second);
      sessions.erase(it);
      empty = sessions.empty();
    }
    if (bound_live && bound == id) release_bindings(nullptr);
    closing.reset();
    // With no session left, the engine's caches are let go too, so the device
    // holds what it held after load.
    if (empty) (void)backend::trim_device_memory();
    return true;
  }

  json sessions_json() const {
    json list = json::array();
    std::lock_guard held(sessions_lock);
    for (const auto& [id, e] : sessions) {
      list.push_back({{"id", id},
                      {"tokens", e->tokens},
                      {"bytes", e->bytes},
                      {"requests", e->requests},
                      {"last_used", e->last_used}});
    }
    return list;
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
    if (mtp != nullptr)
      draft = {{"kind", "mtp"}, {"depth", opt.mtp_depth}, {"adaptive", opt.adaptive_mtp}};
    if (dflash2 != nullptr)
      draft = {{"kind", "dflash2"},
               {"block_size", dflash2->block_size()},
               {"depth", runtime::dflash2_verify_depth(dflash2->block_size())},
               {"adaptive", opt.adaptive_dflash2},
               {"tree", runtime::draft_trees_enabled(opt.dflash2_tree, opt.adaptive_dflash2)}};
    return json{{"context_length", c.kv_capacity()},
                {"max_position_embeddings", c.train_seq_len},
                {"kv_len", c.kv_capacity()},
                {"kv_cache_dtype", std::string(kv::to_string(c.kv_cache_dtype))},
                {"max_tokens", opt.max_tokens_cap > 0 ? json(opt.max_tokens_cap) : json(nullptr)},
                {"generation_defaults", c.sampling_defaults.to_json()},
                {"thinking", opt.thinking.to_json()},
                {"draft", std::move(draft)}};
  }

  // Bytes the engine holds through its backends right now.
  static json allocated() {
    const auto device = backend::allocation_totals(backend::MemoryClass::kDevice);
    const auto staging = backend::allocation_totals(backend::MemoryClass::kStaging);
    json by_site = json::object();
    for (const auto& site : backend::allocation_sites()) {
      if (site.live == 0 && site.peak == 0) continue;
      by_site[backend::to_string(site.site)] = {{"bytes", site.live}, {"peak_bytes", site.peak},
                                                {"allocations", site.allocations}};
    }
    return json{{"device_bytes", device.live},
                {"device_peak_bytes", device.peak},
                {"device_allocations", device.allocations},
                {"device_by_site", std::move(by_site)},
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
    if (body.contains("session_id") && !body["session_id"].is_null()) {
      if (!body["session_id"].is_string() || body["session_id"].get<std::string>().empty() ||
          body["session_id"].get<std::string>().find('/') != std::string::npos) {
        send_error(res, 400, "session_id must be a nonempty string without '/'",
                   "invalid_request_error", "session_id");
        return LSE_ERROR(kInvalidArgument, "session_id");
      }
      r.session_id = body["session_id"].get<std::string>();
    }
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

    // Thinking fields. They are resolved against the model's chat template on
    // the chat route (see thinking_request.hpp); a raw completion only uses
    // them to decide whether to split reasoning out of the output.
    if (auto bad = detail::read_thinking_request(body, r.thinking)) {
      send_coded_error(res, bad->message, bad->code, bad->param);
      return LSE_ERROR(kInvalidArgument, bad->code);
    }
    r.thinking_enabled = r.thinking.enable.value_or(true) && r.thinking.level != "none";

    // No output limit unless the client asks for one: generation runs to a
    // stop token, a stop sequence or a full context. See output_limit().
    for (const char* key : {"max_completion_tokens", "max_tokens"}) {
      const auto at = body.find(key);
      if (at == body.end() || at->is_null()) continue;
      if (!at->is_number_integer() || at->get<std::int64_t>() <= 0 ||
          at->get<std::int64_t>() > std::numeric_limits<std::int32_t>::max()) {
        send_error(res, 400, std::string(key) + " must be a positive integer", "invalid_request_error", key);
        return LSE_ERROR(kInvalidArgument, key);
      }
      r.max_tokens = static_cast<std::int32_t>(at->get<std::int64_t>());  // max_tokens wins
    }
    if (r.max_tokens && opt.max_tokens_cap > 0 && *r.max_tokens > opt.max_tokens_cap) {
      send_error(res, 400,
                 "max_tokens " + std::to_string(*r.max_tokens) + " exceeds this server's cap of " +
                     std::to_string(opt.max_tokens_cap),
                 "invalid_request_error", "max_tokens");
      return LSE_ERROR(kInvalidArgument, "max_tokens cap");
    }
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
      // A request that names its depth gets that depth every step.
      r.limits.adaptive_mtp = false;
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

  // Generated tokens a request may produce, from the first of: the client's
  // max_tokens; the operator's --max-tokens cap; the model's
  // generation_config max_new_tokens; else no limit. The model's
  // generation_config max_length (prompt plus completion) bounds the last
  // three. The context bounds all of them, in the generator.
  std::int32_t output_limit(const Request& r) const {
    if (r.max_tokens) return *r.max_tokens;
    const models::SamplingDefaults& d = model.config().sampling_defaults;
    std::int32_t limit = runtime::kNoTokenLimit;
    if (opt.max_tokens_cap > 0) limit = opt.max_tokens_cap;
    else if (d.max_new_tokens) limit = *d.max_new_tokens;
    if (d.max_length) {
      const auto room = static_cast<std::int64_t>(*d.max_length) - static_cast<std::int64_t>(r.prompt.size());
      limit = static_cast<std::int32_t>(std::clamp<std::int64_t>(room, 1, limit));
    }
    return limit;
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
  const std::uint64_t fallbacks_before = graph::cpu_fallback_sequence();

  runtime::Generator gen(impl.model, r.sampling, impl.opt.prefill);
  if (impl.mtp != nullptr) gen.use_mtp(*impl.mtp, impl.opt.adaptive_mtp ? &impl.mtp_widths : nullptr);
  if (impl.dflash2 != nullptr)
    gen.use_dflash2(*impl.dflash2, impl.opt.adaptive_dflash2 ? &impl.widths : nullptr,
                    runtime::draft_trees_enabled(impl.opt.dflash2_tree, impl.opt.adaptive_dflash2));

  // The request's session: a named one is found or created, and continues
  // its own prefix (the Generator checks the exact tokens before reusing
  // state); a one-shot request gets a session of its own.
  const bool ephemeral = r.session_id.empty();
  std::unique_ptr<SessionEntry> one_shot;
  SessionEntry* entry = nullptr;
  if (ephemeral) {
    one_shot = std::make_unique<SessionEntry>("", impl.model.state_slots());
    entry = one_shot.get();
  } else {
    // Room for this session first: beyond the session cap or the byte budget,
    // least recently used idle sessions go.
    bool known = false;
    {
      std::lock_guard sessions_held(impl.sessions_lock);
      known = impl.sessions.count(r.session_id) != 0;
    }
    const std::size_t cap = impl.opt.max_sessions == 0 ? SIZE_MAX
                            : known ? impl.opt.max_sessions : impl.opt.max_sessions - 1;
    (void)impl.evict_idle(cap, impl.opt.session_memory_budget);
    std::lock_guard sessions_held(impl.sessions_lock);
    auto& slot = impl.sessions[r.session_id];
    if (!slot) slot = std::make_unique<SessionEntry>(r.session_id, impl.model.state_slots());
    entry = slot.get();
    entry->last_used = ++impl.session_clock;
    ++entry->requests;
    impl.active = r.session_id;
    impl.active_live = true;
  }
  struct ActiveScope {
    Router::Impl& impl;
    ~ActiveScope() {
      std::lock_guard held(impl.sessions_lock);
      impl.active_live = false;
      impl.active.clear();
    }
  } active_scope{impl};
  runtime::Session& session = entry->session;
  // The model's retained passes and the draft modules serve one session at a
  // time. Another session's are let go; that session keeps its KV unless a
  // draft module makes it start cold.
  if (impl.bound_live && (ephemeral || impl.bound != r.session_id)) {
    std::unique_ptr<SessionEntry>* previous = nullptr;
    {
      std::lock_guard sessions_held(impl.sessions_lock);
      auto it = impl.sessions.find(impl.bound);
      if (it != impl.sessions.end()) previous = &it->second;
    }
    impl.release_bindings(previous != nullptr ? &(*previous)->session : nullptr);
  }
  if (!impl.bound_live && !ephemeral && (impl.mtp != nullptr || impl.dflash2 != nullptr))
    session.clear();  // the draft's context is not this session's
  impl.bound = r.session_id;
  impl.bound_live = true;

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

  auto ids = gen.generate(session, r.prompt, r.limits, on_token);
  // A one-shot session is gone once its request ends, with what the model
  // built for it. A failed request's session starts cold next time, and
  // nothing the failed pass built is kept: that is what lets the engine
  // recover from running out of memory without a restart.
  if (!ids.ok() || ephemeral) {
    session.clear();
    impl.release_bindings(nullptr);
    one_shot.reset();
    bool none_left = false;
    {
      std::lock_guard sessions_held(impl.sessions_lock);
      none_left = impl.sessions.empty();
    }
    // A failed request leaves nothing behind: what it built is released
    // above and the caches go too, so the next request starts clean.
    if (!ids.ok() || none_left) (void)backend::trim_device_memory();
  }
  if (!ephemeral) {
    std::lock_guard sessions_held(impl.sessions_lock);
    entry->snapshot();
  }
  if (!ids.ok()) return ids.status();
  out.completion_tokens = static_cast<int>(ids->size());
  const runtime::GenerationStats& st = gen.stats();
  out.stopped_by_string = stopped_by_string;
  out.stop = st.stop_reason;
  out.hit_limit = !stopped_by_string && (st.stop_reason == runtime::StopReason::kMaxTokens ||
                                         st.stop_reason == runtime::StopReason::kContextFull);
  out.context_tokens = out.prompt_tokens + out.completion_tokens;
  out.context_length = st.context_length;

  out.jit = detail::JitTotals::from(st);
  out.device_groups = st.device_groups;
  out.host_groups = st.host_groups;
  out.host_fallbacks = st.host_fallbacks;
  out.mtp_depth = st.mtp_depth;
  out.dflash2_depth = st.dflash2_depth;
  out.spec_steps = st.spec_steps;
  out.spec_tested = st.spec_tested;
  out.spec_accepted = st.spec_accepted;
  out.spec_proposed = st.spec_proposed;
  out.spec_plain_steps = st.spec_plain_steps;
  out.tree_steps = st.tree_steps;
  out.tree_rows = st.tree_rows;
  out.spec_mean_width = st.mean_verify_width();
  out.spec_adaptive = st.spec_adaptive;
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
  out.cpu_fallbacks = graph::cpu_fallback_events(fallbacks_before);
  impl.record(out);
  // After every request: what the engine holds and the device runtime's own
  // count, so growth across requests shows in the log of a run that does
  // not fail.
  {
    // The runtime's totals and size histogram: its first two lines.
    std::string runtime = backend::runtime_memory_report();
    std::size_t end = 0;
    for (int line = 0; line < 2 && end != std::string::npos; ++line) {
      end = runtime.find('\n', end);
      if (end != std::string::npos) ++end;
    }
    if (end != std::string::npos) runtime.resize(end);
    std::fprintf(stderr, "lse-server: device memory after request: %s\n%s",
                 backend::describe_device_allocations().c_str(), runtime.c_str());
  }

  std::fprintf(stderr,
               "lse-server: prompt %d in %.2fs (%.1f tok/s) | decode %d in "
               "%.2fs (%.1f tok/s)%s\n",
               out.prefill_tokens, static_cast<double>(out.prefill_ns) / 1e9, out.prompt_per_second,
               out.decode_tokens, static_cast<double>(out.decode_ns) / 1e9,
               out.decode_per_second,
               out.acceptance >= 0.0
                   ? (" | accepted " + std::to_string(
                          static_cast<int>(out.acceptance * 100.0)) + "% (" +
                      std::to_string(out.spec_accepted) + "/" + std::to_string(out.spec_tested) +
                      " tested, " + std::to_string(out.spec_proposed) + " proposed, mean width " +
                      std::to_string(out.spec_mean_width).substr(0, 4) + ")").c_str()
                   : "");
  return out;
}

namespace {


// Not part of the OpenAI schema: how full the context is, so a client can
// offer to compact before (or once) it fills. tokens_used is the prompt plus
// the completion; context_length the most the engine holds (kv_len).
json context_of(int tokens_used, int context_length) {
  return json{{"tokens_used", tokens_used},
              {"context_length", context_length},
              {"tokens_remaining", std::max(0, context_length - tokens_used)}};
}

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
  if (o.mtp_depth != 0 || o.dflash2_depth != 0) t["spec_adaptive"] = o.spec_adaptive;
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
    t["spec_proposed"] = o.spec_proposed;
    t["spec_plain_steps"] = o.spec_plain_steps;
    t["tree_steps"] = o.tree_steps;
    t["tree_rows"] = o.tree_rows;
    t["spec_mean_width"] = o.spec_mean_width;
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
json chat_choice(const Outcome& o, const Request& r, const std::string& id) {
  detail::ChatResponseParser parser(r.framing.reasoning, r.chat, id);
  parser.push(o.text);
  parser.finish(o.hit_limit);
  return json{{"index", 0}, {"message", parser.message()}, {"logprobs", nullptr},
              {"finish_reason", parser.finish_reason(o.hit_limit)},
              {"stop_reason", stop_reason_name(o.stop, o.stopped_by_string)}};
}

json text_choice(const Outcome& o, bool expose_thinking) {
  const std::string& text = o.text;
  const bool hit_limit = o.hit_limit;
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
              {"finish_reason", finish_reason(hit_limit)},
              {"stop_reason", stop_reason_name(o.stop, o.stopped_by_string)}};
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
    const detail::ThinkingChoice thinking = detail::choose_thinking(impl.opt.thinking, r.thinking);
    if (thinking.error) {
      json extra{{"levels", json::array()}};
      for (const auto& l : impl.opt.thinking.levels) extra["levels"].push_back(l.id);
      send_coded_error(res, thinking.error->message, thinking.error->code, thinking.error->param,
                       std::move(extra));
      return;
    }
    r.framing = thinking.framing;
    try {
      r.chat = detail::prepare_chat(body, r.framing);
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

  // The prompt must leave room for at least one generated token.
  const int context_length = impl.model.config().kv_capacity();
  const int prompt_tokens = static_cast<int>(r.prompt.size());
  if (prompt_tokens >= context_length) {
    send_coded_error(res,
                     "the prompt is " + std::to_string(prompt_tokens) + " tokens and the context holds " +
                         std::to_string(context_length) + "; compact or shorten the conversation",
                     "context_full", chat ? "messages" : "prompt",
                     json{{"lse_context", context_of(prompt_tokens, context_length)}});
    return;
  }
  r.limits.max_tokens = impl.output_limit(r);

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
                {"choices", json::array({chat ? chat_choice(*out, r, id)
                                              : text_choice(*out, r.thinking_enabled)})},
                {"usage", usage_of(*out)},
                {"lse_context", context_of(out->context_tokens, out->context_length)},
                {"timings", timings_of(*out)}};
      if (!out->cpu_fallbacks.empty()) resp["lse_warnings"] = warnings_of(*out);
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

        detail::ChatResponseParser parser(r.framing.reasoning, r.chat, id);
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

        const char* why = stop_reason_name(out->stop, out->stopped_by_string);
        json last = chat
            ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", parser.finish_reason(out->hit_limit)},
                   {"stop_reason", why}}
            : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason(out->hit_limit)},
                   {"stop_reason", why}};
        json final{{"id", id},
                   {"object", chunk_object},
                   {"created", created},
                   {"model", impl.opt.model_id},
                   {"choices", json::array({last})},
                   {"lse_context", context_of(out->context_tokens, out->context_length)},
                   {"timings", timings_of(*out)}};
        if (!out->cpu_fallbacks.empty()) final["lse_warnings"] = warnings_of(*out);
        send(final);
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

Status Router::prepare_kernels() {
  std::lock_guard<std::mutex> held(impl_->generate_lock);
  // What a request that names no sampling settings runs with.
  LSE_ASSIGN_OR(runtime::SamplingParams sampling,
                detail::request_sampling(json::object(),
                                         impl_->model.config().sampling_defaults));
  runtime::Generator gen(impl_->model, sampling, impl_->opt.prefill);
  if (impl_->mtp != nullptr)
    gen.use_mtp(*impl_->mtp, impl_->opt.adaptive_mtp ? &impl_->mtp_widths : nullptr);
  if (impl_->dflash2 != nullptr)
    gen.use_dflash2(*impl_->dflash2, impl_->opt.adaptive_dflash2 ? &impl_->widths : nullptr,
                    runtime::draft_trees_enabled(impl_->opt.dflash2_tree, impl_->opt.adaptive_dflash2));
  return gen.prepare_kernels();
}

Result<runtime::PerplexityReport> Router::perplexity(
    std::span<const std::uint32_t> tokens, const runtime::PerplexityOptions& options,
    const runtime::PerplexityProgress& progress) {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> held(impl.generate_lock);
  if (impl.bound_live) {
    std::unique_ptr<Impl::SessionEntry>* previous = nullptr;
    {
      std::lock_guard sessions_held(impl.sessions_lock);
      auto it = impl.sessions.find(impl.bound);
      if (it != impl.sessions.end()) previous = &it->second;
    }
    impl.release_bindings(previous != nullptr ? &(*previous)->session : nullptr);
  }
  // Sampling settings do not enter scoring; the model's defaults fill them.
  LSE_ASSIGN_OR(runtime::SamplingParams sampling,
                detail::request_sampling(json::object(), impl.model.config().sampling_defaults));
  runtime::Generator gen(impl.model, sampling, impl.opt.prefill);
  auto report = runtime::score_perplexity(gen, impl.model, tokens, options, progress);
  (void)impl.model.drop_retained_passes();
  return report;
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
                         {"POST", "/v1/completions"},
                         {"GET", "/v1/lse/sessions"},
                         {"DELETE", "/v1/lse/sessions/:id"}};
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
                                                 ? runtime::dflash2_verify_depth(impl.dflash2->block_size()) : 0},
                            {"dflash2_adaptive", impl.dflash2 != nullptr && impl.opt.adaptive_dflash2},
                            {"mtp_adaptive", impl.mtp != nullptr && impl.opt.adaptive_mtp}});
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
    if (method == "GET" && path == "/v1/lse/sessions") {
      set_content(res, json{{"object", "list"}, {"data", impl.sessions_json()}});
      return res;
    }
    if (method == "DELETE" && matches("/v1/lse/sessions/:id", path)) {
      const std::string id(path.substr(std::string_view("/v1/lse/sessions/").size()));
      if (!impl.close_session(id)) {
        send_error(res, 404, "no session '" + id + "'", "invalid_request_error", "id");
        return res;
      }
      set_content(res, json{{"id", id}, {"object", "lse.session"}, {"deleted", true}});
      return res;
    }
    if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions")) {
      // A suspended or lost device answers at once rather than queueing
      // work it cannot run.
      if (auto refused = power_refusal(backend::device_power_state())) return std::move(*refused);
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

bool Router::close_session(const std::string& id) { return impl_->close_session(id); }

std::string Router::sessions_json() const { return impl_->sessions_json().dump(); }

std::string Router::metrics_json() const {
  std::lock_guard held(impl_->metrics_lock);
  json m{{"model", impl_->opt.model_id},
         {"generations", impl_->generations},
         {"mtp_enabled", impl_->mtp != nullptr},
         {"dflash2_enabled", impl_->dflash2 != nullptr}};
  m["last_timings"] = impl_->last_timings.is_null() ? json(nullptr) : impl_->last_timings;
  m["sessions"] = impl_->sessions_json();
  {
    json events = json::array();
    std::uint64_t total = 0;
    for (const auto& e : graph::cpu_fallback_events()) {
      events.push_back({{"cause", e.cause}, {"count", e.count}});
      total += e.count;
    }
    m["cpu_fallback"] = {{"allowed", graph::cpu_fallback_allowed()}, {"total", total},
                         {"events", std::move(events)}};
  }
  {
    std::lock_guard sessions_held(impl_->sessions_lock);
    m["session_evictions"] = impl_->evictions;
  }
  return m.dump();
}

std::optional<RouteReply> power_refusal(const std::optional<backend::DevicePowerState>& power) {
  if (!power) return std::nullopt;
  if (power->lost()) {
    RouteReply r;
    r.status = 503;
    json e = error_json(
        "the GPU was reset: its memory went with a host sleep. The engine must be opened again "
        "(the model reloads); the conversation itself is kept by the client",
        "device_lost", "");
    e["error"]["code"] = "device_lost";
    e["power"] = backend::power_state_name(power->state);
    r.body = e.dump();
    return r;
  }
  if (power->paused()) {
    RouteReply r;
    r.status = 503;
    json e = error_json(
        std::string("the GPU is ") + backend::power_state_name(power->state) +
            " (the app is in the background or the host is going to sleep); nothing was started, "
            "retry once it resumes",
        "engine_suspended", "");
    e["error"]["code"] = "suspended";
    e["power"] = backend::power_state_name(power->state);
    e["retry_after"] = 1;
    r.body = e.dump();
    r.headers.emplace_back("Retry-After", "1");
    return r;
  }
  return std::nullopt;
}

}  // namespace lse::server
