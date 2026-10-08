// The OpenAI-shaped request surface over one loaded model, independent of how
// a request arrives. The HTTP server (http_server.hpp) and the in-process C
// API (lse/lse.h) are both thin adapters over this: the same JSON in, the same
// JSON out, the same queueing for the device.
//
// One model, one device, so generation is serialized: requests queue rather
// than interleave. That is a property of this surface and not of the engine,
// which decodes several sequences in one step; a batching front end belongs
// here later and does not change the wire format.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/core/status.hpp"
#include "lse/runtime/perplexity.hpp"
#include "lse/runtime/prefill_batch.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/models/thinking_controls.hpp"
#include "lse/tokenizer/tokenizer.hpp"

namespace lse::server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  // Reported as the model id, and what a request's "model" field is matched
  // against. A request naming something else is still served, since there is
  // only one model loaded, and the response says which one answered.
  std::string model_id;
  // When set, every HTTP request must carry `Authorization: Bearer <key>`.
  std::string api_key;
  // An operator's cap on generated tokens per request: a request asking for
  // more is refused, one asking for none is held to it. 0 (the default): no
  // cap. LSE imposes no output limit of its own; generation ends at a stop
  // token, a stop sequence, the request's max_tokens, a limit the model's own
  // generation_config sets, or a full context.
  std::int32_t max_tokens_cap = 0;
  // The thinking levels the model's chat template defines.
  models::ThinkingControls thinking;
  std::uint32_t mtp_depth = 3;
  // DFlash2 verifies the prefix of each draft block that DraftWidthPolicy
  // picks (true), or the whole block every step (false).
  bool adaptive_dflash2 = true;
  // DFlash2 steps may verify a draft tree (runtime::DraftTree): positive on,
  // zero off, negative the device's default (runtime::draft_trees_enabled).
  int dflash2_tree = -1;
  // A sampled MTP request chains as deep as DraftWidthPolicy picks, up to
  // runtime::kMaxMtpDepth (true), or mtp_depth every step (false).
  bool adaptive_mtp = true;
  runtime::PrefillBatch prefill;
  // Where the served model and its draft came from, and what they run on, for
  // /v1/lse/model_info and /v1/lse/estimate. Those routes describe the loaded
  // model only: a server never inspects a path a client names. An empty
  // model_path answers them 404.
  std::string model_path;
  std::string draft_path;    // the MTP module or DFlash2 checkpoint, if any
  std::string device_arch;   // e.g. "gfx1201"; empty when unknown
  bool fragmented_kv = true; // K/V in Loom fragments rather than contiguous pools
  // Sessions kept between requests (keyed by a request's session_id). Beyond
  // either limit, least recently used idle sessions are evicted; an evicted
  // session's next request prefills again. 0: no limit. Under memory
  // pressure idle sessions are evicted whatever the limits.
  std::size_t max_sessions = 8;
  std::size_t session_memory_budget = 0;  // bytes of KV and state, all sessions
};

// What a route answers. A streaming answer has status 200 and carries
// `stream`, which runs the generation and hands each event to `send` as it
// resolves; `send` returns false to abandon the stream (a disconnected
// client). The stream function returns true when it completed normally, in
// which case the transport's terminator ("data: [DONE]") follows.
struct RouteReply {
  enum class Event { kChunk, kError };
  using Send = std::function<bool(Event, const std::string& json)>;
  int status = 200;
  std::string body;  // JSON; empty only for a stream
  std::function<bool(const Send&)> stream;
  // Extra HTTP headers (Retry-After on a 503 while the device is suspended).
  std::vector<std::pair<std::string, std::string>> headers;
};

// What a completion request is answered while the device cannot take work,
// or nullopt when it can (or its power is not tracked):
//  - suspended (the host app is in the background, or the host is going to
//    sleep): 503, type "engine_suspended", with Retry-After; nothing was
//    started, so the client retries once the device resumes;
//  - lost (the device's memory went with a host sleep): 503, type
//    "device_lost"; not retryable until the engine is opened again.
[[nodiscard]] std::optional<RouteReply> power_refusal(const std::optional<backend::DevicePowerState>& power);

// A route this surface answers: the method and the path pattern (":id" names
// one path segment), so a transport can register exactly these.
struct Route {
  const char* method;
  const char* pattern;
};

class Router {
 public:
  Router(model::HybridLM& model, tokenizer::Tokenizer& tok, ServerOptions options);
  ~Router();
  Router(const Router&) = delete;
  Router& operator=(const Router&) = delete;

  // Speculative decoding, when the checkpoint shipped a module.
  void use_mtp(model::MtpModule& mtp) noexcept;
  void use_dflash2(model::DFlash2Module& draft) noexcept;

  // Makes resident, before the first request, every kernel a request can
  // launch with this router's options and draft module
  // (runtime::Generator::prepare_kernels). Called once at load, after the
  // draft module is attached.
  Status prepare_kernels();

  [[nodiscard]] const ServerOptions& options() const noexcept;

  // Every route handle() answers, in registration order.
  [[nodiscard]] static const std::vector<Route>& routes();

  // Answers one request. `stopping`, when given, is watched for the life of
  // the request: once set, a request that has not started answers 503 and a
  // generation in flight (or queued for the device) ends early. A path no
  // route matches answers 404.
  [[nodiscard]] RouteReply handle(std::string_view method, std::string_view path,
                                  std::string_view body,
                                  const std::atomic<bool>* stopping = nullptr);

  // Scores `tokens` with the target model alone (runtime::score_perplexity),
  // queued behind any generation in flight as a request is. A draft module is
  // not used: it does not change the target's logits. What the model retained
  // for the last session is released first; that session keeps its KV unless
  // a draft module is attached, as when another request runs.
  [[nodiscard]] Result<runtime::PerplexityReport> perplexity(
      std::span<const std::uint32_t> tokens, const runtime::PerplexityOptions& options,
      const runtime::PerplexityProgress& progress = {});

  // Counters and the timings of the last completed generation, as JSON.
  [[nodiscard]] std::string metrics_json() const;

  // Releases a session's KV, state and what the model holds for it, waiting
  // for a generation in flight to finish. False when there is no such
  // session. The same as DELETE /v1/lse/sessions/{id}.
  bool close_session(const std::string& id);
  // The live sessions as a JSON array: id, tokens, bytes, requests.
  [[nodiscard]] std::string sessions_json() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The error envelope clients parse. Anything non-2xx wears it.
[[nodiscard]] std::string error_body(const std::string& message,
                                     const std::string& type = "invalid_request_error",
                                     const std::string& param = "");

}  // namespace lse::server
