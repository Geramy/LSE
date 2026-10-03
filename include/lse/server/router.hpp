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
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/runtime/prefill_batch.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/mtp.hpp"
#include "lse/model/dflash2.hpp"
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
  // Refused above this, so one request cannot take the whole KV pool.
  std::int32_t max_tokens_cap = 4096;
  std::uint32_t mtp_depth = 3;
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
};

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
