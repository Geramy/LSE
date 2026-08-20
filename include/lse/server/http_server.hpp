// An OpenAI-shaped HTTP surface over one loaded model.
//
// One model, one device, so generation is serialized: requests queue rather
// than interleave. That is a property of this server and not of the engine,
// which decodes several sequences in one step; a batching front end belongs
// here later and does not change the wire format.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "lse/core/status.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/mtp.hpp"
#include "lse/tokenizer/tokenizer.hpp"

namespace lse::server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  // Reported as the model id, and what a request's "model" field is matched
  // against. A request naming something else is still served, since there is
  // only one model loaded, and the response says which one answered.
  std::string model_id;
  // When set, every request must carry `Authorization: Bearer <key>`.
  std::string api_key;
  // Refused above this, so one request cannot take the whole KV pool.
  std::int32_t max_tokens_cap = 4096;
};

class HttpServer {
 public:
  HttpServer(model::HybridLM& model, tokenizer::Tokenizer& tok,
             ServerOptions options);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Speculative decoding, when the checkpoint shipped a module.
  void use_mtp(model::MtpModule& mtp) noexcept;

  // Blocks until stop() is called or the listen fails.
  Status listen();
  void stop();

 private:
  struct Impl;
  struct Run;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::server
