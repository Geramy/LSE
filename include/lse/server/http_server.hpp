// The OpenAI-shaped HTTP surface: a thin adapter that carries requests from a
// socket to a Router and its answers back, adding only what is HTTP's own
// (CORS, the bearer key, server-sent-event framing).
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "lse/core/status.hpp"
#include "lse/server/router.hpp"

namespace lse::server {

class HttpServer {
 public:
  // Serves `router` on host:port, requiring router.options().api_key when it
  // is set.
  HttpServer(Router& router, std::string host, int port);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds the listening socket without serving yet, so a caller can report a
  // port that is taken before it commits to listening on another thread.
  // Port 0 takes any free port; port() then names it.
  Status bind();
  [[nodiscard]] int port() const noexcept;
  // Blocks until stop() is called or the listen fails. Binds first unless
  // bind() already did.
  Status listen();
  // Stops the listener and ends HTTP generations in flight; requests that
  // arrive meanwhile answer 503. Idempotent and safe before listen starts.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lse::server
