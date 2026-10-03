#include "lse/server/http_server.hpp"
#include "lse/server/shutdown.hpp"

#include <atomic>
#include <string>
#include <utility>

// httplib falls back to select() without this, and select() refuses any
// socket whose descriptor is >= FD_SETSIZE (1024). A loaded model holds
// well over a thousand descriptors, so the listening socket and every
// connection land above that line and are closed without a reply. poll()
// has no such ceiling.
#define CPPHTTPLIB_USE_POLL
#include "httplib.h"

namespace lse::server {

struct HttpServer::Impl {
  Router& router;
  std::string host;
  int port;
  httplib::Server http;
  // Set once the server is shutting down. A decode in flight does not watch
  // the listening socket, so without this it holds the worker pool open for as
  // long as its completion takes and the server outlives the ^C that asked it
  // to stop.
  std::atomic<bool> stopping{false};
  std::atomic<bool> listener_stop_issued{false};
  bool bound = false;
  bool routed = false;

  Impl(Router& r, std::string h, int p) : router(r), host(std::move(h)), port(p) {}

  [[nodiscard]] bool authorized(const httplib::Request& req) const {
    const std::string& key = router.options().api_key;
    if (key.empty()) return true;
    auto it = req.headers.find("Authorization");
    return it != req.headers.end() && it->second == "Bearer " + key;
  }

  // Hands one request to the router and writes its answer: a JSON body, or a
  // server-sent-event stream that frames each event as "data: <json>".
  void forward(const httplib::Request& req, httplib::Response& res) {
    RouteReply reply = router.handle(req.method, req.path, req.body, &stopping);
    res.status = reply.status;
    for (const auto& [name, value] : reply.headers) res.set_header(name, value);
    if (!reply.stream) {
      res.set_content(reply.body, "application/json");
      return;
    }
    // The generation runs inside the provider so a delta reaches the socket
    // as it resolves rather than at the end.
    res.set_chunked_content_provider(
        "text/event-stream",
        [stream = std::move(reply.stream)](std::size_t, httplib::DataSink& sink) {
          const bool completed = stream([&sink](RouteReply::Event, const std::string& json) {
            const std::string frame = "data: " + json + "\n\n";
            return sink.write(frame.data(), frame.size());
          });
          if (completed) {
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
          }
          sink.done();
          return true;
        });
  }

  void install_routes() {
    if (routed) return;
    routed = true;
    http.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
          res.status = 500;
          res.set_content(error_body("internal error", "server_error"), "application/json");
        });

    // Any client the user points at this is entitled to ask from a browser.
    http.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          res.set_header("Access-Control-Allow-Origin", "*");
          res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
          res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
          if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
          }
          if (!authorized(req)) {
            res.status = 401;
            res.set_content(error_body("missing or invalid Authorization header",
                                       "invalid_request_error"),
                            "application/json");
            return httplib::Server::HandlerResponse::Handled;
          }
          return httplib::Server::HandlerResponse::Unhandled;
        });

    // Exactly the router's routes; anything else is httplib's own 404.
    auto handler = [this](const httplib::Request& req, httplib::Response& res) {
      forward(req, res);
    };
    for (const Route& route : Router::routes()) {
      const std::string_view method(route.method);
      if (method == "GET") http.Get(route.pattern, handler);
      else if (method == "DELETE") http.Delete(route.pattern, handler);
      else http.Post(route.pattern, handler);
    }
  }
};

HttpServer::HttpServer(Router& router, std::string host, int port)
    : impl_(std::make_unique<Impl>(router, std::move(host), port)) {}

HttpServer::~HttpServer() = default;

void HttpServer::stop() {
  impl_->stopping.store(true, std::memory_order_relaxed);
  detail::stop_listener_once(impl_->http, impl_->listener_stop_issued);
}

Status HttpServer::bind() {
  Impl& impl = *impl_;
  impl.install_routes();
  if (impl.bound) return OkStatus();
  if (impl.port == 0) {
    const int any = impl.http.bind_to_any_port(impl.host);
    if (any <= 0) {
      return LSE_ERROR(kIoError, "could not listen on ", impl.host, ":0");
    }
    impl.port = any;
  } else if (!impl.http.bind_to_port(impl.host, impl.port)) {
    return LSE_ERROR(kIoError, "could not listen on ", impl.host, ":",
                     std::to_string(impl.port));
  }
  impl.bound = true;
  return OkStatus();
}

int HttpServer::port() const noexcept { return impl_->port; }

Status HttpServer::listen() {
  Impl& impl = *impl_;
  impl.install_routes();
  if (impl.stopping.load()) return OkStatus();
  if (!impl.bound) {
    if (!impl.http.listen(impl.host, impl.port)) {
      return LSE_ERROR(kIoError, "could not listen on ", impl.host, ":",
                       std::to_string(impl.port));
    }
    return OkStatus();
  }
  if (!impl.http.listen_after_bind()) {
    return LSE_ERROR(kIoError, "could not listen on ", impl.host, ":",
                     std::to_string(impl.port));
  }
  return OkStatus();
}

}  // namespace lse::server
