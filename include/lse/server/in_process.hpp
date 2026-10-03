// Requests handed to a Router from inside the process: the transport behind
// lse_request. Each request runs on its own worker, as each HTTP request runs
// on a server worker, so requests queue for the device exactly as they do
// behind the HTTP server, and each can be cancelled while queued or running.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "lse/server/router.hpp"

namespace lse::server {

class InProcess {
 public:
  enum class Event {
    kResponse,  // a complete 2xx body (final)
    kChunk,     // one stream event: the JSON an SSE "data:" line carries
    kDone,      // the stream completed ("[DONE]"); no data (final)
    kError,     // the error envelope, with the HTTP status it maps to (final)
  };
  // `data` is valid only during the call; null for kDone.
  using Callback = std::function<void(std::uint64_t id, Event event, int status,
                                      const std::string* data)>;

  struct Counters {
    std::uint64_t started = 0, active = 0, completed = 0, failed = 0, cancelled = 0;
  };

  explicit InProcess(Router& router);
  // Cancels whatever is still running and waits for it.
  ~InProcess();
  InProcess(const InProcess&) = delete;
  InProcess& operator=(const InProcess&) = delete;

  // Starts one request and returns its id at once; 0 when shutting down.
  std::uint64_t submit(std::string method, std::string path, std::string body,
                       Callback callback);
  // False when `id` is unknown or already finished. A cancelled request ends
  // with kError, status 499, and nothing after the cancel.
  bool cancel(std::uint64_t id);
  // Refuses new requests, cancels every running one and waits for them all.
  void shutdown();

  [[nodiscard]] Counters counters() const;

 private:
  struct Request {
    std::uint64_t id = 0;
    std::atomic<bool> cancel{false};
    std::atomic<bool> finished{false};
    std::thread worker;
  };
  void run(Request& r, std::string method, std::string path, std::string body,
           const Callback& callback);
  void reap_locked();

  Router& router_;
  mutable std::mutex lock_;
  std::condition_variable changed_;
  std::map<std::uint64_t, std::shared_ptr<Request>> requests_;
  std::uint64_t next_id_ = 1;
  bool closing_ = false;
  std::atomic<std::uint64_t> started_{0}, completed_{0}, failed_{0}, cancelled_{0};
};

}  // namespace lse::server
