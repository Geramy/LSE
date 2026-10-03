#include "lse/server/in_process.hpp"

#include <optional>
#include <utility>

namespace lse::server {

InProcess::InProcess(Router& router) : router_(router) {}

InProcess::~InProcess() { shutdown(); }

void InProcess::reap_locked() {
  for (auto it = requests_.begin(); it != requests_.end();) {
    if (it->second->finished.load()) {
      if (it->second->worker.joinable()) it->second->worker.join();
      it = requests_.erase(it);
    } else {
      ++it;
    }
  }
}

std::uint64_t InProcess::submit(std::string method, std::string path, std::string body,
                                Callback callback) {
  auto r = std::make_shared<Request>();
  std::lock_guard held(lock_);
  if (closing_) return 0;
  reap_locked();
  r->id = next_id_++;
  requests_.emplace(r->id, r);
  ++started_;
  // Started under the lock so shutdown() always sees a joinable worker.
  r->worker = std::thread([this, req = r.get(), method = std::move(method),
                           path = std::move(path), body = std::move(body),
                           callback = std::move(callback)] {
    run(*req, method, path, body, callback);
  });
  return r->id;
}

bool InProcess::cancel(std::uint64_t id) {
  std::lock_guard held(lock_);
  const auto it = requests_.find(id);
  if (it == requests_.end() || it->second->finished.load()) return false;
  it->second->cancel.store(true);
  return true;
}

void InProcess::shutdown() {
  std::unique_lock held(lock_);
  closing_ = true;
  for (auto& [id, r] : requests_) r->cancel.store(true);
  changed_.wait(held, [&] {
    for (auto& [id, r] : requests_)
      if (!r->finished.load()) return false;
    return true;
  });
  reap_locked();
}

InProcess::Counters InProcess::counters() const {
  Counters c;
  c.started = started_.load();
  c.completed = completed_.load();
  c.failed = failed_.load();
  c.cancelled = cancelled_.load();
  std::lock_guard held(lock_);
  for (const auto& [id, r] : requests_) c.active += r->finished.load() ? 0 : 1;
  return c;
}

void InProcess::run(Request& r, std::string method, std::string path, std::string body,
                    const Callback& callback) {
  const auto emit = [&](Event event, int status, const std::string* data) {
    if (callback) callback(r.id, event, status, data);
  };
  const auto cancelled = [&] {
    ++cancelled_;
    const std::string b = error_body("request cancelled", "cancelled");
    emit(Event::kError, 499, &b);
  };
  try {
    RouteReply reply = router_.handle(method, path, body, &r.cancel);
    if (!reply.stream) {
      if (r.cancel.load()) {
        cancelled();
      } else if (reply.status >= 200 && reply.status < 300) {
        ++completed_;
        emit(Event::kResponse, reply.status, &reply.body);
      } else {
        ++failed_;
        emit(Event::kError, reply.status, &reply.body);
      }
    } else {
      // The HTTP adapter writes an error event into the stream and ends it
      // without "[DONE]"; here the same object is the final kError.
      std::optional<std::string> failure;
      const bool completed = reply.stream([&](RouteReply::Event event, const std::string& json) {
        if (r.cancel.load()) return false;
        if (event == RouteReply::Event::kError) {
          failure = json;
          return true;
        }
        emit(Event::kChunk, 200, &json);
        return true;
      });
      if (r.cancel.load()) {
        cancelled();
      } else if (failure) {
        ++failed_;
        emit(Event::kError, 500, &*failure);
      } else if (completed) {
        ++completed_;
        emit(Event::kDone, 200, nullptr);
      } else {
        ++failed_;
        const std::string b = error_body("the stream ended early", "server_error");
        emit(Event::kError, 500, &b);
      }
    }
  } catch (const std::exception&) {
    ++failed_;
    const std::string b = error_body("internal error", "server_error");
    emit(Event::kError, 500, &b);
  }
  // Under the lock, so a reaper never joins a worker still waiting for it.
  std::lock_guard held(lock_);
  r.finished.store(true);
  changed_.notify_all();
}

}  // namespace lse::server
