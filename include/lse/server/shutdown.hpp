#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

namespace lse::server::detail {

// httplib keeps is_running true while its workers drain, after closing the
// listener. Calling its stop twice in that interval asserts. Leave the flag
// unset before startup so a subsequent poll can close a late-starting listener.
template <class Server>
void stop_listener_once(Server& server, std::atomic<bool>& issued) {
  if (server.is_running() && !issued.exchange(true)) server.stop();
}

// The owner must finish/join this watchdog before destroying the stop target.
// A timeout deliberately skips destructors: handlers may still own GPU work.
class ShutdownWatch {
 public:
  ShutdownWatch(const std::atomic<bool>& requested,
                std::chrono::milliseconds grace, std::function<void()> stop)
      : worker_([this, &requested, grace, stop = std::move(stop)] {
          bool draining = false;
          std::chrono::steady_clock::time_point deadline;
          while (!finished_.load()) {
            if (requested.load()) {
              if (!draining) {
                draining = true;
                deadline = std::chrono::steady_clock::now() + grace;
                std::fputs("lse-server: stopping; draining active requests\n", stderr);
              }
              stop();
              if (!finished_.load() && std::chrono::steady_clock::now() >= deadline) {
                std::fputs("lse-server: shutdown grace expired; active resources retained until process exit\n", stderr);
                std::fflush(stderr);
                std::_Exit(1);
              }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }) {}
  ShutdownWatch(const ShutdownWatch&) = delete;
  ShutdownWatch& operator=(const ShutdownWatch&) = delete;
  ~ShutdownWatch() { finish(); }
  void finish() {
    finished_.store(true);
    if (worker_.joinable()) worker_.join();
  }
 private:
  std::atomic<bool> finished_{false};
  std::thread worker_;
};

}  // namespace lse::server::detail
