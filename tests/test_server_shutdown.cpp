// CPU-only lifecycle tests: no model, backend, runtime, or device is opened.
#define CPPHTTPLIB_USE_POLL
#include "httplib.h"
#include "lse/server/shutdown.hpp"
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

using namespace std::chrono_literals;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::abort(); } } while (false)

template <class Predicate> void wait_for(Predicate pred) {
  const auto limit = std::chrono::steady_clock::now() + 3s;
  while (!pred()) { CHECK(std::chrono::steady_clock::now() < limit); std::this_thread::sleep_for(1ms); }
}

int main() {
  // A stuck handler must produce a failing exit without running resource
  // destructors. Run before creating threads so fork inherits no held locks.
  int marker[2]; CHECK(pipe(marker) == 0);
  const pid_t child = fork(); CHECK(child >= 0);
  if (child == 0) {
    close(marker[0]);
    struct Resource { int fd; ~Resource() { const char c = 'x'; (void)write(fd, &c, 1); } } resource{marker[1]};
    std::atomic<bool> request{true};
    lse::server::detail::ShutdownWatch watch(request, 80ms, [] {});
    std::this_thread::sleep_for(5s);
    return 99;
  }
  close(marker[1]);
  int status = 0; CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
  char c = 0; CHECK(read(marker[0], &c, 1) == 0); close(marker[0]);

  // Finishing normally (including bind failure) must join a watcher even when
  // no shutdown signal ever arrived.
  {
    std::atomic<bool> request{false};
    int stops = 0;
    lse::server::detail::ShutdownWatch watch(request, 1s, [&] { ++stops; });
    watch.finish(); CHECK(stops == 0);
  }

  // Shutdown requested before listen starts is retried after readiness.
  {
    httplib::Server server;
    const int port = server.bind_to_any_port("127.0.0.1"); CHECK(port > 0);
    std::atomic<bool> request{true}, issued{false};
    lse::server::detail::ShutdownWatch watch(request, 2s, [&] {
      lse::server::detail::stop_listener_once(server, issued);
    });
    std::this_thread::sleep_for(30ms); CHECK(!issued.load());
    CHECK(server.listen_after_bind());
    watch.finish(); CHECK(issued.load());
  }

  // Real socket worker stays active beyond several watchdog polls. Listening
  // cannot finish (and backing cannot be destroyed) until that handler exits.
  {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> entered{false}, release{false}, returned{false};
    std::atomic<bool> request{false}, issued{false};
    std::atomic<int> stops{0};
    server.Get("/hold", [&](const httplib::Request&, httplib::Response& res) {
      entered.store(true);
      while (!release.load()) std::this_thread::sleep_for(1ms);
      res.set_content("drained", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1"); CHECK(port > 0);
    std::thread listener([&] { CHECK(server.listen_after_bind()); returned.store(true); });
    wait_for([&] { return server.is_running(); });
    auto client = std::async(std::launch::async, [&] {
      httplib::Client http("127.0.0.1", port);
      auto result = http.Get("/hold"); CHECK(result && result->body == "drained");
    });
    wait_for([&] { return entered.load(); });
    lse::server::detail::ShutdownWatch watch(request, 2s, [&] {
      ++stops;
      lse::server::detail::stop_listener_once(server, issued);
    });
    request.store(true);
    wait_for([&] { return issued.load(); });
    std::this_thread::sleep_for(100ms);
    CHECK(stops.load() >= 3); CHECK(!returned.load());
    release.store(true); client.get(); listener.join(); watch.finish();
    CHECK(returned.load());
  }
  std::puts("server shutdown: timeout, normal return, startup race, socket worker drain PASS");
}
