// CPU fallback outside a request: it is refused unless the run chose the CPU
// backend, and then the device set and the scheduler refuse the host
// interpreter and say why. --no-cpu-fallback refuses it even then.
//
// The per-group refusal inside a request is covered by test_capture_prefill,
// which has a device that declines one op. Here the only backend is the cpu
// one, so nothing opens a GPU.
#include <cstdlib>
#include <string>

#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/place/devices.hpp"

using namespace lse;
using namespace lse::graph;

namespace {

bool mentions(const Status& s, const char* text) {
  return s.to_string().find(text) != std::string::npos;
}

// Fallback disallowed for one scope, restored on every exit.
struct NoCpuFallback {
  bool previous = cpu_fallback_allowed();
  NoCpuFallback() { set_cpu_fallback_allowed(false); }
  ~NoCpuFallback() { set_cpu_fallback_allowed(previous); }
};

// Fallback allowed for one scope, as for a run that chose the CPU backend.
struct CpuBackendChosen {
  bool previous = cpu_fallback_allowed();
  CpuBackendChosen() { set_cpu_fallback_allowed(true); }
  ~CpuBackendChosen() { set_cpu_fallback_allowed(previous); }
};

// One environment variable for one scope.
struct ScopedEnv {
  std::string name;
  bool had = false;
  std::string saved;
  ScopedEnv(const char* n, const char* value) : name(n) {
    if (const char* v = std::getenv(n)) { had = true; saved = v; }
    if (value != nullptr) ::setenv(n, value, 1); else ::unsetenv(n);
  }
  ~ScopedEnv() {
    if (had) ::setenv(name.c_str(), saved.c_str(), 1); else ::unsetenv(name.c_str());
  }
};

}  // namespace

// First, while nothing has set the policy: a process whose environment chose
// no CPU backend starts with fallback refused.
LSE_TEST(a_process_that_chose_no_cpu_backend_starts_with_fallback_refused) {
  if (cpu_backend_selected(std::getenv("LSE_BACKEND"), std::getenv("LSE_POOL"))) {
    lse::test::skip("the environment chose the cpu backend");
    return;
  }
  LSE_EXPECT(!cpu_fallback_allowed());
}

LSE_TEST(cpu_fallback_is_allowed_only_for_a_run_that_chose_the_cpu_backend) {
  ScopedEnv strict_unset("LSE_REQUIRE_DEVICE_KERNELS", nullptr);
  LSE_EXPECT(!cpu_fallback_default(nullptr, nullptr));
  LSE_EXPECT(!cpu_fallback_default("hrx", nullptr));
  LSE_EXPECT(!cpu_fallback_default(nullptr, "hrx:0"));
  LSE_EXPECT(!cpu_fallback_default(nullptr, "hrx:0,hrx:1"));
  LSE_EXPECT(!cpu_fallback_default(nullptr, "cpux:0"));
  LSE_EXPECT(cpu_fallback_default("cpu", nullptr));
  LSE_EXPECT(cpu_fallback_default(nullptr, "cpu"));
  LSE_EXPECT(cpu_fallback_default(nullptr, "cpu:0"));
  LSE_EXPECT(cpu_fallback_default(nullptr, "hrx:0, cpu:0"));
  // The qualification switch refuses it even for a CPU run.
  ScopedEnv strict("LSE_REQUIRE_DEVICE_KERNELS", "1");
  LSE_EXPECT(!cpu_fallback_default("cpu", nullptr));
  LSE_EXPECT(!cpu_fallback_default(nullptr, "cpu:0"));
}

LSE_TEST(a_cpu_pool_member_opens_while_cpu_fallback_is_allowed) {
  CpuBackendChosen chosen;
  auto opened = place::Devices::open("cpu:0");
  LSE_EXPECT_OK(opened.status());
}

LSE_TEST(a_cpu_pool_member_is_refused_when_cpu_fallback_is_disabled) {
  NoCpuFallback off;
  auto opened = place::Devices::open("cpu:0");
  LSE_EXPECT(!opened.ok());
  if (opened.ok()) return;
  std::printf("       %s\n", opened.status().to_string().c_str());
  LSE_EXPECT(opened.status().code() == StatusCode::kDeviceError);
  // Names the member, the reason and how fallback is chosen.
  LSE_EXPECT(mentions(opened.status(), "cpu:0"));
  LSE_EXPECT(mentions(opened.status(), "host interpreter"));
  LSE_EXPECT(mentions(opened.status(), "CPU fallback is refused"));
}

LSE_TEST(the_default_device_never_falls_back_to_the_cpu_when_disabled) {
  // The default order ends at the cpu backend. With only that one to try, the
  // run gets no device instead of a quiet host-interpreter run.
  ScopedEnv backend("LSE_BACKEND", "cpu");
  {
    CpuBackendChosen chosen;
    auto opened = place::Devices::open("");
    LSE_EXPECT_OK(opened.status());
  }
  NoCpuFallback off;
  auto opened = place::Devices::open("");
  LSE_EXPECT(!opened.ok());
  if (opened.ok()) return;
  std::printf("       %s\n", opened.status().to_string().c_str());
  LSE_EXPECT(mentions(opened.status(), "no device backend came up"));
  LSE_EXPECT(mentions(opened.status(), "CPU fallback is"));
  LSE_EXPECT(mentions(opened.status(), "LSE_BACKEND=cpu"));
  // The backend that was passed over is listed with its reason.
  LSE_EXPECT(mentions(opened.status(), "cpu: refused"));
}

LSE_TEST(a_scheduler_with_no_code_generator_refuses_a_step_when_disabled) {
  auto be = backend::create_backend("cpu");
  LSE_EXPECT(be.ok());
  if (!be.ok()) return;
  LSE_EXPECT_OK((*be)->init(0));
  Scheduler sched(**be);
  LSE_EXPECT(sched.mode() == Scheduler::Mode::kHostOnly);

  Array x = Array::full(Shape{64}, DType::kF32, 1.5f);
  {
    NoCpuFallback off;
    Array y = x + x;
    const NodePtr roots[] = {y.node()};
    const Status refused = sched.eval(roots, true);
    LSE_EXPECT(!refused.ok());
    std::printf("       %s\n", refused.to_string().c_str());
    LSE_EXPECT(mentions(refused, "CPU fallback disabled"));
    LSE_EXPECT(mentions(refused, "host interpreter"));
  }
  // Allowed again, the same scheduler runs the step on the host.
  CpuBackendChosen chosen;
  Array y = x + x;
  const NodePtr roots[] = {y.node()};
  LSE_EXPECT_OK(sched.eval(roots, true));
}

LSE_TEST(require_device_kernels_is_read_as_the_same_switch) {
  {
    ScopedEnv strict("LSE_REQUIRE_DEVICE_KERNELS", "1");
    LSE_EXPECT(device_kernels_required_by_environment());
  }
  {
    ScopedEnv other("LSE_REQUIRE_DEVICE_KERNELS", "0");
    LSE_EXPECT(!device_kernels_required_by_environment());
  }
  ScopedEnv unset("LSE_REQUIRE_DEVICE_KERNELS", nullptr);
  LSE_EXPECT(!device_kernels_required_by_environment());
}

LSE_TEST_MAIN()
