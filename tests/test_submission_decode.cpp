// Reuse the synthetic tiny model and state fixtures; select only the cases
// below, and bind an explicit host-only backend before any graph exists.
#define main runtime_fixture_unused_main
#include "test_runtime.cpp"
#undef main
#include "lse/backends/cpu/cpu_backend.hpp"

namespace {
struct ObservedCpu : lse::backend::CpuBackend {
  bool advertise = false, active = false, fail_begin = false, fail_end = false;
  unsigned begins = 0, ends = 0, cancelled = 0;
  std::span<const lse::graph::KernelToolchain> toolchains() const noexcept {
    static const lse::graph::KernelToolchain loom{lse::graph::Dialect::kLoom, nullptr, nullptr};
    return advertise ? std::span(&loom, 1) : std::span<const lse::graph::KernelToolchain>{};
  }
  lse::Status begin_decode_sample_impl(std::uint64_t) {
    LSE_EXPECT(!active);
    active = true;
    ++begins;
    if (fail_begin) return LSE_ERROR(kDeviceError, "injected begin drain failure");
    return lse::OkStatus();
  }
  lse::Status end_decode_sample_impl(std::uint64_t elapsed, bool eligible) {
    LSE_EXPECT(active && elapsed > 0);
    // The interpreter is deliberately ineligible for device policy evidence.
    LSE_EXPECT(!eligible);
    ++ends;
    if (fail_end) return LSE_ERROR(kDeviceError, "injected end drain failure");
    active = false;
    return lse::OkStatus();
  }
  void cancel_decode_sample_impl() noexcept { active = false; ++cancelled; }
};
lse::backend::BackendAdapter<ObservedCpu> observed_backend;
lse::backend::SingleDevice observed_device(observed_backend);
lse::backend::IDeviceSet* host_set() { return &observed_device; }
}

LSE_TEST(submission_decode_observation_preserves_tokens_and_state) {
  auto& backend = observed_backend.impl();
  MtpFixture fx = build_mtp_fixture();
  LSE_EXPECT(fx.ok);
  if (!fx.ok) return;
  GenerationLimits limits;
  limits.max_tokens = 5;
  const std::vector<std::uint32_t> prompt{2, 11, 33};
  Generator reference(*fx.lm, greedy_params());
  Session reference_session("batch-reference", fx.lm->state_slots());
  auto expected = reference.generate(reference_session, prompt, limits);
  LSE_EXPECT(expected.ok());
  if (!expected.ok()) return;
  LSE_EXPECT_EQ(backend.begins, 0u);
  backend.advertise = true;
  Generator observed(*fx.lm, greedy_params());
  Session observed_session("batch-observed", fx.lm->state_slots());
  auto actual = observed.generate(observed_session, prompt, limits);
  LSE_EXPECT(actual.ok());
  if (!actual.ok()) return;
  LSE_EXPECT(*actual == *expected);
  LSE_EXPECT(observed_session.history() == reference_session.history());
  LSE_EXPECT_EQ(observed_session.position(), reference_session.position());
  LSE_EXPECT_EQ(backend.begins, 4u);
  LSE_EXPECT_EQ(backend.ends, 4u);
  LSE_EXPECT(!backend.active);
  // Prefill-only and cancellation after its token must never create a decode
  // observation or run an extra token for calibration.
  limits.max_tokens = 0;
  LSE_EXPECT(observed.generate(prompt, limits).ok());
  limits.max_tokens = 5;
  auto cancelled = observed.generate(prompt, limits, [](std::uint32_t) { return false; });
  LSE_EXPECT(cancelled.ok() && cancelled->size() == 1);
  LSE_EXPECT_EQ(backend.begins, 4u);
  LSE_EXPECT_EQ(backend.ends, 4u);
  LSE_EXPECT_EQ(backend.cancelled, 0u);
}

LSE_TEST(submission_decode_boundary_failure_cancels_the_observation) {
  auto& backend = observed_backend.impl();
  for (bool fail_at_begin : {true, false}) {
    backend.advertise = true;
    backend.begins = backend.ends = backend.cancelled = 0;
    backend.fail_begin = fail_at_begin;
    backend.fail_end = !fail_at_begin;
    MtpFixture fx = build_mtp_fixture();
    LSE_EXPECT(fx.ok);
    if (!fx.ok) return;
    Generator gen(*fx.lm, greedy_params());
    GenerationLimits limits;
    limits.max_tokens = 3;
    auto result = gen.generate({2, 11, 33}, limits);
    LSE_EXPECT(!result.ok());
    LSE_EXPECT_EQ(backend.begins, 1u);
    LSE_EXPECT_EQ(backend.ends, fail_at_begin ? 0u : 1u);
    LSE_EXPECT_EQ(backend.cancelled, 1u);
    LSE_EXPECT(!backend.active);
    backend.fail_begin = backend.fail_end = false;
  }
}

int main() {
  if (!observed_backend.init(0).ok()) return 1;
  lse::graph::register_device_set_factory(host_set);
  auto* scheduler = lse::graph::default_scheduler();
  if (scheduler == nullptr || &scheduler->backend() != &observed_backend) return 2;
  scheduler->set_mode(lse::graph::Scheduler::Mode::kHostOnly);
  auto& cases = lse::test::Registry::get().cases;
  std::erase_if(cases, [](const auto& test) { return !test.name.starts_with("submission_decode_"); });
  return lse::test::run_all();
}
