// Device power (backend::DevicePower, lse_power_prepare/resume, lse_status
// "power", and the router's answer while the device cannot take work)
// against a fake runtime: a registered DevicePower whose state the test
// moves through active, suspended and lost the way the mac_linuxgpu HSA
// runtime reports a background suspend and a sleep that lost device memory.
#include "harness.hpp"

#include <string>

#include <nlohmann/json.hpp>

#include "lse/backend/backend.hpp"
#include "lse/lse.h"
#include "lse/server/router.hpp"

using namespace lse;
using json = nlohmann::json;

namespace {

// The fake driver: prepare suspends with device memory kept, resume makes
// it active again, unless a "sleep" lost the device meanwhile.
struct FakeDevice {
  backend::DevicePowerState state{};
  int prepares = 0;
  int resumes = 0;
  std::uint32_t last_drain = 0;

  void install() {
    state.state = backend::PowerState::kActive;
    backend::DevicePower p;
    p.state = [this] { return std::optional<backend::DevicePowerState>(state); };
    p.prepare = [this](std::uint32_t drain) -> Result<backend::DevicePowerState> {
      ++prepares;
      last_drain = drain;
      if (state.lost()) return state;
      state.state = backend::PowerState::kSuspended;
      state.flags = 1u << 0 | 1u << 3 | 1u << 4;  // vram_preserved, client_hold, quiesced
      ++state.generation;
      ++state.quiesces;
      return state;
    };
    p.resume = [this]() -> Result<backend::DevicePowerState> {
      ++resumes;
      if (state.lost()) return LSE_ERROR(kDeviceError, "device lost: the GPU's memory went with a host sleep");
      state.state = backend::PowerState::kActive;
      state.flags = 0;
      ++state.generation;
      return state;
    };
    backend::register_device_power(std::move(p));
  }

  void host_slept() {
    state.state = backend::PowerState::kLost;
    state.flags = 1u << 1;  // system_sleep
    ++state.generation;
    ++state.losses;
  }
};

json status_json() {
  char* out = nullptr;
  LSE_EXPECT(lse_status(nullptr, &out) == LSE_OK);
  json j = json::parse(out);
  lse_free(out);
  return j;
}

}  // namespace

LSE_TEST(untracked_power_reports_unknown_and_refuses_nothing) {
  backend::clear_device_power();
  LSE_EXPECT(!backend::device_power_state().has_value());
  const json s = status_json();
  LSE_EXPECT(s["power"]["state"] == "unknown");
  LSE_EXPECT(s["power"]["tracked"] == false);
  LSE_EXPECT(!server::power_refusal(backend::device_power_state()).has_value());

  char* out = nullptr;
  char* err = nullptr;
  LSE_EXPECT(lse_power_prepare(nullptr, 100, &out, &err) == LSE_ERR_FAILED);
  LSE_EXPECT(err != nullptr && std::string(err).find("low-power") != std::string::npos);
  lse_free(out);
  lse_free(err);
}

LSE_TEST(prepare_suspends_and_requests_are_refused_retryably) {
  FakeDevice dev;
  dev.install();
  LSE_EXPECT(status_json()["power"]["state"] == "active");
  LSE_EXPECT(!server::power_refusal(backend::device_power_state()).has_value());

  char* out = nullptr;
  char* err = nullptr;
  LSE_EXPECT(lse_power_prepare(nullptr, 1500, &out, &err) == LSE_OK);
  LSE_EXPECT(err == nullptr);
  LSE_EXPECT_EQ(dev.last_drain, 1500u);
  const json prepared = json::parse(out);
  lse_free(out);
  LSE_EXPECT(prepared["state"] == "suspended");
  LSE_EXPECT(prepared["flags"].size() == 3);
  LSE_EXPECT(prepared["quiesces"] == 1);

  const auto refused = server::power_refusal(backend::device_power_state());
  LSE_EXPECT(refused.has_value());
  LSE_EXPECT_EQ(refused->status, 503);
  const json body = json::parse(refused->body);
  LSE_EXPECT(body["error"]["type"] == "engine_suspended");
  LSE_EXPECT(body["error"]["code"] == "suspended");
  LSE_EXPECT(body["retry_after"] == 1);
  LSE_EXPECT(refused->headers.size() == 1 && refused->headers[0].first == "Retry-After");

  LSE_EXPECT(lse_power_resume(nullptr, &out, &err) == LSE_OK);
  LSE_EXPECT(json::parse(out)["state"] == "active");
  lse_free(out);
  LSE_EXPECT(!server::power_refusal(backend::device_power_state()).has_value());
  LSE_EXPECT_EQ(dev.prepares, 1);
  LSE_EXPECT_EQ(dev.resumes, 1);
  backend::clear_device_power();
}

LSE_TEST(a_lost_device_is_reported_and_resume_says_so) {
  FakeDevice dev;
  dev.install();
  char* out = nullptr;
  char* err = nullptr;
  LSE_EXPECT(lse_power_prepare(nullptr, 0, &out, &err) == LSE_OK);
  lse_free(out);
  dev.host_slept();

  const json s = status_json();
  LSE_EXPECT(s["power"]["state"] == "lost");
  LSE_EXPECT(s["power"]["losses"] == 1);

  const auto refused = server::power_refusal(backend::device_power_state());
  LSE_EXPECT(refused.has_value());
  LSE_EXPECT_EQ(refused->status, 503);
  const json body = json::parse(refused->body);
  LSE_EXPECT(body["error"]["type"] == "device_lost");
  LSE_EXPECT(!body.contains("retry_after"));
  LSE_EXPECT(refused->headers.empty());

  LSE_EXPECT(lse_power_resume(nullptr, &out, &err) == LSE_ERR_STATE);
  LSE_EXPECT(out != nullptr && json::parse(out)["state"] == "lost");
  LSE_EXPECT(err != nullptr && std::string(err).find("device lost") != std::string::npos);
  lse_free(out);
  lse_free(err);
  backend::clear_device_power();
}

LSE_TEST_MAIN()
