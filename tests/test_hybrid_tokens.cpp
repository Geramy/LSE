#include "harness.hpp"
#include "lse/backend/backend.hpp"
#include "lse/graph/graph.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/hybrid_lm.hpp"
#include "lse/model/weights.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unistd.h>
using namespace lse;
namespace {
class TokenBackend final : public backend::Backend<TokenBackend> {
 public:
  static constexpr std::string_view kName = "fixture.token_replay";
  Status init_impl(int) { return OkStatus(); }
  void shutdown_impl() noexcept {}
  const backend::DeviceInfo& device_info_impl() const noexcept { return info_; }
  Result<backend::DeviceBuffer> allocate_impl(std::size_t bytes, backend::MemoryClass, backend::Stream) {
    backend::DeviceBuffer b;
    b.storage = std::shared_ptr<void>(new std::byte[bytes], [](void* p) { delete[] static_cast<std::byte*>(p); });
    b.handle = reinterpret_cast<std::uint64_t>(b.storage.get()); b.size_bytes = bytes;
    return b;
  }
  void deallocate_impl(backend::DeviceBuffer& b) noexcept { b = {}; }
  Status copy_h2d_impl(const void* source, backend::DeviceBuffer& destination, std::size_t bytes, std::size_t offset) {
    std::memcpy(address(destination) + offset, source, bytes); return OkStatus();
  }
  Status copy_d2h_impl(const backend::DeviceBuffer& source, void* destination, std::size_t bytes, std::size_t offset) {
    std::memcpy(destination, address(source) + offset, bytes); ++downloads; return OkStatus();
  }
  Result<backend::KernelHandle> load_executable_impl(std::string_view, std::span<const std::byte>) {
    return LSE_ERROR(kUnimplemented, "fixture uses the host interpreter");
  }
  Status launch_impl(const backend::KernelHandle&, const backend::LaunchDims&, const backend::DispatchArgs&) {
    return LSE_ERROR(kUnimplemented, "fixture uses the host interpreter");
  }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const graph::KernelToolchain> toolchains_impl() const noexcept { return {}; }
  std::size_t downloads = 0;
 private:
  static std::byte* address(const backend::DeviceBuffer& b) { return reinterpret_cast<std::byte*>(b.handle) + b.offset; }
  backend::DeviceInfo info_;
};
backend::IDeviceSet* token_devices() {
  static backend::BackendAdapter<TokenBackend> be;
  static const auto initialized = be.init(0);
  static backend::SingleDevice set(be);
  return initialized.ok() ? &set : nullptr;
}
class ZeroMixer final : public model::IMixer {
 public:
  explicit ZeroMixer(std::vector<ops::AttentionExecutionPhase>* phases = nullptr) : phases_(phases) {}
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override { return OkStatus(); }
  Result<graph::Array> forward(const graph::Array& x, model::MixerState*, const model::LayerContext& ctx) override {
    if (phases_) phases_->push_back(ctx.attention_phase);
    return graph::Array::zeros(x.shape(), DType::kF32);
  }
  std::string_view name() const noexcept override { return "fixture.zero_mixer"; }
 private:
  std::vector<ops::AttentionExecutionPhase>* phases_;
};
class ZeroFfn final : public model::IFeedForward {
 public:
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override { return OkStatus(); }
  Result<graph::Array> forward(const graph::Array& x, graph::Array*, const model::LayerContext&) override { return graph::Array::zeros(x.shape(), DType::kF32); }
  std::string_view name() const noexcept override { return "fixture.zero_ffn"; }
};
LSE_TEST(opaque_tokens_replay_the_uploaded_ids_without_a_host_mirror) {
  auto* scheduler = graph::default_scheduler(); LSE_EXPECT(scheduler); if (!scheduler) return;
  scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);
  const auto directory = std::filesystem::temp_directory_path() / ("lse-token-replay-" + std::to_string(getpid()));
  std::filesystem::create_directories(directory);
  struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); } } cleanup{directory};
  nlohmann::json header;
  std::vector<float> weights;
  const auto add = [&](const std::string& name, std::vector<std::int64_t> shape, bool embed) {
    const auto start = weights.size() * 4; std::size_t count = 1; for (auto d : shape) count *= static_cast<std::size_t>(d);
    for (std::size_t i = 0; i < count; ++i) weights.push_back(embed ? static_cast<float>(1 + i % 8 + i / 8 * 2) : 1.0f);
    header[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {start, weights.size() * 4}}};
  };
  add("embed.weight", {16, 8}, true); add("final_norm.weight", {8}, false);
  add("blocks.0.norm1.weight", {8}, false); add("blocks.0.norm2.weight", {8}, false);
  auto text = header.dump(); text.append((8 - text.size() % 8) % 8, ' ');
  { std::ofstream out(directory / "model.safetensors", std::ios::binary); const std::uint64_t bytes = text.size();
    out.write(reinterpret_cast<const char*>(&bytes), 8); out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * 4)); }
  auto opened = model::SafeTensors::open((directory / "model.safetensors").string()); LSE_EXPECT(opened.ok()); if (!opened.ok()) return;
  model::Config config; config.hidden_size = 8; config.vocab_size = 16; config.num_layers = 1; config.full_attention_interval = 1;
  model::HybridLMSpec spec; spec.zero_centered_norm = false;
  std::vector<ops::AttentionExecutionPhase> phases;
  model::HybridLM lm(config, spec, [&phases](std::int32_t) -> Result<std::unique_ptr<model::HybridBlock>> {
    return std::make_unique<model::HybridBlock>(std::make_unique<ZeroMixer>(&phases), std::make_unique<ZeroFfn>(), false);
  });
  model::WeightBinder binder(*opened); LSE_EXPECT_OK(lm.load(binder));
  auto states = lm.make_states(); graph::NodePtr retained;
  for (std::size_t pass = 0; pass < 3; ++pass) {
    std::vector<float> ids(8); for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<float>((pass * 5 + i) % 16);
    auto allocated = scheduler->backend().allocate(9 * sizeof(float), backend::MemoryClass::kDevice); LSE_EXPECT(allocated.ok()); if (!allocated.ok()) return;
    auto b = allocated.release(); LSE_EXPECT_OK(scheduler->backend().copy_h2d(ids.data(), b, ids.size() * 4, 4));
    b.offset = 4; b.size_bytes = ids.size() * 4;
    auto input = graph::Array::from_buffer(std::move(b), {1, 8}, DType::kF32);
    LSE_EXPECT(input.node()->host_mirror.empty()); LSE_EXPECT(input.node()->device_dirty);
    auto result = lm.hidden(input, &states, nullptr); LSE_EXPECT(result.ok()); if (!result.ok()) return;
    if (pass == 0) retained = result->node(); else LSE_EXPECT(result->node() == retained);
    std::vector<float> actual(64); LSE_EXPECT_OK(result->to_host(actual.data(), actual.size() * 4));
    for (std::size_t row = 0; row < 8; ++row) {
      double sum = 0; for (std::size_t d = 0; d < 8; ++d) { const double v = 1.0 + static_cast<double>(d) + static_cast<double>(ids[row]) * 2.0; sum += v * v; }
      const auto inverse = 1 / std::sqrt(sum / 8 + static_cast<double>(config.rms_eps));
      for (std::size_t d = 0; d < 8; ++d) LSE_EXPECT_NEAR(actual[row * 8 + d], (1.0 + static_cast<double>(d) + static_cast<double>(ids[row]) * 2.0) * inverse, 2e-6);
    }
    LSE_EXPECT_EQ(states[0].position, static_cast<std::int32_t>((pass + 1) * 8));
  }
  // Equal-width prompt and verification passes must have separate retained
  // programs, while repeated verification can still replay its own program.
  using Phase = ops::AttentionExecutionPhase;
  auto input = graph::Array::zeros(Shape{1, 8}, DType::kF32);
  auto prompt = lm.hidden(input, &states, nullptr, nullptr, nullptr, false,
                          nullptr, false, Phase::kPrefill);
  LSE_EXPECT(prompt.ok()); if (!prompt.ok()) return;
  LSE_EXPECT(prompt->node() != retained);
  LSE_EXPECT(phases.back() == Phase::kPrefill);
  auto verify = lm.hidden(input, &states, nullptr, nullptr, nullptr, false,
                          nullptr, false, Phase::kSpeculative);
  LSE_EXPECT(verify.ok()); if (!verify.ok()) return;
  LSE_EXPECT(verify->node() != prompt->node());
  LSE_EXPECT(phases.back() == Phase::kSpeculative);
  auto replay = lm.hidden(input, &states, nullptr, nullptr, nullptr, false,
                          nullptr, false, Phase::kSpeculative);
  LSE_EXPECT(replay.ok()); if (!replay.ok()) return;
  LSE_EXPECT(replay->node() == verify->node());
  const auto position = states[0].position;
  auto invalid = lm.hidden(input, &states, nullptr, nullptr, nullptr, true,
                           nullptr, false, Phase::kPrefill);
  LSE_EXPECT(!invalid.ok());
  LSE_EXPECT(invalid.status().to_string().find("cannot change attention phase") != std::string::npos);
  LSE_EXPECT_EQ(states[0].position, position);
}
}
int main() { graph::register_device_set_factory(token_devices); return lse::test::run_all(); }
