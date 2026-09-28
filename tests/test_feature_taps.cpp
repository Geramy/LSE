#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/model/hybrid_lm.hpp"

using namespace lse;

namespace {

constexpr std::int32_t kLayers = 4;
constexpr std::int64_t kWidth = 4;

class TestMixer final : public model::IMixer {
 public:
  explicit TestMixer(std::shared_ptr<std::size_t> builds)
      : builds_(std::move(builds)) {}

  Status load(model::WeightBinder&, std::string_view,
              const model::LayerContext&) override {
    return OkStatus();
  }

  Result<graph::Array> forward(const graph::Array& x, model::MixerState* state,
                               const model::LayerContext& ctx) override {
    ++*builds_;
    graph::Array y = x * graph::Array::full(
        Shape{1}, DType::kF32, 0.05f * static_cast<float>(ctx.layer_index + 1));
    if (state != nullptr) {
      const auto batch = x.shape().dim(0);
      const graph::Array old = state->gdn_state;
      y = y + graph::reshape(old, Shape{batch, 1, 1}) *
                  graph::Array::full(Shape{1}, DType::kF32, 0.01f);
      const graph::Array current = graph::mean(graph::mean(x, -1, true), 1, true);
      state->gdn_state = graph::reshape(current, Shape{batch, 1, 1, 1}) +
                         old * graph::Array::full(Shape{1}, DType::kF32, 0.5f);
    }
    return y;
  }

  std::string_view name() const noexcept override { return "feature_test_mixer"; }

 private:
  std::shared_ptr<std::size_t> builds_;
};

class TestFeedForward final : public model::IFeedForward {
 public:
  Status load(model::WeightBinder&, std::string_view,
              const model::LayerContext&) override {
    return OkStatus();
  }

  Result<graph::Array> forward(const graph::Array& x, graph::Array*,
                               const model::LayerContext& ctx) override {
    return x * graph::Array::full(
        Shape{1}, DType::kF32, 0.03f * static_cast<float>(ctx.layer_index + 1));
  }

  std::string_view name() const noexcept override { return "feature_test_ffn"; }
};

struct Fixture {
  std::filesystem::path path;
  std::shared_ptr<std::size_t> builds = std::make_shared<std::size_t>(0);
  std::unique_ptr<model::HybridLM> model;

  explicit Fixture(std::int32_t layers = kLayers) {
    model::Config config;
    config.hidden_size = static_cast<std::int32_t>(kWidth);
    config.vocab_size = 8;
    config.num_layers = layers;
    config.full_attention_interval = kLayers + 1;
    config.global_attention_layers.clear();
    config.gdn_qk_heads = config.gdn_v_heads = config.gdn_head_dim = 1;
    config.gdn_conv_kernel = 1;
    config.kv_length = 64;
    model::HybridLMSpec spec;
    spec.zero_centered_norm = false;
    const auto count = builds;
    model = std::make_unique<model::HybridLM>(
        config, spec, [count](std::int32_t) -> Result<std::unique_ptr<model::HybridBlock>> {
          return std::make_unique<model::HybridBlock>(
              std::make_unique<TestMixer>(count),
              std::make_unique<TestFeedForward>(), false);
        });

    struct Tensor {
      std::string name;
      std::vector<std::int64_t> dims;
      std::vector<float> values;
    };
    std::vector<Tensor> tensors;
    Tensor embed{"embed.weight", {config.vocab_size, kWidth}, {}};
    for (std::int32_t row = 0; row < config.vocab_size; ++row)
      for (std::int64_t d = 0; d < kWidth; ++d)
        embed.values.push_back(
            0.125f * static_cast<float>((row * 5 + d * 3) % 13 - 6));
    tensors.push_back(std::move(embed));
    tensors.push_back({"final_norm.weight", {kWidth}, {1.0f, 1.1f, 0.9f, 1.2f}});
    for (std::int32_t layer = 0; layer < layers; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer);
      tensors.push_back({prefix + ".norm1.weight", {kWidth}, {1, 1, 1, 1}});
      tensors.push_back({prefix + ".norm2.weight", {kWidth}, {1, 1, 1, 1}});
    }
    std::string header = "{";
    std::size_t offset = 0;
    for (const Tensor& tensor : tensors) {
      if (header.size() > 1) header += ",";
      header += "\"" + tensor.name + "\":{\"dtype\":\"F32\",\"shape\":[";
      for (std::size_t d = 0; d < tensor.dims.size(); ++d) {
        if (d != 0) header += ",";
        header += std::to_string(tensor.dims[d]);
      }
      const std::size_t bytes = tensor.values.size() * sizeof(float);
      header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
                std::to_string(offset + bytes) + "]}";
      offset += bytes;
    }
    header += "}";
    while (header.size() % 8 != 0) header += " ";
    static std::size_t serial = 0;
    path = std::filesystem::temp_directory_path() /
           ("lse-feature-taps-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(++serial) + ".safetensors");
    {
      std::ofstream out(path, std::ios::binary);
      const std::uint64_t size = header.size();
      out.write(reinterpret_cast<const char*>(&size), sizeof(size));
      out.write(header.data(), static_cast<std::streamsize>(header.size()));
      for (const Tensor& tensor : tensors)
        out.write(reinterpret_cast<const char*>(tensor.values.data()),
                  static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    }
    auto weights = model::SafeTensors::open(path.string());
    LSE_EXPECT_OK(weights.status());
    if (!weights.ok()) return;
    model::WeightBinder binder(*weights);
    LSE_EXPECT_OK(model->load(binder));
  }

  ~Fixture() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

graph::Array tokens(std::int64_t batch, std::int64_t width, std::int32_t first) {
  graph::Scheduler* scheduler = graph::default_scheduler();
  graph::Array out = graph::Array::zeros(Shape{batch, width}, DType::kF32);
  LSE_EXPECT_OK(graph::interpreter::ensure_output_buffer(*out.node(), scheduler->backend()));
  for (std::size_t i = 0; i < out.shape().elem_count(); ++i)
    graph::interpreter::store_element(*out.node(), i,
                                     static_cast<float>((first + static_cast<std::int32_t>(i)) % 8));
  out.node()->materialized = true;
  LSE_EXPECT_OK(graph::interpreter::sync_to_device(*out.node(), scheduler->backend()));
  return out;
}

std::vector<float> values(graph::Array array) {
  std::vector<float> out(array.shape().elem_count());
  LSE_EXPECT_OK(array.to_host(out.data(), out.size() * sizeof(float)));
  return out;
}

void expect_features(graph::Array features, const std::vector<graph::Array>& trace,
                     std::span<const std::int32_t> ids) {
  LSE_EXPECT(features.dtype() == DType::kF32);
  LSE_EXPECT_EQ(features.shape().dim(2), static_cast<std::int64_t>(ids.size()) * kWidth);
  const auto actual = values(features);
  const std::size_t positions = actual.size() / (ids.size() * kWidth);
  for (std::size_t tap = 0; tap < ids.size(); ++tap) {
    const auto expected = values(trace[static_cast<std::size_t>(ids[tap])]);
    for (std::size_t position = 0; position < positions; ++position)
      for (std::size_t d = 0; d < static_cast<std::size_t>(kWidth); ++d)
        LSE_EXPECT_NEAR(actual[(position * ids.size() + tap) * kWidth + d],
                        expected[position * kWidth + d], 1e-5);
  }
}

}  // namespace

LSE_TEST(feature_capture_preserves_order_beyond_one_concat_group) {
  Fixture fixture(9);
  const std::array<std::int32_t, 9> ids{0, 1, 2, 3, 4, 5, 6, 7, 8};
  model::FeatureCapture capture{ids, {}};
  std::vector<graph::Array> trace;
  auto hidden = fixture.model->hidden(tokens(1, 2, 1), nullptr, nullptr,
                                       &trace, nullptr, false, &capture);
  LSE_EXPECT_OK(hidden.status());
  if (hidden.ok()) expect_features(capture.features, trace, ids);
}

LSE_TEST(selected_features_replay_and_carry_match_full_trace) {
  Fixture captured;
  Fixture reference;
  auto states = captured.model->make_states();
  auto reference_states = reference.model->make_states();
  const std::array<std::int32_t, 2> ids{0, 3};
  model::FeatureCapture capture{ids, {}};
  graph::Node* feature_node = nullptr;
  std::vector<float> previous;
  for (std::int32_t step = 0; step < 4; ++step) {
    capture.features = {};
    auto result = captured.model->hidden(tokens(1, 1, step), &states, nullptr,
                                         nullptr, nullptr, false, &capture);
    LSE_EXPECT_OK(result.status());
    if (!result.ok()) return;
    std::vector<graph::Array> trace;
    auto full = reference.model->hidden(tokens(1, 1, step), &reference_states,
                                       nullptr, &trace);
    LSE_EXPECT_OK(full.status());
    if (!full.ok()) return;
    expect_features(capture.features, trace, ids);
    LSE_EXPECT_EQ(*captured.builds, static_cast<std::size_t>(kLayers));
    LSE_EXPECT_EQ(states[0].position, step + 1);
    if (step == 0) feature_node = capture.features.node().get();
    LSE_EXPECT(capture.features.node().get() == feature_node);
    const auto& roots = captured.model->retained_program().roots();
    LSE_EXPECT(std::any_of(roots.begin(), roots.end(), [&](const graph::NodePtr& node) {
      return node.get() == feature_node;
    }));
    const auto current = values(capture.features);
    if (!previous.empty()) LSE_EXPECT(current != previous);
    previous = current;
  }
}

LSE_TEST(feature_lists_are_cache_identity_and_replacement_is_atomic) {
  Fixture fixture;
  auto states = fixture.model->make_states();
  std::array<std::int32_t, 2> ids{0, 3};
  model::FeatureCapture capture{ids, {}};
  LSE_EXPECT_OK(fixture.model->hidden(tokens(1, 1, 1), &states, nullptr,
                                     nullptr, nullptr, false, &capture).status());
  const graph::NodePtr original = capture.features.node();
  const auto original_values = values(capture.features);
  const graph::NodePtr state = states[0].gdn_state.node();
  const auto position = states[0].position;
  ids = {1, 2};
  const auto replacement = fixture.model->hidden(tokens(1, 1, 2), &states, nullptr,
                                                nullptr, nullptr, true, &capture);
  LSE_EXPECT(!replacement.ok());
  LSE_EXPECT(replacement.status().code() == StatusCode::kInvalidArgument);
  LSE_EXPECT_EQ(states[0].position, position);
  LSE_EXPECT(states[0].gdn_state.node() == state);
  LSE_EXPECT(capture.features.node() == original);
  LSE_EXPECT(values(capture.features) == original_values);
  LSE_EXPECT_EQ(*fixture.builds, static_cast<std::size_t>(kLayers));
  LSE_EXPECT_OK(fixture.model->hidden(tokens(1, 1, 2), &states, nullptr,
                                     nullptr, nullptr, false, &capture).status());
  LSE_EXPECT(capture.features.node() != original);
  LSE_EXPECT_EQ(*fixture.builds, static_cast<std::size_t>(2 * kLayers));
  const auto changed = capture.features.node();
  LSE_EXPECT_OK(fixture.model->hidden(tokens(1, 1, 3), &states, nullptr,
                                     nullptr, nullptr, false, &capture).status());
  LSE_EXPECT(capture.features.node() == changed);
  LSE_EXPECT_EQ(*fixture.builds, static_cast<std::size_t>(2 * kLayers));
  LSE_EXPECT_OK(fixture.model->hidden(tokens(1, 1, 4), &states, nullptr, nullptr).status());
  LSE_EXPECT_EQ(*fixture.builds, static_cast<std::size_t>(3 * kLayers));
  LSE_EXPECT_OK(fixture.model->hidden(tokens(1, 1, 5), &states, nullptr,
                                     nullptr, nullptr, false, &capture).status());
  LSE_EXPECT_EQ(*fixture.builds, static_cast<std::size_t>(4 * kLayers));
}

LSE_TEST(feature_capture_rejects_invalid_ids_before_state_changes) {
  Fixture fixture;
  auto states = fixture.model->make_states();
  const std::vector<std::vector<std::int32_t>> invalid{{}, {-1}, {4}, {1, 1}, {2, 0}};
  for (const auto& ids : invalid) {
    model::FeatureCapture capture{ids, {}};
    const auto result = fixture.model->hidden(tokens(1, 1, 1), &states, nullptr,
                                             nullptr, nullptr, false, &capture);
    LSE_EXPECT(!result.ok());
    LSE_EXPECT(result.status().code() == StatusCode::kInvalidArgument);
    LSE_EXPECT_EQ(states[0].position, 0);
    LSE_EXPECT(!states[0].gdn_state.valid());
    LSE_EXPECT(!capture.features.valid());
    LSE_EXPECT_EQ(*fixture.builds, 0u);
  }
}

LSE_TEST(feature_capture_preserves_batch_and_token_width_carry_handoffs) {
  Fixture captured;
  Fixture reference;
  auto states = captured.model->make_states();
  auto reference_states = reference.model->make_states();
  const std::array<std::int32_t, 2> ids{1, 3};
  model::FeatureCapture capture{ids, {}};
  std::int32_t first = 0;
  for (const std::int64_t width : {2, 2, 1, 1, 2}) {
    auto result = captured.model->hidden(tokens(2, width, first), &states, nullptr,
                                         nullptr, nullptr, false, &capture);
    LSE_EXPECT_OK(result.status());
    if (!result.ok()) return;
    std::vector<graph::Array> trace;
    auto full = reference.model->hidden(tokens(2, width, first), &reference_states,
                                        nullptr, &trace);
    LSE_EXPECT_OK(full.status());
    if (!full.ok()) return;
    LSE_EXPECT_EQ(capture.features.shape().dim(0), 2);
    LSE_EXPECT_EQ(capture.features.shape().dim(1), width);
    expect_features(capture.features, trace, ids);
    first += static_cast<std::int32_t>(width);
    LSE_EXPECT_EQ(states[0].position, first);
  }
}

int main() {
  setenv("LSE_BACKEND", "cpu", 1);
  graph::Scheduler* scheduler = graph::default_scheduler();
  if (scheduler == nullptr || scheduler->backend().name() != "cpu") {
    std::fprintf(stderr, "feature taps require the CPU backend\n");
    return 1;
  }
  scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);
  return test::run_all();
}
