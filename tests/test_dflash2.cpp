#include "harness.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/weights.hpp"
#if defined(LSE_DFLASH2_RUNTIME_TESTS)
#include "lse/runtime/generator.hpp"
#endif
#include "lse/graph/interpreter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#if defined(LSE_HRX_LINKED) && LSE_HAVE_LOOMC
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>
#include <vector>

using namespace lse;
namespace {
graph::Array filled(Shape shape, const std::vector<float>& values) {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return {};
  auto allocated = scheduler->backend().allocate(values.size() * sizeof(float), backend::MemoryClass::kDevice);
  if (!allocated.ok()) return {};
  auto buffer = allocated.release();
  if (!scheduler->backend().copy(buffer, values.data(), values.size() * sizeof(float)).ok()) return {};
  return graph::Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
}
std::vector<float> read(graph::Array array) {
  std::vector<float> values(array.shape().elem_count());
  LSE_EXPECT_OK(array.to_host(values.data(), values.size() * sizeof(float)));
  return values;
}
class OpaqueBufferBackend final : public backend::Backend<OpaqueBufferBackend> {
 public:
  static constexpr std::string_view kName = "fixture.opaque_buffer";
  Status init_impl(int) { return OkStatus(); }
  void shutdown_impl() noexcept {}
  const backend::DeviceInfo& device_info_impl() const noexcept { return info_; }
  Result<backend::DeviceBuffer> allocate_impl(std::size_t bytes, backend::MemoryClass, backend::Stream) {
    auto storage = std::make_shared<std::vector<std::byte>>(bytes);
    backend::DeviceBuffer buffer;
    buffer.handle = reinterpret_cast<std::uint64_t>(storage->data());
    buffer.size_bytes = bytes;
    buffer.storage = std::move(storage);
    return buffer;
  }
  void deallocate_impl(backend::DeviceBuffer& buffer) noexcept { buffer = {}; }
  Status copy_h2d_impl(const void* source, backend::DeviceBuffer& destination,
                       std::size_t bytes, std::size_t offset) {
    std::memcpy(address(destination) + offset, source, bytes);
    return OkStatus();
  }
  Status copy_d2h_impl(const backend::DeviceBuffer& source, void* destination,
                       std::size_t bytes, std::size_t offset) {
    std::memcpy(destination, address(source) + offset, bytes);
    ++downloads;
    return OkStatus();
  }
  Result<backend::KernelHandle> load_executable_impl(std::string_view, std::span<const std::byte>) {
    return LSE_ERROR(kUnimplemented, "fixture has no device compiler");
  }
  Status launch_impl(const backend::KernelHandle&, const backend::LaunchDims&, const backend::DispatchArgs&) {
    return LSE_ERROR(kUnimplemented, "fixture has no device execution");
  }
  Status synchronize_impl() { return OkStatus(); }
  std::span<const graph::KernelToolchain> toolchains_impl() const noexcept { return {}; }
  std::size_t downloads = 0;
 private:
  static std::byte* address(const backend::DeviceBuffer& buffer) {
    return reinterpret_cast<std::byte*>(buffer.handle) + buffer.offset;
  }
  backend::DeviceInfo info_;
};
nlohmann::json config_json() {
  return {{"architectures", {"DFlash2DraftModel"}}, {"is_causal", false},
    {"hidden_size", 8}, {"vocab_size", 16}, {"num_hidden_layers", 1},
    {"num_target_layers", 1}, {"num_attention_heads", 2}, {"num_key_value_heads", 1},
    {"head_dim", 4}, {"intermediate_size", 12}, {"sliding_window", 4},
    {"rms_norm_eps", 1e-6}, {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 10000}}},
    {"layer_types", {"sliding_attention"}},
    {"dflash_config", {{"block_size", 3}, {"mask_token_id", 15},
      {"target_layer_ids", {0}}, {"conv_kernel_size", 2}, {"conv_group_size", 2},
      {"selector_rank", 4}, {"selector_top_k", 3}}}};
}
model::Config parent_config() {
  model::Config c;
  c.hidden_size = 8; c.vocab_size = 16; c.num_layers = 1; c.kv_length = 32;
  c.attn_q_heads = 2; c.attn_kv_heads = 1; c.attn_head_dim = 4; c.rope_dim = 4;
  return c;
}
const graph::KernelPrimitiveBase* attention_primitive() {
  return dynamic_cast<const graph::KernelPrimitiveBase*>(graph::find_primitive("dflash2.attention"));
}
std::vector<float> attention_reference(const std::vector<float>& q, const std::vector<float>& k,
    const std::vector<float>& v, const std::vector<float>& meta,
    std::size_t heads, std::size_t kv_heads, std::size_t dim, std::size_t queries, std::size_t window) {
  auto* primitive = attention_primitive();
  LSE_EXPECT(primitive != nullptr);
  if (!primitive) return {};
  std::vector<float> output(heads * queries * dim);
  const float* inputs[] = {q.data(), k.data(), v.data(), meta.data()};
  primitive->eval_cpu(inputs, output.data(), output.size(),
      {static_cast<float>(heads), static_cast<float>(kv_heads), static_cast<float>(dim), static_cast<float>(window)});
  return output;
}
class ZeroMixer final : public model::IMixer {
 public:
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override { return OkStatus(); }
  Result<graph::Array> forward(const graph::Array& x, model::MixerState*, const model::LayerContext&) override {
    return graph::Array::zeros(x.shape(), DType::kF32);
  }
  std::string_view name() const noexcept override { return "fixture.zero_mixer"; }
};
class ZeroFfn final : public model::IFeedForward {
 public:
  Status load(model::WeightBinder&, std::string_view, const model::LayerContext&) override { return OkStatus(); }
  Result<graph::Array> forward(const graph::Array& x, graph::Array*, const model::LayerContext&) override {
    return graph::Array::zeros(x.shape(), DType::kF32);
  }
  std::string_view name() const noexcept override { return "fixture.zero_ffn"; }
};
struct Named { std::string name; std::vector<std::int64_t> shape; };
enum class FixtureWeights {
  kDefault, kConstantTokens, kStopTransition, kRejectedTokens, kRejectedStopTransition
};
void write_weights(const std::filesystem::path& path, const std::vector<Named>& names,
                   FixtureWeights mode = FixtureWeights::kDefault) {
  nlohmann::json header;
  std::vector<float> payload;
  for (std::size_t t = 0; t < names.size(); ++t) {
    const auto& tensor = names[t]; std::size_t count = 1;
    for (auto dim : tensor.shape) count *= static_cast<std::size_t>(dim);
    const auto start = payload.size() * sizeof(float);
    const bool normalizer = tensor.name.find("norm") != std::string::npos;
    for (std::size_t i = 0; i < count; ++i) {
      float value = std::sin(static_cast<float>((i + 1) * (t + 3)) * 0.371f) * 0.13f;
      if (normalizer) value = 1.0f;
      if (tensor.name.ends_with("base_kernel")) value = (i / 8) % 2 == 0 ? 1.0f : 0.17f;
      if (mode != FixtureWeights::kDefault) {
        value = normalizer ? 1.0f : 0.0f;
        const bool constant = mode == FixtureWeights::kConstantTokens ||
                              mode == FixtureWeights::kRejectedTokens;
        if (tensor.name == "embed.weight") {
          const auto token = i / 8;
          const auto dimension = mode == FixtureWeights::kRejectedStopTransition
              ? std::min<std::size_t>(token, 3) : (token == 0 ? 0u : 1u);
          value = constant ? 0.25f : (i % 8 == dimension ? 1.0f : 0.0f);
        } else if (tensor.name == "lm_head.weight") {
          const auto token = i / 8, dimension = i % 8;
          const bool transition = mode == FixtureWeights::kRejectedStopTransition
              ? ((token == 1 && dimension == 0) || (token == 2 && dimension == 1) ||
                 (token == 3 && (dimension == 2 || dimension == 3)))
              : ((token == 1 && dimension == 0) || (token == 2 && dimension == 1));
          value = transition ? 1.0f : 0.0f;
        } else if (mode == FixtureWeights::kRejectedTokens) {
          if (tensor.name == "candidate_selector.hidden_projection.weight" ||
              tensor.name == "candidate_selector.predecessor_codebook") value = 1.0f;
          if (tensor.name == "candidate_selector.successor_codebook")
            value = i / 4 == 1 ? 1.0f : 0.0f;
        }
      }
      payload.push_back(value);
    }
    header[tensor.name] = {{"dtype", "F32"}, {"shape", tensor.shape},
                          {"data_offsets", {start, payload.size() * sizeof(float)}}};
  }
  auto raw = header.dump(); raw.append((8 - raw.size() % 8) % 8, ' ');
  const std::uint64_t bytes = raw.size(); std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes)); out.write(raw.data(), static_cast<std::streamsize>(raw.size()));
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size() * sizeof(float)));
}
struct Fixture {
  std::filesystem::path directory;
  model::Config config = parent_config();
  model::SafeTensors weights;
  std::unique_ptr<model::HybridLM> target;
  std::unique_ptr<model::DFlash2Module> draft;
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
  Status open(FixtureWeights mode = FixtureWeights::kDefault) {
    // ZeroMixer has no recurrence for the generator to retain.
    if (mode != FixtureWeights::kDefault) config.full_attention_interval = 1;
    static unsigned serial = 0;
    directory = std::filesystem::temp_directory_path() /
        ("lse-dflash2-fixture-" + std::to_string(getpid()) + "-" + std::to_string(++serial));
    std::filesystem::create_directories(directory / "draft");
    std::vector<Named> target_names{{"embed.weight", {16, 8}}, {"final_norm.weight", {8}},
                                  {"blocks.0.norm1.weight", {8}}, {"blocks.0.norm2.weight", {8}}};
    if (mode == FixtureWeights::kStopTransition || mode == FixtureWeights::kRejectedStopTransition) {
      config.tie_word_embeddings = false;
      target_names.push_back({"lm_head.weight", {16, 8}});
    }
    write_weights(directory / "model.safetensors", target_names, mode);
    LSE_ASSIGN_OR(weights, model::SafeTensors::open((directory / "model.safetensors").string()));
    model::HybridLMSpec spec; spec.zero_centered_norm = false;
    if (mode == FixtureWeights::kStopTransition || mode == FixtureWeights::kRejectedStopTransition) spec.lm_head_name = "lm_head.weight";
    target = std::make_unique<model::HybridLM>(config, spec,
        [](std::int32_t) -> Result<std::unique_ptr<model::HybridBlock>> {
          return std::make_unique<model::HybridBlock>(std::make_unique<ZeroMixer>(), std::make_unique<ZeroFfn>(), false);
        });
    model::WeightBinder binder(weights); LSE_RETURN_IF_ERROR(target->load(binder));
    const std::vector<Named> names{
      {"fc.weight", {8,8}}, {"hidden_norm.weight", {8}}, {"norm.weight", {8}},
      {"candidate_selector.hidden_projection.weight", {4,8}},
      {"candidate_selector.predecessor_codebook", {16,4}}, {"candidate_selector.successor_codebook", {16,4}},
      {"layers.0.input_layernorm.weight", {8}}, {"layers.0.post_attention_layernorm.weight", {8}},
      {"layers.0.self_attn.q_proj.weight", {8,8}}, {"layers.0.self_attn.k_proj.weight", {4,8}},
      {"layers.0.self_attn.v_proj.weight", {4,8}}, {"layers.0.self_attn.o_proj.weight", {8,8}},
      {"layers.0.self_attn.q_norm.weight", {4}}, {"layers.0.self_attn.k_norm.weight", {4}},
      {"layers.0.mlp.gate_proj.weight", {12,8}}, {"layers.0.mlp.up_proj.weight", {12,8}},
      {"layers.0.mlp.down_proj.weight", {8,12}}, {"layers.0.attention_conv.base_kernel", {2,2,8}},
      {"layers.0.attention_conv.kernel_projection.weight", {16,8}}, {"layers.0.mlp_conv.base_kernel", {2,2,8}},
      {"layers.0.mlp_conv.kernel_projection.weight", {16,8}}};
    write_weights(directory / "draft" / "model.safetensors", names, mode);
    std::ofstream(directory / "draft" / "config.json") << config_json().dump();
    LSE_ASSIGN_OR(draft, model::DFlash2Module::open((directory / "draft").string(), config, *target));
    return OkStatus();
  }
};
}

LSE_TEST(dflash2_config_checks_target_and_attention_contracts) {
  auto result = model::DFlash2Config::from_json_string(config_json().dump());
  LSE_EXPECT(result.ok()); if (!result.ok()) return;
  LSE_EXPECT_OK(result->validate(parent_config()));
  auto wrong_target = parent_config(); ++wrong_target.hidden_size;
  LSE_EXPECT(!result->validate(wrong_target).ok());
  auto config = config_json(); config["is_causal"] = true;
  LSE_EXPECT(!model::DFlash2Config::from_json_string(config.dump()).ok());
  config = config_json(); config["dflash_config"]["target_layer_ids"] = {0,0};
  auto repeated = model::DFlash2Config::from_json_string(config.dump());
  LSE_EXPECT(repeated.ok() && !repeated->validate(parent_config()).ok());
  config = config_json(); config["dflash_config"]["block_size"] = 9;
  LSE_EXPECT(!model::DFlash2Config::from_json_string(config.dump()).ok());
}
LSE_TEST(dflash2_selector_follows_predecessor_conditioned_scores) {
  const std::vector<std::uint32_t> candidates{10,11,20,21,30,31};
  const std::vector<float> scores{0,2,0,2, 10,0,0,10, 1,0,0,1};
  auto path = model::dflash2_select_path(scores, candidates, 3, 2);
  LSE_EXPECT(path.ok()); if (!path.ok()) return;
  LSE_EXPECT((*path) == std::vector<std::uint32_t>({11,21,31}));
  LSE_EXPECT(!model::dflash2_select_path(scores, candidates, 0, 2).ok());
  auto poison = scores; poison[1] = std::numeric_limits<float>::quiet_NaN();
  LSE_EXPECT(!model::dflash2_select_path(poison, candidates, 3, 2).ok());
}
LSE_TEST(dflash2_attention_exposes_future_block_and_slides_only_context) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> q(2*3*2, 0.0f);
  const std::vector<float> k(6*2, 0.0f);
  const std::vector<float> v{nan,nan, 2,4, 6,8, 10,12, 14,16, 18,20};
  auto output = attention_reference(q, k, v, {2,3}, 2,1,2,3,3);
  LSE_EXPECT_EQ(output.size(), 12u);
  auto graph_output = graph::custom("dflash2.attention", {filled({1,2,3,2},q),
      filled({1,1,6,2},k), filled({1,1,6,2},v), filled({2},{2,3})}, {2,1,2,3});
  LSE_EXPECT(graph_output.ok()); if (!graph_output.ok()) return;
  const auto interpreted = read(graph_output.release());
  LSE_EXPECT(interpreted == output);
  const float expected[]{10,12,12,14,14,16};
  for (std::size_t h = 0; h < 2; ++h)
    for (std::size_t i = 0; i < 6; ++i) LSE_EXPECT(std::fabs(output[h*6+i] - expected[i]) < 1e-6f);
}
LSE_TEST(dflash2_host_attention_reads_initialized_opaque_buffer_windows) {
  backend::BackendAdapter<OpaqueBufferBackend> backend;
  LSE_EXPECT_OK(backend.init(0));
  graph::Scheduler scheduler(backend);
  scheduler.set_mode(graph::Scheduler::Mode::kHostOnly);
  auto upload = [&](Shape shape, const std::vector<float>& values) -> graph::Array {
    auto allocation = backend.allocate((values.size() + 1) * sizeof(float), backend::MemoryClass::kDevice, backend::kDefaultStream);
    LSE_EXPECT(allocation.ok());
    if (!allocation.ok()) return {};
    auto buffer = allocation.release();
    LSE_EXPECT_OK(backend.copy({buffer, sizeof(float)}, values.data(), values.size() * sizeof(float)));
    buffer.offset = sizeof(float);
    buffer.size_bytes = values.size() * sizeof(float);
    return graph::Array::from_buffer(std::move(buffer), std::move(shape), DType::kF32);
  };
  const std::vector<float> q(2*3*2, 0.0f), k(6*2, 0.0f);
  const std::vector<float> v{0,0, 2,4, 6,8, 10,12, 14,16, 18,20}, meta{2,3};
  auto output = graph::custom("dflash2.attention", {upload({1,2,3,2},q), upload({1,1,6,2},k),
      upload({1,1,6,2},v), upload({2},meta)}, {2,1,2,3});
  LSE_EXPECT(output.ok());
  if (!output.ok()) return;
  const graph::NodePtr roots[] = {output->node()};
  LSE_EXPECT_OK(scheduler.eval(roots, true));
  std::vector<float> actual(output->shape().elem_count());
  LSE_EXPECT_OK(graph::interpreter::read_raw(*output->node(), actual.data(), actual.size() * sizeof(float)));
  LSE_EXPECT(actual == attention_reference(q,k,v,meta,2,1,2,3,3));
  LSE_EXPECT_EQ(backend.impl().downloads, 4u);
}
LSE_TEST(dflash2_dynamic_convolution_shares_groups_and_respects_left_boundary) {
  const std::vector<float> x{1,2,3,4, 5,6,7,8, 9,10,11,12};
  const std::vector<float> dynamic{.1f,.2f,.3f,.4f, .5f,.6f,.7f,.8f, .9f,1,1.1f,1.2f};
  const std::vector<float> base{1,2,3,4, 5,6,7,8};
  auto y = model::dflash2_convolve(filled({1,3,4}, x), filled({1,3,2,2}, dynamic), filled({2,4}, base), 2);
  LSE_EXPECT(y.valid()); if (!y.valid()) return;
  auto output = read(y);
  for (std::size_t t = 0; t < 3; ++t)
    for (std::size_t d = 0; d < 4; ++d) {
      float expected = 0;
      for (std::size_t tap = 0; tap < 2; ++tap)
        if (tap <= t) expected += (base[tap*4+d] + dynamic[(t*2+tap)*2+d/2]) * x[(t-tap)*4+d];
      LSE_EXPECT(std::fabs(output[t*4+d] - expected) < 1e-5f);
    }
}
LSE_TEST(dflash2_context_overwrite_matches_fresh_prefix_and_does_not_alias_features) {
  Fixture fixture;
  auto opened = fixture.open(); LSE_EXPECT_OK(opened); if (!opened.ok()) return;
  std::vector<float> input(8*8);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i) * .71f);
  auto features = filled({1,8,8}, input);
  LSE_EXPECT_OK(fixture.draft->append_context(features, 0));
  LSE_EXPECT_EQ(fixture.draft->context_position(), 8);
  std::vector<std::vector<std::uint32_t>> original;
  for (std::uint32_t anchor = 0; anchor < 8; ++anchor) {
    auto drafted = fixture.draft->draft(anchor, 8, 2); LSE_EXPECT(drafted.ok()); if (!drafted.ok()) return;
    original.push_back(drafted.release());
    auto prefix = fixture.draft->draft(anchor, 8, 1);
    LSE_EXPECT(prefix.ok()); if (!prefix.ok()) return;
    LSE_EXPECT(*prefix == std::vector<std::uint32_t>{original.back()[0]});
  }
  auto tail = filled({1,2,8}, std::vector<float>(16, 5.0f));
  LSE_EXPECT_OK(fixture.draft->append_context(tail, 8));
  LSE_EXPECT_EQ(fixture.draft->context_position(), 10);
  LSE_EXPECT_OK(fixture.draft->rewind(8));
  for (std::uint32_t anchor = 0; anchor < 8; ++anchor) {
    auto drafted = fixture.draft->draft(anchor, 8, 2); LSE_EXPECT(drafted.ok()); if (!drafted.ok()) return;
    LSE_EXPECT((*drafted) == original[anchor]);
  }
  LSE_EXPECT_OK(fixture.draft->append_context(tail, 8));
  auto changed = filled({1,3,8}, std::vector<float>(24, -.7f));
  LSE_EXPECT_OK(fixture.draft->append_context(changed, 8));
  auto before = fixture.draft->draft(1, 11, 2); LSE_EXPECT(before.ok()); if (!before.ok()) return;
  LSE_EXPECT_OK(graph::default_scheduler()->backend().copy(changed.node()->buffer, std::vector<float>(24, 9.0f).data(), 24*sizeof(float)));
  auto after = fixture.draft->draft(1, 11, 2); LSE_EXPECT(after.ok()); if (!after.ok()) return;
  LSE_EXPECT((*before) == (*after));
  fixture.draft->reset();
  LSE_EXPECT_EQ(fixture.draft->context_position(), 0);
  LSE_EXPECT(!fixture.draft->draft(1, 0, 2).ok());
  LSE_EXPECT_OK(fixture.draft->append_context(features, 0));
  LSE_EXPECT_OK(fixture.draft->append_context(changed = filled({1,3,8}, std::vector<float>(24, -.7f)), 8));
  auto fresh = fixture.draft->draft(1, 11, 2); LSE_EXPECT(fresh.ok());
  if (fresh.ok()) LSE_EXPECT((*fresh) == (*before));
  LSE_EXPECT(!fixture.draft->rewind(3).ok());
}


#if defined(LSE_DFLASH2_RUNTIME_TESTS)
namespace {
runtime::SamplingParams greedy_sampling() {
  runtime::SamplingParams params;
  params.temperature = 0.0f;
  return params;
}
void expect_retained_dflash(const Fixture& fixture, runtime::Session& session,
                           std::int32_t covered) {
  LSE_EXPECT_EQ(session.position(), covered);
  LSE_EXPECT_EQ(fixture.draft->context_position(), covered);
  LSE_EXPECT(static_cast<std::size_t>(covered) <= session.history().size());
  LSE_EXPECT(session.history().size() - static_cast<std::size_t>(covered) <= 1u);
  for (const auto& state : session.states()) {
    LSE_EXPECT(!state.gdn_state.valid());
    LSE_EXPECT_EQ(state.position, covered);
  }
}
void terminal_and_next_turn(FixtureWeights mode, std::int32_t max_tokens,
                            std::size_t cancel_after,
                            const std::vector<std::uint32_t>& stops,
                            const std::vector<std::uint32_t>& expected,
                            std::int32_t covered, std::uint32_t passes,
                            std::uint32_t accepted, std::uint32_t tested) {
  Fixture fixture;
  const auto opened = fixture.open(mode);
  LSE_EXPECT_OK(opened); if (!opened.ok()) return;
  runtime::Session session("retained", 1);
  const std::vector<std::uint32_t> prompt =
      (mode == FixtureWeights::kStopTransition || mode == FixtureWeights::kRejectedStopTransition)
      ? std::vector<std::uint32_t>{0} : std::vector<std::uint32_t>{1, 2};
  runtime::GenerationLimits limits;
  limits.max_tokens = max_tokens;
  limits.stop_tokens = stops;
  std::vector<std::uint32_t> delivered;
  {
    runtime::Generator request(*fixture.target, greedy_sampling());
    request.use_dflash2(*fixture.draft);
    const auto output = request.generate(session, prompt, limits, [&](std::uint32_t token) {
      delivered.push_back(token);
      return cancel_after == 0 || delivered.size() < cancel_after;
    });
    LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
    LSE_EXPECT(*output == expected);
    LSE_EXPECT(delivered == expected);
    LSE_EXPECT_EQ(request.stats().generated_tokens, static_cast<std::int32_t>(expected.size()));
    LSE_EXPECT_EQ(request.stats().prompt_tokens, static_cast<std::int32_t>(prompt.size()));
    LSE_EXPECT_EQ(request.stats().spec_verify_passes, passes);
    LSE_EXPECT_EQ(request.stats().spec_accepted, accepted);
    LSE_EXPECT_EQ(request.stats().spec_tested, tested);
  }
  auto history = prompt;
  history.insert(history.end(), expected.begin(), expected.end());
  LSE_EXPECT(session.history() == history);
  expect_retained_dflash(fixture, session, covered);

  auto next_prompt = history;
  next_prompt.insert(next_prompt.end(), {4, 5});
  runtime::GenerationLimits next_limits;
  next_limits.max_tokens = 4;
  std::vector<std::uint32_t> continued;
  {
    // HTTP creates a Generator for each request while retaining model and session.
    runtime::Generator request(*fixture.target, greedy_sampling());
    request.use_dflash2(*fixture.draft);
    auto output = request.generate(session, next_prompt, next_limits);
    LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
    continued = output.release();
    LSE_EXPECT_EQ(request.stats().prompt_tokens,
                  static_cast<std::int32_t>(next_prompt.size()) - covered);
    LSE_EXPECT(request.stats().prompt_tokens < static_cast<std::int32_t>(next_prompt.size()));
    LSE_EXPECT_EQ(request.stats().generated_tokens, 4);
  }
  history = next_prompt;
  history.insert(history.end(), continued.begin(), continued.end());
  LSE_EXPECT(session.history() == history);
  expect_retained_dflash(fixture, session, static_cast<std::int32_t>(next_prompt.size()) + 3);

  Fixture cold;
  const auto cold_opened = cold.open(mode);
  LSE_EXPECT_OK(cold_opened); if (!cold_opened.ok()) return;
  runtime::Session fresh("fresh", 1);
  runtime::Generator request(*cold.target, greedy_sampling());
  request.use_dflash2(*cold.draft);
  const auto output = request.generate(fresh, next_prompt, next_limits);
  LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
  LSE_EXPECT(*output == continued);
  LSE_EXPECT_EQ(request.stats().prompt_tokens, static_cast<std::int32_t>(next_prompt.size()));
  LSE_EXPECT(fresh.history() == session.history());
  expect_retained_dflash(cold, fresh, session.position());
}
}
LSE_TEST(dflash2_generator_retains_full_terminal_pass_for_next_request) {
  terminal_and_next_turn(FixtureWeights::kConstantTokens, 4, 0, {}, {0, 0, 0, 0}, 5, 1, 2, 2);
}
LSE_TEST(dflash2_generator_commits_one_terminal_row_after_callback_cancel) {
  terminal_and_next_turn(FixtureWeights::kConstantTokens, 6, 2, {}, {0, 0}, 3, 1, 0, 0);
}
LSE_TEST(dflash2_generator_commits_two_terminal_rows_after_callback_cancel) {
  terminal_and_next_turn(FixtureWeights::kConstantTokens, 6, 3, {}, {0, 0, 0}, 4, 1, 1, 1);
}
LSE_TEST(dflash2_generator_retains_prefill_after_first_callback_cancel) {
  terminal_and_next_turn(FixtureWeights::kConstantTokens, 6, 1, {}, {0}, 2, 0, 0, 0);
}
LSE_TEST(dflash2_generator_stop_inside_verifier_commits_only_consumed_prefix) {
  terminal_and_next_turn(FixtureWeights::kStopTransition, 6, 0, {2}, {1}, 2, 1, 0, 0);
}
LSE_TEST(dflash2_generator_callback_after_rejection_preserves_next_turn) {
  terminal_and_next_turn(FixtureWeights::kRejectedTokens, 6, 3, {}, {0, 0, 0}, 4, 2, 0, 1);
}
LSE_TEST(dflash2_generator_stop_after_rejection_preserves_next_turn) {
  terminal_and_next_turn(FixtureWeights::kRejectedStopTransition, 6, 0, {3}, {1, 2}, 3, 2, 0, 1);
}
LSE_TEST(dflash2_generator_prefill_stop_retains_fully_covered_history) {
  terminal_and_next_turn(FixtureWeights::kConstantTokens, 6, 0, {0}, {}, 2, 0, 0, 0);
}
LSE_TEST(dflash2_generator_repeated_fully_cached_stop_request_coldstarts) {
  Fixture fixture;
  const auto opened = fixture.open(FixtureWeights::kConstantTokens);
  LSE_EXPECT_OK(opened); if (!opened.ok()) return;
  runtime::Session session("repeated-stop", 1);
  const std::vector<std::uint32_t> prompt{1, 2};
  runtime::GenerationLimits limits;
  limits.max_tokens = 6;
  limits.stop_tokens = {0};
  for (unsigned turn = 0; turn < 2; ++turn) {
    runtime::Generator request(*fixture.target, greedy_sampling());
    request.use_dflash2(*fixture.draft);
    std::size_t callbacks = 0;
    const auto output = request.generate(session, prompt, limits, [&](std::uint32_t) {
      ++callbacks;
      return true;
    });
    LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
    LSE_EXPECT(output->empty());
    LSE_EXPECT_EQ(callbacks, 0u);
    LSE_EXPECT_EQ(request.stats().prompt_tokens, 2);
    LSE_EXPECT_EQ(request.stats().spec_verify_passes, 0u);
    LSE_EXPECT(session.history() == prompt);
    expect_retained_dflash(fixture, session, 2);
  }
  limits.stop_tokens.clear();
  limits.max_tokens = 4;
  runtime::Generator request(*fixture.target, greedy_sampling());
  request.use_dflash2(*fixture.draft);
  const auto output = request.generate(session, prompt, limits);
  LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
  LSE_EXPECT(*output == std::vector<std::uint32_t>({0, 0, 0, 0}));
  LSE_EXPECT_EQ(request.stats().prompt_tokens, 2);
  expect_retained_dflash(fixture, session, 5);
}
LSE_TEST(dflash2_generator_unrelated_prefix_coldstarts_retained_request) {
  Fixture fixture;
  const auto opened = fixture.open(FixtureWeights::kConstantTokens);
  LSE_EXPECT_OK(opened); if (!opened.ok()) return;
  runtime::Session session("unrelated", 1);
  runtime::GenerationLimits limits;
  limits.max_tokens = 4;
  {
    runtime::Generator first(*fixture.target, greedy_sampling());
    first.use_dflash2(*fixture.draft);
    const auto output = first.generate(session, {1, 2}, limits);
    LSE_EXPECT(output.ok()); if (!output.ok()) return;
  }
  expect_retained_dflash(fixture, session, 5);
  const std::vector<std::uint32_t> unrelated{7, 6, 5, 4, 3, 2, 1};
  runtime::Generator next(*fixture.target, greedy_sampling());
  next.use_dflash2(*fixture.draft);
  const auto output = next.generate(session, unrelated, limits);
  LSE_EXPECT(output.ok()); if (!output.ok()) { LSE_EXPECT_OK(output.status()); return; }
  LSE_EXPECT_EQ(next.stats().prompt_tokens, static_cast<std::int32_t>(unrelated.size()));
  auto expected_history = unrelated;
  expected_history.insert(expected_history.end(), output->begin(), output->end());
  LSE_EXPECT(session.history() == expected_history);
  expect_retained_dflash(fixture, session, 10);

  Fixture cold;
  const auto cold_opened = cold.open(FixtureWeights::kConstantTokens);
  LSE_EXPECT_OK(cold_opened); if (!cold_opened.ok()) return;
  runtime::Session fresh("fresh-unrelated", 1);
  runtime::Generator reference(*cold.target, greedy_sampling());
  reference.use_dflash2(*cold.draft);
  const auto expected_output = reference.generate(fresh, unrelated, limits);
  LSE_EXPECT(expected_output.ok()); if (!expected_output.ok()) return;
  LSE_EXPECT(*output == *expected_output);
  LSE_EXPECT(fresh.history() == session.history());
  expect_retained_dflash(cold, fresh, session.position());
}
#endif

std::vector<float> selector_reference(const std::vector<float>& predecessor, const std::vector<float>& gate,
    const std::vector<float>& successor, const std::vector<float>& unary,
    std::size_t rows, std::size_t kp, std::size_t ks, std::size_t rank) {
  std::vector<float> scores(rows * kp * ks);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t p = 0; p < kp; ++p)
      for (std::size_t s = 0; s < ks; ++s) {
        double sum = 0;
        for (std::size_t r = 0; r < rank; ++r)
          sum += static_cast<double>(predecessor[(row * kp + p) * rank + r]) *
              static_cast<double>(gate[row * rank + r]) *
              static_cast<double>(successor[(row * ks + s) * rank + r]);
        scores[(row * kp + p) * ks + s] = static_cast<float>(sum + static_cast<double>(unary[row * ks + s]));
      }
  return scores;
}
LSE_TEST(selector_native_primitive_cpu_preserves_batch_position_and_candidate_axes) {
  constexpr std::size_t batch = 2, positions = 3, kp = 3, ks = 4, rank = 5;
  std::vector<float> pred(batch * positions * kp * rank), succ(batch * positions * ks * rank);
  std::vector<float> gate(batch * positions * rank), unary(batch * positions * ks);
  for (std::size_t i = 0; i < pred.size(); ++i) pred[i] = static_cast<float>(i % 13) * .043f - .2f;
  for (std::size_t i = 0; i < succ.size(); ++i) succ[i] = static_cast<float>(i % 17) * .027f - .1f;
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = static_cast<float>(i % 11) * -.039f;
  for (std::size_t i = 0; i < unary.size(); ++i) unary[i] = static_cast<float>(i) * .031f;
  auto output = graph::custom("dflash2.selector", {filled({batch,positions,kp,rank},pred), filled({batch,positions,rank},gate),
      filled({batch,positions,ks,rank},succ), filled({batch,positions,ks},unary)},
      {static_cast<float>(positions),static_cast<float>(kp),static_cast<float>(ks),static_cast<float>(rank)});
  LSE_EXPECT(output.ok()); if (!output.ok()) return;
  auto got = read(output.release());
  auto expected = selector_reference(pred,gate,succ,unary,batch*positions,kp,ks,rank);
  for (std::size_t i = 0; i < got.size(); ++i) LSE_EXPECT_NEAR(got[i], expected[i], 2e-7);
}
int gpu_selector() {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(graph::Dialect::kLoom);
  constexpr std::size_t positions = 7, candidates = 16, rank = 256;
  std::vector<float> pred(positions*candidates*rank), succ(pred.size()), gate(positions*rank), unary(positions*candidates);
  for (std::size_t i = 0; i < pred.size(); ++i) { pred[i] = std::sin(static_cast<float>(i)*.013f)*.31f; succ[i] = std::cos(static_cast<float>(i)*.021f)*.29f; }
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = std::sin(static_cast<float>(i)*.037f)*.41f;
  for (std::size_t i = 0; i < unary.size(); ++i) unary[i] = std::cos(static_cast<float>(i)*.017f)*.23f;
  auto expected = selector_reference(pred,gate,succ,unary,positions,candidates,candidates,rank);
  auto output = graph::custom("dflash2.selector", {filled({1,positions,candidates,rank},pred),filled({1,positions,rank},gate),
      filled({1,positions,candidates,rank},succ),filled({1,positions,candidates},unary)},
      {static_cast<float>(positions),static_cast<float>(candidates),static_cast<float>(candidates),static_cast<float>(rank)});
  LSE_EXPECT(output.ok()); if (!output.ok()) return 1;
  LSE_EXPECT(output->shape() == Shape{1,positions,candidates,candidates});
  scheduler->reset_accumulated_trace(); auto got = read(output.release());
  const auto trace = scheduler->accumulated_trace();
  LSE_EXPECT(trace.device_groups > 0); LSE_EXPECT(trace.kernels_launched > 0);
  LSE_EXPECT_EQ(trace.host_groups,0u); LSE_EXPECT_EQ(trace.host_fallbacks,0u);
  LSE_EXPECT_EQ(got.size(), expected.size());
  if (got.size() != expected.size()) return 1;
  const double unit_roundoff = static_cast<double>(std::numeric_limits<float>::epsilon()) / 2;
  const double operations = static_cast<double>(rank + 2);
  const double gamma = operations * unit_roundoff / (1 - operations * unit_roundoff);
  const double relative = 2 * unit_roundoff / (1 - unit_roundoff);
  double maximum = 0, maximum_bound = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    LSE_EXPECT(std::isfinite(got[i])); LSE_EXPECT(std::isfinite(expected[i]));
    if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) continue;
    const auto row = i / (candidates * candidates);
    const auto p = (i / candidates) % candidates, s = i % candidates;
    double absolute_terms = 0;
    for (std::size_t r = 0; r < rank; ++r)
      absolute_terms += std::fabs(static_cast<double>(pred[(row*candidates+p)*rank+r]) *
          static_cast<double>(gate[row*rank+r]) * static_cast<double>(succ[(row*candidates+s)*rank+r]));
    // Product rounding and rank FMAs use the dot-product forward bound;
    // unary addition and reference narrowing supply the relative term.
    const double bound = gamma * absolute_terms + relative * std::fabs(static_cast<double>(expected[i]));
    const double error = std::fabs(static_cast<double>(got[i]) - static_cast<double>(expected[i]));
    maximum = std::max(maximum,error); maximum_bound = std::max(maximum_bound,bound);
    LSE_EXPECT_NEAR(got[i],expected[i],bound);
  }
  std::printf("DFlash2 selector shape=7x16x16 rank=256 max_absolute_error=%g max_rounding_bound=%g device_groups=%u host_groups=%u host_fallbacks=%u\n",
      maximum,maximum_bound,trace.device_groups,trace.host_groups,trace.host_fallbacks);
  return lse::test::Registry::get().failures ? 1 : 0;
}
int gpu_attention() {
  auto* scheduler = graph::default_scheduler();
  if (!scheduler) return 1;
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(graph::Dialect::kLoom);
  constexpr std::size_t heads = 4, kv_heads = 2, queries = 8, dim = 128, capacity = 2055;
  std::vector<float> q(heads*queries*dim), k(kv_heads*(capacity+queries)*dim), v(k.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i)*.013f)*.35f;
  for (std::size_t i = 0; i < k.size(); ++i) { k[i] = std::cos(static_cast<float>(i)*.019f)*.2f; v[i] = std::sin(static_cast<float>(i)*.017f); }
  for (float live : {17.0f, 2055.0f}) {
    const std::vector<float> meta{live, static_cast<float>(capacity)};
    auto expected = attention_reference(q,k,v,meta,heads,kv_heads,dim,queries,2048);
    auto result = graph::custom("dflash2.attention", {filled({1,heads,queries,dim},q),
        filled({1,kv_heads,capacity+queries,dim},k), filled({1,kv_heads,capacity+queries,dim},v), filled({2},meta)},
        {static_cast<float>(heads),static_cast<float>(kv_heads),static_cast<float>(dim),2048.0f});
    LSE_EXPECT(result.ok()); if (!result.ok()) return 1;
    scheduler->reset_accumulated_trace(); auto got = read(result.release());
    const auto trace = scheduler->accumulated_trace();
    LSE_EXPECT(trace.device_groups > 0);
    LSE_EXPECT(trace.kernels_launched > 0);
    LSE_EXPECT_EQ(trace.host_groups, 0u);
    LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
    float maximum = 0;
    for (std::size_t i = 0; i < got.size(); ++i) maximum = std::max(maximum,std::fabs(got[i]-expected[i]));
    std::printf("DFlash2 attention live=%.0f max_absolute_error=%g host_groups=%u\n",static_cast<double>(live),static_cast<double>(maximum),trace.host_groups);
    LSE_EXPECT(maximum < 2e-5f);
  }
  return lse::test::Registry::get().failures ? 1 : 0;
}
LSE_TEST(ring_cache_writes_only_new_rows_and_keeps_rewind_context) {
  constexpr std::size_t heads=2,capacity=11,dim=5,queries=4,window=8;
  auto cache_k=filled({1,heads,capacity,dim},std::vector<float>(heads*capacity*dim,0));
  auto cache_v=filled(cache_k.shape(),std::vector<float>(heads*capacity*dim,0));
  std::vector<float> expected_k(heads*capacity*dim,0),expected_v(expected_k.size(),0);
  std::int32_t position=0,live=0;
  const auto value=[](std::size_t h,std::int32_t t,std::size_t d,bool key) {
    const auto at=static_cast<float>(h*1000+static_cast<std::size_t>(t)*17+d);
    return key ? std::sin(at*.037f)*.3f : std::cos(at*.019f)*.7f;
  };
  const auto check_attention=[&] {
    std::vector<float> q(4*queries*dim),prop_k(heads*queries*dim),prop_v(prop_k.size());
    for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(static_cast<float>(i)*.071f)*.4f;
    for(std::size_t h=0;h<heads;++h)for(std::size_t t=0;t<queries;++t)for(std::size_t d=0;d<dim;++d) {
      prop_k[(h*queries+t)*dim+d]=value(h,position+static_cast<std::int32_t>(t),d,true);
      prop_v[(h*queries+t)*dim+d]=value(h,position+static_cast<std::int32_t>(t),d,false);
    }
    std::vector<float> linear_k(heads*(capacity+queries)*dim,0),linear_v(linear_k.size(),0);
    for(std::size_t h=0;h<heads;++h) {
      for(std::size_t t=capacity-static_cast<std::size_t>(live);t<capacity;++t)
        for(std::size_t d=0;d<dim;++d) {
          const auto absolute=position-static_cast<std::int32_t>(capacity-t);
          linear_k[(h*(capacity+queries)+t)*dim+d]=value(h,absolute,d,true);
          linear_v[(h*(capacity+queries)+t)*dim+d]=value(h,absolute,d,false);
        }
      for(std::size_t t=0;t<queries;++t)for(std::size_t d=0;d<dim;++d) {
        linear_k[(h*(capacity+queries)+capacity+t)*dim+d]=prop_k[(h*queries+t)*dim+d];
        linear_v[(h*(capacity+queries)+capacity+t)*dim+d]=prop_v[(h*queries+t)*dim+d];
      }
    }
    auto output=graph::custom("dflash2.ring_attention.v2",{filled({1,4,queries,dim},q),cache_k,cache_v,
        filled({1,heads,queries,dim},prop_k),filled({1,heads,queries,dim},prop_v),
        filled({3},{static_cast<float>(live),static_cast<float>(capacity),static_cast<float>(position%static_cast<std::int32_t>(capacity))})},
        {4,static_cast<float>(heads),static_cast<float>(dim),static_cast<float>(window)});
    LSE_EXPECT(output.ok());if(!output.ok())return;
    const auto actual=read(output.release());
    const auto expected=attention_reference(q,linear_k,linear_v,{static_cast<float>(live),static_cast<float>(capacity)},4,heads,dim,queries,window);
    LSE_EXPECT(actual==expected);
  };
  const auto append=[&](std::size_t rows) {
    std::vector<float> update_k(heads*rows*dim),update_v(update_k.size());
    for(std::size_t h=0;h<heads;++h)for(std::size_t t=0;t<rows;++t)for(std::size_t d=0;d<dim;++d) {
      const auto absolute=position+static_cast<std::int32_t>(t);
      update_k[(h*rows+t)*dim+d]=value(h,absolute,d,true);
      update_v[(h*rows+t)*dim+d]=value(h,absolute,d,false);
      const auto at=(h*capacity+static_cast<std::size_t>(absolute)%capacity)*dim+d;
      expected_k[at]=update_k[(h*rows+t)*dim+d];expected_v[at]=update_v[(h*rows+t)*dim+d];
    }
    auto first=filled({1},{static_cast<float>(position)});
    auto keys=graph::custom("dflash2.cache_write",{cache_k,filled({1,heads,static_cast<std::int64_t>(rows),dim},update_k),first},
        {static_cast<float>(heads),static_cast<float>(capacity),static_cast<float>(dim),static_cast<float>(rows)});
    auto values=graph::custom("dflash2.cache_write",{cache_v,filled({1,heads,static_cast<std::int64_t>(rows),dim},update_v),first},
        {static_cast<float>(heads),static_cast<float>(capacity),static_cast<float>(dim),static_cast<float>(rows)});
    LSE_EXPECT(keys.ok());LSE_EXPECT(values.ok());if(!keys.ok()||!values.ok())return;
    LSE_EXPECT(read(*keys)==expected_k);LSE_EXPECT(read(*values)==expected_v);
    LSE_EXPECT(keys->node()->buffer.handle==cache_k.node()->buffer.handle);
    LSE_EXPECT(values->node()->buffer.handle==cache_v.node()->buffer.handle);
    position+=static_cast<std::int32_t>(rows);live=std::min(static_cast<std::int32_t>(capacity),live+static_cast<std::int32_t>(rows));
    check_attention();
  };
  append(11);
  for(int round=0;round<6;++round) {
    append(4);
    const auto drop=round%4+1;
    position-=drop;live-=drop;
    check_attention();
    append(static_cast<std::size_t>(drop));
    append(2);
  }
}
LSE_TEST(ring_cache_and_attention_emit_native_loom_without_context_concatenation) {
  const auto leaf=[](Shape shape) {
    auto node=std::make_shared<graph::Node>();node->shape=shape;node->dtype=DType::kF32;node->materialized=true;
    return graph::Array(node);
  };
  auto cache=leaf({1,8,2055,128}),update=leaf({1,8,8,128});
  auto written=graph::custom("dflash2.cache_write",{cache,update,leaf({1})},{8,2055,128,8});
  auto attention=graph::custom("dflash2.ring_attention.v2",{leaf({1,32,8,128}),cache,cache,update,update,leaf({3})},{32,8,128,2048});
  LSE_EXPECT(written.ok());LSE_EXPECT(attention.ok());if(!written.ok()||!attention.ok())return;
  backend::DeviceInfo device;device.arch="gfx1201";device.wavefront_size=32;
  device.max_threads_per_workgroup=1024;device.lds_bytes_per_workgroup=65536;
  backend::LoomEmitter emitter;
  for(const auto& output:{*written,*attention}) {
    const graph::NodePtr roots[]={output.node()};
    const auto groups=graph::Partitioner::partition(roots);LSE_EXPECT_EQ(groups.size(),1u);
    for(const auto& group:groups) {
      auto emitted=emitter.emit(group,device);LSE_EXPECT(emitted.ok());
      if(!emitted.ok())std::fprintf(stderr,"%s\n",emitted.status().to_string().c_str());
    }
  }
}
int gpu_ring_cache() {
  auto* scheduler=graph::default_scheduler();
  if(!scheduler)return 1;
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(graph::Dialect::kLoom);
  constexpr std::size_t heads=4,kv_heads=2,capacity=2055,dim=128,queries=8;
  std::vector<float> q(heads*queries*dim),initial_k(kv_heads*capacity*dim),initial_v(initial_k.size());
  std::vector<float> update_k(kv_heads*queries*dim),update_v(update_k.size());
  for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(static_cast<float>(i)*.013f)*.35f;
  for(std::size_t i=0;i<initial_k.size();++i) {
    initial_k[i]=std::cos(static_cast<float>(i)*.019f)*.2f;
    initial_v[i]=std::sin(static_cast<float>(i)*.017f);
  }
  for(std::size_t i=0;i<update_k.size();++i) {
    update_k[i]=std::sin(static_cast<float>(i)*.023f)*.31f;
    update_v[i]=std::cos(static_cast<float>(i)*.029f)*.43f;
  }
  const std::size_t first=capacity-2,end=(first+queries)%capacity;
  auto expected_k=initial_k,expected_v=initial_v;
  for(std::size_t h=0;h<kv_heads;++h)for(std::size_t r=0;r<queries;++r)for(std::size_t d=0;d<dim;++d) {
    expected_k[(h*capacity+(first+r)%capacity)*dim+d]=update_k[(h*queries+r)*dim+d];
    expected_v[(h*capacity+(first+r)%capacity)*dim+d]=update_v[(h*queries+r)*dim+d];
  }
  auto cache_k=filled({1,kv_heads,capacity,dim},initial_k),cache_v=filled(cache_k.shape(),initial_v);
  auto key_update=filled({1,kv_heads,queries,dim},update_k),value_update=filled(key_update.shape(),update_v);
  auto offset=filled({1},{static_cast<float>(first)});
  auto written_k=graph::custom("dflash2.cache_write",{cache_k,key_update,offset},
      {static_cast<float>(kv_heads),static_cast<float>(capacity),static_cast<float>(dim),static_cast<float>(queries)});
  auto written_v=graph::custom("dflash2.cache_write",{cache_v,value_update,offset},
      {static_cast<float>(kv_heads),static_cast<float>(capacity),static_cast<float>(dim),static_cast<float>(queries)});
  LSE_EXPECT(written_k.ok());LSE_EXPECT(written_v.ok());if(!written_k.ok()||!written_v.ok())return 1;
  scheduler->reset_accumulated_trace();
  const graph::NodePtr write_roots[]={written_k->node(),written_v->node()};
  LSE_EXPECT_OK(scheduler->eval(write_roots,false));
  LSE_EXPECT_OK(scheduler->drain());
  LSE_EXPECT(written_k->node()->buffer.handle==cache_k.node()->buffer.handle);
  LSE_EXPECT(written_v->node()->buffer.handle==cache_v.node()->buffer.handle);
  LSE_EXPECT(read(*written_k)==expected_k);LSE_EXPECT(read(*written_v)==expected_v);
  float maximum=0;
  for(std::size_t live:{17u,2055u})for(std::size_t rewind:{0u,3u}) {
    const auto visible=live-rewind,ring_end=(end+capacity-rewind)%capacity;
    std::vector<float> linear_k(kv_heads*(capacity+queries)*dim),linear_v(linear_k.size());
    for(std::size_t h=0;h<kv_heads;++h) {
      for(std::size_t j=0;j<capacity;++j)for(std::size_t d=0;d<dim;++d) {
        linear_k[(h*(capacity+queries)+j)*dim+d]=expected_k[(h*capacity+(ring_end+j)%capacity)*dim+d];
        linear_v[(h*(capacity+queries)+j)*dim+d]=expected_v[(h*capacity+(ring_end+j)%capacity)*dim+d];
      }
      for(std::size_t j=0;j<queries;++j)for(std::size_t d=0;d<dim;++d) {
        linear_k[(h*(capacity+queries)+capacity+j)*dim+d]=update_k[(h*queries+j)*dim+d];
        linear_v[(h*(capacity+queries)+capacity+j)*dim+d]=update_v[(h*queries+j)*dim+d];
      }
    }
    const auto expected=attention_reference(q,linear_k,linear_v,{static_cast<float>(visible),static_cast<float>(capacity)},
        heads,kv_heads,dim,queries,2048);
    auto output=graph::custom("dflash2.ring_attention.v2",{filled({1,heads,queries,dim},q),*written_k,*written_v,key_update,value_update,
        filled({3},{static_cast<float>(visible),static_cast<float>(capacity),static_cast<float>(ring_end)})},
        {static_cast<float>(heads),static_cast<float>(kv_heads),static_cast<float>(dim),2048});
    LSE_EXPECT(output.ok());if(!output.ok())return 1;
    const auto actual=read(output.release());
    LSE_EXPECT_EQ(actual.size(),expected.size());if(actual.size()!=expected.size())return 1;
    for(std::size_t i=0;i<actual.size();++i) {
      LSE_EXPECT(std::isfinite(actual[i]));
      if(std::isfinite(actual[i]))maximum=std::max(maximum,std::fabs(actual[i]-expected[i]));
    }
  }
  const auto trace=scheduler->accumulated_trace();
  LSE_EXPECT(trace.device_groups>=6u);LSE_EXPECT(trace.kernels_launched>=6u);
  LSE_EXPECT_EQ(trace.host_groups,0u);LSE_EXPECT_EQ(trace.host_fallbacks,0u);
  LSE_EXPECT(maximum<2e-5f);
  std::printf("DFlash2 ring cache wrapped_write=8 live=17,2055 rewind=0,3 max_absolute_error=%g device_groups=%u host_groups=%u host_fallbacks=%u\n",
      static_cast<double>(maximum),trace.device_groups,trace.host_groups,trace.host_fallbacks);
  return lse::test::Registry::get().failures?1:0;
}

namespace {
constexpr const char *kPartial = "dflash2.ring_partial256.v1";
constexpr const char *kMerge = "dflash2.ring_merge256.v1";
graph::Array split_ring_output(std::vector<graph::Array> inputs,
                               std::size_t heads, std::size_t kvheads,
                               std::size_t queries, std::size_t dim,
                               std::size_t window) {
  auto partial =
      graph::custom(kPartial, inputs,
                    {static_cast<float>(heads), static_cast<float>(kvheads),
                     static_cast<float>(dim), static_cast<float>(window)});
  LSE_EXPECT(partial.ok());
  if (!partial.ok())
    return {};
  const auto parts = partial->shape().dim(2);
  auto output =
      graph::custom(kMerge, {*partial},
                    {static_cast<float>(heads), static_cast<float>(queries),
                     static_cast<float>(dim), static_cast<float>(parts)});
  LSE_EXPECT(output.ok());
  return output.ok() ? output.release() : graph::Array{};
}
LSE_TEST(
    split_ring_cpu_preserves_wrapped_cache_masks_rewinds_and_empty_partitions) {
  constexpr std::size_t heads = 2, kvheads = 1, capacity = 515, dim = 5,
                        queries = 4, window = 508;
  std::vector<float> q(heads * queries * dim), keys(kvheads * capacity * dim),
      values(keys.size()), pk(kvheads * queries * dim), pv(pk.size());
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = std::sin(static_cast<float>(i) * .13f) * .37f;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = std::cos(static_cast<float>(i) * .19f) * .2f;
    values[i] = std::sin(static_cast<float>(i) * .17f);
  }
  for (std::size_t i = 0; i < pk.size(); ++i) {
    pk[i] = std::sin(static_cast<float>(i) * .23f) * .31f;
    pv[i] = std::cos(static_cast<float>(i) * .29f) * .43f;
  }
  auto ck = filled({1, kvheads, capacity, dim}, keys),
       cv = filled(ck.shape(), values),
       kp = filled({1, kvheads, queries, dim}, pk), vp = filled(kp.shape(), pv);
  const std::size_t first = capacity - 2, end = (first + queries) % capacity;
  auto offset = filled({1}, {static_cast<float>(first)});
  auto wk =
      graph::custom("dflash2.cache_write", {ck, kp, offset},
                    {1, static_cast<float>(capacity), static_cast<float>(dim),
                     static_cast<float>(queries)});
  auto wv =
      graph::custom("dflash2.cache_write", {cv, vp, offset},
                    {1, static_cast<float>(capacity), static_cast<float>(dim),
                     static_cast<float>(queries)});
  LSE_EXPECT(wk.ok());
  LSE_EXPECT(wv.ok());
  if (!wk.ok() || !wv.ok())
    return;
  for (std::size_t r = 0; r < queries; ++r)
    for (std::size_t d = 0; d < dim; ++d) {
      keys[((first + r) % capacity) * dim + d] = pk[r * dim + d];
      values[((first + r) % capacity) * dim + d] = pv[r * dim + d];
    }
  LSE_EXPECT(read(*wk) == keys);
  LSE_EXPECT(read(*wv) == values);
  for (std::size_t live : {7u, 515u})
    for (std::size_t rewind : {0u, 3u}) {
      const auto visible = live - rewind,
                 ringend = (end + capacity - rewind) % capacity;
      std::vector<float> linear_k(kvheads * (capacity + queries) * dim),
          linear_v(linear_k.size());
      for (std::size_t j = 0; j < capacity; ++j)
        for (std::size_t d = 0; d < dim; ++d) {
          linear_k[j * dim + d] = keys[((ringend + j) % capacity) * dim + d];
          linear_v[j * dim + d] = values[((ringend + j) % capacity) * dim + d];
        }
      std::copy(pk.begin(), pk.end(),
                linear_k.begin() + static_cast<std::ptrdiff_t>(capacity * dim));
      std::copy(pv.begin(), pv.end(),
                linear_v.begin() + static_cast<std::ptrdiff_t>(capacity * dim));
      std::vector<graph::Array> inputs{
          filled({1, heads, queries, dim}, q),
          *wk,
          *wv,
          kp,
          vp,
          filled({3},
                 {static_cast<float>(visible), static_cast<float>(capacity),
                  static_cast<float>(ringend)})};
      auto partial =
          graph::custom(kPartial, inputs,
                        {static_cast<float>(heads), static_cast<float>(kvheads),
                         static_cast<float>(dim), static_cast<float>(window)});
      LSE_EXPECT(partial.ok());
      if (!partial.ok())
        return;
      LSE_EXPECT((partial->shape() == Shape{1, heads, 3, queries, dim + 2}));
      const auto parts = read(*partial);
      if (live == 7)
        for (std::size_t h = 0; h < heads; ++h)
          for (std::size_t row = 0; row < queries; ++row) {
            const auto base = (h * 3 * queries + row) * (dim + 2);
            LSE_EXPECT(parts[base] == -std::numeric_limits<float>::infinity());
            for (std::size_t d = 1; d < dim + 2; ++d)
              LSE_EXPECT_EQ(parts[base + d], 0.0f);
          }
      auto result =
          graph::custom(kMerge, {*partial},
                        {static_cast<float>(heads), static_cast<float>(queries),
                         static_cast<float>(dim), 3});
      LSE_EXPECT(result.ok());
      if (!result.ok())
        return;
      auto actual = read(result.release()),
           expected = attention_reference(
               q, linear_k, linear_v,
               {static_cast<float>(visible), static_cast<float>(capacity)},
               heads, kvheads, dim, queries, window);
      LSE_EXPECT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        LSE_EXPECT(std::isfinite(actual[i]));
        LSE_EXPECT_NEAR(actual[i], expected[i], 2e-6);
      }
    }
}
LSE_TEST(split_ring_merge_all_empty_records_return_finite_zero) {
  std::vector<float> data(2 * 3 * 4 * 7, 0);
  for (std::size_t row = 0; row < data.size() / 7; ++row)
    data[row * 7] = -std::numeric_limits<float>::infinity();
  auto output =
      graph::custom(kMerge, {filled({1, 2, 3, 4, 7}, data)}, {2, 4, 5, 3});
  LSE_EXPECT(output.ok());
  if (!output.ok())
    return;
  const auto actual = read(output.release());
  for (float v : actual) {
    LSE_EXPECT(std::isfinite(v));
    LSE_EXPECT_EQ(v, 0.0f);
  }
}
LSE_TEST(split_ring_partial_and_merge_emit_native_loom_at_actual_geometry) {
  const auto leaf = [](Shape shape) {
    auto n = std::make_shared<graph::Node>();
    n->shape = shape;
    n->dtype = DType::kF32;
    n->materialized = true;
    return graph::Array(n);
  };
  std::vector<graph::Array> inputs{
      leaf({1, 32, 8, 128}), leaf({1, 8, 2055, 128}), leaf({1, 8, 2055, 128}),
      leaf({1, 8, 8, 128}),  leaf({1, 8, 8, 128}),    leaf({3})};
  auto out = split_ring_output(inputs, 32, 8, 8, 128, 2048);
  LSE_EXPECT(out.node() != nullptr);
  if (!out.node())
    return;
  const graph::NodePtr roots[]{out.node()};
  auto groups = graph::Partitioner::partition(roots);
  LSE_EXPECT_EQ(groups.size(), 2u);
  backend::DeviceInfo device;
  device.arch = "gfx1201";
  device.wavefront_size = 32;
  device.max_threads_per_workgroup = 1024;
  device.lds_bytes_per_workgroup = 65536;
  backend::LoomEmitter emitter;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    auto emitted = emitter.emit(groups[i], device);
    LSE_EXPECT(emitted.ok());
    if (!emitted.ok()) {
      std::fprintf(stderr, "%s\n", emitted.status().to_string().c_str());
      return;
    }
    std::ofstream("/private/tmp/lse-dflash2-split-ring-" + std::to_string(i) +
                  ".loom")
        << emitted->source;
#if defined(LSE_HRX_LINKED) && LSE_HAVE_LOOMC
    backend::LoomcCompiler compiler;
    auto compiled = compiler.compile(emitted->source, "gfx1201");
    if (!compiled.ok())
      std::fprintf(stderr, "split_stage%zu: %s\n", i,
                   compiled.status().to_string().c_str());
    LSE_EXPECT(compiled.ok());
#endif
    LSE_EXPECT_EQ(emitted->dims.workgroup_count[0], i == 0 ? 288u : 256u);
  }
}
int gpu_ring_split() {
  auto *scheduler = graph::default_scheduler();
  if (!scheduler)
    return 1;
  scheduler->set_mode(graph::Scheduler::Mode::kDeviceFirst);
  scheduler->set_dialect(graph::Dialect::kLoom);
  constexpr std::size_t heads = 32, kvheads = 8, capacity = 2055, dim = 128,
                        queries = 8, window = 2048;
  std::vector<float> q(heads * queries * dim), keys(kvheads * capacity * dim),
      values(keys.size()), pk(kvheads * queries * dim), pv(pk.size());
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = std::sin(static_cast<float>(i) * .013f) * .35f;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = std::cos(static_cast<float>(i) * .019f) * .2f;
    values[i] = std::sin(static_cast<float>(i) * .017f);
  }
  for (std::size_t i = 0; i < pk.size(); ++i) {
    pk[i] = std::sin(static_cast<float>(i) * .023f) * .31f;
    pv[i] = std::cos(static_cast<float>(i) * .029f) * .43f;
  }
  const std::size_t first = capacity - 2, end = (first + queries) % capacity;
  auto qleaf = filled({1, heads, queries, dim}, q),
       ck = filled({1, kvheads, capacity, dim}, keys),
       cv = filled(ck.shape(), values),
       kp = filled({1, kvheads, queries, dim}, pk), vp = filled(kp.shape(), pv),
       offset = filled({1}, {static_cast<float>(first)});
  auto wk =
      graph::custom("dflash2.cache_write", {ck, kp, offset},
                    {static_cast<float>(kvheads), static_cast<float>(capacity),
                     static_cast<float>(dim), static_cast<float>(queries)});
  auto wv =
      graph::custom("dflash2.cache_write", {cv, vp, offset},
                    {static_cast<float>(kvheads), static_cast<float>(capacity),
                     static_cast<float>(dim), static_cast<float>(queries)});
  LSE_EXPECT(wk.ok());
  LSE_EXPECT(wv.ok());
  if (!wk.ok() || !wv.ok())
    return 1;
  scheduler->reset_accumulated_trace();
  for (std::size_t h = 0; h < kvheads; ++h)
    for (std::size_t r = 0; r < queries; ++r)
      for (std::size_t d = 0; d < dim; ++d) {
        keys[(h * capacity + (first + r) % capacity) * dim + d] =
            pk[(h * queries + r) * dim + d];
        values[(h * capacity + (first + r) % capacity) * dim + d] =
            pv[(h * queries + r) * dim + d];
      }
  LSE_EXPECT(read(*wk) == keys);
  LSE_EXPECT(read(*wv) == values);
  auto frozen_k = graph::Array::from_buffer(wk->node()->buffer, wk->shape(),
                                            DType::kF32),
       frozen_v = graph::Array::from_buffer(wv->node()->buffer, wv->shape(),
                                            DType::kF32);
  float maximum = 0;
  const std::vector<std::size_t> live_cases{17, 1024, 2055};
  for (std::size_t live : live_cases)
    for (std::size_t rewind : {0u, 3u}) {
      const auto visible = live - rewind,
                 ringend = (end + capacity - rewind) % capacity;
      std::vector<float> linear_k(kvheads * (capacity + queries) * dim),
          linear_v(linear_k.size());
      for (std::size_t h = 0; h < kvheads; ++h) {
        for (std::size_t j = 0; j < capacity; ++j)
          for (std::size_t d = 0; d < dim; ++d) {
            linear_k[(h * (capacity + queries) + j) * dim + d] =
                keys[(h * capacity + (ringend + j) % capacity) * dim + d];
            linear_v[(h * (capacity + queries) + j) * dim + d] =
                values[(h * capacity + (ringend + j) % capacity) * dim + d];
          }
        for (std::size_t r = 0; r < queries; ++r)
          for (std::size_t d = 0; d < dim; ++d) {
            linear_k[(h * (capacity + queries) + capacity + r) * dim + d] =
                pk[(h * queries + r) * dim + d];
            linear_v[(h * (capacity + queries) + capacity + r) * dim + d] =
                pv[(h * queries + r) * dim + d];
          }
      }
      std::vector<graph::Array> inputs{
          qleaf,
          frozen_k,
          frozen_v,
          kp,
          vp,
          filled({3},
                 {static_cast<float>(visible), static_cast<float>(capacity),
                  static_cast<float>(ringend)})};
      auto output =
          split_ring_output(inputs, heads, kvheads, queries, dim, window);
      if (!output.node())
        return 1;
      const auto actual = read(output),
                 expected =
                     attention_reference(q, linear_k, linear_v,
                                         {static_cast<float>(visible),
                                          static_cast<float>(capacity)},
                                         heads, kvheads, dim, queries, window);
      LSE_EXPECT_EQ(actual.size(), expected.size());
      if (actual.size() != expected.size())
        return 1;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        LSE_EXPECT(std::isfinite(actual[i]));
        if (std::isfinite(actual[i]))
          maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
      }
      LSE_EXPECT(read(frozen_k) == keys);
      LSE_EXPECT(read(frozen_v) == values);
    }
  const auto trace = scheduler->accumulated_trace();
  LSE_EXPECT(trace.device_groups >= 2u + 4u * live_cases.size());
  LSE_EXPECT(trace.kernels_launched >= 2u + 4u * live_cases.size());
  LSE_EXPECT_EQ(trace.host_groups, 0u);
  LSE_EXPECT_EQ(trace.host_fallbacks, 0u);
  LSE_EXPECT(maximum < 2e-5f);
  std::printf("DFlash2 split ring32Q/8KV live17,1024,2055 rewind0/3 "
              "max_absolute_error=%g device_groups=%u host_groups=%u "
              "host_fallbacks=%u\n",
              static_cast<double>(maximum), trace.device_groups,
              trace.host_groups, trace.host_fallbacks);
  std::fflush(stdout);
  return lse::test::Registry::get().failures ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--gpu-attention") return gpu_attention();
  if (argc == 2 && std::string(argv[1]) == "--gpu-selector") return gpu_selector();
  if (argc == 2 && std::string(argv[1]) == "--gpu-ring-cache") return gpu_ring_cache();
  if (argc == 2 && std::string(argv[1]) == "--gpu-ring-split") return gpu_ring_split();
  if (auto* scheduler = graph::default_scheduler()) scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);
  return lse::test::run_all();
}
