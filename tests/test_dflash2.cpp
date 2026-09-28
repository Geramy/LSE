#include "harness.hpp"
#include "lse/model/dflash2.hpp"
#include "lse/model/layer.hpp"
#include "lse/model/weights.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/graph/ops.hpp"

#include <algorithm>
#include <cmath>
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
void write_weights(const std::filesystem::path& path, const std::vector<Named>& names) {
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
  Status open() {
    directory = std::filesystem::temp_directory_path() / ("lse-dflash2-fixture-" + std::to_string(getpid()));
    std::filesystem::create_directories(directory / "draft");
    write_weights(directory / "model.safetensors", {{"embed.weight", {16, 8}}, {"final_norm.weight", {8}},
                  {"blocks.0.norm1.weight", {8}}, {"blocks.0.norm2.weight", {8}}});
    LSE_ASSIGN_OR(weights, model::SafeTensors::open((directory / "model.safetensors").string()));
    model::HybridLMSpec spec; spec.zero_centered_norm = false;
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
    write_weights(directory / "draft" / "model.safetensors", names);
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
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--gpu-attention") return gpu_attention();
  if (argc == 2 && std::string(argv[1]) == "--gpu-selector") return gpu_selector();
  if (auto* scheduler = graph::default_scheduler()) scheduler->set_mode(graph::Scheduler::Mode::kHostOnly);
  return lse::test::run_all();
}
