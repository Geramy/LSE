#include "harness.hpp"
#include "lse/models/sampling_defaults.hpp"
#include "request_sampling.hpp"

#include <filesystem>
#include <fstream>
#include <unistd.h>

using namespace lse;
using json = nlohmann::json;
namespace {
struct Files {
  std::filesystem::path dir = std::filesystem::temp_directory_path() /
      ("lse-sampling-defaults-" + std::to_string(getpid()));
  Files() { std::filesystem::create_directories(dir); }
  ~Files() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  void put(const char* name, const std::string& text) const {
    std::ofstream file(dir / name);
    file << text;
  }
};
}

LSE_TEST(sampling_defaults_merge_each_field_in_priority_order) {
  const auto result = models::sampling_defaults_from_json(R"({
    "model_type":"qwen3_5", "temperature":0.7, "top_k":7,
    "generation_config":{"temperature":0.6,"top_p":0.8,"top_k":null}
  })", R"({"temperature":0.5,"top_k":5,"top_p":null,"repetition_penalty":1.1})");
  LSE_EXPECT(result.ok());
  if (!result.ok()) return;
  LSE_EXPECT_NEAR(result->temperature, 0.5, 1e-6);
  LSE_EXPECT_EQ(result->top_k, 5);
  LSE_EXPECT_NEAR(result->top_p, 0.8, 1e-6);
  LSE_EXPECT_NEAR(result->repetition_penalty, 1.1, 1e-6);
}

LSE_TEST(sampling_defaults_are_generic_with_a_missing_field_family_table) {
  auto unknown = models::sampling_defaults_from_json(R"({"model_type":"other"})");
  LSE_EXPECT(unknown.ok());
  if (!unknown.ok()) return;
  LSE_EXPECT_EQ(unknown->temperature, 1.0f);
  LSE_EXPECT_EQ(unknown->top_k, 0);
  LSE_EXPECT_EQ(unknown->top_p, 1.0f);
  for (const char* type : {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}) {
    auto family = models::sampling_defaults_from_json(json{{"model_type",type}}.dump());
    LSE_EXPECT(family.ok());
    if (!family.ok()) continue;
    LSE_EXPECT_EQ(family->temperature, 1.0f);
    LSE_EXPECT_EQ(family->top_k, 20);
    LSE_EXPECT_NEAR(family->top_p, 0.95, 1e-6);
  }
  auto nested = models::sampling_defaults_from_json(R"({"text_config":{"model_type":"qwen3_5_text"},"top_k":0})");
  LSE_EXPECT(nested.ok());
  if (nested.ok()) {
    LSE_EXPECT_EQ(nested->top_k, 0);
    LSE_EXPECT_NEAR(nested->top_p, 0.95, 1e-6);
  }
}

LSE_TEST(sampling_defaults_load_the_sibling_file_for_any_model) {
  Files files;
  files.put("config.json", R"({"model_type":"unregistered","temperature":0.7,"generation_config":{"top_k":4}})");
  auto absent = models::load_sampling_defaults((files.dir / "config.json").string());
  LSE_EXPECT(absent.ok());
  if (absent.ok()) {
    LSE_EXPECT_NEAR(absent->temperature, 0.7, 1e-6);
    LSE_EXPECT_EQ(absent->top_k, 4);
  }
  files.put("generation_config.json", R"({"temperature":1,"top_k":20,"top_p":0.95})");
  auto present = models::load_sampling_defaults((files.dir / "config.json").string());
  LSE_EXPECT(present.ok());
  if (present.ok()) {
    LSE_EXPECT_EQ(present->temperature, 1.0f);
    LSE_EXPECT_EQ(present->top_k, 20);
    LSE_EXPECT_NEAR(present->top_p, 0.95, 1e-6);
  }
  files.put("generation_config.json", "{");
  LSE_EXPECT(!models::load_sampling_defaults((files.dir / "config.json").string()).ok());
  files.put("generation_config.json", "");
  LSE_EXPECT(!models::load_sampling_defaults((files.dir / "config.json").string()).ok());
}

LSE_TEST(sampling_defaults_reject_malformed_supplied_metadata) {
  for (const char* generation : {"[]", "null", "{", "{\"temperature\":\"hot\"}",
      "{\"temperature\":-1}", "{\"top_k\":1.5}", "{\"top_k\":-1}",
      "{\"top_k\":18446744073709551615}", "{\"top_p\":1.5}",
      "{\"repetition_penalty\":0}", "{\"do_sample\":0}"})
    LSE_EXPECT(!models::sampling_defaults_from_json("{}", generation).ok());
  LSE_EXPECT(!models::sampling_defaults_from_json(R"({"generation_config":[]})").ok());
  LSE_EXPECT(!models::sampling_defaults_from_json("[]").ok());
  auto greedy = models::sampling_defaults_from_json(R"({"do_sample":true,"temperature":0.8})",
                                                   R"({"do_sample":false,"temperature":1})");
  LSE_EXPECT(greedy.ok());
  if (greedy.ok()) LSE_EXPECT_EQ(greedy->temperature, 0.0f);
}

LSE_TEST(http_sampling_explicit_zero_overrides_and_null_inherits) {
  const models::SamplingDefaults defaults{1.0f,20,0.95f,1.1f};
  auto omitted = server::detail::request_sampling(json::object(), defaults);
  LSE_EXPECT(omitted.ok());
  if (!omitted.ok()) return;
  LSE_EXPECT_EQ(omitted->temperature, defaults.temperature);
  LSE_EXPECT_EQ(omitted->top_k, defaults.top_k);
  LSE_EXPECT_EQ(omitted->top_p, defaults.top_p);
  LSE_EXPECT_EQ(omitted->repetition_penalty, defaults.repetition_penalty);
  auto nulls = server::detail::request_sampling(
      json{{"temperature",nullptr},{"top_k",nullptr},{"top_p",nullptr},{"frequency_penalty",nullptr}}, defaults);
  LSE_EXPECT(nulls.ok());
  if (nulls.ok()) {
    LSE_EXPECT_EQ(nulls->temperature, defaults.temperature);
    LSE_EXPECT_EQ(nulls->top_k, defaults.top_k);
    LSE_EXPECT_EQ(nulls->top_p, defaults.top_p);
    LSE_EXPECT_EQ(nulls->repetition_penalty, defaults.repetition_penalty);
  }
  auto explicit_values = server::detail::request_sampling(
      json{{"temperature",0},{"top_k",0},{"top_p",1},{"frequency_penalty",0},{"seed",42}}, defaults);
  LSE_EXPECT(explicit_values.ok());
  if (explicit_values.ok()) {
    LSE_EXPECT_EQ(explicit_values->temperature, 0.0f);
    LSE_EXPECT_EQ(explicit_values->top_k, 0);
    LSE_EXPECT_EQ(explicit_values->top_p, 1.0f);
    LSE_EXPECT_EQ(explicit_values->repetition_penalty, 1.0f);
    LSE_EXPECT_EQ(explicit_values->seed, 42u);
  }
  auto penalties = server::detail::request_sampling(
      json{{"frequency_penalty",0.5},{"repetition_penalty",1.3}}, defaults);
  LSE_EXPECT(penalties.ok());
  if (penalties.ok()) LSE_EXPECT_NEAR(penalties->repetition_penalty, 1.3, 1e-6);
  auto override = server::detail::request_sampling(json{{"temperature",0.6},{"top_k",8},{"top_p",0.7}}, defaults);
  LSE_EXPECT(override.ok());
  if (override.ok()) {
    LSE_EXPECT_NEAR(override->temperature, 0.6, 1e-6);
    LSE_EXPECT_EQ(override->top_k, 8);
    LSE_EXPECT_NEAR(override->top_p, 0.7, 1e-6);
  }
  for (const auto& malformed : {json{{"temperature","hot"}},json{{"top_k",1.5}},
      json{{"top_k",-1}},json{{"top_p",2}},json{{"seed",-1}}})
    LSE_EXPECT(!server::detail::request_sampling(malformed, defaults).ok());
}

LSE_TEST(sampling_defaults_reject_float_underflow_and_preserve_literal_zero) {
  for (const char* key : {"temperature", "top_p", "repetition_penalty"}) {
    const auto result = models::sampling_defaults_from_json(
        "{}", json{{key, 1e-50}}.dump());
    LSE_EXPECT(!result.ok());
  }
  auto zero = models::sampling_defaults_from_json(
      "{}", R"({"temperature":0,"top_p":0})");
  LSE_EXPECT(zero.ok());
  if (zero.ok()) {
    LSE_EXPECT_EQ(zero->temperature, 0.0f);
    LSE_EXPECT_EQ(zero->top_p, 0.0f);
  }
  LSE_EXPECT(!models::sampling_defaults_from_json(
      "{}", R"({"repetition_penalty":0})").ok());
}

LSE_TEST(http_sampling_rejects_float_underflow_and_preserves_literal_zero) {
  const models::SamplingDefaults defaults{1.0f, 20, 0.95f, 1.1f};
  for (const char* key : {"temperature", "top_p", "frequency_penalty", "repetition_penalty"}) {
    for (const double value : {1e-50, -1e-50}) {
      const auto result = server::detail::request_sampling(json{{key, value}}, defaults);
      LSE_EXPECT(!result.ok());
    }
  }
  auto zero = server::detail::request_sampling(
      json{{"temperature",0},{"top_p",0},{"frequency_penalty",0}}, defaults);
  LSE_EXPECT(zero.ok());
  if (zero.ok()) {
    LSE_EXPECT_EQ(zero->temperature, 0.0f);
    LSE_EXPECT_EQ(zero->top_p, 0.0f);
    LSE_EXPECT_EQ(zero->repetition_penalty, 1.0f);
  }
  LSE_EXPECT(!server::detail::request_sampling(
      json{{"repetition_penalty",0}}, defaults).ok());
}

LSE_TEST_MAIN()
