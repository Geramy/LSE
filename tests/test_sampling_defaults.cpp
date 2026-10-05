#include "harness.hpp"
#include "lse/models/sampling_defaults.hpp"
#include "request_sampling.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
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

LSE_TEST(sampling_defaults_without_files_are_neutral_and_say_so) {
  // No per-family table: a model that names nothing gets LSE's neutral
  // defaults, reported as such, whatever its model_type.
  for (const char* type : {"other", "qwen3_5", "qwen3_5_moe", "qwen3_5_text"}) {
    auto d = models::sampling_defaults_from_json(json{{"model_type", type}}.dump());
    LSE_EXPECT(d.ok());
    if (!d.ok()) continue;
    LSE_EXPECT_EQ(d->temperature, 1.0f);
    LSE_EXPECT_EQ(d->top_k, 0);
    LSE_EXPECT_EQ(d->top_p, 1.0f);
    LSE_EXPECT_EQ(d->min_p, 0.0f);
    LSE_EXPECT_EQ(d->repetition_penalty, 1.0f);
    LSE_EXPECT_EQ(d->presence_penalty, 0.0f);
    LSE_EXPECT(!d->max_new_tokens && !d->max_length);
    const json j = d->to_json();
    for (const char* key : {"temperature", "top_k", "top_p", "min_p", "repetition_penalty",
                            "presence_penalty", "max_new_tokens", "max_length"})
      LSE_EXPECT(j["sources"][key] == "lse_default");
    LSE_EXPECT(j["max_new_tokens"].is_null());
  }
}

LSE_TEST(generation_config_fields_and_their_sources_are_reported) {
  auto d = models::sampling_defaults_from_json(
      R"({"model_type":"x","temperature":0.3})",
      R"({"temperature":0.7,"top_k":40,"top_p":0.8,"min_p":0.05,"presence_penalty":1.5,
          "repetition_penalty":1.05,"max_new_tokens":2048,"max_length":32768,"bos_token_id":1})");
  LSE_EXPECT(d.ok());
  if (!d.ok()) return;
  LSE_EXPECT_NEAR(d->temperature, 0.7, 1e-6);
  LSE_EXPECT_EQ(d->top_k, 40);
  LSE_EXPECT_NEAR(d->top_p, 0.8, 1e-6);
  LSE_EXPECT_NEAR(d->min_p, 0.05, 1e-6);
  LSE_EXPECT_NEAR(d->presence_penalty, 1.5, 1e-6);
  LSE_EXPECT_NEAR(d->repetition_penalty, 1.05, 1e-6);
  LSE_EXPECT(d->max_new_tokens == 2048);
  LSE_EXPECT(d->max_length == 32768);
  const json j = d->to_json();
  LSE_EXPECT(j["sources"]["temperature"] == "generation_config.json");
  LSE_EXPECT(j["sources"]["min_p"] == "generation_config.json");
  LSE_EXPECT(j["max_new_tokens"] == 2048);
  auto from_config = models::sampling_defaults_from_json(R"({"generation_config":{"top_k":7}})");
  LSE_EXPECT(from_config.ok());
  if (from_config.ok()) {
    LSE_EXPECT(from_config->to_json()["sources"]["top_k"] == "config.json");
    LSE_EXPECT(from_config->to_json()["sources"]["top_p"] == "lse_default");
  }
  // A negative presence penalty is legal (it encourages repeats).
  LSE_EXPECT(models::sampling_defaults_from_json("{}", R"({"presence_penalty":-0.5})").ok());
  for (const char* bad : {R"({"min_p":1.5})", R"({"min_p":-0.1})", R"({"max_new_tokens":0})",
                          R"({"max_length":-5})", R"({"max_new_tokens":1.5})"})
    LSE_EXPECT(!models::sampling_defaults_from_json("{}", bad).ok());
}

LSE_TEST(real_qwen38_generation_config_supplies_the_defaults) {
  // tests/fixtures/chat_templates/qwen3.8/generation_config.json, verbatim.
  std::ifstream in("tests/fixtures/chat_templates/qwen3.8/generation_config.json");
  std::stringstream text;
  text << in.rdbuf();
  auto d = models::sampling_defaults_from_json(R"({"model_type":"qwen3_5"})", text.str());
  LSE_EXPECT(d.ok());
  if (!d.ok()) return;
  LSE_EXPECT_EQ(d->temperature, 1.0f);
  LSE_EXPECT_EQ(d->top_k, 20);
  LSE_EXPECT_NEAR(d->top_p, 0.95, 1e-6);
  LSE_EXPECT(d->to_json()["sources"]["top_k"] == "generation_config.json");
  LSE_EXPECT(d->to_json()["sources"]["min_p"] == "lse_default");
  LSE_EXPECT(!d->max_new_tokens);  // it sets no output limit
}

LSE_TEST(requests_inherit_generation_config_defaults_and_override_them) {
  models::SamplingDefaults defaults;
  defaults.temperature = 0.6f;
  defaults.top_k = 20;
  defaults.top_p = 0.95f;
  defaults.min_p = 0.05f;
  defaults.presence_penalty = 1.5f;
  auto inherited = server::detail::request_sampling(json::object(), defaults);
  LSE_EXPECT(inherited.ok());
  if (inherited.ok()) {
    LSE_EXPECT_NEAR(inherited->temperature, 0.6, 1e-6);
    LSE_EXPECT_EQ(inherited->top_k, 20);
    LSE_EXPECT_NEAR(inherited->top_p, 0.95, 1e-6);
    LSE_EXPECT_NEAR(inherited->min_p, 0.05, 1e-6);
    LSE_EXPECT_NEAR(inherited->presence_penalty, 1.5, 1e-6);
  }
  auto overridden = server::detail::request_sampling(
      json{{"temperature", 0.2}, {"top_k", 3}, {"top_p", 0.5}, {"min_p", 0.1}, {"presence_penalty", 0}},
      defaults);
  LSE_EXPECT(overridden.ok());
  if (overridden.ok()) {
    LSE_EXPECT_NEAR(overridden->temperature, 0.2, 1e-6);
    LSE_EXPECT_EQ(overridden->top_k, 3);
    LSE_EXPECT_NEAR(overridden->top_p, 0.5, 1e-6);
    LSE_EXPECT_NEAR(overridden->min_p, 0.1, 1e-6);
    LSE_EXPECT_EQ(overridden->presence_penalty, 0.0f);
  }
  // top_k -1 is vLLM's "off", like 0.
  auto off = server::detail::request_sampling(json{{"top_k", -1}}, defaults);
  LSE_EXPECT(off.ok() && off->top_k == 0);
  for (const auto& bad : {json{{"top_k", -2}}, json{{"min_p", 2}}, json{{"min_p", "x"}},
                          json{{"presence_penalty", "x"}}})
    LSE_EXPECT(!server::detail::request_sampling(bad, defaults).ok());
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
  models::SamplingDefaults defaults;
  defaults.top_k = 20;
  defaults.top_p = 0.95f;
  defaults.repetition_penalty = 1.1f;
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
      json{{"top_k",-2}},json{{"top_p",2}},json{{"seed",-1}}})
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
  models::SamplingDefaults defaults;
  defaults.top_k = 20;
  defaults.top_p = 0.95f;
  defaults.repetition_penalty = 1.1f;
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
