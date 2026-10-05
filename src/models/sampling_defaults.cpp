#include "lse/models/sampling_defaults.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace lse::models {
namespace {
using json = nlohmann::json;

constexpr const char* kFloatFields[] = {"temperature", "top_p", "min_p", "repetition_penalty",
                                        "presence_penalty"};
constexpr const char* kIntFields[] = {"top_k", "max_new_tokens", "max_length"};

Result<json> object(std::string_view text, const char* source) {
  try {
    auto value = json::parse(text);
    if (!value.is_object())
      return LSE_ERROR(kInvalidArgument, source, " must be a JSON object");
    return value;
  } catch (const json::exception& e) {
    return LSE_ERROR(kInvalidArgument, source, " is not valid JSON: ", e.what());
  }
}

Status merge(const json& source, std::string_view origin, SamplingDefaults& out,
             std::optional<bool>& do_sample, std::string& do_sample_origin) {
  for (const char* key : kFloatFields) {
    const auto at = source.find(key);
    if (at == source.end() || at->is_null()) continue;
    const std::string_view k(key);
    if (!at->is_number())
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' must be numeric");
    const double value = at->get<double>();
    const bool signed_field = k == "presence_penalty";
    if (!std::isfinite(value) || std::fabs(value) > static_cast<double>(std::numeric_limits<float>::max()) ||
        (!signed_field && value < 0.0) || ((k == "top_p" || k == "min_p") && value > 1.0) ||
        (k == "repetition_penalty" && value == 0.0))
      return LSE_ERROR(kInvalidArgument, "invalid sampling field '", key, "'");
    const float converted = static_cast<float>(value);
    if (value != 0.0 && converted == 0.0f)
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' underflows float32");
    if (k == "temperature") out.temperature = converted;
    else if (k == "top_p") out.top_p = converted;
    else if (k == "min_p") out.min_p = converted;
    else if (k == "repetition_penalty") out.repetition_penalty = converted;
    else out.presence_penalty = converted;
    out.sources[key] = std::string(origin);
  }
  for (const char* key : kIntFields) {
    const auto at = source.find(key);
    if (at == source.end() || at->is_null()) continue;
    const std::string_view k(key);
    const bool limit = k != "top_k";
    if (!at->is_number_integer() ||
        (at->is_number_unsigned() && at->get<std::uint64_t>() >
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())))
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' must be a ",
                       limit ? "positive" : "nonnegative", " int32");
    const auto value = at->get<std::int64_t>();
    if (value < (limit ? 1 : 0) || value > std::numeric_limits<std::int32_t>::max())
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' must be a ",
                       limit ? "positive" : "nonnegative", " int32");
    const auto v = static_cast<std::int32_t>(value);
    if (k == "top_k") out.top_k = v;
    else if (k == "max_new_tokens") out.max_new_tokens = v;
    else out.max_length = v;
    out.sources[key] = std::string(origin);
  }
  if (const auto at = source.find("do_sample"); at != source.end() && !at->is_null()) {
    if (!at->is_boolean())
      return LSE_ERROR(kInvalidArgument, "sampling do_sample must be boolean");
    do_sample = at->get<bool>();
    do_sample_origin = std::string(origin);
  }
  return OkStatus();
}

Result<std::string> read_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) return LSE_ERROR(kIoError, "cannot read sampling config '", path.string(), "'");
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) return LSE_ERROR(kIoError, "cannot read sampling config '", path.string(), "'");
  return text.str();
}
}  // namespace

std::string SamplingDefaults::source(std::string_view field) const {
  const auto it = sources.find(std::string(field));
  return it == sources.end() ? std::string(kFromLseDefault) : it->second;
}

nlohmann::json SamplingDefaults::to_json() const {
  json src = json::object();
  for (const char* key : {"temperature", "top_k", "top_p", "min_p", "repetition_penalty",
                          "presence_penalty", "max_new_tokens", "max_length"})
    src[key] = source(key);
  return json{{"temperature", temperature},
              {"top_k", top_k},
              {"top_p", top_p},
              {"min_p", min_p},
              {"repetition_penalty", repetition_penalty},
              {"presence_penalty", presence_penalty},
              {"max_new_tokens", max_new_tokens ? json(*max_new_tokens) : json(nullptr)},
              {"max_length", max_length ? json(*max_length) : json(nullptr)},
              {"sources", std::move(src)}};
}

Result<SamplingDefaults> sampling_defaults_from_json(
    std::string_view model_config, std::string_view generation_config) {
  LSE_ASSIGN_OR(json root, object(model_config, "model config"));
  SamplingDefaults defaults;
  std::optional<bool> do_sample;
  std::string do_sample_origin;
  LSE_RETURN_IF_ERROR(merge(root, kFromModelConfig, defaults, do_sample, do_sample_origin));
  if (const auto at = root.find("generation_config"); at != root.end() && !at->is_null()) {
    if (!at->is_object())
      return LSE_ERROR(kInvalidArgument, "embedded generation_config must be an object");
    LSE_RETURN_IF_ERROR(merge(*at, kFromModelConfig, defaults, do_sample, do_sample_origin));
  }
  if (!generation_config.empty()) {
    LSE_ASSIGN_OR(json generation, object(generation_config, "generation_config.json"));
    LSE_RETURN_IF_ERROR(merge(generation, kFromGenerationConfig, defaults, do_sample, do_sample_origin));
  }
  if (do_sample == false) {
    defaults.temperature = 0.0f;
    defaults.sources["temperature"] = do_sample_origin;
  }
  return defaults;
}

Result<SamplingDefaults> load_sampling_defaults(const std::string& config_path) {
  LSE_ASSIGN_OR(std::string config, read_file(config_path));
  const auto generation_path = std::filesystem::path(config_path).parent_path() / "generation_config.json";
  std::error_code ec;
  const bool present = std::filesystem::exists(generation_path, ec);
  if (ec) return LSE_ERROR(kIoError, "cannot inspect sampling config '", generation_path.string(), "': ", ec.message());
  if (!present) return sampling_defaults_from_json(config);
  LSE_ASSIGN_OR(std::string generation, read_file(generation_path));
  if (generation.empty()) return LSE_ERROR(kInvalidArgument, "generation_config.json is empty");
  return sampling_defaults_from_json(config, generation);
}
}  // namespace lse::models
