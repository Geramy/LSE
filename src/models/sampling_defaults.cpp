#include "lse/models/sampling_defaults.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

#include <nlohmann/json.hpp>

namespace lse::models {
namespace {
using json = nlohmann::json;
struct FamilyDefaults {
  std::string_view type;
  SamplingDefaults values;
};
constexpr std::array kFamilyDefaults{
    FamilyDefaults{"qwen3_5", {1.0f, 20, 0.95f, 1.0f}},
    FamilyDefaults{"qwen3_5_text", {1.0f, 20, 0.95f, 1.0f}},
    FamilyDefaults{"qwen3_5_moe", {1.0f, 20, 0.95f, 1.0f}},
    FamilyDefaults{"qwen3_5_moe_text", {1.0f, 20, 0.95f, 1.0f}}};

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

Status merge(const json& source, SamplingDefaults& out,
             std::optional<bool>& do_sample) {
  for (const char* key : {"temperature", "top_p", "repetition_penalty"}) {
    const auto at = source.find(key);
    if (at == source.end() || at->is_null()) continue;
    if (!at->is_number())
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' must be numeric");
    const double value = at->get<double>();
    if (!std::isfinite(value) || value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < 0.0 || (std::string_view(key) == "top_p" && value > 1.0) ||
        (std::string_view(key) == "repetition_penalty" && value == 0.0))
      return LSE_ERROR(kInvalidArgument, "invalid sampling field '", key, "'");
    const float converted = static_cast<float>(value);
    if (value != 0.0 && converted == 0.0f)
      return LSE_ERROR(kInvalidArgument, "sampling field '", key, "' underflows float32");
    if (std::string_view(key) == "temperature") out.temperature = converted;
    else if (std::string_view(key) == "top_p") out.top_p = converted;
    else out.repetition_penalty = converted;
  }
  if (const auto at = source.find("top_k"); at != source.end() && !at->is_null()) {
    if (!at->is_number_integer() ||
        (at->is_number_unsigned() && at->get<std::uint64_t>() >
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())))
      return LSE_ERROR(kInvalidArgument, "sampling top_k must be a nonnegative int32");
    const auto value = at->get<std::int64_t>();
    if (value < 0 || value > std::numeric_limits<std::int32_t>::max())
      return LSE_ERROR(kInvalidArgument, "sampling top_k must be a nonnegative int32");
    out.top_k = static_cast<std::int32_t>(value);
  }
  if (const auto at = source.find("do_sample"); at != source.end() && !at->is_null()) {
    if (!at->is_boolean())
      return LSE_ERROR(kInvalidArgument, "sampling do_sample must be boolean");
    do_sample = at->get<bool>();
  }
  return OkStatus();
}

std::optional<SamplingDefaults> family(const json& config) {
  const auto at = config.find("model_type");
  if (at == config.end() || !at->is_string()) return std::nullopt;
  const auto type = at->get<std::string>();
  for (const auto& row : kFamilyDefaults)
    if (row.type == type) return row.values;
  return std::nullopt;
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

Result<SamplingDefaults> sampling_defaults_from_json(
    std::string_view model_config, std::string_view generation_config) {
  LSE_ASSIGN_OR(json root, object(model_config, "model config"));
  SamplingDefaults defaults;
  if (auto values = family(root)) defaults = *values;
  else if (const auto at = root.find("text_config"); at != root.end() && at->is_object())
    if (auto nested_values = family(*at)) defaults = *nested_values;
  std::optional<bool> do_sample;
  LSE_RETURN_IF_ERROR(merge(root, defaults, do_sample));
  if (const auto at = root.find("generation_config"); at != root.end() && !at->is_null()) {
    if (!at->is_object())
      return LSE_ERROR(kInvalidArgument, "embedded generation_config must be an object");
    LSE_RETURN_IF_ERROR(merge(*at, defaults, do_sample));
  }
  if (!generation_config.empty()) {
    LSE_ASSIGN_OR(json generation, object(generation_config, "generation_config.json"));
    LSE_RETURN_IF_ERROR(merge(generation, defaults, do_sample));
  }
  if (do_sample == false) defaults.temperature = 0.0f;
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
