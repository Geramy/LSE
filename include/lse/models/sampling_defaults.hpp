// The generation defaults a checkpoint declares: sampling settings and any
// output limit, read from its own files. Nothing here is chosen per model.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "lse/core/status.hpp"

namespace lse::models {

// Where a default came from.
inline constexpr std::string_view kFromGenerationConfig = "generation_config.json";
inline constexpr std::string_view kFromModelConfig = "config.json";
inline constexpr std::string_view kFromLseDefault = "lse_default";
inline constexpr std::string_view kFromServerOption = "server_option";

struct SamplingDefaults {
  // LSE's neutral defaults, used for a field no file names: plain sampling
  // from the model's distribution at temperature 1, no filters, no penalties.
  float temperature = 1.0f;
  std::int32_t top_k = 0;            // 0: off
  float top_p = 1.0f;                // 1: off
  float min_p = 0.0f;                // 0: off
  float repetition_penalty = 1.0f;   // 1: off
  float presence_penalty = 0.0f;     // 0: off
  // Output limits, only when the model's files define them; never an LSE
  // default. max_new_tokens caps generated tokens; max_length caps prompt plus
  // generated tokens (Hugging Face's meaning).
  std::optional<std::int32_t> max_new_tokens;
  std::optional<std::int32_t> max_length;
  // Field name -> one of the kFrom* sources above.
  std::map<std::string, std::string> sources;

  [[nodiscard]] std::string source(std::string_view field) const;
  // The "generation_defaults" object /v1/models and model info report.
  [[nodiscard]] nlohmann::json to_json() const;
};

// Merges, highest priority first: `generation_config` (generation_config.json),
// then the model config's embedded "generation_config" object, then its
// top-level fields. Fields read: temperature, top_k, top_p, min_p,
// repetition_penalty, presence_penalty, do_sample (false makes the default
// greedy), max_new_tokens, max_length. Null fields inherit; malformed values
// are errors.
Result<SamplingDefaults> sampling_defaults_from_json(
    std::string_view model_config, std::string_view generation_config = {});
// The same, with generation_config.json read from beside `config_path`.
Result<SamplingDefaults> load_sampling_defaults(const std::string& config_path);

}  // namespace lse::models
