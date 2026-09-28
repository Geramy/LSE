#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "lse/core/status.hpp"

namespace lse::models {

struct SamplingDefaults {
  float temperature = 1.0f;
  std::int32_t top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
};

// Merge supported sampling fields; null fields inherit the lower-priority value.
Result<SamplingDefaults> sampling_defaults_from_json(
    std::string_view model_config, std::string_view generation_config = {});
Result<SamplingDefaults> load_sampling_defaults(const std::string& config_path);

}  // namespace lse::models
