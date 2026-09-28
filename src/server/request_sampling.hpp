#pragma once

#include <cmath>
#include <limits>
#include <string_view>

#include <nlohmann/json.hpp>

#include "lse/models/sampling_defaults.hpp"
#include "lse/runtime/sampler.hpp"

namespace lse::server::detail {

inline Result<runtime::SamplingParams> request_sampling(
    const nlohmann::json& body, const models::SamplingDefaults& defaults) {
  runtime::SamplingParams out;
  out.temperature = defaults.temperature;
  out.top_k = defaults.top_k;
  out.top_p = defaults.top_p;
  out.repetition_penalty = defaults.repetition_penalty;
  for (const char* key : {"temperature", "top_p", "frequency_penalty", "repetition_penalty"}) {
    const auto at = body.find(key);
    if (at == body.end() || at->is_null()) continue;
    if (!at->is_number())
      return LSE_ERROR(kInvalidArgument, "request sampling field '", key, "' must be numeric");
    const double value = at->get<double>();
    if (!std::isfinite(value) || std::fabs(value) > static_cast<double>(std::numeric_limits<float>::max()) ||
        (std::string_view(key) == "top_p" && (value < 0.0 || value > 1.0)) ||
        (std::string_view(key) == "repetition_penalty" && value <= 0.0))
      return LSE_ERROR(kInvalidArgument, "invalid request sampling field '", key, "'");
    const float converted = static_cast<float>(value);
    if (value != 0.0 && converted == 0.0f)
      return LSE_ERROR(kInvalidArgument, "request sampling field '", key, "' underflows float32");
    if (std::string_view(key) == "temperature") out.temperature = converted;
    else if (std::string_view(key) == "top_p") out.top_p = converted;
    else if (std::string_view(key) == "repetition_penalty") out.repetition_penalty = converted;
    else out.repetition_penalty = converted > 0.0f ? 1.0f + converted : 1.0f;
  }
  if (const auto at = body.find("top_k"); at != body.end() && !at->is_null()) {
    if (!at->is_number_integer() ||
        (at->is_number_unsigned() && at->get<std::uint64_t>() >
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())))
      return LSE_ERROR(kInvalidArgument, "request top_k must be a nonnegative int32");
    const auto value = at->get<std::int64_t>();
    if (value < 0 || value > std::numeric_limits<std::int32_t>::max())
      return LSE_ERROR(kInvalidArgument, "request top_k must be a nonnegative int32");
    out.top_k = static_cast<std::int32_t>(value);
  }
  if (const auto at = body.find("seed"); at != body.end() && !at->is_null()) {
    if (!at->is_number_integer() ||
        (!at->is_number_unsigned() && at->get<std::int64_t>() < 0))
      return LSE_ERROR(kInvalidArgument, "request seed must be a nonnegative integer");
    out.seed = at->get<std::uint64_t>();
  }
  return out;
}
}  // namespace lse::server::detail
