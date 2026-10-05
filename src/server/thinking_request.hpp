// A chat request's thinking fields, resolved against the levels the model's
// chat template defines (lse/models/thinking_controls.hpp).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_protocol.hpp"
#include "lse/models/thinking_controls.hpp"

namespace lse::server::detail {

// The fields a client may send, in any of their spellings.
//   switch: enable_thinking (bool), thinking (bool, or {"type": "enabled" |
//           "disabled"}), chat_template_kwargs.enable_thinking (bool)
//   level:  reasoning_effort, thinking_level, chat_template_kwargs.reasoning_effort,
//           reasoning.effort (strings; null means not sent)
// Spellings of one field that disagree are a conflict.
struct ThinkingRequest {
  std::optional<bool> enable;
  std::optional<std::string> level;
};

// A refusal: HTTP 400 with error.code `code` and error.param `param`.
struct ThinkingError {
  std::string code;  // invalid_thinking | conflicting_thinking |
                     // unsupported_reasoning_effort | thinking_unsupported
  std::string param;
  std::string message;
};

struct ThinkingChoice {
  // The level used; nullptr when the model has no thinking controls.
  const models::ThinkingLevel* level = nullptr;
  ChatFraming framing;
  std::optional<ThinkingError> error;
};

// ChatML's assistant opening, for a model without a chat template.
inline constexpr const char* kChatMlGenerationPrompt = "<|im_start|>assistant\n";

inline std::optional<ThinkingError> read_thinking_request(const nlohmann::json& body,
                                                          ThinkingRequest& out) {
  const auto set_enable = [&](bool v, const char* param) -> std::optional<ThinkingError> {
    if (out.enable && *out.enable != v)
      return ThinkingError{"conflicting_thinking", param,
                           std::string("'") + param + "' disagrees with another thinking switch in the request"};
    out.enable = v;
    return std::nullopt;
  };
  const auto set_level = [&](const nlohmann::json& v, const char* param) -> std::optional<ThinkingError> {
    if (v.is_null()) return std::nullopt;
    if (!v.is_string() || v.get<std::string>().empty())
      return ThinkingError{"invalid_thinking", param, std::string("'") + param + "' must be a nonempty string"};
    if (out.level && *out.level != v.get<std::string>())
      return ThinkingError{"conflicting_thinking", param,
                           std::string("'") + param + "' disagrees with another reasoning level in the request"};
    out.level = v.get<std::string>();
    return std::nullopt;
  };
  const auto boolean = [&](const nlohmann::json& v, const char* param) -> std::optional<ThinkingError> {
    if (v.is_null()) return std::nullopt;
    if (!v.is_boolean())
      return ThinkingError{"invalid_thinking", param, std::string("'") + param + "' must be a boolean"};
    return set_enable(v.get<bool>(), param);
  };
  if (!body.is_object()) return std::nullopt;
  if (const auto at = body.find("enable_thinking"); at != body.end())
    if (auto e = boolean(*at, "enable_thinking")) return e;
  if (const auto at = body.find("thinking"); at != body.end() && !at->is_null()) {
    if (at->is_boolean()) {
      if (auto e = set_enable(at->get<bool>(), "thinking")) return e;
    } else if (at->is_object()) {
      const auto type = at->find("type");
      if (type == at->end() || !type->is_string() ||
          (type->get<std::string>() != "enabled" && type->get<std::string>() != "disabled"))
        return ThinkingError{"invalid_thinking", "thinking",
                             "'thinking.type' must be \"enabled\" or \"disabled\""};
      if (auto e = set_enable(type->get<std::string>() == "enabled", "thinking")) return e;
    } else {
      return ThinkingError{"invalid_thinking", "thinking", "'thinking' must be a boolean or an object"};
    }
  }
  if (const auto at = body.find("chat_template_kwargs"); at != body.end() && !at->is_null()) {
    if (!at->is_object())
      return ThinkingError{"invalid_thinking", "chat_template_kwargs", "'chat_template_kwargs' must be an object"};
    if (const auto e = at->find("enable_thinking"); e != at->end())
      if (auto err = boolean(*e, "chat_template_kwargs.enable_thinking")) return err;
    if (const auto e = at->find("reasoning_effort"); e != at->end())
      if (auto err = set_level(*e, "chat_template_kwargs.reasoning_effort")) return err;
  }
  for (const char* field : {"reasoning_effort", "thinking_level"})
    if (const auto at = body.find(field); at != body.end())
      if (auto e = set_level(*at, field)) return e;
  if (const auto at = body.find("reasoning"); at != body.end() && !at->is_null()) {
    if (!at->is_object())
      return ThinkingError{"invalid_thinking", "reasoning", "'reasoning' must be an object"};
    if (const auto e = at->find("effort"); e != at->end())
      if (auto err = set_level(*e, "reasoning.effort")) return err;
  }
  return std::nullopt;
}

inline std::string level_list(const models::ThinkingControls& c) {
  std::string out;
  for (const auto& l : c.levels) {
    if (!out.empty()) out += ", ";
    out += l.id;
  }
  return out.empty() ? "none" : out;
}

inline ThinkingChoice choose_thinking(const models::ThinkingControls& controls,
                                      const ThinkingRequest& request) {
  ThinkingChoice choice;
  const auto refuse = [&](std::string code, std::string param, std::string message) {
    choice.error = ThinkingError{std::move(code), std::move(param), std::move(message)};
    return choice;
  };
  const bool controlled = controls.supported();
  const models::ThinkingLevel* level = nullptr;
  if (request.level) {
    const std::string& want = *request.level;
    level = controls.find(want);
    if (level == nullptr) {
      // Asking a model that cannot think not to think is already satisfied.
      if (want == "none" && !controlled) {
        if (request.enable == true)
          return refuse("thinking_unsupported", "reasoning_effort",
                        "this model's chat template defines no thinking controls");
      } else if (!controlled) {
        return refuse("thinking_unsupported", "reasoning_effort",
                      "this model's chat template defines no thinking controls, so reasoning_effort '" +
                          want + "' cannot be applied");
      } else {
        return refuse("unsupported_reasoning_effort", "reasoning_effort",
                      "reasoning_effort '" + want + "' is not defined by this model's chat template; levels: " +
                          level_list(controls));
      }
    } else if (request.enable && *request.enable != level->enable_thinking.value_or(true)) {
      return refuse("conflicting_thinking", "reasoning_effort",
                    "reasoning_effort '" + want + "' contradicts the request's thinking switch");
    }
  } else if (request.enable) {
    if (*request.enable) {
      level = controlled ? controls.thinking_on() : nullptr;
      if (level == nullptr)
        return refuse("thinking_unsupported", "enable_thinking",
                      "this model's chat template defines no thinking controls");
    } else if (controlled) {
      level = controls.find("none");
      if (level == nullptr)
        return refuse("thinking_unsupported", "enable_thinking",
                      "this model's chat template cannot switch thinking off; levels: " + level_list(controls));
    }
  } else {
    level = controls.default_thinking();
  }
  choice.level = level;
  if (level != nullptr) {
    choice.framing.instruction = level->instruction;
    choice.framing.generation_prompt = level->generation_prompt;
    choice.framing.reasoning = level->opens_reasoning;
  } else {
    choice.framing.generation_prompt =
        controls.generation_prompt.empty() ? kChatMlGenerationPrompt : controls.generation_prompt;
    choice.framing.reasoning = false;
  }
  return choice;
}

}  // namespace lse::server::detail
