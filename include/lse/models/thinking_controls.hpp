// The thinking levels a model's own chat template defines.
//
// Nothing here is written per model. The template is rendered with probe
// conversations and the results compared:
//
//   - The toggle is the template variable `enable_thinking` (the Hugging Face
//     chat-template keyword for it), when setting it changes the rendering.
//   - Effort levels are the values of the template variable `reasoning_effort`
//     (the Hugging Face keyword for it) that the template compares it with and
//     renders without raising.
//   - A level's instruction is the text the template inserts ahead of the
//     system prompt for it, relative to thinking switched off (or, without a
//     toggle, to what every level shares).
//   - A level's generation prompt is what add_generation_prompt appends for it,
//     e.g. "<|im_start|>assistant\n<think>\n".
//   - The default level is the one whose rendering matches the template's
//     rendering with neither variable set.
//
// Levels, by id:
//   "none"   enable_thinking=false (only when the template has the toggle)
//   <effort> each reasoning_effort value the template accepts, thinking on
//   "on"     thinking on, for a template with the toggle and no effort levels
// A template with neither variable has no levels: the model is served without
// thinking controls, and says so.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/core/status.hpp"

namespace lse::models {

inline constexpr std::string_view kThinkingToggleVariable = "enable_thinking";
inline constexpr std::string_view kReasoningEffortVariable = "reasoning_effort";

struct ThinkingLevel {
  std::string id;
  // What the level passes to the template's variables; nullopt where the
  // template has no such variable.
  std::optional<bool> enable_thinking;
  std::optional<std::string> reasoning_effort;
  // The template's instruction for this level ("" when it inserts none).
  std::string instruction;
  // The assistant turn's opening for this level.
  std::string generation_prompt;
  // The generation prompt leaves a <think> block open: the reply starts as
  // reasoning.
  bool opens_reasoning = false;
};

struct ThinkingControls {
  // The file the template came from; empty when the model has no template.
  std::string template_path;
  bool has_toggle = false;
  bool has_effort = false;
  std::vector<ThinkingLevel> levels;
  // The template's own default: the level used when a request asks for none.
  // Empty when the model has no levels.
  std::string default_level;
  // The assistant turn's opening when the model has no levels ("" without a
  // template: ChatML's opening applies).
  std::string generation_prompt;

  [[nodiscard]] bool supported() const noexcept { return !levels.empty(); }
  [[nodiscard]] const ThinkingLevel* find(std::string_view id) const noexcept;
  [[nodiscard]] const ThinkingLevel* default_thinking() const noexcept { return find(default_level); }
  // The level a request switching thinking on without naming an effort gets:
  // the default when it thinks, else the first level that does.
  [[nodiscard]] const ThinkingLevel* thinking_on() const noexcept;
  // The "thinking" object /v1/models and model info report.
  [[nodiscard]] nlohmann::json to_json() const;
};

// Reads the controls from a template's text. `path` is reported as the source.
Result<ThinkingControls> thinking_controls_from_template(std::string_view text, std::string path);

// Finds the model directory's template (see find_chat_template) and reads it.
// A directory without one has no controls.
Result<ThinkingControls> load_thinking_controls(const std::string& model_dir);

}  // namespace lse::models
