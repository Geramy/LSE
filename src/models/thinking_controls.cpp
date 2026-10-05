#include "lse/models/thinking_controls.hpp"

#include <algorithm>

#include "lse/models/chat_template.hpp"

namespace lse::models {
namespace {

using ojson = nlohmann::ordered_json;

// Probe conversation. The system text starts with a character no template
// instruction starts with, so where a rendering first differs from the
// baseline is where the template's own text begins.
constexpr std::string_view kProbeSystem = "\xE2\x9F\xA6lse-probe-system\xE2\x9F\xA7";
constexpr std::string_view kProbeUser = "\xE2\x9F\xA6lse-probe-user\xE2\x9F\xA7";

struct Setting {
  std::optional<bool> enable_thinking;
  std::optional<std::string> reasoning_effort;
};

ojson context(const Setting& s, bool generation_prompt) {
  ojson c{{"messages", ojson::array({{{"role", "system"}, {"content", kProbeSystem}},
                                     {{"role", "user"}, {"content", kProbeUser}}})},
          {"add_generation_prompt", generation_prompt},
          {"bos_token", ""},
          {"eos_token", ""}};
  if (s.enable_thinking) c[std::string(kThinkingToggleVariable)] = *s.enable_thinking;
  if (s.reasoning_effort) c[std::string(kReasoningEffortVariable)] = *s.reasoning_effort;
  return c;
}

std::string trim(std::string_view s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string_view::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return std::string(s.substr(b, e - b + 1));
}

std::size_t common_prefix(std::string_view a, std::string_view b) {
  std::size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}

// The text `rendered` carries before the probe system prompt that `baseline`
// does not.
Result<std::string> instruction(const std::string& rendered, const std::string& baseline) {
  const std::size_t at = rendered.find(kProbeSystem);
  if (at == std::string::npos)
    return LSE_ERROR(kInvalidArgument, "the chat template drops the system message");
  const std::size_t from = std::min(common_prefix(rendered, baseline), at);
  return trim(std::string_view(rendered).substr(from, at - from));
}

struct Probe {
  const ChatTemplate& tmpl;
  Result<std::string> render(const Setting& s, bool generation_prompt = true) const {
    return tmpl.render(context(s, generation_prompt));
  }
  // What add_generation_prompt appends.
  Result<std::string> generation_prompt(const Setting& s) const {
    LSE_ASSIGN_OR(std::string with, render(s, true));
    LSE_ASSIGN_OR(std::string without, render(s, false));
    if (!with.starts_with(without))
      return LSE_ERROR(kInvalidArgument,
                       "the chat template's generation prompt is not an append to the conversation");
    return with.substr(without.size());
  }
};

bool opens(const std::string& prompt) {
  const auto open = prompt.rfind("<think>");
  return open != std::string::npos && prompt.find("</think>", open) == std::string::npos;
}

}  // namespace

const ThinkingLevel* ThinkingControls::find(std::string_view id) const noexcept {
  for (const ThinkingLevel& l : levels)
    if (l.id == id) return &l;
  return nullptr;
}

const ThinkingLevel* ThinkingControls::thinking_on() const noexcept {
  const ThinkingLevel* d = default_thinking();
  if (d != nullptr && d->enable_thinking.value_or(true)) return d;
  for (const ThinkingLevel& l : levels)
    if (l.enable_thinking.value_or(true)) return &l;
  return nullptr;
}

nlohmann::json ThinkingControls::to_json() const {
  nlohmann::json list = nlohmann::json::array();
  for (const ThinkingLevel& l : levels) {
    list.push_back({{"id", l.id},
                    {"enable_thinking", l.enable_thinking ? nlohmann::json(*l.enable_thinking) : nlohmann::json(nullptr)},
                    {"reasoning_effort", l.reasoning_effort ? nlohmann::json(*l.reasoning_effort) : nlohmann::json(nullptr)},
                    {"instruction", l.instruction.empty() ? nlohmann::json(nullptr) : nlohmann::json(l.instruction)},
                    {"opens_reasoning", l.opens_reasoning},
                    {"default", l.id == default_level}});
  }
  return nlohmann::json{
      {"supported", supported()},
      {"source", template_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(template_path)},
      {"toggle", has_toggle ? nlohmann::json(kThinkingToggleVariable) : nlohmann::json(nullptr)},
      {"effort", has_effort ? nlohmann::json(kReasoningEffortVariable) : nlohmann::json(nullptr)},
      {"default_level", default_level.empty() ? nlohmann::json(nullptr) : nlohmann::json(default_level)},
      {"levels", std::move(list)}};
}

Result<ThinkingControls> thinking_controls_from_template(std::string_view text, std::string path) {
  LSE_ASSIGN_OR(const ChatTemplate tmpl, ChatTemplate::parse(text));
  const Probe probe{tmpl};
  ThinkingControls out;
  out.template_path = std::move(path);

  LSE_ASSIGN_OR(const std::string plain, probe.render({}));
  if (plain.find(kProbeUser) == std::string::npos)
    return LSE_ERROR(kInvalidArgument, "the chat template drops the user message");

  if (tmpl.references(kThinkingToggleVariable)) {
    LSE_ASSIGN_OR(const std::string on, probe.render({true, std::nullopt}));
    LSE_ASSIGN_OR(const std::string off, probe.render({false, std::nullopt}));
    out.has_toggle = on != off;
  }
  const std::optional<bool> thinking = out.has_toggle ? std::optional<bool>(true) : std::nullopt;

  std::vector<std::string> efforts;
  if (tmpl.references(kReasoningEffortVariable)) {
    for (const std::string& candidate : tmpl.literals_compared_with(kReasoningEffortVariable)) {
      if (probe.render({thinking, candidate}).ok()) efforts.push_back(candidate);
    }
    // A variable whose values all render alike selects nothing.
    if (efforts.size() >= 2 || (efforts.size() == 1 && out.has_toggle)) {
      bool differs = false;
      LSE_ASSIGN_OR(const std::string first, probe.render({thinking, efforts.front()}));
      for (std::size_t i = 1; i < efforts.size() && !differs; ++i) {
        LSE_ASSIGN_OR(const std::string other, probe.render({thinking, efforts[i]}));
        differs = other != first;
      }
      out.has_effort = differs || (efforts.size() == 1 && out.has_toggle);
    }
    if (!out.has_effort) efforts.clear();
  }

  if (!out.has_toggle && !out.has_effort) {
    LSE_ASSIGN_OR(out.generation_prompt, probe.generation_prompt({}));
    return out;
  }

  // The baseline instructions are measured against: thinking off, or what
  // every effort level's rendering shares.
  std::string baseline;
  if (out.has_toggle) {
    LSE_ASSIGN_OR(baseline, probe.render({false, std::nullopt}));
  } else {
    LSE_ASSIGN_OR(baseline, probe.render({std::nullopt, efforts.front()}));
    std::size_t shared = baseline.find(kProbeSystem);
    for (const std::string& e : efforts) {
      LSE_ASSIGN_OR(const std::string r, probe.render({std::nullopt, e}));
      shared = std::min(shared, common_prefix(baseline, r));
    }
    baseline.resize(shared);
  }

  const auto add = [&](std::string id, Setting s) -> Status {
    ThinkingLevel level;
    level.id = std::move(id);
    level.enable_thinking = s.enable_thinking;
    level.reasoning_effort = s.reasoning_effort;
    LSE_ASSIGN_OR(const std::string rendered, probe.render(s));
    if (!(out.has_toggle && s.enable_thinking == false)) {
      LSE_ASSIGN_OR(level.instruction, instruction(rendered, baseline));
    }
    LSE_ASSIGN_OR(level.generation_prompt, probe.generation_prompt(s));
    level.opens_reasoning = opens(level.generation_prompt);
    out.levels.push_back(std::move(level));
    return OkStatus();
  };
  if (out.has_toggle) LSE_RETURN_IF_ERROR(add("none", {false, std::nullopt}));
  if (out.has_effort) {
    for (const std::string& e : efforts) {
      if (e == "none" && out.has_toggle) continue;  // the toggle's level already
      LSE_RETURN_IF_ERROR(add(e, {thinking, e}));
    }
  } else {
    LSE_RETURN_IF_ERROR(add("on", {true, std::nullopt}));
  }

  // The template's default: the level whose rendering matches the rendering
  // with nothing set.
  for (const ThinkingLevel& l : out.levels) {
    LSE_ASSIGN_OR(const std::string r, probe.render({l.enable_thinking, l.reasoning_effort}));
    if (r == plain) {
      out.default_level = l.id;
      break;
    }
  }
  if (out.default_level.empty())
    return LSE_ERROR(kInvalidArgument,
                     "the chat template's default rendering matches none of its thinking levels");
  return out;
}

Result<ThinkingControls> load_thinking_controls(const std::string& model_dir) {
  LSE_ASSIGN_OR(ChatTemplateSource source, find_chat_template(model_dir));
  if (source.text.empty()) return ThinkingControls{};
  auto controls = thinking_controls_from_template(source.text, source.path);
  if (!controls.ok())
    return LSE_ERROR(kInvalidArgument, "reading thinking controls from '", source.path, "': ",
                     controls.status().message());
  return controls;
}

}  // namespace lse::models
