// A checkpoint's chat template (Jinja), evaluated.
//
// Hugging Face checkpoints describe their conversation format as a Jinja
// template, in chat_template.jinja or tokenizer_config.json's "chat_template".
// This is a Jinja interpreter for the subset those templates use: text,
// {{ }} output, {% if/elif/else %}, {% for %} with loop variables and
// loop controls, {% set %} (including namespace attributes and block sets),
// {% macro %}, comments, whitespace control, the filters, tests and string
// and mapping methods chat templates call, raise_exception, namespace(),
// range() and strftime_now(). It renders the way transformers'
// apply_chat_template does: trim_blocks and lstrip_blocks on, tojson with
// Python's separators.
//
// The engine uses it to read what a template defines, by rendering it with
// probe conversations (see thinking_controls.hpp). A construct outside the
// subset is an error naming it, never a guess.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "lse/core/status.hpp"

namespace lse::models {

namespace jinja {
struct Program;
}

class ChatTemplate {
 public:
  // Parses `source`. Syntax errors name the line.
  static Result<ChatTemplate> parse(std::string_view source);

  // Renders with `context` as the template's variables (messages, tools,
  // add_generation_prompt, enable_thinking, ...). An absent key is Jinja's
  // undefined. raise_exception() in the template is an error carrying its
  // message.
  [[nodiscard]] Result<std::string> render(const nlohmann::ordered_json& context) const;

  // Whether the template reads a variable named `name` anywhere.
  [[nodiscard]] bool references(std::string_view name) const;

  // String literals the template compares `name` against (==, !=, in, not in)
  // or gives it as a default (`name|default('x')`), following names assigned
  // from expressions that read it. In order of first appearance.
  [[nodiscard]] std::vector<std::string> literals_compared_with(std::string_view name) const;

 private:
  std::shared_ptr<const jinja::Program> program_;
};

// The template text a model directory carries, and the file it came from:
// chat_template.jinja, else tokenizer_config.json's "chat_template" (a string,
// or a list of named templates of which "default" is taken), else
// chat_template.json's "chat_template". An empty text means the directory has
// none. Malformed files are errors.
struct ChatTemplateSource {
  std::string text;
  std::string path;
};
Result<ChatTemplateSource> find_chat_template(const std::string& model_dir);

}  // namespace lse::models
