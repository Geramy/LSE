// The Jinja subset chat templates use, and the thinking levels read from real
// checkpoints' templates (tests/fixtures/chat_templates, copied verbatim).
// Templates found in the local Hugging Face cache are read too when present.
#include "harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "lse/models/chat_template.hpp"
#include "lse/models/thinking_controls.hpp"

using namespace lse;
using namespace lse::models;
using ojson = nlohmann::ordered_json;

#define LSE_EXPECT_STR(a, b)                                                   \
  do {                                                                         \
    const std::string _sa = (a), _sb = (b);                                   \
    if (_sa != _sb) ::lse::test::fail(__FILE__, __LINE__, "\"" + _sa + "\" vs \"" + _sb + "\""); \
  } while (0)

namespace {

std::string render(std::string_view tmpl, const ojson& ctx = ojson::object()) {
  auto t = ChatTemplate::parse(tmpl);
  if (!t.ok()) return "<parse error: " + t.status().message() + ">";
  auto r = t->render(ctx);
  if (!r.ok()) return "<render error: " + r.status().message() + ">";
  return *r;
}

const char* kFixtures = "tests/fixtures/chat_templates";

std::string read(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

constexpr const char* kXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr const char* kLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

}  // namespace

LSE_TEST(jinja_renders_expressions_filters_and_tests) {
  LSE_EXPECT_STR(render("{{ 'a' ~ (1 + 2) ~ 'b' * 2 }}"), "a3bb");
  LSE_EXPECT_STR(render("{{ x|default('d') }}|{{ y|default('d') }}", {{"y", "v"}}), "d|v");
  LSE_EXPECT_STR(render("{{ '  pad  '|trim }}"), "pad");
  LSE_EXPECT_STR(render("{{ [1, 2, 3]|length }}"), "3");
  LSE_EXPECT_STR(render("{{ {'b': [1, 2], 'a': 'x'}|tojson }}"), "{\"b\": [1, 2], \"a\": \"x\"}");
  LSE_EXPECT_STR(render("{{ 'a' if x is defined else 'b' }}"), "b");
  LSE_EXPECT_STR(render("{{ x is not none }}", {{"x", nullptr}}), "False");
  LSE_EXPECT_STR(render("{{ 'low' not in ('xhigh', 'low') }}"), "False");
  LSE_EXPECT_STR(render("{{ 'a<think>b</think>c'.split('</think>')[0].split('<think>')[-1] }}"), "b");
  LSE_EXPECT_STR(render("{{ 'abcdef'[::-1] }}{{ [1,2,3,4][1:3] }}"), "fedcba[2, 3]");
  LSE_EXPECT_STR(render("{{ 'x'.startswith(('y', 'x')) }} {{ '\\n a\\n'.rstrip('\\n') }}"), "True \n a");
  LSE_EXPECT_STR(render("{% for k, v in d|items %}{{ k }}={{ v }};{% endfor %}", {{"d", {{"a", 1}, {"b", "x"}}}}),
                 "a=1;b=x;");
  LSE_EXPECT_STR(render("{{ none }}{{ true }}{{ 1.0 }}"), "NoneTrue1.0");
  // Truthiness: empty strings, lists and mappings, zero, none and undefined are false.
  LSE_EXPECT_STR(render("{% if '' or [] or {} or 0 or none or nothing %}T{% elif 'x' %}x{% endif %}"), "x");
}

LSE_TEST(jinja_scopes_loops_namespaces_and_macros) {
  LSE_EXPECT_STR(render("{% set ns = namespace(n=0) %}{% for i in range(4) %}{% set ns.n = ns.n + i %}"
                        "{% set leaked = i %}{% endfor %}{{ ns.n }}{{ leaked|default('-') }}"),
                 "6-");
  LSE_EXPECT_STR(render("{% for m in ms %}{{ loop.index0 }}{{ 'F' if loop.first }}{{ 'L' if loop.last }}"
                        "{{ loop.previtem|default('^') }}{% endfor %}",
                        {{"ms", {"a", "b"}}}),
                 "0F^1La");
  LSE_EXPECT_STR(render("{% macro m(a, b='B') %}[{{ a }}{{ b }}]{% endmacro %}{{ m('x') }}{{ m('y', b='z')|trim }}"),
                 "[xB][yz]");
  LSE_EXPECT_STR(render("{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}"
                        "{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}"),
                 "02");
  LSE_EXPECT_STR(render("{% for x in [] %}a{% else %}empty{% endfor %}"), "empty");
  LSE_EXPECT_STR(render("{% set s %}in{{ 1 }}{% endset %}{{ s }}"), "in1");
}

LSE_TEST(jinja_whitespace_follows_trim_and_lstrip_blocks) {
  // transformers renders with trim_blocks and lstrip_blocks on.
  LSE_EXPECT_STR(render("a\n  {% if true %}\n  b\n  {% endif %}\nc"), "a\n  b\nc");
  LSE_EXPECT_STR(render("a  {%- if true -%}  b  {%- endif %}"), "ab");
  LSE_EXPECT_STR(render("{# note #}\nx {{- ' y' }}"), "x y");
}

LSE_TEST(jinja_errors_are_reported_not_guessed) {
  auto raised = ChatTemplate::parse("{{ raise_exception('nope') }}");
  LSE_EXPECT(raised.ok());
  if (raised.ok()) {
    auto r = raised->render(ojson::object());
    LSE_EXPECT(!r.ok());
    LSE_EXPECT(!r.ok() && r.status().message().find("nope") != std::string::npos);
  }
  LSE_EXPECT(!ChatTemplate::parse("{% if x %}unclosed").ok());
  LSE_EXPECT(!ChatTemplate::parse("{% include 'other' %}").ok());
  auto filter = ChatTemplate::parse("{{ x|nosuchfilter }}");
  LSE_EXPECT(filter.ok() && !filter->render(ojson::object()).ok());
}

LSE_TEST(literals_compared_with_a_variable_follow_its_aliases) {
  auto t = ChatTemplate::parse(read(std::filesystem::path(kFixtures) / "qwen3.8" / "chat_template.jinja"));
  LSE_EXPECT(t.ok());
  if (!t.ok()) return;
  const auto lits = t->literals_compared_with("reasoning_effort");
  LSE_EXPECT_EQ(lits.size(), std::size_t{3});
  if (lits.size() == 3) {
    LSE_EXPECT_STR(lits[0], "xhigh");
    LSE_EXPECT_STR(lits[1], "medium");
    LSE_EXPECT_STR(lits[2], "low");
  }
  LSE_EXPECT(t->references("enable_thinking"));
  LSE_EXPECT(!t->references("thinking_budget"));
}

LSE_TEST(qwen38_template_defines_none_xhigh_medium_and_low) {
  auto c = load_thinking_controls((std::filesystem::path(kFixtures) / "qwen3.8").string());
  LSE_EXPECT_OK(c.status());
  if (!c.ok()) return;
  LSE_EXPECT(c->supported());
  LSE_EXPECT(c->has_toggle);
  LSE_EXPECT(c->has_effort);
  LSE_EXPECT(c->template_path.ends_with("chat_template.jinja"));
  LSE_EXPECT_EQ(c->levels.size(), std::size_t{4});
  LSE_EXPECT_STR(c->default_level, "xhigh");
  const ThinkingLevel* none = c->find("none");
  const ThinkingLevel* xhigh = c->find("xhigh");
  const ThinkingLevel* medium = c->find("medium");
  const ThinkingLevel* low = c->find("low");
  LSE_EXPECT(none && xhigh && medium && low);
  LSE_EXPECT(c->find("high") == nullptr);  // not defined: no alias
  if (!(none && xhigh && medium && low)) return;
  LSE_EXPECT(none->enable_thinking == false);
  LSE_EXPECT(!none->reasoning_effort);
  LSE_EXPECT_STR(none->instruction, "");
  LSE_EXPECT_STR(none->generation_prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  LSE_EXPECT(!none->opens_reasoning);
  LSE_EXPECT_STR(xhigh->instruction, kXhigh);
  LSE_EXPECT_STR(low->instruction, kLow);
  LSE_EXPECT_STR(medium->instruction, "");
  LSE_EXPECT(medium->reasoning_effort == std::optional<std::string>("medium"));
  LSE_EXPECT_STR(xhigh->generation_prompt, "<|im_start|>assistant\n<think>\n");
  LSE_EXPECT(xhigh->opens_reasoning && low->opens_reasoning && medium->opens_reasoning);
  LSE_EXPECT(c->thinking_on() == xhigh);

  const auto j = c->to_json();
  LSE_EXPECT(j["supported"].get<bool>());
  LSE_EXPECT_STR(j["toggle"].get<std::string>(), "enable_thinking");
  LSE_EXPECT_STR(j["effort"].get<std::string>(), "reasoning_effort");
  LSE_EXPECT_STR(j["default_level"].get<std::string>(), "xhigh");
  LSE_EXPECT_STR(j["levels"][0]["id"].get<std::string>(), "none");
  LSE_EXPECT(j["levels"][2]["instruction"].is_null());  // medium
}

LSE_TEST(qwen36_template_defines_only_the_toggle) {
  auto c = load_thinking_controls((std::filesystem::path(kFixtures) / "qwen3.6").string());
  LSE_EXPECT_OK(c.status());
  if (!c.ok()) return;
  LSE_EXPECT(c->template_path.ends_with("tokenizer_config.json"));
  LSE_EXPECT(c->has_toggle);
  LSE_EXPECT(!c->has_effort);
  LSE_EXPECT_EQ(c->levels.size(), std::size_t{2});
  LSE_EXPECT_STR(c->default_level, "on");
  const ThinkingLevel* on = c->find("on");
  const ThinkingLevel* none = c->find("none");
  LSE_EXPECT(on && none);
  if (!(on && none)) return;
  LSE_EXPECT_STR(on->instruction, "");
  LSE_EXPECT_STR(on->generation_prompt, "<|im_start|>assistant\n<think>\n");
  LSE_EXPECT(on->opens_reasoning);
  LSE_EXPECT(!none->opens_reasoning);
  LSE_EXPECT(c->find("low") == nullptr && c->find("xhigh") == nullptr);
}

LSE_TEST(a_template_without_controls_reports_none) {
  auto c = load_thinking_controls((std::filesystem::path(kFixtures) / "smollm2").string());
  LSE_EXPECT_OK(c.status());
  if (!c.ok()) return;
  LSE_EXPECT(!c->supported());
  LSE_EXPECT(!c->has_toggle && !c->has_effort);
  LSE_EXPECT(c->levels.empty());
  LSE_EXPECT_STR(c->default_level, "");
  LSE_EXPECT_STR(c->generation_prompt, "<|im_start|>assistant\n");
  const auto j = c->to_json();
  LSE_EXPECT(!j["supported"].get<bool>());
  LSE_EXPECT(j["levels"].empty());
  LSE_EXPECT(j["default_level"].is_null());

  // No template at all: no controls and no source.
  const auto empty = std::filesystem::temp_directory_path() / "lse-no-template";
  std::filesystem::create_directories(empty);
  auto none = load_thinking_controls(empty.string());
  LSE_EXPECT_OK(none.status());
  if (none.ok()) {
    LSE_EXPECT(!none->supported());
    LSE_EXPECT(none->to_json()["source"].is_null());
  }
  std::filesystem::remove_all(empty);
}

LSE_TEST(effort_levels_without_a_toggle_are_read_too) {
  // gpt-oss style: the effort goes into the system header and is never off.
  const char* tmpl =
      "{%- set effort = reasoning_effort|default('medium') %}"
      "{%- if effort not in ['low', 'medium', 'high'] %}{{ raise_exception('bad effort') }}{% endif %}"
      "<|start|>system<|message|>Reasoning: {{ effort }}\n"
      "{%- for m in messages %}{% if m.role == 'system' %}{{ m.content }}{% endif %}{% endfor %}<|end|>"
      "{%- for m in messages %}{% if m.role == 'user' %}<|start|>user<|message|>{{ m.content }}<|end|>{% endif %}{% endfor %}"
      "{%- if add_generation_prompt %}<|start|>assistant{% endif %}";
  auto c = thinking_controls_from_template(tmpl, "inline");
  LSE_EXPECT_OK(c.status());
  if (!c.ok()) return;
  LSE_EXPECT(!c->has_toggle && c->has_effort);
  LSE_EXPECT_EQ(c->levels.size(), std::size_t{3});
  LSE_EXPECT_STR(c->default_level, "medium");
  LSE_EXPECT(c->find("none") == nullptr);
  if (const auto* high = c->find("high")) {
    LSE_EXPECT_STR(high->instruction, "high");
    LSE_EXPECT(!high->enable_thinking.has_value());
  } else {
    LSE_EXPECT(false);
  }
}

// Templates in the local Hugging Face cache, and in any model directories
// listed (colon-separated) in LSE_TEST_TEMPLATE_DIRS, must parse and give
// consistent controls.
LSE_TEST(cached_checkpoint_templates_are_read) {
  std::vector<std::filesystem::path> dirs;
  std::error_code ec;
  if (const char* home = std::getenv("HOME")) {
    const auto hub = std::filesystem::path(home) / ".cache/huggingface/hub";
    if (std::filesystem::exists(hub, ec))
      for (const auto& model : std::filesystem::directory_iterator(hub, ec)) {
        const auto snapshots = model.path() / "snapshots";
        if (!std::filesystem::exists(snapshots, ec)) continue;
        for (const auto& snap : std::filesystem::directory_iterator(snapshots, ec)) dirs.push_back(snap.path());
      }
  }
  if (const char* extra = std::getenv("LSE_TEST_TEMPLATE_DIRS")) {
    std::stringstream list(extra);
    for (std::string d; std::getline(list, d, ':');)
      if (!d.empty()) dirs.emplace_back(d);
  }
  int read_any = 0;
  for (const auto& dir : dirs) {
    auto source = find_chat_template(dir.string());
    LSE_EXPECT_OK(source.status());
    if (!source.ok() || source->text.empty()) continue;
    auto c = thinking_controls_from_template(source->text, source->path);
    LSE_EXPECT_OK(c.status());
    if (c.ok()) {
      if (c->supported()) LSE_EXPECT(c->default_thinking() != nullptr);
      std::fprintf(stderr, "    %s: %s\n", source->path.c_str(), c->to_json().dump().c_str());
    }
    ++read_any;
  }
  if (read_any == 0) LSE_SKIP("no local checkpoint carries a chat template");
}

LSE_TEST_MAIN()
