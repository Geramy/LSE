#include "harness.hpp"
#include "chat_protocol.hpp"
#include "thinking_request.hpp"
#include "lse/models/thinking_controls.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "lse/models/chat_template.hpp"
using namespace lse::server::detail;
using json = ChatJson;
namespace {
// The framing the Qwen3.8 checkpoint's own chat template defines for a level.
const lse::models::ThinkingControls& qwen38() {
  static const lse::models::ThinkingControls controls =
      lse::models::load_thinking_controls("tests/fixtures/chat_templates/qwen3.8").release();
  return controls;
}
ChatFraming framing(const std::string& level) {
  ThinkingRequest r;
  r.level = level;
  const auto choice = choose_thinking(qwen38(), r);
  LSE_EXPECT(!choice.error);
  return choice.framing;
}
json request() {
  return json::parse(R"({"messages":[{"role":"user","content":"Look up a value"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"},"count":{"type":"integer"},"options":{"type":"object"}}}}}]})");
}
const std::string call = "<tool_call>\n<function=lookup>\n<parameter=key>\n007\n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=options>\n{\"enabled\":true,\"values\":[1,null]}\n</parameter>\n</function>\n</tool_call>";
}
LSE_TEST(thinking_and_tools_survive_every_stream_boundary) {
  const auto req = prepare_chat(request(), framing("low"));
  const std::string raw = "Check café.\n</think>\n\n" + call;
  ChatResponseParser whole(true, req, "same");
  whole.push(raw); whole.finish(false);
  const auto expected = whole.message();
  LSE_EXPECT(expected["content"].is_null());
  LSE_EXPECT(expected["reasoning_content"] == "Check café.\n");
  const auto args = json::parse(expected["tool_calls"][0]["function"]["arguments"].get<std::string>());
  LSE_EXPECT(args["key"] == "007"); LSE_EXPECT(args["count"] == 2);
  LSE_EXPECT(args["options"]["enabled"] == true);
  for (std::size_t split = 0; split <= raw.size(); ++split) {
    ChatResponseParser parser(true, req, "same");
    parser.push(raw.substr(0, split)); parser.push(raw.substr(split)); parser.finish(false);
    LSE_EXPECT(parser.message() == expected);
    LSE_EXPECT(parser.finish_reason(false) == "tool_calls");
  }
  ChatResponseParser chars(true, req, "same");
  std::size_t tool_deltas = 0;
  for (const char c : raw) for (const auto& d : chars.push(std::string(1, c))) {
    if (d.contains("tool_calls")) {
      ++tool_deltas;
      LSE_EXPECT(d["tool_calls"][0]["index"] == 0);
    }
  }
  chars.finish(false); LSE_EXPECT_EQ(tool_deltas, 1u);
  LSE_EXPECT(chars.message() == expected);
}
LSE_TEST(reasoning_streams_before_close_and_truncation_stays_reasoning) {
  ChatResponseParser parser(true, prepare_chat(request(), framing("low")), "r");
  const auto first = parser.push("working on it");
  LSE_EXPECT_EQ(first.size(), 1u);
  LSE_EXPECT(first[0]["reasoning_content"] == "working on it");
  parser.push("</thi"); parser.finish(true);
  LSE_EXPECT(parser.message()["content"] == "");
  LSE_EXPECT(parser.message()["reasoning_content"] == "working on it</thi");
  LSE_EXPECT(parser.finish_reason(true) == "length");
}
LSE_TEST(explicit_think_tag_and_plain_non_thinking_answer) {
  auto req = prepare_chat(request(), framing("low"));
  ChatResponseParser p(true, req, "r");
  p.push("<thi"); p.push("nk>\nreason</think>\n\nanswer"); p.finish(false);
  LSE_EXPECT(p.message()["reasoning_content"] == "reason");
  LSE_EXPECT(p.message()["content"] == "answer");
  ChatResponseParser plain(false, req, "r");
  LSE_EXPECT(!plain.push("Hello").empty()); plain.finish(false);
  LSE_EXPECT(!plain.message().contains("reasoning_content"));
}
LSE_TEST(multiple_tool_calls_have_distinct_ids_and_stable_indices) {
  auto req = prepare_chat(request(), framing("none"));
  ChatResponseParser p(false, req, "r");
  auto d = p.push(call + "\n" + call); p.finish(false);
  const auto calls = p.message()["tool_calls"];
  LSE_EXPECT_EQ(calls.size(), 2u);
  LSE_EXPECT(calls[0]["id"] != calls[1]["id"]);
  LSE_EXPECT(d.back()["tool_calls"][0]["index"] == 1);
}
LSE_TEST(tool_result_history_and_developer_role_render_as_trained_template) {
  auto b = request();
  b["messages"] = json::parse(R"([
    {"role":"developer","content":"Be concise"},
    {"role":"user","content":"Read the key"},
    {"role":"assistant","content":null,"reasoning_content":"Need lookup",
     "tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":{"key":"007"}}}]},
    {"role":"tool","tool_call_id":"a","content":"value=42"}])");
  const auto req = prepare_chat(b, framing("low"));
  LSE_EXPECT(req.prompt.find("# Tools") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<|im_start|>developer") == std::string::npos);
  LSE_EXPECT(req.prompt.find("<think>\nNeed lookup\n</think>") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<parameter=key>\n007\n</parameter>") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<|im_start|>user\n<tool_response>\nvalue=42\n</tool_response><|im_end|>") != std::string::npos);
}
LSE_TEST(incomplete_tool_call_never_becomes_an_executable_call) {
  ChatResponseParser p(false, prepare_chat(request(), framing("none")), "r");
  p.push("<tool_call><function=lookup><parameter=key>unfinished"); p.finish(true);
  LSE_EXPECT(!p.message().contains("tool_calls"));
  LSE_EXPECT(p.finish_reason(true) == "length");
}
LSE_TEST(tool_choice_none_and_named_choice_are_respected) {
  auto b = request(); b["tool_choice"] = "none";
  auto req = prepare_chat(b, framing("none"));
  LSE_EXPECT(req.tools.empty()); LSE_EXPECT(req.prompt.find("# Tools") == std::string::npos);
  b["tool_choice"] = {{"type", "function"}, {"function", {{"name", "lookup"}}}};
  req = prepare_chat(b, framing("none"));
  ChatResponseParser p(false, req, "r"); p.push(call); p.finish(false);
  LSE_EXPECT(p.finish_reason(false) == "tool_calls");
  bool rejected = false;
  try { ChatResponseParser missing(false, req, "r"); missing.push("No call"); missing.finish(false); }
  catch (const std::exception&) { rejected = true; }
  LSE_EXPECT(rejected);
}
LSE_TEST(malformed_requests_and_unsupported_strict_mode_are_rejected) {
  for (int mode = 0; mode < 5; ++mode) {
    auto b = request();
    if (mode == 0) b["tools"][0]["function"]["strict"] = true;
    if (mode == 1) b["messages"].push_back({{"role", "tool"}, {"tool_call_id", "missing"}, {"content", "42"}});
    if (mode == 2) b["tool_choice"] = "invalid";
    if (mode == 3) b["messages"][0]["content"] = json::array({{{"type", "image_url"}, {"image_url", "x"}}});
    if (mode == 4) b["tools"][0]["function"]["name"] = "bad>name";
    bool rejected = false;
    try { prepare_chat(b, framing("none")); } catch (const std::exception&) { rejected = true; }
    LSE_EXPECT(rejected);
  }
}
LSE_TEST(json_style_calls_and_parallel_false) {
  auto b = request(); b["parallel_tool_calls"] = false;
  ChatResponseParser p(false, prepare_chat(b, framing("none")), "r");
  p.push("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"key\":\"a\"}}</tool_call>");
  bool rejected = false;
  try { p.push(call); } catch (const std::exception&) { rejected = true; }
  LSE_EXPECT(rejected);
}
LSE_TEST(parallel_results_are_rendered_in_call_order) {
  auto b = request();
  b["messages"] = json::parse(R"([
    {"role":"user","content":"Look up both keys"},
    {"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[
      {"id":"first","function":{"name":"lookup","arguments":"{\"key\":\"a\"}"}},
      {"id":"second","function":{"name":"lookup","arguments":"{\"key\":\"b\"}"}}]},
    {"role":"tool","tool_call_id":"second","content":"RESULT_B"},
    {"role":"tool","tool_call_id":"first","content":"RESULT_A"}])");
  const auto req = prepare_chat(b, framing("none"));
  LSE_EXPECT(req.prompt.find("RESULT_A") < req.prompt.find("RESULT_B"));
  LSE_EXPECT(req.prompt.find("</tool_response>\n<tool_response>") != std::string::npos);
}
LSE_TEST(nullable_and_referenced_string_parameters_keep_numeric_text) {
  for (int mode = 0; mode < 3; ++mode) {
    auto b = request();
    auto& schema = b["tools"][0]["function"]["parameters"];
    if (mode == 0) schema["properties"]["key"] = {{"type", json::array({"string", "null"})}};
    if (mode == 1) schema["properties"]["key"] = {{"anyOf", json::array({json{{"type", "string"}}, json{{"type", "null"}}})}};
    if (mode == 2) {
      schema["$defs"]["Text"] = {{"type", "string"}};
      schema["properties"]["key"] = {{"$ref", "#/$defs/Text"}};
    }
    ChatResponseParser parser(false, prepare_chat(b, framing("none")), "r");
    parser.push("<tool_call><function=lookup><parameter=key>42</parameter></function></tool_call>");
    parser.finish(false);
    const auto args = json::parse(parser.message()["tool_calls"][0]["function"]["arguments"].get<std::string>());
    LSE_EXPECT(args["key"] == "42");
  }
}

LSE_TEST(assistant_reasoning_round_trip_preserves_the_cached_prompt_prefix) {
  const json initial{{"messages", json::array({
      {{"role", "user"}, {"content", "First turn"}}})}};
  const auto first = prepare_chat(initial, framing("low"));
  for (const std::string tail : {std::string("\n"), std::string("\n\n")}) {
    const std::string raw = "Consider it." + tail + "</think>\n\nDone.";
    for (std::size_t split = 0; split <= raw.size(); ++split) {
      ChatResponseParser parser(true, first, "roundtrip");
      parser.push(raw.substr(0, split));
      parser.push(raw.substr(split));
      parser.finish(false);
      const auto assistant = parser.message();
      LSE_EXPECT(assistant["reasoning_content"] == "Consider it." + tail);
      json next = initial;
      next["messages"].push_back(assistant);
      next["messages"].push_back({{"role", "user"}, {"content", "Next turn"}});
      const auto second = prepare_chat(next, framing("low"));
      LSE_EXPECT(second.prompt.starts_with(first.prompt + raw + "<|im_end|>\n"));
    }
  }
}

// The prompt LSE builds for each thinking level is what the checkpoint's own
// template renders for the same conversation.
LSE_TEST(each_qwen38_level_frames_the_prompt_exactly_as_its_template_does) {
  std::ifstream in("tests/fixtures/chat_templates/qwen3.8/chat_template.jinja");
  std::stringstream text;
  text << in.rdbuf();
  auto tmpl = lse::models::ChatTemplate::parse(text.str());
  LSE_EXPECT(tmpl.ok());
  if (!tmpl.ok()) return;
  for (const bool with_system : {false, true}) {
    json body{{"messages", json::array()}};
    if (with_system) body["messages"].push_back({{"role", "system"}, {"content", "Be brief."}});
    body["messages"].push_back({{"role", "user"}, {"content", "What is 2+3?"}});
    for (const auto& level : qwen38().levels) {
      nlohmann::ordered_json ctx{{"messages", nlohmann::ordered_json::parse(body["messages"].dump())},
                                 {"add_generation_prompt", true}};
      if (level.enable_thinking) ctx["enable_thinking"] = *level.enable_thinking;
      if (level.reasoning_effort) ctx["reasoning_effort"] = *level.reasoning_effort;
      const auto expected = tmpl->render(ctx);
      LSE_EXPECT(expected.ok());
      if (!expected.ok()) continue;
      const auto ours = prepare_chat(body, framing(level.id)).prompt;
      if (ours != *expected)
        std::fprintf(stderr, "level %s:\n--- lse\n%s\n--- template\n%s\n", level.id.c_str(), ours.c_str(),
                     expected->c_str());
      LSE_EXPECT(ours == *expected);
    }
  }
}

LSE_TEST(thinking_levels_resolve_from_the_template_with_no_aliases) {
  const auto& c = qwen38();
  const auto pick = [&](const json& body) {
    ThinkingRequest r;
    const auto bad = read_thinking_request(body, r);
    LSE_EXPECT(!bad);
    return choose_thinking(c, r);
  };
  // Nothing sent: the template's default (xhigh), reasoning opened.
  auto d = pick(json::object());
  LSE_EXPECT(d.level != nullptr && d.level->id == "xhigh" && d.framing.reasoning);
  LSE_EXPECT(d.framing.instruction.starts_with("Reasoning effort is set to xhigh."));
  // Every spelling of the switch and the level.
  LSE_EXPECT(pick(json{{"enable_thinking", false}}).level->id == "none");
  LSE_EXPECT(pick(json{{"thinking", {{"type", "disabled"}}}}).level->id == "none");
  LSE_EXPECT(pick(json{{"chat_template_kwargs", {{"enable_thinking", false}}}}).level->id == "none");
  LSE_EXPECT(pick(json{{"enable_thinking", true}}).level->id == "xhigh");
  LSE_EXPECT(pick(json{{"reasoning_effort", "low"}}).level->id == "low");
  LSE_EXPECT(pick(json{{"thinking_level", "medium"}}).level->id == "medium");
  LSE_EXPECT(pick(json{{"chat_template_kwargs", {{"reasoning_effort", "low"}}}}).level->id == "low");
  LSE_EXPECT(pick(json{{"reasoning", {{"effort", "low"}}}}).level->id == "low");
  LSE_EXPECT(pick(json{{"reasoning_effort", "none"}}).level->id == "none");
  LSE_EXPECT(pick(json{{"reasoning_effort", "low"}, {"enable_thinking", true}}).level->id == "low");
  LSE_EXPECT(pick(json{{"reasoning_effort", "none"}}).framing.instruction.empty());
  // Levels the template does not define are refused, not aliased.
  for (const char* undefined : {"high", "minimal", "max", "on"}) {
    const auto r = pick(json{{"reasoning_effort", undefined}});
    LSE_EXPECT(r.error && r.error->code == "unsupported_reasoning_effort");
    LSE_EXPECT(r.error && r.error->message.find("none, xhigh, medium, low") != std::string::npos);
  }
  // Contradictions and malformed fields.
  LSE_EXPECT(pick(json{{"reasoning_effort", "low"}, {"enable_thinking", false}}).error->code ==
             "conflicting_thinking");
  ThinkingRequest r;
  LSE_EXPECT(read_thinking_request(json{{"enable_thinking", true}, {"thinking", false}}, r)->code ==
             "conflicting_thinking");
  ThinkingRequest r2;
  LSE_EXPECT(read_thinking_request(json{{"reasoning_effort", "low"}, {"thinking_level", "medium"}}, r2)->code ==
             "conflicting_thinking");
  ThinkingRequest r3;
  LSE_EXPECT(read_thinking_request(json{{"enable_thinking", "yes"}}, r3)->code == "invalid_thinking");
  ThinkingRequest r4;
  LSE_EXPECT(read_thinking_request(json{{"thinking", {{"type", "sometimes"}}}}, r4)->code == "invalid_thinking");
}

LSE_TEST(toggle_only_and_uncontrolled_templates_resolve_their_own_way) {
  const auto qwen36 = lse::models::load_thinking_controls("tests/fixtures/chat_templates/qwen3.6").release();
  const auto smol = lse::models::load_thinking_controls("tests/fixtures/chat_templates/smollm2").release();
  const auto pick = [](const lse::models::ThinkingControls& c, const json& body) {
    ThinkingRequest r;
    LSE_EXPECT(!read_thinking_request(body, r));
    return choose_thinking(c, r);
  };
  LSE_EXPECT(pick(qwen36, json::object()).level->id == "on");
  LSE_EXPECT(pick(qwen36, json{{"enable_thinking", false}}).level->id == "none");
  LSE_EXPECT(pick(qwen36, json{{"reasoning_effort", "on"}}).level->id == "on");
  LSE_EXPECT(pick(qwen36, json{{"reasoning_effort", "low"}}).error->code == "unsupported_reasoning_effort");
  // No controls: no thinking, the template's own assistant opening, and a
  // request to think is refused; asking it not to think is already true.
  const auto plain = pick(smol, json::object());
  LSE_EXPECT(plain.level == nullptr && !plain.error && !plain.framing.reasoning);
  LSE_EXPECT(plain.framing.generation_prompt == "<|im_start|>assistant\n");
  LSE_EXPECT(!pick(smol, json{{"enable_thinking", false}}).error);
  LSE_EXPECT(!pick(smol, json{{"reasoning_effort", "none"}}).error);
  LSE_EXPECT(pick(smol, json{{"enable_thinking", true}}).error->code == "thinking_unsupported");
  LSE_EXPECT(pick(smol, json{{"reasoning_effort", "low"}}).error->code == "thinking_unsupported");
  // No template at all: ChatML's opening.
  const lse::models::ThinkingControls none;
  LSE_EXPECT(pick(none, json::object()).framing.generation_prompt == "<|im_start|>assistant\n");
}

LSE_TEST_MAIN()
