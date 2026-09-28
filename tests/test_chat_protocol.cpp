#include "harness.hpp"
#include "chat_protocol.hpp"
#include <stdexcept>
using namespace lse::server::detail;
using json = ChatJson;
namespace {
json request() {
  return json::parse(R"({"messages":[{"role":"user","content":"Look up a value"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"},"count":{"type":"integer"},"options":{"type":"object"}}}}}]})");
}
const std::string call = "<tool_call>\n<function=lookup>\n<parameter=key>\n007\n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=options>\n{\"enabled\":true,\"values\":[1,null]}\n</parameter>\n</function>\n</tool_call>";
}
LSE_TEST(thinking_and_tools_survive_every_stream_boundary) {
  const auto req = prepare_chat(request(), true, "low");
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
  ChatResponseParser parser(true, prepare_chat(request(), true, "low"), "r");
  const auto first = parser.push("working on it");
  LSE_EXPECT_EQ(first.size(), 1u);
  LSE_EXPECT(first[0]["reasoning_content"] == "working on it");
  parser.push("</thi"); parser.finish(true);
  LSE_EXPECT(parser.message()["content"] == "");
  LSE_EXPECT(parser.message()["reasoning_content"] == "working on it</thi");
  LSE_EXPECT(parser.finish_reason(true) == "length");
}
LSE_TEST(explicit_think_tag_and_plain_non_thinking_answer) {
  auto req = prepare_chat(request(), true, "low");
  ChatResponseParser p(true, req, "r");
  p.push("<thi"); p.push("nk>\nreason</think>\n\nanswer"); p.finish(false);
  LSE_EXPECT(p.message()["reasoning_content"] == "reason");
  LSE_EXPECT(p.message()["content"] == "answer");
  ChatResponseParser plain(false, req, "r");
  LSE_EXPECT(!plain.push("Hello").empty()); plain.finish(false);
  LSE_EXPECT(!plain.message().contains("reasoning_content"));
}
LSE_TEST(multiple_tool_calls_have_distinct_ids_and_stable_indices) {
  auto req = prepare_chat(request(), false, "");
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
  const auto req = prepare_chat(b, true, "low");
  LSE_EXPECT(req.prompt.find("# Tools") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<|im_start|>developer") == std::string::npos);
  LSE_EXPECT(req.prompt.find("<think>\nNeed lookup\n</think>") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<parameter=key>\n007\n</parameter>") != std::string::npos);
  LSE_EXPECT(req.prompt.find("<|im_start|>user\n<tool_response>\nvalue=42\n</tool_response><|im_end|>") != std::string::npos);
}
LSE_TEST(incomplete_tool_call_never_becomes_an_executable_call) {
  ChatResponseParser p(false, prepare_chat(request(), false, ""), "r");
  p.push("<tool_call><function=lookup><parameter=key>unfinished"); p.finish(true);
  LSE_EXPECT(!p.message().contains("tool_calls"));
  LSE_EXPECT(p.finish_reason(true) == "length");
}
LSE_TEST(tool_choice_none_and_named_choice_are_respected) {
  auto b = request(); b["tool_choice"] = "none";
  auto req = prepare_chat(b, false, "");
  LSE_EXPECT(req.tools.empty()); LSE_EXPECT(req.prompt.find("# Tools") == std::string::npos);
  b["tool_choice"] = {{"type", "function"}, {"function", {{"name", "lookup"}}}};
  req = prepare_chat(b, false, "");
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
    try { prepare_chat(b, false, ""); } catch (const std::exception&) { rejected = true; }
    LSE_EXPECT(rejected);
  }
}
LSE_TEST(json_style_calls_and_parallel_false) {
  auto b = request(); b["parallel_tool_calls"] = false;
  ChatResponseParser p(false, prepare_chat(b, false, ""), "r");
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
  const auto req = prepare_chat(b, false, "");
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
    ChatResponseParser parser(false, prepare_chat(b, false, ""), "r");
    parser.push("<tool_call><function=lookup><parameter=key>42</parameter></function></tool_call>");
    parser.finish(false);
    const auto args = json::parse(parser.message()["tool_calls"][0]["function"]["arguments"].get<std::string>());
    LSE_EXPECT(args["key"] == "42");
  }
}

LSE_TEST(assistant_reasoning_round_trip_preserves_the_cached_prompt_prefix) {
  const json initial{{"messages", json::array({
      {{"role", "user"}, {"content", "First turn"}}})}};
  const auto first = prepare_chat(initial, true, "low");
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
      const auto second = prepare_chat(next, true, "low");
      LSE_EXPECT(second.prompt.starts_with(first.prompt + raw + "<|im_end|>\n"));
    }
  }
}

LSE_TEST_MAIN()
