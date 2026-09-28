#include "chat_protocol.hpp"
#include "lse/server/chat.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>

namespace lse::server::detail {
namespace {
using json = ChatJson;
std::string trim(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return {};
  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
[[noreturn]] void bad(const std::string& s) { throw std::invalid_argument(s); }
std::string text_content(const json& m) {
  if (!m.contains("content") || m["content"].is_null()) return {};
  const auto& c = m["content"];
  if (c.is_string()) return c.get<std::string>();
  if (!c.is_array()) bad("message content must be text or text parts");
  std::string out;
  for (const auto& p : c) {
    if (!p.is_object() || p.value("type", "") != "text" ||
        !p.contains("text") || !p["text"].is_string())
      bad("only text content parts are supported");
    out += p["text"].get<std::string>();
  }
  return out;
}
bool name_ok(const std::string& s) {
  return !s.empty() && s.size() <= 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-';
  });
}
json arguments(json a) {
  if (a.is_string()) a = json::parse(a.get<std::string>());
  if (!a.is_object()) bad("function arguments must be a JSON object");
  return a;
}
std::string call_text(const json& f) {
  const auto name = f.at("name").get<std::string>();
  if (!name_ok(name)) bad("invalid function name");
  const auto args = arguments(f.at("arguments"));
  std::string out = "<tool_call>\n<function=" + name + ">\n";
  for (const auto& [key, value] : args.items()) {
    if (key.find_first_of("<>\r\n") != std::string::npos) bad("invalid parameter name");
    out += "<parameter=" + key + ">\n";
    out += value.is_string() ? value.get<std::string>() : value.dump();
    out += "\n</parameter>\n";
  }
  return out + "</function>\n</tool_call>";
}
bool string_parameter(const json& property, const json& schema, unsigned depth = 0) {
  if (!property.is_object() || depth > 16) return false;
  const auto type = property.value("type", json{});
  if (type == "string" || (type.is_array() &&
      std::find(type.begin(), type.end(), "string") != type.end())) return true;
  for (const char* key : {"anyOf", "oneOf", "allOf"})
    if (property.contains(key) && property[key].is_array())
      for (const auto& option : property[key])
        if (string_parameter(option, schema, depth + 1)) return true;
  if (property.contains("$ref") && property["$ref"].is_string()) {
    const auto ref = property["$ref"].get<std::string>();
    if (ref.starts_with("#/"))
      return string_parameter(schema.at(json::json_pointer(ref.substr(1))), schema, depth + 1);
  }
  return false;
}
std::size_t held_suffix(const std::string& s, std::string_view marker) {
  for (std::size_t n = std::min(s.size(), marker.size() - 1); n > 0; --n)
    if (s.compare(s.size() - n, n, marker.substr(0, n)) == 0) return n;
  return 0;
}
}  // namespace

ChatRequest prepare_chat(const ChatJson& body, bool thinking, const std::string& effort) {
  ChatRequest result;
  if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
    bad("messages must be a non-empty array");
  if (body.contains("tools")) {
    if (!body["tools"].is_array()) bad("tools must be an array");
    result.tools = body["tools"];
  }
  std::set<std::string> names;
  for (const auto& t : result.tools) {
    if (!t.is_object() || t.value("type", "") != "function" ||
        !t.contains("function") || !t["function"].is_object())
      bad("only function tools are supported");
    const auto& f = t["function"];
    const auto name = f.value("name", "");
    if (!name_ok(name) || !names.insert(name).second) bad("invalid or duplicate tool name");
    if (f.contains("parameters") && !f["parameters"].is_object()) bad("parameters must be an object");
    if (f.value("strict", false)) bad("strict schema-constrained generation is not supported; use strict:false");
  }
  const auto choice = body.value("tool_choice", json("auto"));
  if (choice.is_string()) {
    const auto s = choice.get<std::string>();
    if (s == "none") result.tools.clear();
    else if (s == "required") result.require_tool = true;
    else if (s != "auto") bad("invalid tool_choice");
  } else if (choice.is_object() && choice.value("type", "") == "function") {
    result.required_tool = choice.at("function").at("name").get<std::string>();
    if (!names.contains(result.required_tool)) bad("tool_choice names an undeclared function");
    result.require_tool = true;
  } else bad("invalid tool_choice");
  if (result.require_tool && result.tools.empty()) bad("tool_choice requires a tool definition");
  result.parallel = body.value("parallel_tool_calls", true);

  std::vector<ChatMessage> messages;
  std::string system;
  std::set<std::string> pending;
  bool seen_conversation = false;
  std::vector<std::string> pending_order;
  std::map<std::string, std::string> tool_results;
  for (const auto& m : body["messages"]) {
    if (!m.is_object()) bad("messages must contain objects");
    const auto role = m.value("role", "user");
    const auto content = trim(text_content(m));
    if (role == "system" || role == "developer") {
      if (seen_conversation) bad("system and developer messages must precede the conversation");
      if (!system.empty()) system += "\n\n";
      system += content;
      continue;
    }
    seen_conversation = true;
    if (role == "tool") {
      const auto id = m.value("tool_call_id", "");
      if (id.empty() || !pending.erase(id)) bad("tool result needs a matching pending tool_call_id");
      tool_results[id] = content;
      if (pending.empty()) {
        std::string rendered;
        for (const auto& call_id : pending_order) {
          if (!rendered.empty()) rendered += "\n";
          rendered += "<tool_response>\n" + tool_results.at(call_id) + "\n</tool_response>";
        }
        messages.push_back({"user", rendered});
        pending_order.clear(); tool_results.clear();
      }
      continue;
    }
    if (!pending.empty()) bad("missing tool result before the next message");
    if (role == "user") messages.push_back({role, content});
    else if (role == "assistant") {
      const auto reasoning = m.contains("reasoning_content") && !m["reasoning_content"].is_null()
          ? m.at("reasoning_content").get<std::string>() : std::string{};
      std::string rendered = "<think>\n" + reasoning;
      if (reasoning.empty() || !reasoning.ends_with('\n')) rendered += '\n';
      rendered += "</think>\n\n" + content;
      if (m.contains("tool_calls")) {
        if (!m["tool_calls"].is_array()) bad("tool_calls must be an array");
        for (const auto& call : m["tool_calls"]) {
          if (call.value("type", "function") != "function") bad("only function calls are supported");
          const auto id = call.at("id").get<std::string>();
          if (id.empty() || !pending.insert(id).second) bad("invalid or duplicate tool call id");
          pending_order.push_back(id);
          if (!rendered.empty() && !rendered.ends_with("\n\n")) rendered += "\n\n";
          rendered += call_text(call.at("function"));
        }
      }
      messages.push_back({role, rendered});
    } else bad("unsupported message role: " + role);
  }
  if (!pending.empty()) bad("missing tool results at end of conversation");
  std::string prefix = thinking ? reasoning_effort_instructions(effort.empty() ? "xhigh" : effort) : "";
  if (!result.tools.empty()) {
    if (!prefix.empty()) prefix += "\n\n";
    prefix += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto& t : result.tools) prefix += "\n" + t.dump();
    prefix += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
              "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
              "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
              "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
              "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
              "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    if (result.require_tool) prefix += result.required_tool.empty()
        ? "\nYou must call at least one of the available functions."
        : "\nYou must call the function " + result.required_tool + ".";
    if (!result.parallel) prefix += "\nCall at most one function in this turn.";
  }
  if (!system.empty()) { if (!prefix.empty()) prefix += "\n\n"; prefix += system; }
  if (!prefix.empty()) messages.insert(messages.begin(), {"system", prefix});
  result.prompt = render_chatml(messages, true, thinking);
  return result;
}

ChatResponseParser::ChatResponseParser(bool thinking, const ChatRequest& request, std::string id)
    : request_(request), id_(std::move(id)), thinking_(thinking) {}

void ChatResponseParser::emit_text(std::vector<ChatJson>& out, const char* field, std::string text) {
  if (first_) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return;
    text.erase(0, start);
    first_ = false;
  }
  if (text.empty()) return;
  (thinking_ ? reasoning_ : content_) += text;
  out.push_back({{field, std::move(text)}});
}
void ChatResponseParser::emit_call(std::vector<ChatJson>& out, const std::string& raw) {
  json args = json::object();
  std::string name;
  const auto body = trim(raw);
  if (body.starts_with("{")) {
    const auto f = json::parse(body);
    name = f.at("name").get<std::string>();
    args = arguments(f.at("arguments"));
  } else {
    if (!body.starts_with("<function=")) bad("malformed function call");
    const auto end = body.find('>');
    if (end == std::string::npos || !body.ends_with("</function>")) bad("incomplete function call");
    name = body.substr(10, end - 10);
    auto tail = trim(body.substr(end + 1, body.size() - end - 12));
    const json* schema = nullptr;
    for (const auto& t : request_.tools)
      if (t["function"]["name"] == name && t["function"].contains("parameters"))
        schema = &t["function"]["parameters"];
    while (!tail.empty()) {
      if (!tail.starts_with("<parameter=")) bad("malformed function parameter");
      const auto key_end = tail.find('>'), value_end = tail.find("</parameter>");
      if (key_end == std::string::npos || value_end == std::string::npos || value_end < key_end) bad("incomplete function parameter");
      const auto key = tail.substr(11, key_end - 11);
      if (args.contains(key)) bad("duplicate function parameter");
      auto value = tail.substr(key_end + 1, value_end - key_end - 1);
      if (value.starts_with("\n")) value.erase(0, 1);
      if (value.ends_with("\n")) value.pop_back();
      bool string_type = schema && schema->contains("properties") &&
          (*schema)["properties"].contains(key) && string_parameter((*schema)["properties"][key], *schema);
      auto parsed = json::parse(value, nullptr, false);
      args[key] = string_type || parsed.is_discarded() ? json(value) : parsed;
      tail = trim(tail.substr(value_end + 12));
    }
  }
  const auto declared = std::find_if(request_.tools.begin(), request_.tools.end(), [&](const auto& t) {
    return t["function"]["name"] == name;
  });
  if (declared == request_.tools.end()) bad("generated call to undeclared function: " + name);
  if (!request_.required_tool.empty() && request_.required_tool != name) bad("generated call does not match tool_choice");
  if (!request_.parallel && !calls_.empty()) bad("generated multiple calls with parallel_tool_calls:false");
  json call{{"id", "call_" + id_ + "_" + std::to_string(calls_.size())}, {"type", "function"},
            {"function", {{"name", name}, {"arguments", args.dump()}}}};
  calls_.push_back(call);
  call["index"] = calls_.size() - 1;
  out.push_back({{"tool_calls", json::array({call})}});
}
std::vector<ChatJson> ChatResponseParser::drain(bool final) {
  std::vector<ChatJson> out;
  while (!pending_.empty()) {
    if (thinking_) {
      if (first_) {
        if (pending_.starts_with("<think>")) pending_.erase(0, 7);
        else if (!final && std::string_view("<think>").starts_with(pending_)) break;
      }
      const auto end = pending_.find("</think>");
      if (end != std::string::npos) {
        emit_text(out, "reasoning_content", pending_.substr(0, end));
        pending_.erase(0, end + 8);
        thinking_ = false; first_ = true;
        continue;
      }
      const auto take = pending_.size() - (final ? 0 : held_suffix(pending_, "</think>"));
      emit_text(out, "reasoning_content", pending_.substr(0, take));
      pending_.erase(0, take);
      break;
    }
    if (in_tool_) {
      const auto end = pending_.find("</tool_call>");
      if (end == std::string::npos) {
        if (final) { emit_text(out, "content", "<tool_call>" + pending_); pending_.clear(); }
        break;
      }
      emit_call(out, pending_.substr(0, end));
      pending_.erase(0, end + 12); in_tool_ = false;
      continue;
    }
    const auto start = request_.tools.empty() ? std::string::npos : pending_.find("<tool_call>");
    if (start != std::string::npos) {
      emit_text(out, "content", pending_.substr(0, start));
      pending_.erase(0, start + 11); in_tool_ = true;
      continue;
    }
    const auto take = pending_.size() - (final || request_.tools.empty() ? 0 : held_suffix(pending_, "<tool_call>"));
    emit_text(out, "content", pending_.substr(0, take));
    pending_.erase(0, take);
    break;
  }
  return out;
}
std::vector<ChatJson> ChatResponseParser::push(const std::string& text) {
  pending_ += text;
  return drain(false);
}
std::vector<ChatJson> ChatResponseParser::finish(bool hit_limit) {
  auto out = drain(true);
  if (!hit_limit && request_.require_tool && calls_.empty()) bad("model did not produce the required tool call");
  return out;
}
ChatJson ChatResponseParser::message() const {
  json m{{"role", "assistant"}, {"content", content_.empty() && !calls_.empty() ? json(nullptr) : json(content_)}};
  if (!reasoning_.empty()) m["reasoning_content"] = reasoning_;
  if (!calls_.empty()) m["tool_calls"] = calls_;
  return m;
}
std::string ChatResponseParser::finish_reason(bool hit_limit) const {
  return hit_limit ? "length" : !calls_.empty() ? "tool_calls" : "stop";
}
}  // namespace lse::server::detail
