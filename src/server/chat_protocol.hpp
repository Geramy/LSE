#pragma once
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace lse::server::detail {
using ChatJson = nlohmann::json;
struct ChatRequest {
  std::string prompt;
  ChatJson tools = ChatJson::array();
  std::string required_tool;
  bool require_tool = false;
  bool parallel = true;
};
// Throws invalid_argument for unsupported or malformed wire fields.
ChatRequest prepare_chat(const ChatJson&, bool thinking, const std::string& effort);

class ChatResponseParser {
 public:
  ChatResponseParser(bool thinking, const ChatRequest&, std::string id);
  std::vector<ChatJson> push(const std::string&);
  std::vector<ChatJson> finish(bool hit_limit);
  ChatJson message() const;
  std::string finish_reason(bool hit_limit) const;
 private:
  void emit_text(std::vector<ChatJson>&, const char*, std::string);
  void emit_call(std::vector<ChatJson>&, const std::string&);
  std::vector<ChatJson> drain(bool final);
  ChatRequest request_;
  std::string id_, pending_, content_, reasoning_;
  ChatJson calls_ = ChatJson::array();
  bool thinking_, first_ = true, in_tool_ = false;
};
}  // namespace lse::server::detail
