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
// How the prompt is framed for the requested thinking level, as the model's
// chat template defines it.
struct ChatFraming {
  // Template instruction placed first in the system prompt ("" for none).
  std::string instruction;
  // Appended after the conversation to open the assistant turn.
  std::string generation_prompt = "<|im_start|>assistant\n";
  // The generation prompt leaves a <think> block open.
  bool reasoning = false;
};
// Throws invalid_argument for unsupported or malformed wire fields.
ChatRequest prepare_chat(const ChatJson&, const ChatFraming& framing);

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
