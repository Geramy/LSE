#include "lse/server/chat.hpp"

namespace lse::server {

std::string render_chatml(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt) {
  std::string out;
  for (const ChatMessage& m : messages) {
    out += "<|im_start|>";
    out += m.role.empty() ? "user" : m.role;
    out += '\n';
    out += m.content;
    out += "<|im_end|>\n";
  }
  if (add_generation_prompt) out += "<|im_start|>assistant\n";
  return out;
}

std::vector<std::uint32_t> chat_stop_tokens(const tokenizer::Tokenizer& tok) {
  std::vector<std::uint32_t> stops;
  for (const char* name : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>"}) {
    if (auto id = tok.token_to_id(name); id.ok()) stops.push_back(*id);
  }
  return stops;
}

}  // namespace lse::server
