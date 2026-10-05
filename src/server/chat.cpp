#include "lse/server/chat.hpp"

#include <string_view>

namespace lse::server {

std::string render_chatml(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt, bool enable_thinking) {
  std::string out;
  for (const ChatMessage& m : messages) {
    out += "<|im_start|>";
    out += m.role.empty() ? "user" : m.role;
    out += '\n';
    out += m.content;
    out += "<|im_end|>\n";
  }
  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    // Qwen3.x reasoning gate (from the checkpoint chat_template): for a
    // thinking turn emit an open think tag; for a non-thinking turn emit an
    // empty think block so the model answers directly. Inert for a non-reasoner.
    if (enable_thinking) {
      out += "<think>\n";
    } else {
      out += "<think>\n\n</think>\n\n";
    }
  }
  return out;
}

std::vector<std::uint32_t> chat_stop_tokens(const tokenizer::Tokenizer& tok) {
  std::vector<std::uint32_t> stops;
  for (const char* name : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>"}) {
    if (auto id = tok.token_to_id(name); id.ok()) stops.push_back(*id);
  }
  return stops;
}


namespace detail {
namespace {
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kWhitespace = " \t\r\n";
}  // namespace

std::pair<std::string, std::string> split_thinking(const std::string& text) {
  const std::size_t end = text.find(kThinkClose);
  if (end == std::string::npos) {
    const std::size_t open = text.find(kThinkOpen);
    if (open != std::string::npos)
      return {"", text.substr(open + kThinkOpen.size())};
    return {text, ""};
  }

  std::string reasoning = text.substr(0, end);
  const std::size_t open = reasoning.find(kThinkOpen);
  if (open != std::string::npos)
    reasoning = reasoning.substr(open + kThinkOpen.size());
  const std::size_t reasoning_start = reasoning.find_first_not_of(kWhitespace);
  reasoning = reasoning_start == std::string::npos
                  ? "" : reasoning.substr(reasoning_start);

  std::string answer = text.substr(end + kThinkClose.size());
  const std::size_t answer_start = answer.find_first_not_of(kWhitespace);
  answer = answer_start == std::string::npos ? "" : answer.substr(answer_start);
  return {std::move(answer), std::move(reasoning)};
}

}  // namespace detail

}  // namespace lse::server
