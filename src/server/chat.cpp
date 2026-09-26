#include "lse/server/chat.hpp"

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


// Qwen3.x reasoners steer thinking depth with a system prompt, not a token.
// These instructions are verbatim from the checkpoint chat_template, which
// ships exactly two of them: the default (xhigh) and low. `medium` is a valid
// effort level that carries no instruction (the model's middle default), and
// an empty/unknown level is likewise uninstructed. "" = no instruction.
std::string reasoning_effort_instructions(const std::string& level) {
  if (level == "low") {
    return "Reasoning effort is set to low. Keep your thinking brief and focused, "
           "moving directly to the conclusion without unnecessary elaboration.";
  }
  if (level == "xhigh" || level == "high") {
    // "high" is accepted as an alias for the template's top level "xhigh"; the
    // instruction text is the template's xhigh string, which is all the model
    // was trained with at this depth.
    return "Reasoning effort is set to xhigh. Please think carefully through the "
           "task, validate key assumptions, consider plausible alternatives, and "
           "prioritize correctness, consistency, and clarity in the final answer.";
  }
  // "medium" and anything else: no instruction (the template injects nothing).
  return "";
}

}  // namespace lse::server
