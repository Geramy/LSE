// Chat messages to a prompt, and the stop tokens that end one.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/tokenizer/tokenizer.hpp"

namespace lse::server {

struct ChatMessage {
  std::string role;
  std::string content;
};

// ChatML, which is what every model this engine targets is trained on:
// Qwen3.5, Qwen3.8 and lemonseed. It is written out here rather than rendered
// from the checkpoint's own chat_template, which is Jinja and would need an
// engine to evaluate; a checkpoint trained on some other framing would need
// its own renderer rather than a different template string.
[[nodiscard]] std::string render_chatml(const std::vector<ChatMessage>& messages,
                                        bool add_generation_prompt = true);

// Ids that end a turn: the tokenizer's own end-of-sequence plus ChatML's
// <|im_end|>. Missing ones are skipped, so a tokenizer without them simply
// runs to the token limit.
[[nodiscard]] std::vector<std::uint32_t> chat_stop_tokens(
    const tokenizer::Tokenizer& tok);

}  // namespace lse::server
