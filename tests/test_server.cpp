// The server's own logic: turning messages into a prompt, and the ids that
// end a turn. The HTTP surface itself is exercised against a live model.
#include "harness.hpp"
#include "lse/server/chat.hpp"
#include <algorithm>

using namespace lse;
using lse::server::ChatMessage;

LSE_TEST(chatml_frames_every_message_and_opens_the_assistant_turn) {
  const std::string out = server::render_chatml(
      {{"system", "You are terse."}, {"user", "Hi"}});

  LSE_EXPECT(out ==
             "<|im_start|>system\nYou are terse.<|im_end|>\n"
             "<|im_start|>user\nHi<|im_end|>\n"
             "<|im_start|>assistant\n");
}

LSE_TEST(chatml_can_leave_the_assistant_turn_unopened) {
  // Scoring a completed exchange rather than continuing it: the trailing
  // header would be a token the text never had.
  const std::string out =
      server::render_chatml({{"user", "Hi"}}, /*add_generation_prompt=*/false);
  LSE_EXPECT(out == "<|im_start|>user\nHi<|im_end|>\n");
}

LSE_TEST(a_message_with_no_role_is_from_the_user) {
  // The wire format says role is required; a client that omits it gets the
  // reading that cannot silently become a system instruction.
  const std::string out = server::render_chatml({{"", "Hi"}});
  LSE_EXPECT(out.find("<|im_start|>user\nHi") == 0);
}

LSE_TEST(chat_stop_tokens_skip_what_a_tokenizer_does_not_have) {
  // Not every tokenizer spells every terminator. The ones it does not have are
  // dropped rather than turning into an id that means something else, and a
  // tokenizer with none simply runs to the token limit.
  auto tok = tokenizer::Tokenizer::from_model(
      std::string(tokenizer::kQwen36TokenizerRepo));
  if (!tok.ok()) return;  // no tokenizer cached; nothing to assert against

  const std::vector<std::uint32_t> stops = server::chat_stop_tokens(*tok);
  LSE_EXPECT(!stops.empty());
  for (std::uint32_t id : stops) {
    LSE_EXPECT(id < tok->vocab_size());
    LSE_EXPECT(tok->is_special(id));
  }
  // <|im_end|> is what ends a ChatML turn, so it has to be among them.
  auto im_end = tok->token_to_id("<|im_end|>");
  LSE_EXPECT(im_end.ok());
  LSE_EXPECT(std::find(stops.begin(), stops.end(), *im_end) != stops.end());
}

LSE_TEST_MAIN()
