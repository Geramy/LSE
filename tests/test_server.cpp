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
             "<|im_start|>assistant\n<think>\n");
}

LSE_TEST(chatml_can_disable_thinking_in_the_generation_prompt) {
  const std::string out = server::render_chatml(
      {{"user", "Hi"}}, /*add_generation_prompt=*/true,
      /*enable_thinking=*/false);
  LSE_EXPECT(out == "<|im_start|>user\nHi<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n");
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

LSE_TEST(marker_free_stream_matches_nonstream_content) {
  server::detail::ThinkingStreamSplitter stream(/*expose=*/true);
  LSE_EXPECT(stream.push("Hello ").empty());
  LSE_EXPECT(stream.push("world").empty());
  const auto deltas = stream.finish();
  LSE_EXPECT_EQ(deltas.size(), 1u);
  LSE_EXPECT(std::string(deltas[0].first) == "content");
  LSE_EXPECT(deltas[0].second == "Hello world");
  const auto whole = server::detail::split_thinking("Hello world");
  LSE_EXPECT(whole.first == deltas[0].second);
  LSE_EXPECT(whole.second.empty());
}

LSE_TEST(thinking_stream_handles_split_close_marker_and_answer) {
  server::detail::ThinkingStreamSplitter stream(/*expose=*/true);
  LSE_EXPECT(stream.push("<think>\nreason").empty());
  LSE_EXPECT(stream.push("</thi").empty());
  const auto boundary = stream.push("nk>\n\nanswer");
  LSE_EXPECT_EQ(boundary.size(), 2u);
  LSE_EXPECT(std::string(boundary[0].first) == "reasoning");
  LSE_EXPECT(boundary[0].second == "reason");
  LSE_EXPECT(std::string(boundary[1].first) == "content");
  LSE_EXPECT(boundary[1].second == "answer");
  const auto continuation = stream.push(" continues");
  LSE_EXPECT_EQ(continuation.size(), 1u);
  LSE_EXPECT(std::string(continuation[0].first) == "content");
  LSE_EXPECT(continuation[0].second == " continues");
  LSE_EXPECT(stream.finish().empty());
}

LSE_TEST(unclosed_explicit_thinking_marker_finishes_as_reasoning) {
  server::detail::ThinkingStreamSplitter stream(/*expose=*/true);
  LSE_EXPECT(stream.push("<think>unfinished").empty());
  const auto deltas = stream.finish();
  LSE_EXPECT_EQ(deltas.size(), 1u);
  LSE_EXPECT(std::string(deltas[0].first) == "reasoning");
  LSE_EXPECT(deltas[0].second == "unfinished");
}

LSE_TEST(non_thinking_stream_emits_content_immediately) {
  server::detail::ThinkingStreamSplitter stream(/*expose=*/false);
  const auto deltas = stream.push("direct answer");
  LSE_EXPECT_EQ(deltas.size(), 1u);
  LSE_EXPECT(std::string(deltas[0].first) == "content");
  LSE_EXPECT(deltas[0].second == "direct answer");
  LSE_EXPECT(stream.finish().empty());
}

LSE_TEST_MAIN()
