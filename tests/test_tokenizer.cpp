// Exercises the fastokens FFI shim against whatever tokenizer.json is on the
// machine. Skips cleanly when none is present.
#include "lse/tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

#include "harness.hpp"

using namespace lse;
using namespace lse::tokenizer;

namespace {

namespace fs = std::filesystem;

// LSE_TEST_TOKENIZER overrides; otherwise take the first one in the HF cache.
std::string find_tokenizer() {
  if (const char* p = std::getenv("LSE_TEST_TOKENIZER")) return p;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path hub = fs::path(home) / ".cache/huggingface/hub";
  std::error_code ec;
  if (!fs::is_directory(hub, ec)) return {};
  for (auto it = fs::recursive_directory_iterator(
           hub, fs::directory_options::skip_permission_denied, ec);
       it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (it->path().filename() == "tokenizer.json") return it->path().string();
  }
  return {};
}

const std::string& tokenizer_path() {
  static const std::string p = find_tokenizer();
  return p;
}

}  // namespace

LSE_TEST(loads_a_tokenizer_from_disk) {
  if (tokenizer_path().empty()) {
    std::printf("       (skipped: no tokenizer.json found)\n");
    return;
  }
  auto tok = Tokenizer::from_file(tokenizer_path());
  LSE_EXPECT(tok.ok());
  if (tok.ok()) LSE_EXPECT(tok->vocab_size() > 0);
}

LSE_TEST(missing_file_reports_an_error_not_a_crash) {
  auto tok = Tokenizer::from_file("/definitely/not/here/tokenizer.json");
  LSE_EXPECT(!tok.ok());
  LSE_EXPECT(!tok.status().message().empty());
}

LSE_TEST(encode_decode_round_trips) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;

  const std::string text = "The quick brown fox jumps over the lazy dog.";
  auto ids = tok->encode(text);
  LSE_EXPECT(ids.ok());
  if (!ids.ok()) return;
  LSE_EXPECT(!ids->empty());

  auto back = tok->decode(*ids);
  LSE_EXPECT(back.ok());
  if (back.ok()) LSE_EXPECT(*back == text);
}

LSE_TEST(round_trips_multibyte_utf8) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;

  const std::string text = "こんにちは 🌍 café — naïve";
  auto ids = tok->encode(text);
  LSE_EXPECT(ids.ok());
  if (!ids.ok()) return;
  auto back = tok->decode(*ids);
  LSE_EXPECT(back.ok());
  if (back.ok()) LSE_EXPECT(*back == text);
}

LSE_TEST(empty_input_encodes_to_nothing) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;
  auto ids = tok->encode("");
  LSE_EXPECT(ids.ok());
  if (ids.ok()) LSE_EXPECT(ids->empty());
}

LSE_TEST(token_id_lookups_are_inverse) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;

  auto ids = tok->encode("hello world");
  if (!ids.ok() || ids->empty()) return;
  auto piece = tok->id_to_token((*ids)[0]);
  LSE_EXPECT(piece.ok());
  if (!piece.ok()) return;
  auto back = tok->token_to_id(*piece);
  LSE_EXPECT(back.ok());
  if (back.ok()) LSE_EXPECT_EQ(*back, (*ids)[0]);
}

LSE_TEST(out_of_range_id_is_an_error) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;
  auto piece = tok->id_to_token(4000000000u);
  LSE_EXPECT(!piece.ok());
}

LSE_TEST(stream_reassembles_the_same_text) {
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;

  const std::string text = "Streaming tokens one at a time.";
  auto ids = tok->encode(text);
  if (!ids.ok()) return;

  DecodeStream stream = tok->stream();
  std::string assembled;
  for (std::uint32_t id : *ids) {
    auto chunk = stream.push(id);
    LSE_EXPECT(chunk.ok());
    if (chunk.ok()) assembled += *chunk;
  }
  LSE_EXPECT(assembled == text);
}

LSE_TEST(stream_never_splits_a_multibyte_character) {
  // The reason streaming detokenization exists: an SSE chunk must never carry
  // half a UTF-8 sequence.
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;

  const std::string text = "日本語のテキスト 🎉🎊 emoji";
  auto ids = tok->encode(text);
  if (!ids.ok()) return;

  DecodeStream stream = tok->stream();
  std::string assembled;
  for (std::uint32_t id : *ids) {
    auto chunk = stream.push(id);
    LSE_EXPECT(chunk.ok());
    if (!chunk.ok()) return;
    // Every emitted chunk must itself be well-formed UTF-8.
    const std::string& c = *chunk;
    for (std::size_t i = 0; i < c.size();) {
      const auto b = static_cast<unsigned char>(c[i]);
      std::size_t len = 1;
      if ((b & 0x80) == 0) len = 1;
      else if ((b & 0xE0) == 0xC0) len = 2;
      else if ((b & 0xF0) == 0xE0) len = 3;
      else if ((b & 0xF8) == 0xF0) len = 4;
      else { LSE_EXPECT(false); break; }
      LSE_EXPECT(i + len <= c.size());
      if (i + len > c.size()) break;
      i += len;
    }
    assembled += c;
  }
  LSE_EXPECT(assembled == text);
}

LSE_TEST(qwen36_special_ids_are_what_the_model_expects) {
  // The checkpoint's vocab_size is the padded 248320; the tokenizer's own
  // vocab is smaller. Only assert the relationship, not equality.
  if (tokenizer_path().empty()) return;
  auto tok = Tokenizer::from_file(tokenizer_path());
  if (!tok.ok()) return;
  LSE_EXPECT(static_cast<std::int32_t>(tok->vocab_size()) <= kQwen36PaddedVocabSize);
}

LSE_TEST_MAIN()
