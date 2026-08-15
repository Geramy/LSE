// RAII wrapper over the fastokens C shim.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::tokenizer {

// Qwen3.6, padded. Must match the checkpoint's vocab_size.
inline constexpr std::int32_t kQwen36PaddedVocabSize = 248320;
// The repo that ships the tokenizer.json these ids come from. lemonseed trains
// against this vocabulary and ships no tokenizer of its own.
inline constexpr std::string_view kQwen36TokenizerRepo = "Qwen/Qwen3.6-27B";
inline constexpr std::uint32_t kQwen36Eos = 248046;  // <|im_end|>
inline constexpr std::uint32_t kQwen36Pad = 248044;  // <|endoftext|>

class DecodeStream;

class Tokenizer {
 public:
  Tokenizer() = default;
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  static Result<Tokenizer> from_file(const std::string& tokenizer_json);
  // HF repo id; resolves through the local cache, downloading only if absent.
  static Result<Tokenizer> from_model(const std::string& repo_id);

  // Looks for tokenizer.json beside the weights, then falls back to the repo id
  // the config names.
  static Result<Tokenizer> for_model_dir(const std::string& dir,
                                         const std::string& fallback_repo_id);

  Result<std::vector<std::uint32_t>> encode(std::string_view text,
                                            bool add_special_tokens = false) const;
  Result<std::string> decode(const std::vector<std::uint32_t>& ids,
                             bool skip_special_tokens = true) const;

  [[nodiscard]] std::size_t vocab_size() const noexcept;
  Result<std::uint32_t> token_to_id(std::string_view token) const;
  Result<std::string> id_to_token(std::uint32_t id) const;
  [[nodiscard]] bool is_special(std::uint32_t id) const noexcept;

  [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }
  [[nodiscard]] void* raw() const noexcept { return handle_; }

  DecodeStream stream(bool skip_special_tokens = true) const;

 private:
  void* handle_ = nullptr;
};

// Incremental detokenization for streaming. Returns text only once a token
// boundary resolves, so a multi-byte character never splits across SSE chunks.
class DecodeStream {
 public:
  explicit DecodeStream(const Tokenizer& tok, bool skip_special_tokens = true);
  ~DecodeStream();
  DecodeStream(DecodeStream&&) noexcept;
  DecodeStream& operator=(DecodeStream&&) noexcept;
  DecodeStream(const DecodeStream&) = delete;
  DecodeStream& operator=(const DecodeStream&) = delete;

  // Empty string means "not a boundary yet, feed more tokens".
  Result<std::string> push(std::uint32_t id);
  Result<std::string> push(const std::vector<std::uint32_t>& ids);

 private:
  const Tokenizer* tok_ = nullptr;
  void* handle_ = nullptr;
};

}  // namespace lse::tokenizer
