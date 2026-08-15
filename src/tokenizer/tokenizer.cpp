#include "lse/tokenizer/tokenizer.hpp"

#include <filesystem>
#include <utility>

#include "fastokens_ffi.h"

namespace lse::tokenizer {

namespace {

namespace fs = std::filesystem;

std::string last_error() {
  const char* e = fk_last_error();
  return e ? std::string(e) : std::string("unknown fastokens error");
}

fk_tokenizer* as_tok(void* p) { return static_cast<fk_tokenizer*>(p); }

}  // namespace

Tokenizer::~Tokenizer() {
  if (handle_) fk_free(as_tok(handle_));
}

Tokenizer::Tokenizer(Tokenizer&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

Tokenizer& Tokenizer::operator=(Tokenizer&& other) noexcept {
  if (this != &other) {
    if (handle_) fk_free(as_tok(handle_));
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

Result<Tokenizer> Tokenizer::from_file(const std::string& tokenizer_json) {
  fk_tokenizer* t = fk_from_file(tokenizer_json.c_str());
  if (t == nullptr) {
    return LSE_ERROR(kIoError, "cannot load tokenizer '", tokenizer_json,
                     "': ", last_error());
  }
  Tokenizer out;
  out.handle_ = t;
  return out;
}

Result<Tokenizer> Tokenizer::from_model(const std::string& repo_id) {
  fk_tokenizer* t = fk_from_model(repo_id.c_str());
  if (t == nullptr) {
    return LSE_ERROR(kNotFound, "cannot load tokenizer for '", repo_id,
                     "': ", last_error());
  }
  Tokenizer out;
  out.handle_ = t;
  return out;
}

Result<Tokenizer> Tokenizer::for_model_dir(const std::string& dir,
                                           const std::string& fallback_repo_id) {
  std::error_code ec;
  const fs::path local = fs::path(dir) / "tokenizer.json";
  if (fs::exists(local, ec)) return from_file(local.string());
  return from_model(fallback_repo_id);
}

Result<std::vector<std::uint32_t>> Tokenizer::encode(std::string_view text,
                                                     bool add_special_tokens) const {
  if (!handle_) return LSE_ERROR(kInternal, "tokenizer is not loaded");
  fk_ids ids{};
  if (fk_encode(as_tok(handle_), text.data(), text.size(), add_special_tokens,
                &ids) != 0) {
    return LSE_ERROR(kInternal, "encode failed: ", last_error());
  }
  std::vector<std::uint32_t> out(ids.data, ids.data + ids.len);
  fk_ids_free(&ids);
  return out;
}

Result<std::string> Tokenizer::decode(const std::vector<std::uint32_t>& ids,
                                      bool skip_special_tokens) const {
  if (!handle_) return LSE_ERROR(kInternal, "tokenizer is not loaded");
  fk_text text{};
  if (fk_decode(as_tok(handle_), ids.data(), ids.size(), skip_special_tokens,
                &text) != 0) {
    return LSE_ERROR(kInternal, "decode failed: ", last_error());
  }
  std::string out(text.data, text.len);
  fk_text_free(&text);
  return out;
}

std::size_t Tokenizer::vocab_size() const noexcept {
  return handle_ ? fk_vocab_size(as_tok(handle_)) : 0;
}

Result<std::uint32_t> Tokenizer::token_to_id(std::string_view token) const {
  if (!handle_) return LSE_ERROR(kInternal, "tokenizer is not loaded");
  std::uint32_t id = 0;
  if (fk_token_to_id(as_tok(handle_), token.data(), token.size(), &id) != 0) {
    return LSE_ERROR(kNotFound, last_error());
  }
  return id;
}

Result<std::string> Tokenizer::id_to_token(std::uint32_t id) const {
  if (!handle_) return LSE_ERROR(kInternal, "tokenizer is not loaded");
  fk_text text{};
  if (fk_id_to_token(as_tok(handle_), id, &text) != 0) {
    return LSE_ERROR(kOutOfRange, last_error());
  }
  std::string out(text.data, text.len);
  fk_text_free(&text);
  return out;
}

bool Tokenizer::is_special(std::uint32_t id) const noexcept {
  return handle_ && fk_is_special(as_tok(handle_), id);
}

DecodeStream Tokenizer::stream(bool skip_special_tokens) const {
  return DecodeStream(*this, skip_special_tokens);
}

DecodeStream::DecodeStream(const Tokenizer& tok, bool skip_special_tokens)
    : tok_(&tok), handle_(fk_stream_new(skip_special_tokens)) {}

DecodeStream::~DecodeStream() {
  if (handle_) fk_stream_free(static_cast<fk_decode_stream*>(handle_));
}

DecodeStream::DecodeStream(DecodeStream&& other) noexcept
    : tok_(std::exchange(other.tok_, nullptr)),
      handle_(std::exchange(other.handle_, nullptr)) {}

DecodeStream& DecodeStream::operator=(DecodeStream&& other) noexcept {
  if (this != &other) {
    if (handle_) fk_stream_free(static_cast<fk_decode_stream*>(handle_));
    tok_ = std::exchange(other.tok_, nullptr);
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

Result<std::string> DecodeStream::push(std::uint32_t id) {
  return push(std::vector<std::uint32_t>{id});
}

Result<std::string> DecodeStream::push(const std::vector<std::uint32_t>& ids) {
  if (!handle_ || tok_ == nullptr || !tok_->valid()) {
    return LSE_ERROR(kInternal, "decode stream is not initialized");
  }
  fk_text text{};
  if (fk_stream_step(static_cast<fk_decode_stream*>(handle_), as_tok(tok_->raw()),
                     ids.data(), ids.size(), &text) != 0) {
    return LSE_ERROR(kInternal, "decode stream failed: ", last_error());
  }
  if (text.data == nullptr || text.len == 0) return std::string{};
  std::string out(text.data, text.len);
  fk_text_free(&text);
  return out;
}

}  // namespace lse::tokenizer
