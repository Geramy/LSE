// C ABI over the fastokens Rust crate, which ships only a PyO3 binding.
//
// Every call returns 0 on success and -1 on failure; fk_last_error() then holds
// a NUL-terminated, thread-local message. Panics are caught at the boundary and
// converted to -1 rather than unwinding into C++.
#ifndef FASTOKENS_FFI_H_
#define FASTOKENS_FFI_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fk_tokenizer fk_tokenizer;
typedef struct fk_decode_stream fk_decode_stream;

// Owned token-id buffer. Release with fk_ids_free.
typedef struct {
  uint32_t* data;
  size_t len;
} fk_ids;

// Owned UTF-8 buffer, not NUL-terminated. Release with fk_text_free.
typedef struct {
  char* data;
  size_t len;
} fk_text;

const char* fk_last_error(void);

fk_tokenizer* fk_from_file(const char* path);
// Repo id, e.g. "Qwen/Qwen3.6-27B". Resolves through the HF cache, downloading
// tokenizer.json only if absent.
fk_tokenizer* fk_from_model(const char* repo_id);
void fk_free(fk_tokenizer* tok);

int fk_encode(const fk_tokenizer* tok, const char* text, size_t text_len,
              bool add_special_tokens, fk_ids* out);
int fk_decode(const fk_tokenizer* tok, const uint32_t* ids, size_t count,
              bool skip_special_tokens, fk_text* out);

size_t fk_vocab_size(const fk_tokenizer* tok);
int fk_token_to_id(const fk_tokenizer* tok, const char* token, size_t token_len,
                   uint32_t* out);
int fk_id_to_token(const fk_tokenizer* tok, uint32_t id, fk_text* out);
bool fk_is_special(const fk_tokenizer* tok, uint32_t id);

// Incremental detokenization. Emits text only once a token boundary resolves,
// so a multi-byte UTF-8 character is never split across chunks.
fk_decode_stream* fk_stream_new(bool skip_special_tokens);
// out->len == 0 means "no text yet, feed more tokens".
int fk_stream_step(fk_decode_stream* stream, const fk_tokenizer* tok,
                   const uint32_t* ids, size_t count, fk_text* out);
void fk_stream_free(fk_decode_stream* stream);

void fk_ids_free(fk_ids* ids);
void fk_text_free(fk_text* text);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FASTOKENS_FFI_H_
