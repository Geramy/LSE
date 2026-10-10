#pragma once

// Block-sparse decode attention (ops::gated_attention, kernels/sparse_decode.cpp).
//
// A decode or verify pass at long context reads only part of the KV cache:
// the sink pages, the recent window, and the far pages whose key summaries
// score highest against the pass's query rows, up to a token budget. The
// summaries are per 16-token page and per key head (key center and radius,
// bf16), kept current by every pass that writes keys.

#include <cstdint>

namespace lse::dispatch::sparse_decode {

// Tokens per page: the KV block size (kv::kBlockSize).
inline constexpr std::uint32_t kPage = 16;
// Always read exactly: the first pages (attention sinks) and the recent
// window, which also holds every row of a verify pass (up to 64).
inline constexpr std::uint32_t kSinkPages = 4;
inline constexpr std::uint32_t kRecentPages = 32;
// Default token budget, sink and recent window included.
inline constexpr std::uint32_t kDefaultBudgetTokens = 4096;
inline constexpr std::uint32_t kMinBudgetTokens = (kSinkPages + kRecentPages + 4) * kPage;
// Every pass that writes keys refreshes the summaries of the pages from this
// many tokens before its first row to its last: a page's final summary is
// taken after its last write, whatever a rejected draft or a tree commit
// rewrote below the pass.
inline constexpr std::uint32_t kRefreshSlackTokens = 64;
// The score of a far page for one query row and head is relative to that
// row's bound over the newest pages (always read), so heads with large and
// small logits compete on attention mass rather than on raw scale.
inline constexpr std::uint32_t kReferencePages = 8;
// Score kernel: one thread per page, query rows staged in LDS this many at a
// time (48 rows of 256 floats is 48 KiB).
inline constexpr std::uint32_t kScoreThreads = 256;
inline constexpr std::uint32_t kScoreRowChunk = 48;
// Select kernel: one workgroup per sequence and key head.
inline constexpr std::uint32_t kSelectThreads = 256;
// Selection keys carry the page index in their low bits (unique keys, so the
// threshold selects exactly the budget): at most 2^14 pages, 256K tokens.
inline constexpr std::uint32_t kPageIndexBits = 14;
inline constexpr std::uint32_t kMaxPages = 1u << kPageIndexBits;

// Pages a refresh pass covers for a pass of `rows` rows.
[[nodiscard]] constexpr std::uint32_t refresh_pages(std::uint32_t rows) noexcept {
  return (rows + kRefreshSlackTokens + kPage - 1) / kPage + 1;
}

// The optimizer takes sparse only where it is predicted to save at least this
// share of the exact attention time: a smaller gain is not worth a lossy
// read.
inline constexpr double kMinSavedShare = 0.2;

}  // namespace lse::dispatch::sparse_decode
