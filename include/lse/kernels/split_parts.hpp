// How many partitions a split attention pass cuts its block table into, and
// how many key windows each covers, computed in a kernel from the table's
// run-time size (the stride kv::step_meta_stride_index carries) the way
// dispatch::attention_shapes computes them on the host from a shape. The two
// agree by construction: the host's numbers size the partition records, the
// kernel's walk them.
#pragma once

#include <cstdint>

#include "lse/dispatch/attention_tuneconfig.h"
#include "lse/ir/env.hpp"
#include "lse/math.hpp"
#include "lse/kv/block.hpp"

namespace lse::kernels {

// The block table's row stride, in blocks, for a pass of `rows` rows. Taken
// modulo kv::kMaxTableStride (which writers never reach), so the address
// arithmetic built on it is proven to fit 32 bits.
template <class Meta>
[[nodiscard]] graph::kir::Val<graph::kir::u32> table_stride(graph::env::Emit& e, const Meta& meta,
                                                            std::uint32_t rows) {
  return e.let(graph::kir::cast<graph::kir::u32>(
      meta[static_cast<std::uint32_t>(kv::step_meta_stride_index(static_cast<std::int32_t>(rows)))]) %
      static_cast<std::uint32_t>(kv::kMaxTableStride));
}

// Partitions of `keys` keys each (the scalar split's 128, BLASST's 1024).
[[nodiscard]] inline graph::kir::Val<graph::kir::u32> keyed_parts(
    graph::env::Emit& e, const graph::kir::Val<graph::kir::u32>& capacity, std::uint32_t keys) {
  return e.let((capacity + (keys - 1u)) / keys);
}

// The bound every partition, window and share count of a split pass stays
// below (a table under kv::kMaxTableStride blocks has under 2^17 128-key
// partitions).
inline constexpr std::uint32_t kSplitIndexBound = 1u << 18;

// min and max of two counts, computed in 32-bit words (an index has no min
// the target's integer contract can carry through) and brought back into the
// address domain bounded by `bound`, which they stay below, so the address
// arithmetic built on them is proven to fit 32 bits.
[[nodiscard]] inline graph::kir::Val<graph::kir::u32> bounded_min(
    graph::env::Emit& e, const graph::kir::Val<graph::kir::u32>& a,
    const graph::kir::Val<graph::kir::u32>& b, std::uint32_t bound = kSplitIndexBound) {
  return e.let(graph::kir::cast<graph::kir::u32>(math::min(a, b)) % bound);
}
[[nodiscard]] inline graph::kir::Val<graph::kir::u32> bounded_max(
    graph::env::Emit& e, const graph::kir::Val<graph::kir::u32>& a,
    const graph::kir::Val<graph::kir::u32>& b, std::uint32_t bound = kSplitIndexBound) {
  return e.let(graph::kir::cast<graph::kir::u32>(math::max(a, b)) % bound);
}

struct FlashSplitParts {
  // The shares, the windows each covers, and the windows the table holds
  // (the last share's end, at most).
  graph::kir::Val<graph::kir::u32> parts, share, windows;
};

// The most shares a flash split takes (the host's `want`), and a power of
// two above it: the bound a share count and index are stated within.
[[nodiscard]] constexpr std::uint32_t flash_split_want(std::uint32_t units, std::uint32_t compute_units) {
  return units == 0 ? 1u : (8u * compute_units + units - 1u) / units;
}
[[nodiscard]] constexpr std::uint32_t flash_split_part_bound(std::uint32_t units, std::uint32_t compute_units) {
  std::uint32_t bound = 1;
  while (bound <= flash_split_want(units, compute_units)) bound *= 2u;
  return bound;
}

// dispatch::attention_shapes::flash_split_parts and flash_split_share.
[[nodiscard]] inline FlashSplitParts flash_split_parts(graph::env::Emit& e,
                                                       const graph::kir::Val<graph::kir::u32>& capacity,
                                                       std::uint32_t units, std::uint32_t compute_units) {
  using graph::kir::u32;
  constexpr std::uint32_t kWindow = dispatch::attention_shapes::kFlashKeyWindow;
  const auto windows = e.let((capacity + (kWindow - 1u)) / kWindow);
  const std::uint32_t want = units == 0 ? 0u : (8u * compute_units + units - 1u) / units;
  // The target divides integers only by constants, so the divisions by
  // run-time values are done in f32, which is exact here: every operand is
  // below 2^21 (windows of a table of under kv::kMaxTableStride blocks), so
  // a quotient's truncation is the integer floor and a product of a
  // quotient and its divisor is exact. Every divisor is at least one by
  // construction, and the values match the host's (which returns one part
  // when there are no windows).
  using graph::kir::cast;
  using graph::kir::f32;
  constexpr std::uint32_t kBound = 1u << 21;
  const auto one = e.u32(1);
  const auto ceil_div = [&](const graph::kir::Val<u32>& a, const graph::kir::Val<u32>& b) {
    const auto af = e.let(cast<f32>(a)), bf = e.let(cast<f32>(b));
    const auto q = e.let(cast<u32>(af / bf) % kBound);
    return e.let(q + select(cast<f32>(q) * bf < af, e.u32(1), e.u32(0)));
  };
  const auto first = want == 0 ? one : bounded_max(e, bounded_min(e, windows, e.u32(want)), one);
  const auto first_share = bounded_max(e, ceil_div(windows, first), one);
  const auto parts = bounded_max(e, ceil_div(windows, first_share), one,
                                 flash_split_part_bound(units, compute_units));
  const auto share = e.let(ceil_div(windows, parts) % kSplitIndexBound);
  return {parts, share, windows};
}

}  // namespace lse::kernels
