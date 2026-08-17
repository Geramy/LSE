// The device half of GroupAffine, written on the env surface: one body serves
// `env::Emit` (recording HIP) and `env::Cpu` (running on the host, where it is
// the reference the emitted kernel is checked against).
//
// Dequantization is a register operation and belongs at the point of use, not
// at load. `dequant_chunk` hands each decoded weight to a sink as a live value;
// the GEMV multiplies it into the accumulator and it never reaches memory.
// Nothing in this file writes a widened weight anywhere.
//
// The bit arithmetic is `/` `%` `*` `+` rather than shifts and masks: kir has
// no bitwise operators, and on unsigned values these are the same operation.
// Every divisor and modulus is a compile-time constant because a chunk is the
// smallest run of lanes holding a whole number of codes, so within it each
// code's lane and bit offset are fixed.
#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine.hpp"

namespace lse::quant {

namespace env = graph::env;
namespace kir = graph::kir;

// bits=5 is the worst case: 5 lanes carry 32 codes.
inline constexpr int kMaxChunkWords = 5;

namespace detail {

template <class E>
using F = decltype(std::declval<E&>().f32(0.0f));
template <class E>
using U = decltype(std::declval<E&>().u32(0u));

}  // namespace detail

// Every dialect row the codec reaches for. A backend checks this before
// offering the kernel: KernelBody::call answers a missing row with an empty
// Val, which would emit silently broken source.
inline constexpr std::array<std::string_view, 1> kGroupAffineSymbols{"fma"};

// Decodes the `q.values_per_chunk()` codes packed into the `q.words_per_chunk()`
// lanes at `word_base`, and hands each one to `sink(c, value)` as
// `code * scale + bias`. One chunk always lies inside one group — group_size is
// a multiple of 32 and values_per_chunk divides 32 — so the caller loads the
// scale and bias once and they stay in register for the whole chunk.
template <class E, class Idx, class Fn>
void dequant_chunk(E& e, const env::In<std::uint32_t, E>& packed,
                   const GroupAffine& q, const Idx& word_base,
                   const detail::F<E>& scale, const detail::F<E>& bias,
                   Fn&& sink) {
  const int words = q.words_per_chunk();
  std::array<detail::U<E>, kMaxChunkWords> lane;
  for (int w = 0; w < words; ++w) {
    lane[static_cast<std::size_t>(w)] =
        e.let(packed[word_base + static_cast<std::uint32_t>(w)]);
  }
  for (int c = 0; c < q.values_per_chunk(); ++c) {
    const auto w0 = static_cast<std::size_t>(q.chunk_word(c));
    const int off = q.chunk_bit(c);
    const int carry = q.chunk_carry(c);
    const std::uint32_t shift = 1u << off;
    detail::U<E> code = lane[w0] / shift;
    if (carry > 0) {
      // The code straddles two lanes: the low 32-off bits are already
      // right-aligned above, the remaining `carry` bits come from the next.
      const std::uint32_t carry_mod = 1u << carry;
      const std::uint32_t carry_place = 1u << (32 - off);
      code = code + (lane[w0 + 1] % carry_mod) * carry_place;
    } else {
      code = code % (q.max_code() + 1u);
    }
    sink(c, lse::math::fma(lse::math::cast<kir::f32>(e.let(code)), scale, bias));
  }
}

}  // namespace lse::quant
