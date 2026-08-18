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

// ---------------------------------------------------------------------------
// The 4-bit codes as a dot-product operand
// ---------------------------------------------------------------------------
//
// Per group g, the contraction against a group-affine weight is
//
//   sum_i x_i * (c_i * scale_g + bias_g)
//       = scale_g * sum_i (x_i * c_i)  +  bias_g * sum_i x_i
//
// The first sum is an integer dot product once the activation is quantized,
// and scale_g and bias_g then enter the accumulator once per group instead of
// once per weight. That is the whole of why this path is faster: `dequant_chunk`
// above spends an fma per weight turning a code into a float, and this one
// spends none.
//
// 4 bits only. At 8 bits the packed word is already four unsigned bytes — a
// dot4 operand with nothing to unpack — so that width is a different (and
// smaller) change, and it stays on the fma codec until it is measured.

// Rows the integer path spells that the fma codec does not. A target missing
// any of them has no integer path and takes the codec above.
inline constexpr std::array<std::string_view, 4> kGroupAffineDotSymbols{
    "dot4.i32.iu8", "rint", "max", "abs"};

// A 4-bit chunk is one lane holding this many codes, which is also how many
// activations one pair of dot4 operands consumes.
inline constexpr int kDot4Bits = 4;
inline constexpr int kDot4ChunkCodes = 8;

// Which code of the chunk lands in byte `b` of operand word `p`.
//
// The instruction pairs byte i of one operand with byte i of the other, low
// byte first (measured on gfx1151). Byte b of plane p must therefore hold the
// same code index on both sides, and a 4-bit chunk splits into exactly two
// planes because four bytes take four of its eight nibbles: the even nibbles
// are `w & 0x0f0f0f0f` and the odd ones `(w >> 4) & 0x0f0f0f0f`. Both the
// weight side and the activation side derive their layout from this one
// function, so they cannot drift apart.
[[nodiscard]] constexpr int dot4_operand_slot(int p, int b) noexcept {
  return 2 * b + p;
}

// Plane `p` of a packed 4-bit word: four unsigned codes, one per byte, ready
// to be a dot4 operand. Written as division and modulus like the codec above;
// the backend folds it back into one shift and one mask.
template <class E>
[[nodiscard]] detail::U<E> dot4_code_plane(E& e, const detail::U<E>& word,
                                           int p) {
  detail::U<E> plane =
      (word / (1u << (4 * dot4_operand_slot(p, 0)))) % 16u;
  for (int b = 1; b < 4; ++b) {
    plane = plane + ((word / (1u << (4 * dot4_operand_slot(p, b)))) % 16u) *
                        (1u << (8 * b));
  }
  return e.let(plane);
}

// The activation operand that plane `p` multiplies. `byte_of(j)` hands back
// activation j of the chunk as the unsigned byte its int8 code occupies.
template <class E, class ByteOf>
[[nodiscard]] detail::U<E> dot4_activation_word(E& e, const ByteOf& byte_of,
                                                int p) {
  detail::U<E> word = byte_of(dot4_operand_slot(p, 0));
  for (int b = 1; b < 4; ++b) {
    word = word + byte_of(dot4_operand_slot(p, b)) * (1u << (8 * b));
  }
  return e.let(word);
}

}  // namespace lse::quant
