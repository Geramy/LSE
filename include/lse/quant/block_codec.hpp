// The device half of QuantScheme: the same schemes, the same BlockQ* layouts,
// written on the env surface.
//
// quant/traits.hpp packs a block on the host from real pointers. This packs the
// same bytes from env values, so one body serves `env::Emit` (recording HIP)
// and `env::Cpu` (running on the host, where it is the reference the device
// output is checked against, bit for bit).
//
// One generic body, parameterized on the scheme. A new scheme — Q5, or fp8 once
// a target spells the conversion — is a BlockCodec specialization and a table
// row, never a new pack function and never a new switch arm carrying new logic.
// `dispatch_block_codec` turns a runtime dtype into the right instantiation on
// the host at emit time; no kernel ever branches on the scheme.
//
// Two properties the code cannot state:
//   * The byte arithmetic is deliberately `/` `%` `*` `+` rather than shifts
//     and masks — kir has no bitwise operators, and on unsigned values these
//     are the same operation. Every intermediate stays inside u32.
//   * Bit-identity with QuantScheme is a hard requirement, not a nice property:
//     the scale written to the block is the f32 block scale rounded once to
//     fp16, while the codes divide by the *unrounded* f32 scale. Swapping those
//     two moves the low bit of most codes.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include "lse/core/dtype.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/math.hpp"
#include "lse/quant/traits.hpp"

namespace lse::quant {

namespace env = graph::env;
namespace kir = graph::kir;

inline constexpr std::uint32_t kBlockElemsU =
    static_cast<std::uint32_t>(kBlockElems);

// Buffers as a codec sees them: values on one side, packed bytes on the other.
template <class E>
struct PackArgs {
  env::In<kir::f32, E> x;
  env::Out<std::uint8_t, E> packed;
};

template <class E>
struct UnpackArgs {
  env::In<std::uint8_t, E> packed;
  env::Out<kir::f32, E> x;
};

namespace detail {

// Value types of the env being instantiated: `Val<f32>`/`Val<u32>` when
// recording, plain `float`/`uint32_t` on the host.
template <class E>
using F = decltype(std::declval<E&>().f32(0.0f));
template <class E>
using U = decltype(std::declval<E&>().u32(0u));

template <class E>
struct Scale {
  F<E> inv;   // 1/scale in f32, or 0 for an all-zero block
  U<E> bits;  // the same scale rounded to fp16, as its 16 bits
};

template <class E, class Idx>
U<E> byte_at(E& e, const UnpackArgs<E>& a, const Idx& index) {
  return e.let(lse::math::cast<kir::u32>(a.packed[index]));
}

}  // namespace detail

// Every dialect row the codec bodies reach for. A backend checks this before
// offering the codec: `KernelBody::call` answers a missing row with an empty
// Val, which would emit silently broken source, so the gap has to be caught
// before anything is recorded.
inline constexpr std::array<std::string_view, 6> kBlockCodecSymbols{
    "abs", "round", "min", "max", lse::math::bits_key<lse::f16>,
    lse::math::value_key<lse::f16>};

// Specialization point. No primary definition: a scheme with no codec is a
// compile error here, never a fallthrough to whichever one came first.
template <class Scheme>
struct BlockCodec;

// Everything identical across the block-scaled schemes, exactly as QuantScheme
// owns it on the host: the absmax reduction, the fp16 scale in the block, and
// the round/clamp of one value. A scheme supplies only how its codes land in
// the block's remaining bytes.
template <class Derived, class Block>
struct ScaledBlockCodec {
  static constexpr std::uint32_t kBytes =
      static_cast<std::uint32_t>(sizeof(Block));
  static constexpr std::uint32_t kCodeOffset = sizeof(std::uint16_t);

  template <class E, class Idx>
  static void pack(E& e, const PackArgs<E>& a, const Idx& src, const Idx& dst) {
    const detail::Scale<E> s = scale_of(e, a, src);
    a.packed[dst] = lse::math::cast<std::uint8_t>(s.bits % 256u);
    a.packed[dst + 1u] = lse::math::cast<std::uint8_t>(s.bits / 256u);
    Derived::pack_codes(e, a, src, dst, s.inv);
  }

  template <class E, class Idx>
  static void unpack(E& e, const UnpackArgs<E>& a, const Idx& src,
                     const Idx& dst) {
    const detail::U<E> bits = e.let(detail::byte_at(e, a, src) +
                                    detail::byte_at(e, a, src + 1u) * 256u);
    const detail::F<E> scale = e.let(lse::math::from_bits<lse::f16>(bits));
    Derived::unpack_codes(e, a, src, dst, scale);
  }

  // One code as a float in [-kMaxQ, kMaxQ], rounded the way the host scheme
  // rounds it.
  template <class E, class V, class S>
  static detail::F<E> code(E& e, const V& x, const S& scale_inv) {
    return e.let(lse::math::min(
        lse::math::max(lse::math::round(x * scale_inv), e.f32(-Derived::kMaxQ)),
        e.f32(Derived::kMaxQ)));
  }

 private:
  template <class E, class Idx>
  static detail::Scale<E> scale_of(E& e, const PackArgs<E>& a,
                                   const Idx& src) {
    auto m = e.let(lse::math::abs(a.x[src]));
    for (std::uint32_t i = 1; i < kBlockElemsU; ++i) {
      m = e.let(lse::math::max(m, lse::math::abs(a.x[src + i])));
    }
    const detail::F<E> scale = e.let(m / Derived::kMaxQ);
    return {e.let(lse::math::select(scale != 0.0f, 1.0f / scale, e.f32(0.0f))),
            e.let(lse::math::bits_of<lse::f16>(scale))};
  }
};

// Q8 — 32 signed bytes after the scale. The stored byte is the code's two's
// complement, which `+256 then wrap` produces without a bitwise and; the code
// is integral so the u32 conversion is exact.
template <>
struct BlockCodec<Q8> : ScaledBlockCodec<BlockCodec<Q8>, BlockQ8> {
  static constexpr float kMaxQ = static_cast<float>(Q8::kMaxQ);

  template <class E, class Idx, class S>
  static void pack_codes(E& e, const PackArgs<E>& a, const Idx& src,
                         const Idx& dst, const S& inv) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const auto q = code(e, a.x[src + i], inv);
      a.packed[dst + kCodeOffset + i] = lse::math::cast<std::uint8_t>(
          lse::math::cast<kir::u32>(q + 256.0f) % 256u);
    }
  }

  template <class E, class Idx, class S>
  static void unpack_codes(E& e, const UnpackArgs<E>& a, const Idx& src,
                           const Idx& dst, const S& scale) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const auto byte = detail::byte_at(e, a, src + kCodeOffset + i);
      const auto q =
          e.let(lse::math::cast<kir::f32>((byte + 128u) % 256u) - 128.0f);
      a.x[dst + i] = q * scale;
    }
  }
};

// Q6 — 16 B of 4 low bits, then 8 B of 2 high bits. Codes are biased into
// [1, 63], so `code / 16` is already the 2-bit high field and needs no mask.
template <>
struct BlockCodec<Q6> : ScaledBlockCodec<BlockCodec<Q6>, BlockQ6> {
  static constexpr float kMaxQ = static_cast<float>(Q6::kMaxQ);
  static constexpr float kBias = static_cast<float>(Q6::kBias);
  static constexpr std::uint32_t kHighOffset = kCodeOffset + kBlockElemsU / 2u;

  template <class E, class Idx, class S>
  static void pack_codes(E& e, const PackArgs<E>& a, const Idx& src,
                         const Idx& dst, const S& inv) {
    detail::U<E> biased[kBlockElemsU];
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      biased[i] = e.let(
          lse::math::cast<kir::u32>(code(e, a.x[src + i], inv) + kBias));
    }
    for (std::uint32_t j = 0; j < kBlockElemsU / 2u; ++j) {
      a.packed[dst + kCodeOffset + j] = lse::math::cast<std::uint8_t>(
          biased[2 * j] % 16u + (biased[2 * j + 1] % 16u) * 16u);
    }
    for (std::uint32_t j = 0; j < kBlockElemsU / 4u; ++j) {
      const auto high = e.let(biased[4 * j] / 16u +
                              (biased[4 * j + 1] / 16u) * 4u +
                              (biased[4 * j + 2] / 16u) * 16u +
                              (biased[4 * j + 3] / 16u) * 64u);
      a.packed[dst + kHighOffset + j] = lse::math::cast<std::uint8_t>(high);
    }
  }

  template <class E, class Idx, class S>
  static void unpack_codes(E& e, const UnpackArgs<E>& a, const Idx& src,
                           const Idx& dst, const S& scale) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const std::uint32_t low_field = (i % 2 == 0) ? 1u : 16u;
      const std::uint32_t high_field = 1u << (2u * (i % 4u));
      const auto low = e.let(
          detail::byte_at(e, a, src + kCodeOffset + i / 2u) / low_field % 16u);
      const auto high = e.let(
          detail::byte_at(e, a, src + kHighOffset + i / 4u) / high_field % 4u);
      const auto q =
          e.let(lse::math::cast<kir::f32>(low + high * 16u) - kBias);
      a.x[dst + i] = q * scale;
    }
  }
};

// Q4 — 16 B of nibbles. Codes are biased into [1, 15].
template <>
struct BlockCodec<Q4> : ScaledBlockCodec<BlockCodec<Q4>, BlockQ4> {
  static constexpr float kMaxQ = static_cast<float>(Q4::kMaxQ);
  static constexpr float kBias = static_cast<float>(Q4::kBias);

  template <class E, class Idx, class S>
  static void pack_codes(E& e, const PackArgs<E>& a, const Idx& src,
                         const Idx& dst, const S& inv) {
    for (std::uint32_t j = 0; j < kBlockElemsU / 2u; ++j) {
      const auto low = code(e, a.x[src + 2u * j], inv);
      const auto high = code(e, a.x[src + 2u * j + 1u], inv);
      a.packed[dst + kCodeOffset + j] = lse::math::cast<std::uint8_t>(
          lse::math::cast<kir::u32>(low + kBias) +
          lse::math::cast<kir::u32>(high + kBias) * 16u);
    }
  }

  template <class E, class Idx, class S>
  static void unpack_codes(E& e, const UnpackArgs<E>& a, const Idx& src,
                           const Idx& dst, const S& scale) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const std::uint32_t field = (i % 2 == 0) ? 1u : 16u;
      const auto nibble = e.let(
          detail::byte_at(e, a, src + kCodeOffset + i / 2u) / field % 16u);
      const auto q = e.let(lse::math::cast<kir::f32>(nibble) - kBias);
      a.x[dst + i] = q * scale;
    }
  }
};

// f16 — the uncompressed reference. It carries no block scale, so it is not a
// ScaledBlockCodec; the block survives only so every scheme shares one
// thread-to-work mapping.
template <>
struct BlockCodec<lse::f16> {
  static constexpr std::uint32_t kBytes = kBlockElemsU * 2u;

  template <class E, class Idx>
  static void pack(E& e, const PackArgs<E>& a, const Idx& src, const Idx& dst) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const auto bits = e.let(lse::math::bits_of<lse::f16>(a.x[src + i]));
      a.packed[dst + 2u * i] = lse::math::cast<std::uint8_t>(bits % 256u);
      a.packed[dst + 2u * i + 1u] = lse::math::cast<std::uint8_t>(bits / 256u);
    }
  }

  template <class E, class Idx>
  static void unpack(E& e, const UnpackArgs<E>& a, const Idx& src,
                     const Idx& dst) {
    for (std::uint32_t i = 0; i < kBlockElemsU; ++i) {
      const auto bits = e.let(detail::byte_at(e, a, src + 2u * i) +
                              detail::byte_at(e, a, src + 2u * i + 1u) * 256u);
      a.x[dst + i] = lse::math::from_bits<lse::f16>(bits);
    }
  }
};

// One thread per block, for both directions. The two index temporaries are
// bound before the call because argument evaluation order is unspecified and
// the emitted text has to be identical run to run — it is the JIT cache key.
template <class Scheme, class E, class N>
void pack_blocks(E& e, PackArgs<E>& a, const N& nblocks) {
  using C = BlockCodec<Scheme>;
  auto b = e.thread_id();
  if (auto in_range = e.when(b < nblocks)) {
    const auto src = e.let(b * kBlockElemsU);
    const auto dst = e.let(b * C::kBytes);
    C::pack(e, a, src, dst);
  }
}

template <class Scheme, class E, class N>
void unpack_blocks(E& e, UnpackArgs<E>& a, const N& nblocks) {
  using C = BlockCodec<Scheme>;
  auto b = e.thread_id();
  if (auto in_range = e.when(b < nblocks)) {
    const auto src = e.let(b * C::kBytes);
    const auto dst = e.let(b * kBlockElemsU);
    C::unpack(e, a, src, dst);
  }
}

// Routes a runtime dtype to the compile-time-specialized codec, the same shape
// dispatch_scheme uses for the host schemes. Returns false for a dtype with no
// codec, which is how a caller declines rather than substituting one.
template <typename Fn>
bool dispatch_block_codec(DType dtype, Fn&& fn) {
  switch (dtype) {
    case DType::kF16: fn.template operator()<lse::f16>(); return true;
    case DType::kQ8: fn.template operator()<Q8>(); return true;
    case DType::kQ6: fn.template operator()<Q6>(); return true;
    case DType::kQ4: fn.template operator()<Q4>(); return true;
    default: return false;
  }
}

[[nodiscard]] constexpr bool has_block_codec(DType dtype) noexcept {
  return dtype == DType::kF16 || dtype == DType::kQ8 || dtype == DType::kQ6 ||
         dtype == DType::kQ4;
}

}  // namespace lse::quant
