#pragma once

#include <array>
#include <string>
#include <type_traits>

#include "lse/graph/kernel_args.hpp"
#include "lse/kv/cache_dtype.hpp"
#include "lse/math/fp8.hpp"

namespace lse::kernels {

template <kv::CacheDType Format>
using KvElement = std::conditional_t<
    Format == kv::CacheDType::kF16, lse::f16,
    std::conditional_t<
        Format == kv::CacheDType::kBF16, lse::bf16,
        std::conditional_t<kv::packed_cache(Format), ir::u32, ir::f32>>>;

template <class Fn>
std::string with_kv_storage(DType dtype, float tag, Fn &&fn) {
  const auto format = kv::cache_dtype(dtype, tag);
  if (!kv::valid_storage(dtype, tag))
    return {};
  switch (format) {
  case kv::CacheDType::kF32:
    return fn.template operator()<kv::CacheDType::kF32>();
  case kv::CacheDType::kF16:
    return fn.template operator()<kv::CacheDType::kF16>();
  case kv::CacheDType::kBF16:
    return fn.template operator()<kv::CacheDType::kBF16>();
  case kv::CacheDType::kFP8:
    return fn.template operator()<kv::CacheDType::kFP8>();
  case kv::CacheDType::kBF8:
    return fn.template operator()<kv::CacheDType::kBF8>();
  }
  return {};
}

// Packed pool extents bound valid block IDs below INT32_MAX. Enter the
// 32-bit word domain before composing addresses, so packing division stays u32.
template <kv::CacheDType Format>
ir::Val<ir::u32> kv_block_index(const ir::Val<ir::f32>& block) {
  if constexpr (kv::packed_cache(Format))
    return ir::cast<ir::u32>(ir::cast<ir::i32>(block));
  else
    return ir::cast<ir::u32>(block);
}

// The vector and column stay separate when the caller already knows them.
template <kv::CacheDType Format>
ir::Val<ir::f32>
kv_load_vector(ir::env::Emit &e,
               const ir::env::In<KvElement<Format>, ir::env::Emit> &input,
               const ir::Val<ir::u32> &vector,
               const ir::Val<ir::u32> &column, std::uint32_t width) {
  if constexpr (!kv::packed_cache(Format)) {
    return math::widen(input[e.let(vector * width + column)]);
  } else {
    constexpr auto element = Format == kv::CacheDType::kFP8
                                 ? math::MatrixElem::kFp8
                                 : math::MatrixElem::kBf8;
    const auto base = e.let(vector * (width / 4u + 1u));
    const auto bits = e.let(input[base + column / 4u]);
    const auto byte = e.let(column % 4u);
    const auto value = math::unpack_fp8<element>(bits, ir::cast<ir::i32>(byte));
    const auto scale =
        e.let(math::from_bits<lse::f32>(input[base + width / 4u]));
    return value * scale;
  }
}

// Callers retain logical FP32 element indices; only this load maps physical pitch.
template <kv::CacheDType Format>
ir::Val<ir::f32>
kv_load(ir::env::Emit &e,
        const ir::env::In<KvElement<Format>, ir::env::Emit> &input,
        const ir::Val<ir::u32> &index, std::uint32_t width) {
  if constexpr (!kv::packed_cache(Format)) {
    return math::widen(input[index]);
  } else {
    return kv_load_vector<Format>(e, input, e.let(index / width),
                                  e.let(index % width), width);
  }
}

template <kv::CacheDType Format>
auto kv_load_pair(ir::env::Emit &e,
                  const ir::env::In<KvElement<Format>, ir::env::Emit> &input,
                  const ir::Val<ir::u32> &index, std::uint32_t width) {
  if constexpr (Format == kv::CacheDType::kF32)
    return e.load(input, index, 8u);
  else if constexpr (Format == kv::CacheDType::kF16 || Format == kv::CacheDType::kBF16) {
    const auto pair = e.load(input, index, 4u);
    return std::array<ir::Val<ir::f32>, 2>{math::widen(pair[0]), math::widen(pair[1])};
  } else
    return std::array<ir::Val<ir::f32>, 2>{
        kv_load<Format>(e, input, index, width),
        kv_load<Format>(e, input, index + 1u, width)};
}

} // namespace lse::kernels
