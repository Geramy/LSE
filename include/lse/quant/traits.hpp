// Block quantization schemes: symmetric, block-scaled, fp16 scale per block.
// Symmetric (no zero-point) keeps dequant to a single multiply, which matters
// because dequant sits in the GEMM inner loop.
//
// QuantScheme<Derived, Block> owns everything identical across schemes; a new
// scheme supplies only pack_block, unpack_block and block_absmax_to_scale.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lse/core/dtype.hpp"

namespace lse::quant {

// Every scheme in this family uses 32-element blocks. Kept as a named constant
// because the HIP kernels hard-code it into their LDS tiling.
inline constexpr std::size_t kBlockElems = kQuantBlockElems;

// Memory-mapped straight out of model files: the static_asserts below are the
// on-disk ABI.
#pragma pack(push, 1)

// 8-bit: one int8 per weight. 34 B / 32 w = 8.5 bpw.
struct BlockQ8 {
  std::uint16_t scale;               // fp16 scale
  std::int8_t quants[kBlockElems];   // [-127, 127]
};

// 6-bit: 4 low bits packed two-per-byte, plus 2 high bits packed four-per-byte.
// 26 B / 32 w = 6.5 bpw.
struct BlockQ6 {
  std::uint16_t scale;                    // fp16 scale
  std::uint8_t low[kBlockElems / 2];      // 4 bits each
  std::uint8_t high[kBlockElems / 4];     // 2 bits each
};

// 4-bit: two weights per byte. 18 B / 32 w = 4.5 bpw.
struct BlockQ4 {
  std::uint16_t scale;                    // fp16 scale
  std::uint8_t quants[kBlockElems / 2];   // 4 bits each
};

#pragma pack(pop)

// File-format ABI, and the numbers core's dtype table reports.
static_assert(sizeof(BlockQ8) == kQuantBlockBytesQ8, "BlockQ8 layout is ABI");
static_assert(sizeof(BlockQ6) == kQuantBlockBytesQ6, "BlockQ6 layout is ABI");
static_assert(sizeof(BlockQ4) == kQuantBlockBytesQ4, "BlockQ4 layout is ABI");

// Block is a separate parameter, not `typename Derived::Block`: Derived is
// incomplete while its own base is being instantiated.
template <typename Derived, typename BlockT>
struct QuantScheme {
  using Block = BlockT;

  static constexpr DType dtype() noexcept { return Derived::kDType; }
  static constexpr std::size_t block_elems() noexcept { return kBlockElems; }
  static constexpr std::size_t block_bytes() noexcept { return sizeof(Block); }

  static constexpr double bits_per_weight() noexcept {
    return 8.0 * static_cast<double>(sizeof(Block)) /
           static_cast<double>(kBlockElems);
  }

  // Storage for `count` weights. Requires count % kBlockElems == 0.
  static constexpr std::size_t storage_bytes(std::size_t count) noexcept {
    return (count / kBlockElems) * sizeof(Block);
  }

  // `count` must be a multiple of kBlockElems; the loader pads rows.
  static void quantize_row(const float* src, Block* dst, std::size_t count) {
    const std::size_t nblocks = count / kBlockElems;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* s = src + b * kBlockElems;
      float absmax = 0.0f;
      for (std::size_t i = 0; i < kBlockElems; ++i) {
        absmax = std::max(absmax, std::fabs(s[i]));
      }
      const float scale = Derived::block_absmax_to_scale(absmax);
      dst[b].scale = float16_t::from_float(scale);
      const float scale_inv = (scale != 0.0f) ? 1.0f / scale : 0.0f;
      Derived::pack_block(s, scale_inv, &dst[b]);
    }
  }

  static void dequantize_row(const Block* src, float* dst, std::size_t count) {
    const std::size_t nblocks = count / kBlockElems;
    for (std::size_t b = 0; b < nblocks; ++b) {
      Derived::unpack_block(src[b], dst + b * kBlockElems);
    }
  }

  static double round_trip_rmse(const float* src, std::size_t count,
                                float* scratch, Block* blocks) {
    quantize_row(src, blocks, count);
    dequantize_row(blocks, scratch, count);
    double acc = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double d = static_cast<double>(src[i]) - static_cast<double>(scratch[i]);
      acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(count));
  }
};

struct Q8 : QuantScheme<Q8, BlockQ8> {
  using Block = BlockQ8;
  static constexpr DType kDType = DType::kQ8;
  static constexpr int kMaxQ = 127;

  static float block_absmax_to_scale(float absmax) noexcept {
    return absmax / static_cast<float>(kMaxQ);
  }

  static void pack_block(const float* src, float scale_inv, Block* dst) noexcept {
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      const float q = std::round(src[i] * scale_inv);
      dst->quants[i] = static_cast<std::int8_t>(
          std::clamp(q, static_cast<float>(-kMaxQ), static_cast<float>(kMaxQ)));
    }
  }

  static void unpack_block(const Block& src, float* dst) noexcept {
    float16_t h;
    h.bits = src.scale;
    const float s = h.to_float();
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      dst[i] = static_cast<float>(src.quants[i]) * s;
    }
  }
};

struct Q6 : QuantScheme<Q6, BlockQ6> {
  using Block = BlockQ6;
  static constexpr DType kDType = DType::kQ6;
  static constexpr int kMaxQ = 31;   // signed 6-bit: [-31, 31]
  static constexpr int kBias = 32;   // stored biased into [1, 63]

  static float block_absmax_to_scale(float absmax) noexcept {
    return absmax / static_cast<float>(kMaxQ);
  }

  static void pack_block(const float* src, float scale_inv, Block* dst) noexcept {
    std::memset(dst->low, 0, sizeof(dst->low));
    std::memset(dst->high, 0, sizeof(dst->high));
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      const float q = std::round(src[i] * scale_inv);
      const int qi = static_cast<int>(std::clamp(
                         q, static_cast<float>(-kMaxQ),
                         static_cast<float>(kMaxQ))) + kBias;  // [1, 63]
      const std::uint8_t u = static_cast<std::uint8_t>(qi);
      dst->low[i / 2] |= static_cast<std::uint8_t>((u & 0x0F) << (4 * (i % 2)));
      dst->high[i / 4] |= static_cast<std::uint8_t>(((u >> 4) & 0x03) << (2 * (i % 4)));
    }
  }

  static void unpack_block(const Block& src, float* dst) noexcept {
    float16_t h;
    h.bits = src.scale;
    const float s = h.to_float();
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      const std::uint8_t lo = (src.low[i / 2] >> (4 * (i % 2))) & 0x0F;
      const std::uint8_t hi = (src.high[i / 4] >> (2 * (i % 4))) & 0x03;
      const int q = static_cast<int>(lo | (hi << 4)) - kBias;
      dst[i] = static_cast<float>(q) * s;
    }
  }
};

struct Q4 : QuantScheme<Q4, BlockQ4> {
  using Block = BlockQ4;
  static constexpr DType kDType = DType::kQ4;
  static constexpr int kMaxQ = 7;   // signed 4-bit: [-7, 7]
  static constexpr int kBias = 8;   // stored biased into [1, 15]

  static float block_absmax_to_scale(float absmax) noexcept {
    return absmax / static_cast<float>(kMaxQ);
  }

  static void pack_block(const float* src, float scale_inv, Block* dst) noexcept {
    std::memset(dst->quants, 0, sizeof(dst->quants));
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      const float q = std::round(src[i] * scale_inv);
      const int qi = static_cast<int>(std::clamp(
                         q, static_cast<float>(-kMaxQ),
                         static_cast<float>(kMaxQ))) + kBias;  // [1, 15]
      dst->quants[i / 2] |=
          static_cast<std::uint8_t>((qi & 0x0F) << (4 * (i % 2)));
    }
  }

  static void unpack_block(const Block& src, float* dst) noexcept {
    float16_t h;
    h.bits = src.scale;
    const float s = h.to_float();
    for (std::size_t i = 0; i < kBlockElems; ++i) {
      const int q =
          static_cast<int>((src.quants[i / 2] >> (4 * (i % 2))) & 0x0F) - kBias;
      dst[i] = static_cast<float>(q) * s;
    }
  }
};

// Routes a runtime dtype tag to the compile-time-specialized scheme.
template <typename Fn>
auto dispatch_scheme(DType dtype, Fn&& fn) {
  switch (dtype) {
    case DType::kQ8: return fn.template operator()<Q8>();
    case DType::kQ6: return fn.template operator()<Q6>();
    case DType::kQ4: return fn.template operator()<Q4>();
    default:         return fn.template operator()<Q8>();
  }
}

}  // namespace lse::quant
