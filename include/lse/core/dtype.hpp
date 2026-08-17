// DTypeTraits<T> is the compile-time view, dtype_info(DType) the runtime one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace lse {

enum class DType : std::uint8_t {
  kF32 = 0,
  kF16,
  kBF16,   // the native storage/compute type of the lemonseed checkpoints
  kI32,
  kI8,
  kU8,
  kQ8,     // 8-bit block-quantized  (see quant/traits.hpp)
  kQ6,     // 6-bit block-quantized
  kQ4,     // 4-bit block-quantized
  // A 32-bit lane. Its one use is the packed plane of a group-affine weight
  // (quant/group_affine.hpp), where a lane holds several codes of a bit width
  // the tag cannot name. Appended rather than inserted: the numeric values
  // above are part of the JIT cache key and the stored device profile.
  kU32,
  kCount,
};

struct bfloat16_t {
  std::uint16_t bits = 0;

  bfloat16_t() = default;
  explicit bfloat16_t(float f) noexcept : bits(from_float(f)) {}

  [[nodiscard]] float to_float() const noexcept {
    std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &w, sizeof(out));
    return out;
  }

  explicit operator float() const noexcept { return to_float(); }

  // Round-to-nearest-even, matching what the training stack stores.
  static std::uint16_t from_float(float f) noexcept {
    std::uint32_t w;
    __builtin_memcpy(&w, &f, sizeof(w));
    // NaN: force a quiet NaN rather than letting rounding clear the mantissa.
    if (((w >> 23) & 0xFF) == 0xFF && (w & 0x7FFFFF) != 0) {
      return static_cast<std::uint16_t>((w >> 16) | 0x0040);
    }
    const std::uint32_t rounding_bias = 0x7FFF + ((w >> 16) & 1);
    return static_cast<std::uint16_t>((w + rounding_bias) >> 16);
  }
};

struct float16_t {
  std::uint16_t bits = 0;

  float16_t() = default;
  explicit float16_t(float f) noexcept : bits(from_float(f)) {}

  [[nodiscard]] float to_float() const noexcept;
  explicit operator float() const noexcept { return to_float(); }

  static std::uint16_t from_float(float f) noexcept;
};

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
  static constexpr DType kDType = DType::kF32;
  static constexpr std::size_t kSize = 4;
  static constexpr bool kIsFloat = true;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "f32";
};

template <>
struct DTypeTraits<float16_t> {
  static constexpr DType kDType = DType::kF16;
  static constexpr std::size_t kSize = 2;
  static constexpr bool kIsFloat = true;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "f16";
};

template <>
struct DTypeTraits<bfloat16_t> {
  static constexpr DType kDType = DType::kBF16;
  static constexpr std::size_t kSize = 2;
  static constexpr bool kIsFloat = true;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "bf16";
};

template <>
struct DTypeTraits<std::int32_t> {
  static constexpr DType kDType = DType::kI32;
  static constexpr std::size_t kSize = 4;
  static constexpr bool kIsFloat = false;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "i32";
};

template <>
struct DTypeTraits<std::int8_t> {
  static constexpr DType kDType = DType::kI8;
  static constexpr std::size_t kSize = 1;
  static constexpr bool kIsFloat = false;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "i8";
};

template <>
struct DTypeTraits<std::uint8_t> {
  static constexpr DType kDType = DType::kU8;
  static constexpr std::size_t kSize = 1;
  static constexpr bool kIsFloat = false;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "u8";
};

template <>
struct DTypeTraits<std::uint32_t> {
  static constexpr DType kDType = DType::kU32;
  static constexpr std::size_t kSize = 4;
  static constexpr bool kIsFloat = false;
  static constexpr bool kIsQuantized = false;
  static constexpr std::string_view kName = "u32";
};

struct DTypeInfo {
  DType dtype = DType::kF32;
  std::string_view name;
  // Zero for quantized types: they store block_elems values per block_bytes.
  std::size_t size_bytes = 0;
  std::size_t block_elems = 1;
  std::size_t block_bytes = 0;
  bool is_float = false;
  bool is_quantized = false;
};

const DTypeInfo& dtype_info(DType dtype) noexcept;
std::string_view to_string(DType dtype) noexcept;

// Returns kCount on failure.
DType dtype_from_string(std::string_view name) noexcept;

// Returns 0 when a quantized `count` is not a multiple of the block size.
std::size_t dtype_storage_bytes(DType dtype, std::size_t count) noexcept;

// Block geometry of the quantized dtypes. These live in core because the dtype
// table needs them; lse::quant static_asserts its own layouts against them, so
// the two cannot drift. Without this, core would have to include quant, which
// includes core.
inline constexpr std::size_t kQuantBlockElems = 32;
inline constexpr std::size_t kQuantBlockBytesQ8 = 34;  // fp16 scale + 32 int8
inline constexpr std::size_t kQuantBlockBytesQ6 = 26;  // scale + 16 low + 8 high
inline constexpr std::size_t kQuantBlockBytesQ4 = 18;  // scale + 16 packed

// True only for the block schemes, whose geometry is a property of the tag:
// dtype_storage_bytes can answer for them from block_elems/block_bytes alone.
// kU32 is deliberately excluded. A group-affine weight's bit width and group
// size are per tensor — a mixed-precision checkpoint overrides them by tensor
// path — so no tag could size it, and its scales and biases are separate
// tensors besides. That geometry lives in quant::GroupAffine, beside the
// plane; the plane itself really is just 32-bit lanes.
constexpr bool is_quantized(DType dtype) noexcept {
  return dtype == DType::kQ8 || dtype == DType::kQ6 || dtype == DType::kQ4;
}

}  // namespace lse
