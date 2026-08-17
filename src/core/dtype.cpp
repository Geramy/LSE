#include "lse/core/dtype.hpp"

#include <array>
#include <cstring>


namespace lse {

float float16_t::to_float() const noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  const std::uint32_t exp = (bits >> 10) & 0x1Fu;
  const std::uint32_t mant = bits & 0x3FFu;

  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;  // +/- zero
    } else {
      // Subnormal: renormalize into a normal fp32.
      std::uint32_t e = 0;
      std::uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FFu;
      out = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000u | (mant << 13);  // inf / NaN
  } else {
    out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

std::uint16_t float16_t::from_float(float f) noexcept {
  std::uint32_t w;
  std::memcpy(&w, &f, sizeof(w));

  const std::uint16_t sign = static_cast<std::uint16_t>((w >> 16) & 0x8000u);
  const std::int32_t exp = static_cast<std::int32_t>((w >> 23) & 0xFFu) - 127;
  std::uint32_t mant = w & 0x7FFFFFu;

  if (exp == 128) {  // inf / NaN
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  }
  if (exp > 15) {  // overflow -> inf
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (exp < -14) {
    // Subnormal or underflow to zero.
    if (exp < -25) return sign;
    mant |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(-exp - 14 + 13);
    const std::uint32_t half = 1u << (shift - 1);
    const std::uint32_t rounded = (mant + half) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  // Normal: round-to-nearest-even on the 13 dropped mantissa bits.
  const std::uint32_t rounding_bias = 0x0FFFu + ((mant >> 13) & 1u);
  mant += rounding_bias;
  std::int32_t e = exp;
  if (mant & 0x800000u) {  // rounding carried into the exponent
    mant = 0;
    ++e;
    if (e > 15) return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  const std::uint32_t biased_exp = static_cast<std::uint32_t>(e + 15);
  return static_cast<std::uint16_t>(sign | (biased_exp << 10) | (mant >> 13));
}

namespace {

constexpr std::size_t kNumDTypes = static_cast<std::size_t>(DType::kCount);

const std::array<DTypeInfo, kNumDTypes>& table() {
  static const std::array<DTypeInfo, kNumDTypes> kTable = {{
      {DType::kF32,  "f32",  4, 1, 4, true,  false},
      {DType::kF16,  "f16",  2, 1, 2, true,  false},
      {DType::kBF16, "bf16", 2, 1, 2, true,  false},
      {DType::kI32,  "i32",  4, 1, 4, false, false},
      {DType::kI8,   "i8",   1, 1, 1, false, false},
      {DType::kU8,   "u8",   1, 1, 1, false, false},
      {DType::kQ8,   "q8",   0, kQuantBlockElems, kQuantBlockBytesQ8, false, true},
      {DType::kQ6,   "q6",   0, kQuantBlockElems, kQuantBlockBytesQ6, false, true},
      {DType::kQ4,   "q4",   0, kQuantBlockElems, kQuantBlockBytesQ4, false, true},
      {DType::kU32,  "u32",  4, 1, 4, false, false},
  }};
  return kTable;
}

}  // namespace

const DTypeInfo& dtype_info(DType dtype) noexcept {
  const auto idx = static_cast<std::size_t>(dtype);
  return table()[idx < kNumDTypes ? idx : 0];
}

std::string_view to_string(DType dtype) noexcept { return dtype_info(dtype).name; }

DType dtype_from_string(std::string_view name) noexcept {
  for (const auto& info : table()) {
    if (info.name == name) return info.dtype;
  }
  // Accept the spellings the checkpoints use.
  if (name == "bfloat16") return DType::kBF16;
  if (name == "float16" || name == "half") return DType::kF16;
  if (name == "float32" || name == "float") return DType::kF32;
  return DType::kCount;
}

std::size_t dtype_storage_bytes(DType dtype, std::size_t count) noexcept {
  const DTypeInfo& info = dtype_info(dtype);
  if (!info.is_quantized) return info.size_bytes * count;
  if (info.block_elems == 0 || count % info.block_elems != 0) return 0;
  return (count / info.block_elems) * info.block_bytes;
}

}  // namespace lse
