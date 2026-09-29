#pragma once

#include <cstdint>
#include <string_view>

#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::kv {

enum class CacheDType : std::uint8_t { kF32, kF16, kBF16, kFP8, kBF8 };

[[nodiscard]] constexpr std::string_view to_string(CacheDType format) noexcept {
  switch (format) {
  case CacheDType::kF32:
    return "fp32";
  case CacheDType::kF16:
    return "fp16";
  case CacheDType::kBF16:
    return "bf16";
  case CacheDType::kFP8:
    return "fp8";
  case CacheDType::kBF8:
    return "bf8";
  }
  return "invalid";
}

[[nodiscard]] inline Result<CacheDType>
cache_dtype_from_string(std::string_view name) {
  if (name == "fp32" || name == "f32")
    return CacheDType::kF32;
  if (name == "fp16" || name == "f16")
    return CacheDType::kF16;
  if (name == "bf16" || name == "bfloat16")
    return CacheDType::kBF16;
  if (name == "fp8" || name == "e4m3" || name == "fp8_e4m3")
    return CacheDType::kFP8;
  if (name == "bf8" || name == "e5m2" || name == "fp8_e5m2")
    return CacheDType::kBF8;
  return LSE_ERROR(
      kInvalidArgument,
      "KV cache dtype must be fp32, fp16, bf16, fp8 (E4M3), or bf8 (E5M2)");
}

[[nodiscard]] constexpr bool packed_cache(CacheDType format) noexcept {
  return format == CacheDType::kFP8 || format == CacheDType::kBF8;
}

[[nodiscard]] constexpr DType storage_dtype(CacheDType format) noexcept {
  switch (format) {
  case CacheDType::kF32:
    return DType::kF32;
  case CacheDType::kF16:
    return DType::kF16;
  case CacheDType::kBF16:
    return DType::kBF16;
  case CacheDType::kFP8:
  case CacheDType::kBF8:
    return DType::kU32;
  }
  return DType::kCount;
}

// Each FP8 token/head vector stores packed bytes followed by one FP32 scale
// word.
[[nodiscard]] constexpr std::int64_t
storage_width(CacheDType format, std::int64_t width) noexcept {
  return packed_cache(format)
             ? (width > 0 && width % 4 == 0 ? width / 4 + 1 : 0)
             : width;
}
[[nodiscard]] constexpr std::int64_t
logical_width(CacheDType format, std::int64_t pitch) noexcept {
  return packed_cache(format) ? (pitch > 1 ? (pitch - 1) * 4 : 0) : pitch;
}

[[nodiscard]] constexpr bool valid_storage(DType storage, float tag) noexcept {
  switch (storage) {
  case DType::kF32:
    return tag == 0.0f;
  case DType::kF16:
    return tag == 0.0f || tag == 1.0f;
  case DType::kBF16:
    return tag == 0.0f || tag == 2.0f;
  case DType::kU32:
    return tag == 3.0f || tag == 4.0f;
  default:
    return false;
  }
}

[[nodiscard]] constexpr CacheDType cache_dtype(DType storage,
                                               float tag = 0.0f) noexcept {
  if (storage == DType::kF16)
    return CacheDType::kF16;
  if (storage == DType::kBF16)
    return CacheDType::kBF16;
  if (storage == DType::kU32 && tag == 3.0f)
    return CacheDType::kFP8;
  if (storage == DType::kU32 && tag == 4.0f)
    return CacheDType::kBF8;
  return CacheDType::kF32;
}

} // namespace lse::kv
