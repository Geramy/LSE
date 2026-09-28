#pragma once

#include <cstdint>

#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

namespace lse::dispatch {

enum class QuantMatrix : std::uint8_t { kNone, kInt8, kInt8Lds, kBF16 };

struct QuantPlan {
  const math::MatrixCoreRow* matrix = nullptr;
  QuantMatrix implementation = QuantMatrix::kNone;
  bool int8_activations = false;
  bool rotate_decode_panel = false;
  std::uint32_t decode_columns = 1;
  std::uint32_t prefill_rows = 1;
};

[[nodiscard]] QuantPlan quant_plan(const graph::KernelShapes&, bool indexed = false);
[[nodiscard]] const math::MatrixCoreRow* linear_matrix_row(const graph::KernelShapes&);

inline constexpr std::uint32_t kQ4MatrixLdsBytes = 6656;
inline constexpr std::uint32_t kTableRevision = 1;
[[nodiscard]] constexpr std::uint64_t implementation_id(std::string_view name) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (char c : name) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace lse::dispatch
