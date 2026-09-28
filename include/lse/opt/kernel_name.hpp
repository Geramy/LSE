#pragma once

#include <array>
#include <string_view>

#include "lse/math.hpp"

namespace lse::opt {

constexpr std::string_view matrix_element_name(math::MatrixElem element) {
  using enum math::MatrixElem;
  switch (element) {
    case kF32: return "f32";
    case kF16: return "f16";
    case kBF16: return "bf16";
    case kI32: return "i32";
    case kI8: return "i8";
    case kI4: return "i4";
    case kFp8: return "fp8";
    case kBf8: return "bf8";
    case kSU8: return "su8";
  }
  return {};
}

// Resolve identity before either dialect builds registry, symbol or cache keys.
template <std::size_t N>
consteval auto specialize_kernel_name(std::string_view pattern,
                                     math::MatrixElem element) {
  constexpr std::string_view token = "(MatrixElem)";
  const auto spelling = matrix_element_name(element);
  if (spelling.empty() || pattern.size() > N)
    throw "Invalid matrix kernel identity";
  std::array<char, N + 1> result{};
  std::size_t out = 0;
  while (!pattern.empty()) {
    if (pattern.starts_with(token)) {
      for (char ch : spelling) result[out++] = ch;
      pattern.remove_prefix(token.size());
    } else {
      result[out++] = pattern.front();
      pattern.remove_prefix(1);
    }
  }
  return result;
}

}  // namespace lse::opt
