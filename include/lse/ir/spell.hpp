// How a value is spelled in generated source.
//
// Two things every level of the compiler needs and neither the graph nor a
// backend owns: a float that survives a round trip through text, and the
// `$0`-style template splice a dialect row is written in.
#pragma once

#include <span>
#include <string>
#include <string_view>

namespace lse::ir {

// A float as a source literal. std::to_string is %.6f, which turns 1e-7f into
// "0.000000"; streaming without showpoint turns 1.0f into the invalid "1f".
[[nodiscard]] std::string float_literal(float v);

[[nodiscard]] std::string literal_u32(unsigned int v);
[[nodiscard]] std::string literal_i32(int v);

// Splice `args` into a dialect row. "$N" is input N, parenthesized so a
// template like "$0 * $0" cannot re-associate; "$aN" is attrs[N] as a literal.
[[nodiscard]] std::string substitute(std::string_view tmpl,
                                     std::span<const std::string> args);
[[nodiscard]] std::string substitute(std::string_view tmpl,
                                     std::span<const std::string> args,
                                     std::span<const float> attrs);

}  // namespace lse::ir
