// The Loom dialect's spellings for the built-in primitives.
#pragma once

#include <span>
#include <string>
#include <string_view>

#include "lse/graph/dialect_source.hpp"

namespace lse::backend {

[[nodiscard]] graph::DialectSourceTable loom_sources() noexcept;

// The Loom type of `$r` for a row of that table. Loom statements annotate their
// OPERAND types, so the result type cannot be read back off the template — a
// two-result shuffle annotates three operand types and none of them is what it
// returns. Empty when the primitive has no row.
[[nodiscard]] std::string_view loom_result_type(
    std::string_view primitive) noexcept;

// Splice a row: `$0`..`$9` are operand SSA names, `$a0`..`$a3` attr SSA names,
// `$t0`..`$t15` fresh temporaries, `$r` the result. Unlike ir::substitute this
// does NOT parenthesize — Loom operands are names, and `(%v3)` is not syntax.
[[nodiscard]] std::string loom_splice(std::string_view tmpl,
                                      std::span<const std::string> args,
                                      std::span<const std::string> attrs,
                                      std::string_view result,
                                      std::string_view temp_prefix);

}  // namespace lse::backend
