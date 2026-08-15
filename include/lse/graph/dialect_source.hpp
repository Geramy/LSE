// How a primitive is spelled in one backend's source dialect.
//
// A primitive is a graph concept — `silu` means the same thing everywhere. The
// text `$0 / (1.0f + __expf(-$0))` is not: it is HIP, and belongs to whichever
// backend emits HIP. Each emitter publishes one table keyed by primitive name,
// so a second backend is a new table rather than an edit to every primitive.
//
// A primitive may still carry its own per-dialect source (see DialectExpr) when
// it is written against one target's intrinsics; that takes precedence, which
// is what lets a hand-authored kernel primitive live outside any backend table.
#pragma once

#include <cstdint>

#include "lse/core/enum_names.hpp"
#include <span>
#include <string_view>

namespace lse::graph {

// The source language a kernel is written in. A primitive declares which of
// these it can emit; an emitter declares the one it consumes. Nothing in the
// graph layer assumes a particular value.
#define LSE_DIALECT_LIST(X) \
  X(kHip, "hip")            \
  X(kCuda, "cuda")          \
  X(kSpirv, "spirv")        \
  X(kMetal, "metal")

LSE_DECLARE_ENUM(Dialect, std::uint8_t, LSE_DIALECT_LIST)

struct PrimitiveSource {
  std::string_view primitive;
  std::string_view expr;
};

// Non-owning view over a backend's static table.
class DialectSourceTable {
 public:
  constexpr DialectSourceTable() = default;
  constexpr explicit DialectSourceTable(
      std::span<const PrimitiveSource> entries) noexcept
      : entries_(entries) {}

  // Empty when this dialect has no spelling for `primitive`.
  [[nodiscard]] constexpr std::string_view find(
      std::string_view primitive) const noexcept {
    for (const PrimitiveSource& e : entries_) {
      if (e.primitive == primitive) return e.expr;
    }
    return {};
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return entries_.size();
  }

 private:
  std::span<const PrimitiveSource> entries_;
};

// A primitive that supplies its own source states which dialect it is for.
struct DialectExpr {
  Dialect dialect = Dialect::kHip;
  std::string_view expr;
};

[[nodiscard]] constexpr std::string_view expr_for(
    std::span<const DialectExpr> sources, Dialect dialect) noexcept {
  for (const DialectExpr& s : sources) {
    if (s.dialect == dialect) return s.expr;
  }
  return {};
}

}  // namespace lse::graph
