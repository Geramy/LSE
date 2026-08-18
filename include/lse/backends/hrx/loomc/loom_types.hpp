// Loom's spelling of the element types a kernel can name.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "lse/graph/kernel_ir.hpp"

namespace lse::backend {

[[nodiscard]] graph::kir::TypeTable loom_types() noexcept;

// How Loom spells `s` as the ELEMENT TYPE OF STORAGE — a buffer view or a
// register vector. Distinct from TypeTable::scalar, which answers for a value:
// Loom has no unsigned scalar, so an unsigned index value is `index` while an
// unsigned word in memory is `i32` and stays 32 bits wide. See loom_types.cpp.
[[nodiscard]] std::string_view loom_storage_type(graph::kir::Scalar s) noexcept;

// `view<Nx<elem>, #dense>` — the flat one-dimensional view an LSE buffer is
// seen through. Extents are real: LSE bakes every shape into the kernel, so
// the number is known, and Loom needs it to bound the access.
[[nodiscard]] std::string loom_view_type(graph::kir::Scalar s,
                                         std::uint64_t elements);

// `vector<Nx<elem>>`.
[[nodiscard]] std::string loom_vector_type(graph::kir::Scalar s,
                                           std::uint32_t lanes);

// A float literal Loom's parser accepts: a decimal point is mandatory, and
// infinities are spelled `inf` / `-inf` rather than as C macros.
[[nodiscard]] std::string loom_float_literal(float v);

}  // namespace lse::backend
