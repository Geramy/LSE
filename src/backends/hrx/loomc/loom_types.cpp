// The type half of what the Loom generator supplies, mirroring hipc/hip_types.
//
// Two things make this table more than a column swap of the HIP one.
//
// Loom has thirteen scalar kinds and NONE of them is unsigned: signedness is a
// property of the operation (`shrui`, `divui`, `extui`, `cmp ult`, `fptoui`),
// not of the type. LSE types every index as `Scalar::kU32`, so `kU32` answers
// `index` here — Loom's address type, 64-bit, which is what every thread id,
// loop variable and extent in this engine actually is. The other role `kU32`
// plays, a 32-bit packed weight word whose exact width is the on-disk format,
// must stay 32 bits wide; that answer is `loom_storage_type`, and the printer
// picks between them by where the value came from rather than by guessing.
//
// And Loom has no typedef: a vector is spelled inline as `vector<4xf32>`. So
// `vector_typedef` introduces nothing and the name it returns is the type
// itself, which is exactly the seam TypeTable already draws.
#include "lse/backends/hrx/loomc/loom_types.hpp"

#include <array>
#include <cmath>
#include <cstdio>

#include "lse/core/reflect.hpp"

namespace lse::backend {

using graph::kir::Scalar;

namespace {

struct Spelling {
  Scalar type;
  // The value form: what an SSA value of this element type is.
  std::string_view value;
  // The storage form: what one element of a buffer or vector is.
  std::string_view storage;
};

constexpr std::array<Spelling, 12> kTypes{{
    {Scalar::kU8, "i8", "i8"},
    {Scalar::kI8, "i8", "i8"},
    {Scalar::kU16, "i16", "i16"},
    {Scalar::kI16, "i16", "i16"},
    // The index role. A 32-bit wrap is not reproduced, and that is deliberate:
    // every kU32 value the recorder mints outside a memory read is an address
    // or a bound, and widening it removes the wrap question rather than
    // changing an answer. The word role is the storage column.
    {Scalar::kU32, "index", "i32"},
    {Scalar::kI32, "i32", "i32"},
    {Scalar::kU64, "i64", "i64"},
    {Scalar::kI64, "i64", "i64"},
    {Scalar::kF16, "f16", "f16"},
    {Scalar::kBF16, "bf16", "bf16"},
    {Scalar::kF32, "f32", "f32"},
    {Scalar::kBool, "i1", "i1"},
}};

constexpr bool table_is_ordered() {
  for (std::size_t i = 0; i < kTypes.size(); ++i) {
    if (static_cast<std::size_t>(kTypes[i].type) != i) return false;
  }
  return true;
}
static_assert(table_is_ordered(),
              "kTypes must be indexed by Scalar: add the new element type's "
              "Loom spelling at its enumerator's position");
#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202506L
static_assert(lse::reflected_enum_count<Scalar>() == kTypes.size(),
              "kTypes is missing a Scalar enumerator");
#endif

std::string_view scalar(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kTypes.size() ? kTypes[i].value : std::string_view{};
}

std::string vector_typedef(Scalar s, int n, std::string_view name) {
  (void)s;
  (void)n;
  (void)name;
  // No declaration: Body::vector_typedef records this and hands the caller the
  // name back, and for Loom the name is already the type.
  return {};
}

}  // namespace

graph::kir::TypeTable loom_types() noexcept { return {scalar, vector_typedef}; }

std::string_view loom_storage_type(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kTypes.size() ? kTypes[i].storage : std::string_view{};
}

std::string loom_view_type(Scalar s, std::uint64_t elements) {
  return "view<" + std::to_string(elements) + "x" +
         std::string(loom_storage_type(s)) + ", #dense>";
}

std::string loom_vector_type(Scalar s, std::uint32_t lanes) {
  return "vector<" + std::to_string(lanes) + "x" +
         std::string(loom_storage_type(s)) + ">";
}

std::string loom_float_literal(float v) {
  if (std::isinf(v)) return v < 0.0f ? "-inf" : "inf";
  if (std::isnan(v)) return "nan";
  // Nine significant digits round-trips a float exactly. The decimal point is
  // not optional: without it the parser reads an integer attribute and the
  // result type check fails.
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
  std::string out(buf);
  if (out.find('.') == std::string::npos &&
      out.find('e') == std::string::npos &&
      out.find("inf") == std::string::npos) {
    out += ".0";
  }
  return out;
}

}  // namespace lse::backend
