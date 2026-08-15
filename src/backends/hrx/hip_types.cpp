// The type half of what a backend must supply. Ordered by Scalar so the entry
// for a type is found by indexing, and so adding an element type without a HIP
// spelling fails to compile rather than falling through to a default.
#include "lse/backends/hrx/hip_types.hpp"
#include "lse/core/reflect.hpp"

#include <array>

namespace lse::backend {

using graph::kir::Scalar;

namespace {

struct Spelling {
  Scalar type;
  std::string_view hip;
};

constexpr std::array<Spelling, 12> kTypes{{
    {Scalar::kU8, "unsigned char"},
    {Scalar::kI8, "signed char"},
    {Scalar::kU16, "unsigned short"},
    {Scalar::kI16, "short"},
    {Scalar::kU32, "unsigned int"},
    {Scalar::kI32, "int"},
    {Scalar::kU64, "unsigned long long"},
    {Scalar::kI64, "long long"},
    {Scalar::kF16, "_Float16"},
    {Scalar::kBF16, "__hip_bfloat16"},
    {Scalar::kF32, "float"},
    {Scalar::kBool, "bool"},
}};

constexpr bool table_is_ordered() {
  for (std::size_t i = 0; i < kTypes.size(); ++i) {
    if (static_cast<std::size_t>(kTypes[i].type) != i) return false;
  }
  return true;
}
static_assert(table_is_ordered(),
              "kTypes must be indexed by Scalar: add the new element type's "
              "HIP spelling at its enumerator's position");
#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202603L
static_assert(lse::reflected_enum_count<Scalar>() == kTypes.size(),
              "kTypes is missing a Scalar enumerator");
#endif

std::string_view scalar(Scalar s) noexcept {
  const auto i = static_cast<std::size_t>(s);
  return i < kTypes.size() ? kTypes[i].hip : std::string_view{};
}

std::string vector_typedef(Scalar s, int n, std::string_view name) {
  return "typedef " + std::string(scalar(s)) + " " + std::string(name) +
         " __attribute__((ext_vector_type(" + std::to_string(n) + ")));";
}

}  // namespace

graph::kir::TypeTable hip_types() noexcept { return {scalar, vector_typedef}; }

}  // namespace lse::backend
