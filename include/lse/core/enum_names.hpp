// One declaration list per enum, generating both the enumerators and their
// names.
//
// A hand-written `switch` in a to_string() is a second place to remember: add
// an enumerator, forget the case, and the only symptom is "unknown" in a log or
// a cache key. Here the list is the single source, so the two cannot drift.
//
//   #define MY_KIND_LIST(X)  X(kRead, "read") X(kWrite, "write")
//   LSE_DECLARE_ENUM(MyKind, std::uint8_t, MY_KIND_LIST);
//
// to_string(MyKind::kRead) then works with no further code, found by ADL from
// the enum's own namespace. Adding a third entry to the list is the whole edit.
#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <utility>

#define LSE_ENUM_ENUMERATOR_(name_, text_) name_,
// `E` is the alias the generated lambda body introduces below.
#define LSE_ENUM_ENTRY_(name_, text_) \
  std::pair<E, std::string_view>{E::name_, text_},

#define LSE_DECLARE_ENUM(Enum_, Underlying_, List_)                        \
  enum class Enum_ : Underlying_ { List_(LSE_ENUM_ENUMERATOR_) };          \
                                                                           \
  inline constexpr auto kEnumEntries_##Enum_ = [] {                        \
    using E = Enum_;                                                       \
    return std::array{List_(LSE_ENUM_ENTRY_)};                             \
  }();                                                                     \
                                                                           \
  [[nodiscard]] constexpr std::string_view to_string(Enum_ value) noexcept { \
    for (const auto& entry : kEnumEntries_##Enum_) {                       \
      if (entry.first == value) return entry.second;                       \
    }                                                                      \
    return "unknown";                                                      \
  }                                                                        \
                                                                           \
  [[nodiscard]] constexpr std::size_t enum_count(Enum_) noexcept {         \
    return kEnumEntries_##Enum_.size();                                    \
  }
