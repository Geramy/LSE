// Host-only P2996 helpers. Empty unless the TU is compiled with -freflection
// (GCC 16+). clangd without that flag never sees <meta> or ^^.
#pragma once

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202506L

#include <cstddef>
#include <meta>
#include <string_view>

namespace lse {

template <typename E>
consteval std::size_t reflected_enum_count() {
  return std::define_static_array(std::meta::enumerators_of(^^E)).size();
}

template <typename E>
constexpr std::string_view reflected_enumerator_name(E value) {
  constexpr static auto enumerators =
      std::define_static_array(std::meta::enumerators_of(^^E));
  template for (constexpr auto e : enumerators) {
    if (value == [:e:]) return std::meta::identifier_of(e);
  }
  return {};
}

}  // namespace lse

#endif
