#pragma once
#include <dlfcn.h>

namespace lse::backend::detail {
// Acquire a reference to an existing image only. Darwin's RTLD_NOLOAD lookup
// need not resolve the search-path alias used to load a versioned dylib. A
// globally visible anchor identifies its actual resident path without loading
// another runtime or bypassing an already-installed interposer.
inline void *open_loaded_library(const char *name,
                                 const char *anchor) noexcept {
  if (void *library = dlopen(name, RTLD_LAZY | RTLD_NOLOAD))
    return library;
#if defined(__APPLE__)
  Dl_info image{};
  void *symbol = dlsym(RTLD_DEFAULT, anchor);
  if (symbol != nullptr && dladdr(symbol, &image) != 0 &&
      image.dli_fname != nullptr)
    return dlopen(image.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
#else
  (void)anchor;
#endif
  return nullptr;
}
} // namespace lse::backend::detail
