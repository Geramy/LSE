#include "harness.hpp"
#include "lse/backends/hrx/loaded_library.hpp"
#include <cstring>
namespace {
const char *fixture_path = nullptr;
}
LSE_TEST(loaded_runtime_absence_never_loads_the_requested_library) {
  LSE_EXPECT(dlsym(RTLD_DEFAULT, "lse_runtime_fixture_anchor") == nullptr);
  void *loaded = lse::backend::detail::open_loaded_library(
      fixture_path, "lse_runtime_fixture_anchor");
  LSE_EXPECT(loaded == nullptr);
  LSE_EXPECT(dlsym(RTLD_DEFAULT, "lse_runtime_fixture_anchor") == nullptr);
  if (loaded)
    dlclose(loaded);
}
LSE_TEST(loaded_runtime_resolves_the_existing_versioned_image) {
  void *first = dlopen(fixture_path, RTLD_NOW | RTLD_GLOBAL);
  LSE_EXPECT(first != nullptr);
  if (!first)
    return;
  void *through_name = lse::backend::detail::open_loaded_library(
      "liblse_runtime_fixture.dylib", "lse_runtime_fixture_anchor");
  LSE_EXPECT(through_name != nullptr);
  if (!through_name) {
    dlclose(first);
    return;
  }
  LSE_EXPECT(through_name == first);
  for (const char *name :
       {"lse_runtime_fixture_anchor", "lse_runtime_fixture_copy"})
    LSE_EXPECT(dlsym(through_name, name) == dlsym(first, name));
  // A missing search alias still resolves only the already-resident image.
  void *alias = lse::backend::detail::open_loaded_library(
      "liblse_missing_runtime_alias.dylib", "lse_runtime_fixture_anchor");
  LSE_EXPECT(alias == first);
  if (alias)
    dlclose(alias);
  dlclose(first);
  // The resolver holds its own reference, independent of the original loader.
  auto call = reinterpret_cast<int (*)()>(
      dlsym(through_name, "lse_runtime_fixture_copy"));
  LSE_EXPECT(call != nullptr && call() == 17);
  dlclose(through_name);
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  fixture_path = argv[1];
  return lse::test::run_all();
}
