// A host-only runtime image: no HSA entry points or device initialization.
extern "C" __attribute__((visibility("default"))) int
lse_runtime_fixture_anchor() {
  return 42;
}
extern "C" __attribute__((visibility("default"))) int
lse_runtime_fixture_copy() {
  return 17;
}
