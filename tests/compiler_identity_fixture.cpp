#ifndef LSE_COMPILER_FIXTURE_VALUE
#define LSE_COMPILER_FIXTURE_VALUE 1
#endif
extern "C" __attribute__((visibility("default"))) int
lse_compiler_identity_anchor() {
  return LSE_COMPILER_FIXTURE_VALUE;
}
