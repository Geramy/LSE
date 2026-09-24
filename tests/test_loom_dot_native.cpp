#include "harness.hpp"
#include "loom_dot_fixture.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
namespace {
void compile_fixture(unsigned kind) {
 lse::backend::LoomcCompiler compiler;
 LSE_EXPECT(compiler.available()); if (!compiler.available()) return;
 auto body = dot_fixture::body(kind);
 LSE_EXPECT(body.ok()); if (!body.ok()) return;
 // This native compiler build links the RDNA4 descriptor set. Other
 // architecture capability gates are checked by test_loom_dot.
 auto code = compiler.compile(dot_fixture::kernel(*body), "gfx1201");
 LSE_EXPECT(code.ok());
 if (!code.ok()) std::fprintf(stderr, "fixture%u: %s\n", kind, code.status().to_string().c_str());
 else LSE_EXPECT(!code->code.empty());
}
}
LSE_TEST(signed_mixed_dot_and_ties_even_rounding_emit_shader_bytes) {
 compile_fixture(0);
 compile_fixture(1);
}
LSE_TEST(workgroup_uniform_unsigned_minmax_emit_shader_bytes) {
 compile_fixture(2);
}
LSE_TEST_MAIN()
