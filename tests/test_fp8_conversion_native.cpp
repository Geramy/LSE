#include "harness.hpp"
#include "fp8_conversion_fixture.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include <fstream>
namespace {
template<lse::math::MatrixElem E> void compile_format() {
  lse::backend::LoomcCompiler compiler;
  LSE_EXPECT(compiler.available());
  if (!compiler.available()) return;
  for (unsigned byte = 0; byte != 4; ++byte) {
    auto body = fp8_fixture::body<E>(byte);
    LSE_EXPECT(body.ok());
    if (!body.ok()) { std::fprintf(stderr, "%s\n", body.status().to_string().c_str()); continue; }
    const auto source = dot_fixture::kernel(*body);
    auto code = compiler.compile(source, "gfx1201");
    LSE_EXPECT(code.ok());
    if (!code.ok()) std::fprintf(stderr, "%s\n%s\n", code.status().to_string().c_str(), source.c_str());
    else {
      LSE_EXPECT(!code->code.empty());
      if (const char* dir = std::getenv("LSE_FP8_ARTIFACT_DIR")) {
        const std::string base = std::string(dir) + "/" + (E == lse::math::MatrixElem::kFp8 ? "fp8" : "bf8") + std::to_string(byte);
        std::ofstream(base + ".loom") << source;
        std::ofstream f(base + ".hsaco", std::ios::binary);
        f.write(reinterpret_cast<const char*>(code->code.data()), code->code.size());
      }
    }
  }
  auto hip = lse::backend::hip_sources().find(lse::math::Fp8Format<E>::pack_key);
  LSE_EXPECT(hip.find("__builtin_amdgcn_cvt_pk_") != std::string_view::npos);
  LSE_EXPECT(hip.find("__builtin_isfinite") != std::string_view::npos);
}
}
LSE_TEST(ocp_fp8_shared_pack_unpack_emit_native_gfx1201) { compile_format<lse::math::MatrixElem::kFp8>(); }
LSE_TEST(ocp_bf8_shared_pack_unpack_emit_native_gfx1201) { compile_format<lse::math::MatrixElem::kBf8>(); }
LSE_TEST_MAIN()
