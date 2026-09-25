// Packed-word aliases must not turn shared address constants into i32 values.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/ir/env.hpp"
#include "lse/ir/verify.hpp"

namespace ir = lse::ir;
namespace env = lse::ir::env;

LSE_TEST(packed_words_through_bind_aliases_leave_shared_index_constants_typed) {
  const auto types = lse::backend::loom_types();
  const ir::DialectSourceTable intrinsics{std::span<const ir::PrimitiveSource>{}};
  ir::KernelBody body(types, intrinsics);
  env::Emit e{&body};
  ir::Buffer<ir::u32> packed(&body, &types, "packed");
  const auto at = e.let(e.thread_id() % 8u);
  const auto word = e.let(packed[at].read());
  const auto alias = e.let(word);
  // The same 6u is an index multiplier, a word arithmetic operand, and a loop
  // bound. Conversions belong at word uses, not at the constant definition.
  const auto divisor = e.let((at * 6u) % 32u + 1u);
  const auto code = e.let((alias / divisor) % 64u + 6u);
  auto total = e.var(0.0f);
  for (auto i : e.range(6u)) {
    (void)i;
    total = total.read() + ir::cast<ir::f32>(code);
  }
  e.ret(total.read());
  LSE_EXPECT(ir::verify(body.ir()).ok());
  lse::backend::LoomPrintOptions opts;
  opts.buffers.emplace("packed", lse::backend::LoomBufferView{
                                     ir::Scalar::kU32, 8, "%packed_view"});
  auto result = lse::backend::loom_print(body.ir(), opts);
  LSE_EXPECT(result.ok());
  if (result.ok()) {
    LSE_EXPECT(result->text.find("scalar.divui") != std::string::npos);
    LSE_EXPECT(result->text.find("scalar.remui") != std::string::npos);
    LSE_EXPECT(result->text.find("index to i32") != std::string::npos);
    LSE_EXPECT(result->text.find("index.constant 6 : index") != std::string::npos);
    LSE_EXPECT(result->text.find("scf.for") != std::string::npos);
    LSE_EXPECT(result->text.find("scalar.uitofp") != std::string::npos);
  }
}

LSE_TEST_MAIN()
