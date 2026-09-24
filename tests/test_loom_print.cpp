// Source-only regressions: no Loom compiler, HRX library or GPU is required.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_env.hpp"

namespace ir = lse::ir;
namespace env = lse::ir::env;
namespace {
const ir::TypeTable types = lse::backend::loom_types();
const ir::DialectSourceTable intrinsics{std::span<const ir::PrimitiveSource>{}};
}

LSE_TEST(guarded_index_subtraction_remains_address_only_through_bind_aliases) {
  ir::KernelBody body(types, intrinsics);
  env::Emit e{&body};
  ir::Buffer<ir::f32> input(&body, &types, "input");
  const auto t = e.let(e.thread_id() % 8u);
  const auto source = e.let(t - 3u);
  const auto alias = e.let(source);
  auto result = e.var(0.0f);
  if (auto valid = e.when(t >= 3u)) result = input[alias];
  e.ret(result.read());
  lse::backend::LoomPrintOptions options;
  options.buffers.emplace("input", lse::backend::LoomBufferView{
      ir::Scalar::kF32, 64, "%input_view"});
  auto printed = lse::backend::loom_print(body.ir(), options);
  LSE_EXPECT(printed.ok());
  if (printed.ok()) {
    LSE_EXPECT(printed->text.find("index.max") != std::string::npos);
    LSE_EXPECT(printed->text.find("view.load") != std::string::npos);
  }
}

LSE_TEST(index_subtraction_wrap_observed_through_aliases_still_declines) {
  for (int consumer = 0; consumer < 4; ++consumer) {
    ir::KernelBody body(types, intrinsics);
    env::Emit e{&body};
    const auto source = e.let(e.thread_id() - 3u);
    const auto alias = e.let(source);
    if (consumer == 0) e.ret(alias < 16u);
    if (consumer == 1) e.ret(ir::cast<ir::f32>(alias));
    if (consumer == 2) e.ret(alias);
    if (consumer == 3) e.ret(e.let(alias + 7u) < 16u);
    auto printed = lse::backend::loom_print(body.ir(), {});
    LSE_EXPECT(!printed.ok());
    if (!printed.ok()) {
      LSE_EXPECT(printed.status().message().find("wrapped value is observable") !=
                 std::string::npos);
    }
  }
}

LSE_TEST_MAIN()
