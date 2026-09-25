// Typed dispatch extents retain both SSA dominance and verifier restrictions.
#include "harness.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/ir/env.hpp"
#include "lse/ir/lower.hpp"
#include "lse/ir/verify.hpp"

namespace ir = lse::ir;
namespace env = lse::ir::env;
namespace {
const ir::TypeTable types = lse::backend::loom_types();
const ir::DialectSourceTable intrinsics{std::span<const ir::PrimitiveSource>{}};
lse::backend::LoomPrintOptions options() {
  lse::backend::LoomPrintOptions opts;
  opts.buffers.emplace(
      "meta", lse::backend::LoomBufferView{ir::Scalar::kF32, 5, "%meta_view"});
  opts.buffers.emplace("data", lse::backend::LoomBufferView{
                                   ir::Scalar::kF32, 1024, "%data_view"});
  return opts;
}
}  // namespace

LSE_TEST(typed_runtime_extent_is_a_dominated_metadata_load_in_both_dialects) {
  ir::KernelBody body(types, intrinsics);
  env::Emit e{&body};
  ir::Buffer<ir::f32> meta(&body, &types, "meta");
  auto rows =
      e.runtime_extent("rows", ir::cast<ir::u32>(meta[e.u32(2u)].read()));
  auto total = e.var(0.0f);
  for (auto r : e.range(rows)) {
    (void)r;
    total = total.read() + e.f32(1.0f);
  }
  e.ret(total.read());
  LSE_EXPECT(ir::verify(body.ir()).ok());
  LSE_EXPECT(rows.text().find("meta[2u]") != std::string::npos);
  auto printed = lse::backend::loom_print(body.ir(), options());
  LSE_EXPECT(printed.ok());
  if (printed.ok()) {
    LSE_EXPECT(printed->text.find("view.load") < printed->text.find("scf.for"));
    LSE_EXPECT(printed->text.find("extent_zero") != std::string::npos);
  }
  ir::Body cloned(types, intrinsics);
  cloned.splice(body.ir(), cloned.entry());
  LSE_EXPECT(ir::verify(cloned).ok());
  LSE_EXPECT(cloned.runtime_extents().size() == 1);
}

LSE_TEST(typed_runtime_extents_still_reject_inner_counts_and_variable_strides) {
  for (int use = 0; use < 2; ++use) {
    ir::KernelBody body(types, intrinsics);
    env::Emit e{&body};
    ir::Buffer<ir::f32> meta(&body, &types, "meta");
    ir::Buffer<ir::f32> data(&body, &types, "data");
    auto rows =
        e.runtime_extent("rows", ir::cast<ir::u32>(meta[e.u32(2u)].read()));
    if (use == 0) {
      data[e.thread_id() * rows] = e.f32(1.0f);
    } else {
      for (auto outer : e.range(4u)) {
        for (auto inner : e.range(rows)) {
          data[outer * 4u + inner] = e.f32(1.0f);
        }
      }
    }
    const auto status = ir::verify(body.ir());
    LSE_EXPECT(!status.ok());
    LSE_EXPECT(status.message().find(use == 0 ? "address arithmetic"
                                              : "inner trip count") !=
               std::string::npos);
  }
}

LSE_TEST(typed_extent_rejects_wrong_type_and_missing_runtime_annotation) {
  ir::KernelBody body(types, intrinsics);
  env::Emit e{&body};
  const auto id = body.ir().runtime_extent("bad", e.f32(1.0f).id());
  (void)id;
  LSE_EXPECT(!ir::verify(body.ir()).ok());
  ir::KernelBody other(types, intrinsics);
  env::Emit e2{&other};
  const auto typed = other.ir().runtime_extent("rows", e2.u32(2u).id());
  other.ir().op(other.ir().value(typed).def).flags = 0;
  LSE_EXPECT(!ir::verify(other.ir()).ok());
}

LSE_TEST(untyped_runtime_extent_still_declines_in_loom) {
  ir::KernelBody body(types, intrinsics);
  env::Emit e{&body};
  e.ret(ir::cast<ir::f32>(e.runtime_extent("rows", "k.rows")));
  LSE_EXPECT(!lse::backend::loom_print(body.ir(), {}).ok());
}

LSE_TEST_MAIN()
