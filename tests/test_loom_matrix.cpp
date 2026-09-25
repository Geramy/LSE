#include "lse/kernels/int8_policy.hpp"
#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math.hpp"
using namespace lse;
namespace {
template <math::MatrixElem T, math::MatrixElem C>
Result<backend::LoomBody> print_mma() {
  using Op = math::op::Mma<math::MatrixTarget::kRdna4, C, T, 16, 16, 16>;
  constexpr auto row = Op::kRow;
  const auto types = backend::loom_types();
  const auto sources = backend::loom_sources();
  ir::KernelBody b(types, sources);
  ir::env::Emit e{&b};
  auto a = e.local<math::matrix_scalar_t<row.a_elem>, row.a_len>();
  auto w = e.local<math::matrix_scalar_t<row.b_elem>, row.b_len>();
  auto c = e.local<math::matrix_scalar_t<row.c_elem>, row.c_len>();
  for (auto i : e.unroll(row.a_len))
    a[i] = math::narrow<math::matrix_scalar_t<row.a_elem>>(e.f32(1.0f));
  for (auto i : e.unroll(row.b_len))
    w[i] = math::narrow<math::matrix_scalar_t<row.b_elem>>(e.f32(2.0f));
  for (auto i : e.unroll(row.c_len))
    c[i] = math::narrow<math::matrix_scalar_t<row.c_elem>>(e.f32(0.0f));
  c = math::mma<Op>(a.value(), w.value(), c.value());
  e.ret(c[0].read());
  return backend::loom_print(b.ir(), {});
}
} // namespace
LSE_TEST(loom_matrix_uses_shared_measured_rows_and_declines_other_layouts) {
  const auto table = backend::loom_sources();
  size_t present = 0;
  for (const auto &row : math::matrix_core_table()) {
    const auto *supported = backend::loom_matrix_row(row.key);
    if (supported && row.chained == 1) {
      ++present;
      LSE_EXPECT(supported->emittable());
      LSE_EXPECT(supported->target == math::MatrixTarget::kRdna4);
      LSE_EXPECT(supported->wave == 32 && supported->chained == 1);
      LSE_EXPECT(!table.find(row.key).empty());
    }
  }
  LSE_EXPECT(present == 6);
  LSE_EXPECT(table.find("wmma12.i32.16x16x32.iu4").empty());
  LSE_EXPECT(table.find("wmma.f32.16x16x16.f16").empty());
}
LSE_TEST(loom_matrix_mixed_signedness_is_a_schema_not_a_type_guess) {
  const auto t = backend::loom_sources().find("wmma12.i32.16x16x16.su8");
  LSE_EXPECT(t.find("element_format=i8") != std::string_view::npos);
  LSE_EXPECT(t.find("element_format=u8") != std::string_view::npos);
  LSE_EXPECT(t.find("payload_elements=8, payload_registers=2") !=
             std::string_view::npos);
  LSE_EXPECT(backend::loom_result_type("wmma12.i32.16x16x16.su8") ==
             "vector<8xi32>");
}
LSE_TEST(loom_matrix_shared_ir_preserves_vector_results) {
  auto i8 = print_mma<math::MatrixElem::kI8, math::MatrixElem::kI32>();
  auto su8 = print_mma<math::MatrixElem::kSU8, math::MatrixElem::kI32>();
  auto fp8 = print_mma<math::MatrixElem::kFp8, math::MatrixElem::kF32>();
  auto bf8 = print_mma<math::MatrixElem::kBf8, math::MatrixElem::kF32>();
  for (const auto *result : {&i8, &su8, &fp8, &bf8}) {
    LSE_EXPECT(result->ok());
    if (!result->ok())
      std::fprintf(stderr, "%s\n", result->status().to_string().c_str());
    else
      LSE_EXPECT((*result)->text.find("vector.mma") != std::string::npos);
  }
}

LSE_TEST(loom_register_fragment_updates_are_carried_through_loops_and_guards) {
  const auto types = backend::loom_types();
  const auto table = backend::loom_sources();
  ir::KernelBody body(types, table);
  ir::env::Emit e{&body};
  auto a = e.local<ir::f32, 8>();
  for (auto z : e.unroll(8u))
    a[z] = e.f32(1.0f);
  for (auto k : e.range(0u, 3u, 1u)) {
    if (auto live = e.when(e.thread_id() < 32u)) {
      for (auto z : e.unroll(8u))
        a[z] = a[z].read() + ir::cast<ir::f32>(k);
    }
  }
  e.ret(a[3].read());
  const auto printed = backend::loom_print(body.ir(), {});
  LSE_EXPECT(printed.ok());
  if (printed.ok()) {
    LSE_EXPECT(printed->text.find("scf.for") != std::string::npos);
    LSE_EXPECT(printed->text.find("-> (vector<8xf32>)") != std::string::npos);
    LSE_EXPECT(printed->text.find("scf.yield") != std::string::npos);
    LSE_EXPECT(printed->text.find("vector.insert") != std::string::npos);
    LSE_EXPECT(printed->text.find("buffer.alloca") == std::string::npos);
  }
}
LSE_TEST(loom_register_fragment_dynamic_lane_and_bad_width_decline) {
  const auto types = backend::loom_types();
  const auto table = backend::loom_sources();
  {
    ir::KernelBody body(types, table);
    ir::env::Emit e{&body};
    auto a = e.local<ir::f32, 8>();
    a[e.thread_id()] = e.f32(1.0f);
    e.ret(a[0].read());
    LSE_EXPECT(!backend::loom_print(body.ir(), {}).ok());
  }
  {
    ir::KernelBody body(types, table);
    ir::env::Emit e{&body};
    using Op = math::op::Mma<math::MatrixTarget::kRdna4, math::MatrixElem::kI32,
                             math::MatrixElem::kI8, 16, 16, 16>;
    auto a = e.local<int, 2>();
    auto b = e.local<int, 2>();
    auto c = e.local<int, 8>();
    c = math::mma<Op>(a.value(), b.value(), c.value());
    e.ret(c[0].read());
    body.ir().walk([&](ir::OpId id) {
      auto &op = body.ir().op(id);
      if (op.kind == ir::OpKind::kCall)
        op.type.lanes = 4;
    });
    auto result = backend::loom_print(body.ir(), {});
    LSE_EXPECT(!result.ok());
    if (!result.ok())
      LSE_EXPECT(result.status().message().find("matrix fragment types") !=
                 std::string::npos);
  }
}

LSE_TEST(loom_matrix_availability_does_not_requantize_q6_or_q8) {
  backend::DeviceInfo info;
  backend::AmdDeviceInfo amd;
  info.arch = "gfx1201";
  info.wavefront_size = 32;
  info.max_threads_per_workgroup = 1024;
  info.lds_bytes_per_workgroup = 65536;
  backend::apply_arch_defaults(info, amd);
  info.extension_id = backend::AmdDeviceInfo::kExtensionId;
  info.extension = &amd;
  const auto sources = backend::loom_sources();
  const auto types = backend::loom_types();
  const DType dtypes[] = {DType::kF32, DType::kU32, DType::kBF16, DType::kBF16};
  for (int bits : {4, 6, 8})
    for (int m : {16, 64}) {
      const Shape inputs[] = {Shape{m, 64}, Shape{32, 64 * bits / 32},
                              Shape{32, 1}, Shape{32, 1}};
      graph::KernelShapes s;
      s.inputs = inputs;
      s.input_dtypes = dtypes;
      s.output = Shape{m, 32};
      s.output_dtype = DType::kF32;
      s.iattrs = {bits, 64, 0, 0};
      s.device = &info;
      s.intrinsics = &sources;
      s.types = types;
      const auto *specialized = kernels::wmma_quant_linear_for(s);
      if (bits == 4 && kernels::activation_int8_enabled())
        LSE_EXPECT(specialized != nullptr);
      else
        LSE_EXPECT(specialized == nullptr);
    }
}

LSE_TEST(
    loom_inner_fragment_lifetime_and_unsigned_zero_keep_register_semantics) {
  const auto types = backend::loom_types();
  const auto table = backend::loom_sources();
  ir::KernelBody body(types, table);
  ir::env::Emit e{&body};
  auto total = e.var(0.0f);
  for (auto k : e.range(0u, 2u, 1u)) {
    auto fragment = e.local<ir::u32, 2>();
    for (auto lane : e.unroll(2u))
      fragment[lane] = e.u32(0);
    if (auto live = e.when(k < 1u))
      fragment[1] = e.u32(7);
    total = total.read() + ir::cast<ir::f32>(fragment[1].read());
  }
  e.ret(total.read());
  auto printed = backend::loom_print(body.ir(), {});
  LSE_EXPECT(printed.ok());
  if (printed.ok()) {
    const auto outer = printed->text.find(" = scf.for ");
    const auto brace = printed->text.find('{', outer);
    const auto header = printed->text.substr(outer, brace - outer);
    LSE_EXPECT(header.find("-> (f32)") != std::string::npos);
    LSE_EXPECT(header.find("vector<2xi32>") == std::string::npos);
    LSE_EXPECT(printed->text.find("index to i32") != std::string::npos);
  }
}
LSE_TEST_MAIN()
