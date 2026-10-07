#include "harness.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/kernels/wmma.hpp"
#include "lse/math.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
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
  size_t rdna3 = 0, rdna4 = 0;
  for (const auto &row : math::matrix_core_table()) {
    const auto *supported = backend::loom_matrix_row(row.key);
    if (supported && row.chained == 1) {
      LSE_EXPECT(supported->emittable());
      LSE_EXPECT(supported->target == math::MatrixTarget::kRdna3 ||
                 supported->target == math::MatrixTarget::kRdna4);
      LSE_EXPECT(supported->wave == 32 && supported->chained == 1);
      LSE_EXPECT(!table.find(row.key).empty());
      ++(supported->target == math::MatrixTarget::kRdna3 ? rdna3 : rdna4);
    }
  }
  // RDNA3/3.5: f32 <- f16/bf16, i32 <- iu8/su8/iu4. RDNA4: f32 <- f16/bf16,
  // i32 <- iu8/su8, f32 <- fp8/bf8.
  LSE_EXPECT(rdna3 == 5);
  LSE_EXPECT(rdna4 == 6);
  LSE_EXPECT(table.find("wmma12.i32.16x16x32.iu4").empty());
  // The narrow-accumulate forms have no measured D mapping on either part.
  LSE_EXPECT(table.find("wmma.f16.16x16x16.f16").empty());
  LSE_EXPECT(table.find("wmma12.f16.16x16x16.f16").empty());
  LSE_EXPECT(!table.find("wmma.f32.16x16x16.f16").empty());
}

// The same tile is a different per-lane payload on each generation: RDNA3/3.5
// gives each lane the whole k step (16 values, the half-waves repeating the
// same rows), RDNA4 half of it. The schema is what tells loomc which
// instruction it is, so it has to carry the row's own width.
LSE_TEST(loom_matrix_spells_each_generation_at_its_own_fragment_width) {
  const auto table = backend::loom_sources();
  const auto gfx11 = table.find("wmma.f32.16x16x16.f16");
  const auto gfx12 = table.find("wmma12.f32.16x16x16.f16");
  LSE_EXPECT(gfx11.find("payload_elements=16, payload_registers=8") !=
             std::string_view::npos);
  LSE_EXPECT(gfx11.find("vector<16xf16>") != std::string_view::npos);
  LSE_EXPECT(gfx12.find("payload_elements=8, payload_registers=4") !=
             std::string_view::npos);
  LSE_EXPECT(gfx12.find("vector<8xf16>") != std::string_view::npos);
  const auto iu8 = table.find("wmma.i32.16x16x16.iu8");
  LSE_EXPECT(iu8.find("payload_elements=16, payload_registers=4") !=
             std::string_view::npos);
  const auto iu4 = table.find("wmma.i32.16x16x16.iu4");
  LSE_EXPECT(iu4.find("element_format=u4, payload_elements=16, "
                      "payload_registers=2") != std::string_view::npos);
  for (const auto key : {"wmma.f32.16x16x16.f16", "wmma.f32.16x16x16.bf16",
                         "wmma12.f32.16x16x16.f16"})
    LSE_EXPECT(backend::loom_result_type(key) == "vector<8xf32>");
  LSE_EXPECT(backend::loom_result_type("wmma.i32.16x16x16.iu4") ==
             "vector<8xi32>");
}

namespace {
// Where the hardware reads operand value `v` of lane `lane` from, and where it
// writes accumulator register `e` of lane `lane`, written from AMD's documents
// and NOT from the row's layout fields, so the two can disagree.
//
//   RDNA3/3.5 wave32 (RDNA3 ISA guide, "Wave Matrix Multiply Accumulate";
//   AMD matrix instruction calculator, --architecture rdna3; Loom
//   rdna3_wmmar3_* layouts): A[i][k] sits in lane i AND lane i+16, value k
//   -- the half-waves must hold the same data. B[k][j] likewise in lane j and
//   j+16. D[i][j] is lane j + 16*(i%2), register i/2.
//   RDNA4 wave32 (RDNA4 ISA guide; the calculator, --architecture rdna4;
//   Loom rdna4_wmma_*): A[i][k] is lane i + 16*(k/(K/2)), value k%(K/2): the
//   half-waves split K and nothing repeats. D[i][j] is lane j + 16*(i/8),
//   register i%8.
struct Coord {
  int row, k;
};
Coord documented_operand(const math::MatrixCoreRow &r, int lane, int v) {
  if (r.target == math::MatrixTarget::kRdna3) return {lane % 16, v};
  return {lane % 16, (lane / 16) * (r.k / 2) + v};
}
Coord documented_acc(const math::MatrixCoreRow &r, int lane, int e) {
  if (r.target == math::MatrixTarget::kRdna3)
    return {2 * e + lane / 16, lane % 16};
  return {e + 8 * (lane / 16), lane % 16};
}

// One instruction, executed as the documents describe, on fragments filled
// and an accumulator read back the way every kernel does through
// geometry_of. Returns the largest error against the plain product, or -1
// when the fill breaks a hardware rule (an uncovered k, or half-waves that
// disagree where RDNA3 requires them to repeat).
double run_documented(const math::MatrixCoreRow &r) {
  const int M = r.m, N = r.n, K = r.k, W = r.wave;
  const int per_lane = r.a_len * r.pack;  // operand values in one lane
  std::vector<double> A(M * K), B(K * N);
  for (int i = 0; i < M * K; ++i) A[i] = (i * 7 % 13) - 6;
  for (int i = 0; i < K * N; ++i) B[i] = (i * 5 % 11) - 5;
  // What the kernel puts in each lane, by the row's geometry: lane L takes
  // row L % n starting at k = (L / n) * lane_k on a split layout, 0 otherwise.
  const kernels::TileGeometry g = kernels::geometry_of(r);
  if (static_cast<int>(g.lane_k) != per_lane) return -1.0;
  std::vector<double> af(W * per_lane), bf(W * per_lane);
  for (int lane = 0; lane < W; ++lane) {
    const int lo = lane % r.n, hi = lane / r.n;
    const int k0 = g.split_k ? hi * static_cast<int>(g.lane_k) : 0;
    for (int v = 0; v < per_lane; ++v) {
      af[lane * per_lane + v] = A[lo * K + k0 + v];
      bf[lane * per_lane + v] = B[(k0 + v) * N + lo];
    }
  }
  // What the hardware assembles from them.
  std::vector<double> a(M * K, NAN), b(K * N, NAN);
  for (int lane = 0; lane < W; ++lane)
    for (int v = 0; v < per_lane; ++v) {
      const Coord c = documented_operand(r, lane, v);
      double &ra = a[c.row * K + c.k], &rb = b[c.k * N + c.row];
      if (!std::isnan(ra) && (ra != af[lane * per_lane + v] ||
                              rb != bf[lane * per_lane + v]))
        return -1.0;
      ra = af[lane * per_lane + v];
      rb = bf[lane * per_lane + v];
    }
  for (int i = 0; i < M * K; ++i)
    if (std::isnan(a[i]) || std::isnan(b[i])) return -1.0;
  // D = A * B, written back the documented way and read the kernel's way.
  std::vector<double> out(M * N, NAN);
  for (int lane = 0; lane < W; ++lane)
    for (int e = 0; e < r.c_len; ++e) {
      const Coord d = documented_acc(r, lane, e);
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += a[d.row * K + k] * b[k * N + d.k];
      const int row = e * static_cast<int>(g.slot_step) +
                      (lane / r.n) * static_cast<int>(g.half_rows);
      out[row * N + lane % r.n] = acc;
    }
  double worst = 0.0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double want = 0.0;
      for (int k = 0; k < K; ++k) want += A[i * K + k] * B[k * N + j];
      const double got = out[i * N + j];
      if (std::isnan(got)) return -1.0;
      worst = std::max(worst, std::abs(got - want));
    }
  return worst;
}
}  // namespace

// Every row Loom emits, on either generation, against a CPU reference of the
// instruction as AMD documents it. The kernels never index a lane by hand;
// they go through the row's layout, so this is the check that the layout each
// generation's row carries is the hardware's.
LSE_TEST(matrix_rows_match_the_documented_lane_layouts) {
  size_t checked = 0;
  for (const auto &row : math::matrix_core_table()) {
    if (backend::loom_matrix_row(row.key) != &row) continue;
    const double err = run_documented(row);
    if (err != 0.0)
      std::fprintf(stderr, "%.*s: %g\n", static_cast<int>(row.key.size()),
                   row.key.data(), err);
    LSE_EXPECT(err == 0.0);
    ++checked;
  }
  LSE_EXPECT(checked == 11);
  // And the geometry the tiles read agrees with the row: gfx11 holds the
  // whole k step and interleaves the half-waves' rows, gfx12 splits k and
  // blocks them.
  constexpr auto g11 = kernels::geometry_of(math::matrix_core_row(
      math::MatrixTarget::kRdna3, math::MatrixElem::kF32,
      math::MatrixElem::kF16, 16, 16, 16));
  static_assert(!g11.split_k && g11.lane_k == 16 && g11.frag == 16 &&
                g11.slot_step == 2 && g11.half_rows == 1);
  constexpr auto g12 = kernels::geometry_of(math::matrix_core_row(
      math::MatrixTarget::kRdna4, math::MatrixElem::kF32,
      math::MatrixElem::kF16, 16, 16, 16));
  static_assert(g12.split_k && g12.lane_k == 8 && g12.frag == 8 &&
                g12.slot_step == 1 && g12.half_rows == 8);
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
    for (int m : {1, 8, 9, 15, 16, 17, 32, 511, 512, 513}) {
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
      if (bits == 4 && m >= 16)
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
