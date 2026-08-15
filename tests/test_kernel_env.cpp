// The env authoring surface: one body, recorded as source with env::Emit and
// executed directly with env::Cpu, with the args struct bound by reflection.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/graph/dialect_source.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/math.hpp"

using namespace lse::graph;

namespace {

// A backend-shaped test table: the test is the backend here, so the kernels
// need no HRX to record.
std::string_view test_scalar(kir::Scalar s) noexcept {
  switch (s) {
    case kir::Scalar::kU32: return "unsigned int";
    case kir::Scalar::kF32: return "float";
    case kir::Scalar::kBool: return "bool";
    default: return "float";
  }
}
std::string test_vec(kir::Scalar s, int n, std::string_view name) {
  return "typedef " + std::string(test_scalar(s)) + " " + std::string(name) +
         " __attribute__((ext_vector_type(" + std::to_string(n) + ")));";
}
const kir::TypeTable kTypes{test_scalar, test_vec};

constexpr PrimitiveSource kSources[] = {
    {"fma", "fmaf($0, $1, $2)"},
    {"exp", "__expf($0)"},
};
const DialectSourceTable kTable{std::span<const PrimitiveSource>(kSources)};

// The kernel under test: one thread per output row, straight C++.
template <class E>
struct DotArgs {
  env::In<kir::f32, E> x;
  env::In<kir::f32, E> w;
  env::Out<kir::f32, E> out;
};

template <class E>
void dot_rows(E& e, DotArgs<E>& a, std::uint32_t rows, std::uint32_t cols) {
  auto row = e.thread_id();
  if (auto in_range = e.when(row < rows)) {
    auto acc = e.var(0.0f);
    for (auto t : e.range(cols)) {
      acc = lse::math::fma(a.x[t], a.w[row * cols + t], acc);
    }
    a.out[row] = acc;
  }
}

}  // namespace

LSE_TEST(reflected_args_report_counts_from_the_struct) {
  static_assert(env::input_count<DotArgs<env::Emit>>() == 2);
  static_assert(env::output_count<DotArgs<env::Emit>>() == 1);
}

LSE_TEST(env_emit_records_the_body_as_source) {
  kir::KernelBody k(kTypes, kTable);
  DotArgs<env::Emit> args;
  env::bind(k, args);
  env::Emit e{&k};
  dot_rows(e, args, 4u, 8u);
  const std::string src = k.str();

  // The C++ control flow became device control flow.
  LSE_EXPECT(src.find("if ((i < 4u)) {") != std::string::npos);
  // One counter serves every stem, so the loop var after local v0 is i1.
  LSE_EXPECT(src.find("for (unsigned int i1 = 0u; i1 < 8u; i1 += 1u) {")
             != std::string::npos);
  // Reflection bound the members in declaration order.
  LSE_EXPECT(src.find("in0[") != std::string::npos);
  LSE_EXPECT(src.find("in1[") != std::string::npos);
  LSE_EXPECT(src.find("out[") != std::string::npos);
  // The math library spelled fma from the table.
  LSE_EXPECT(src.find("fmaf(") != std::string::npos);
  // No authored names leaked; locals are generated.
  LSE_EXPECT(src.find("float v0 = 0.0") != std::string::npos ||
             src.find("float v0 = 0.00000000f") != std::string::npos);
}

LSE_TEST(env_cpu_executes_the_same_body) {
  const std::uint32_t rows = 4, cols = 8;
  std::vector<float> x(cols), w(rows * cols), out(rows, -1.0f);
  for (std::uint32_t i = 0; i < cols; ++i) x[i] = 0.5f * float(i) - 1.0f;
  for (std::uint32_t i = 0; i < rows * cols; ++i)
    w[i] = 0.25f * float(i % 7) - 0.5f;

  DotArgs<env::Cpu> args{{x.data()}, {w.data()}, {out.data()}};
  env::run_flat(rows + 2, [&](env::Cpu& e, DotArgs<env::Cpu>& a) {
    dot_rows(e, a, rows, cols);
  }, args);

  for (std::uint32_t r = 0; r < rows; ++r) {
    float want = 0.0f;
    for (std::uint32_t t = 0; t < cols; ++t)
      want = std::fma(x[t], w[r * cols + t], want);
    LSE_EXPECT(std::fabs(out[r] - want) < 1e-6f);
  }
}

LSE_TEST(env_unroll_and_pack_iterate_both_worlds) {
  // Emit: pack load + unroll over its width.
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit> args;
    env::bind(k, args);
    env::Emit e{&k};
    auto acc = e.var(0.0f);
    auto pack = e.load(args.x, e.thread_id() * 4u, 16u);
    for (auto i : e.unroll(pack.width())) {
      acc = lse::math::fma(pack[i], e.f32(2.0f), acc);
    }
    const std::string src = k.str();
    LSE_EXPECT(src.find("#pragma unroll") != std::string::npos);
    LSE_EXPECT(src.find("ext_vector_type(4)") != std::string::npos);
  }
  // Cpu: same statements, real arithmetic.
  {
    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float sink = -1.0f;
    DotArgs<env::Cpu> args{{x.data()}, {x.data()}, {&sink}};
    env::Cpu e{1};
    auto acc = e.var(0.0f);
    auto pack = e.load(args.x, e.thread_id() * 4u, 16u);
    for (auto i : e.unroll(pack.width())) {
      acc = lse::math::fma(pack[i], e.f32(2.0f), acc);
    }
    LSE_EXPECT(std::fabs(float(acc) - 2.0f * (5.0f + 6.0f + 7.0f + 8.0f))
               < 1e-6f);
  }
}

LSE_TEST_MAIN()
