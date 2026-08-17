// The env authoring surface: one body, recorded as source with env::Emit and
// executed directly with env::Cpu, with the args struct bound by reflection.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"
#include "lse/backends/hrx/kernels/vec_mem.hpp"
#include "lse/graph/dialect_source.hpp"
#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/math.hpp"
#include "lse/quant/group_affine_codec.hpp"

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
template <class E, class W = kir::f32>
struct DotArgs {
  env::In<kir::f32, E> x;
  env::In<W, E> w;
  env::Out<kir::f32, E> out;
};

template <class E, class W>
void dot_rows(E& e, DotArgs<E, W>& a, std::uint32_t rows, std::uint32_t cols) {
  auto row = e.thread_id();
  if (auto in_range = e.when(row < rows)) {
    auto acc = e.var(0.0f);
    for (auto t : e.range(cols)) {
      acc = lse::math::fma(a.x[t], lse::math::widen(a.w[row * cols + t]), acc);
    }
    a.out[row] = acc;
  }
}

constexpr lse::DType kF32In[] = {lse::DType::kF32, lse::DType::kF32};
constexpr lse::DType kBf16Weight[] = {lse::DType::kF32, lse::DType::kBF16};

}  // namespace

LSE_TEST(reflected_args_report_counts_from_the_struct) {
  static_assert(env::input_count<DotArgs<env::Emit>>() == 2);
  static_assert(env::output_count<DotArgs<env::Emit>>() == 1);
}

LSE_TEST(env_emit_records_the_body_as_source) {
  kir::KernelBody k(kTypes, kTable);
  DotArgs<env::Emit> args;
  LSE_EXPECT(env::bind(k, args, kF32In, lse::DType::kF32));
  env::Emit e{&k};
  dot_rows<env::Emit, kir::f32>(e, args, 4u, 8u);
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
    dot_rows<env::Cpu, kir::f32>(e, a, rows, cols);
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
    LSE_EXPECT(env::bind(k, args, kF32In, lse::DType::kF32));
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

// The check the bf16 weight scheme rests on: a slot declared over the wrong
// storage type must make the kernel decline, not reinterpret the buffer.
LSE_TEST(bind_refuses_a_slot_whose_dtype_disagrees) {
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit> f32_slots;  // In<f32> over a bf16 buffer
    LSE_EXPECT(!env::bind(k, f32_slots, kBf16Weight, lse::DType::kF32));
  }
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit, lse::bf16> bf16_slot;
    LSE_EXPECT(env::bind(k, bf16_slot, kBf16Weight, lse::DType::kF32));
  }
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit, lse::bf16> bf16_slot;  // In<bf16> over an f32 buffer
    LSE_EXPECT(!env::bind(k, bf16_slot, kF32In, lse::DType::kF32));
  }
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit> args;  // Out<f32> over a bf16 output
    LSE_EXPECT(!env::bind(k, args, kF32In, lse::DType::kBF16));
  }
  // Nothing to check against is not a mismatch.
  {
    kir::KernelBody k(kTypes, kTable);
    DotArgs<env::Emit, lse::bf16> args;
    LSE_EXPECT(env::bind(k, args, {}, lse::DType::kF32));
  }
}

LSE_TEST(env_cpu_widens_narrow_weight_storage) {
  const std::uint32_t rows = 3, cols = 5;
  std::vector<float> x(cols), out(rows, -1.0f);
  std::vector<lse::bfloat16_t> w(rows * cols);
  for (std::uint32_t i = 0; i < cols; ++i) x[i] = 0.5f * float(i) - 1.0f;
  for (std::uint32_t i = 0; i < rows * cols; ++i)
    w[i] = lse::bfloat16_t(0.25f * float(i % 7) - 0.5f);

  DotArgs<env::Cpu, lse::bf16> args{{x.data()}, {w.data()}, {out.data()}};
  env::run_flat(rows, [&](env::Cpu& e, DotArgs<env::Cpu, lse::bf16>& a) {
    dot_rows<env::Cpu, lse::bf16>(e, a, rows, cols);
  }, args);

  for (std::uint32_t r = 0; r < rows; ++r) {
    float want = 0.0f;
    for (std::uint32_t t = 0; t < cols; ++t)
      want = std::fma(x[t], w[r * cols + t].to_float(), want);
    LSE_EXPECT(std::fabs(out[r] - want) < 1e-6f);
  }
}

LSE_TEST(pack_width_divides_the_row_it_walks) {
  for (std::uint32_t k : {1u, 7u, 15u, 16u, 17u, 63u, 64u, 1023u, 1024u,
                          2176u}) {
    for (std::uint32_t eb : {2u, 4u}) {
      const std::uint32_t n =
          lse::backend::hrx_kernels::row_pack(k, 16u, eb);
      LSE_EXPECT(n >= 1 && n <= 16u / eb);
      LSE_EXPECT((n & (n - 1)) == 0);
      // Rows start at multiples of k, so a wider pack would land some row's
      // load off its natural alignment.
      LSE_EXPECT(k % n == 0);
    }
  }
}

// The GEMV splits K into wave-wide spans plus a lane-strided tail. Whatever
// the pack width, every element of the row must be touched exactly once.
LSE_TEST(gemv_k_walk_covers_every_element_once) {
  for (std::uint32_t wave : {32u, 64u}) {
    for (std::uint32_t k : {1u, 7u, 15u, 16u, 17u, 63u, 64u, 1023u, 1024u,
                            2176u}) {
      for (std::uint32_t eb : {2u, 4u}) {
        const std::uint32_t step =
            lse::backend::hrx_kernels::row_pack(k, 16u, eb);
        const std::uint32_t span = wave * step;
        const std::uint32_t aligned = (k / span) * span;
        std::vector<int> hits(k, 0);
        for (std::uint32_t k0 = 0; k0 < aligned; k0 += span) {
          for (std::uint32_t lane = 0; lane < wave; ++lane) {
            const std::uint32_t kk = k0 + lane * step;
            for (std::uint32_t e = 0; e < step; ++e) ++hits[kk + e];
          }
        }
        for (std::uint32_t lane = 0; lane < wave; ++lane) {
          for (std::uint32_t kt = aligned + lane; kt < k; kt += wave) {
            ++hits[kt];
          }
        }
        for (std::uint32_t i = 0; i < k; ++i) LSE_EXPECT(hits[i] == 1);
      }
    }
  }
}

namespace {

// What quant_linear binds: the packed plane plus the two group planes.
template <class E, class S = lse::bf16>
struct QuantArgs {
  env::In<std::uint32_t, E> packed;
  env::In<S, E> scales;
  env::In<S, E> biases;
  env::Out<kir::f32, E> out;
};

constexpr lse::DType kQuantIn[] = {lse::DType::kU32, lse::DType::kBF16,
                                   lse::DType::kBF16};

// A row of pseudo-random codes plus one scale/bias pair per group.
struct QuantRow {
  std::vector<std::uint32_t> packed;
  std::vector<lse::bfloat16_t> scales;
  std::vector<lse::bfloat16_t> biases;
  std::vector<float> want;
};

QuantRow make_row(const lse::quant::GroupAffine& q, std::size_t k) {
  QuantRow r;
  r.packed.assign(q.packed_words(k), 0u);
  r.scales.resize(q.group_count(k));
  r.biases.resize(q.group_count(k));
  r.want.resize(k);
  std::uint32_t state = 12345u;
  for (std::size_t c = 0; c < k; ++c) {
    state = state * 1664525u + 1013904223u;
    q.set_code(r.packed.data(), c, (state >> 13) & q.max_code());
  }
  for (std::size_t g = 0; g < r.scales.size(); ++g) {
    r.scales[g] = lse::bfloat16_t(g % 2 == 0 ? 0.0078125f : -0.015625f);
    r.biases[g] = lse::bfloat16_t(g % 2 == 0 ? -0.0625f : 0.25f);
  }
  q.dequantize_row<lse::bfloat16_t>(r.packed.data(), r.scales.data(),
                                    r.biases.data(), k, r.want.data());
  return r;
}

}  // namespace

// The dequant that quant_linear runs in register, executed on the host. Every
// legal bit width, so the straddling path 3/5/6 bits take is covered too.
LSE_TEST(group_affine_dequant_runs_the_same_on_the_host) {
  for (int bits : lse::quant::kGroupAffineBits) {
    auto spec = lse::quant::GroupAffine::make(bits, 64);
    LSE_EXPECT(spec.ok());
    if (!spec.ok()) continue;
    const lse::quant::GroupAffine q = *spec;
    const std::size_t k = 128;
    const QuantRow r = make_row(q, k);

    const auto vals = static_cast<std::size_t>(q.values_per_chunk());
    const auto words = static_cast<std::size_t>(q.words_per_chunk());
    const std::size_t per_group = 64 / vals;
    std::vector<float> got(k, 1e9f);

    QuantArgs<env::Cpu> a{{r.packed.data()},
                          {r.scales.data()},
                          {r.biases.data()},
                          {got.data()}};
    env::Cpu e{0};
    for (std::size_t chunk = 0; chunk < k / vals; ++chunk) {
      const std::size_t g = chunk / per_group;
      lse::quant::dequant_chunk(
          e, a.packed, q, static_cast<std::uint32_t>(chunk * words),
          r.scales[g].to_float(), r.biases[g].to_float(),
          [&](int c, float v) { got[chunk * vals + static_cast<std::size_t>(c)] = v; });
    }
    for (std::size_t i = 0; i < k; ++i) {
      LSE_EXPECT(std::fabs(got[i] - r.want[i]) < 1e-6f);
    }
  }
}

LSE_TEST(group_affine_dequant_records_shift_free_source) {
  auto spec = lse::quant::GroupAffine::make(4, 64);
  LSE_EXPECT(spec.ok());
  if (!spec.ok()) return;
  kir::KernelBody k(kTypes, kTable);
  QuantArgs<env::Emit> args;
  LSE_EXPECT(env::bind(k, args, kQuantIn, lse::DType::kF32));
  env::Emit e{&k};
  auto acc = e.var(0.0f);
  lse::quant::dequant_chunk(e, args.packed, *spec, e.thread_id(), e.f32(0.25f),
                            e.f32(-1.0f),
                            [&](int, const kir::Val<kir::f32>& v) {
                              acc = v + acc.read();
                            });
  const std::string src = k.str();

  // The lane is read at its own width, not widened on the way in.
  LSE_EXPECT(src.find("unsigned int") != std::string::npos);
  // Field extraction is division and modulus: kir has no bitwise operators,
  // and on unsigned values these are the same operation.
  LSE_EXPECT(src.find(">>") == std::string::npos);
  LSE_EXPECT(src.find("0xF") == std::string::npos);
  // Code 5 of a 4-bit chunk sits at bit 20, so its divisor is 2^20, and every
  // code is masked back to [0, 16).
  LSE_EXPECT(src.find("/ 1048576u") != std::string::npos);
  LSE_EXPECT(src.find("% 16u") != std::string::npos);
  // The weight becomes a value through one fma per code, in register.
  LSE_EXPECT(src.find("fmaf(") != std::string::npos);
}

LSE_TEST_MAIN()
