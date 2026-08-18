// Fake device math library. Calling these records into the kir body that is
// currently being emitted; the HRX (or other) table supplies the spelling.
//
//   acc = lse::math::fma(x, y, acc);
//   acc = lse::math::mma<Op>(a, b, acc);
//
// Every operation is a tag in `op::`. Most carry a fixed dialect key; a few
// carry a `row()` because they have a form per accumulator, operand, tile and
// device generation. `emit<Op>` serves both, so an author calling `math::foo`
// never knows which kind it is and adding variance to an op later is adding
// `row()` to its tag.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

#include "lse/core/dtype.hpp"
#include "lse/core/elem.hpp"
#include "lse/ir/recorder.hpp"

namespace lse::math {

// Host overloads of the same spellings: a kernel body instantiated with
// env::Cpu calls these and executes directly instead of recording.
inline float exp(float x) { return std::exp(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float fma(float a, float b, float c) { return std::fma(a, b, c); }
inline float max(float a, float b) { return a > b ? a : b; }
inline float min(float a, float b) { return a < b ? a : b; }
inline float abs(float x) { return std::fabs(x); }
inline float select(bool c, float a, float b) { return c ? a : b; }
// Half away from zero, matching std::round. A block codec's device output is
// required to be bit-identical to quant::QuantScheme's host pack, which uses
// std::round, so this must NOT become rint/nearbyint (round-half-to-even).
inline float round(float x) { return std::round(x); }
inline float rint(float x) { return std::rint(x); }


// Expression type of this library. The recorder lives in lse::ir; authors
// include this header and never name that namespace.
template <typename T>
using Val = ir::Val<T>;

namespace detail {

template <typename R, typename... A>
Val<R> invoke(std::string_view id, const A&... args) {
  ir::KernelBody* body = ir::KernelBody::try_current();
  if (body == nullptr) return {};
  return body->call<R>(id, args...);
}

}  // namespace detail

// An operation the engine can ask a target to spell. A tag declares either a
// fixed `key` — the overwhelming majority: one form, nothing to choose — or a
// `row()`, when the instruction varies by accumulator, operand, tile shape and
// device generation. `result` is the value type the recorder hands back.
namespace op {

struct Exp {
  static constexpr std::string_view key = "exp";
  using result = lse::f32;
};
struct Sqrt {
  static constexpr std::string_view key = "sqrt";
  using result = lse::f32;
};
struct Rsqrt {
  static constexpr std::string_view key = "rsqrt";
  using result = lse::f32;
};
struct Fma {
  static constexpr std::string_view key = "fma";
  using result = lse::f32;
};
struct Max {
  static constexpr std::string_view key = "max";
  using result = lse::f32;
};
struct Min {
  static constexpr std::string_view key = "min";
  using result = lse::f32;
};
struct NegInf {
  static constexpr std::string_view key = "neg_inf";
  using result = lse::f32;
};
struct Abs {
  static constexpr std::string_view key = "abs";
  using result = lse::f32;
};
struct Round {
  static constexpr std::string_view key = "round";
  using result = lse::f32;
};
struct Rint {
  static constexpr std::string_view key = "rint";
  using result = lse::f32;
};
struct ShflXor {
  static constexpr std::string_view key = "wave.shfl_xor";
  using result = lse::f32;
};
// Four signed bytes against four unsigned bytes into an i32 accumulator. The
// mixed signedness is not a convenience: a group-affine code is unsigned by
// construction and a quantized activation is signed, so the two operands are
// genuinely different types and a same-signedness dot would have to waste a
// bit on one of them.
struct Dot4Iu8 {
  static constexpr std::string_view key = "dot4.i32.iu8";
  using result = ir::i32;
};
struct LocalId {
  static constexpr std::string_view key = "thread.local_id";
  using result = ir::u32;
};
struct WorkgroupIdX {
  static constexpr std::string_view key = "thread.workgroup_id.x";
  using result = ir::u32;
};
struct WorkgroupIdY {
  static constexpr std::string_view key = "thread.workgroup_id.y";
  using result = ir::u32;
};
struct WorkgroupSize {
  static constexpr std::string_view key = "thread.workgroup_size";
  using result = ir::u32;
};
struct GridDimX {
  static constexpr std::string_view key = "thread.grid_dim.x";
  using result = ir::u32;
};

}  // namespace op

namespace detail {

template <class Op>
concept variant_op = requires { Op::row(); };

}  // namespace detail

// The one entry point. `if constexpr` on which kind of tag this is, then a
// single lookup and a single decline path: a target whose table has no row
// yields an empty Val and the kernel that asked declines. Whether the *device*
// has the instruction is a separate question, answered by the gate that reads
// the same row.
template <class Op, typename... A>
[[nodiscard]] inline Val<typename Op::result> emit(const A&... args) {
  if constexpr (detail::variant_op<Op>) {
    constexpr auto row = Op::row();
    return detail::invoke<typename Op::result>(row.key, args...);
  } else {
    return detail::invoke<typename Op::result>(Op::key, args...);
  }
}

inline Val<lse::f32> exp(const Val<lse::f32>& x) { return emit<op::Exp>(x); }
inline Val<lse::f32> sqrt(const Val<lse::f32>& x) { return emit<op::Sqrt>(x); }
inline Val<lse::f32> rsqrt(const Val<lse::f32>& x) { return emit<op::Rsqrt>(x); }
inline Val<lse::f32> fma(const Val<lse::f32>& a, const Val<lse::f32>& b,
                         const Val<lse::f32>& c) {
  return emit<op::Fma>(a, b, c);
}
inline Val<lse::f32> max(const Val<lse::f32>& a, const Val<lse::f32>& b) {
  return emit<op::Max>(a, b);
}
inline Val<lse::f32> min(const Val<lse::f32>& a, const Val<lse::f32>& b) {
  return emit<op::Min>(a, b);
}
inline Val<lse::f32> neg_inf() { return emit<op::NegInf>(); }
inline Val<lse::f32> abs(const Val<lse::f32>& x) { return emit<op::Abs>(x); }
inline Val<lse::f32> round(const Val<lse::f32>& x) { return emit<op::Round>(x); }
// Ties to even, which is the hardware's own rounding and a single
// instruction. `round` is the one a codec that must agree with a host packer
// asks for; this is the one everything else should ask for.
inline Val<lse::f32> rint(const Val<lse::f32>& x) { return emit<op::Rint>(x); }
namespace detail {

// The dialect rows a narrow float format selects when a kernel needs its bit
// pattern rather than its value — a block codec writing a scale into packed
// bytes. A format with no specialization is a compile error here, never a
// fallthrough to whichever width happened to be there first.
template <typename T>
struct bit_codec;
template <>
struct bit_codec<lse::f16> {
  static constexpr std::string_view bits_key = "bits.f16";
  static constexpr std::string_view value_key = "value.f16";
  static std::uint32_t bits(float x) { return float16_t::from_float(x); }
  static float value(std::uint32_t b) {
    float16_t h;
    h.bits = static_cast<std::uint16_t>(b);
    return h.to_float();
  }
};

}  // namespace detail

// The dialect rows a narrow format needs, so a caller that must check the
// target can spell them does not restate the strings and drift from these.
template <typename T>
inline constexpr std::string_view bits_key = detail::bit_codec<T>::bits_key;
template <typename T>
inline constexpr std::string_view value_key = detail::bit_codec<T>::value_key;

// The narrow format is the author's choice of storage, so it is named; the
// value's own type is not, because it comes from the argument.
template <typename T>
[[nodiscard]] inline std::uint32_t bits_of(float x) {
  return detail::bit_codec<T>::bits(x);
}
template <typename T>
[[nodiscard]] inline Val<ir::u32> bits_of(const Val<lse::f32>& x) {
  return detail::invoke<ir::u32>(detail::bit_codec<T>::bits_key, x);
}
template <typename T>
[[nodiscard]] inline float from_bits(std::uint32_t b) {
  return detail::bit_codec<T>::value(b);
}
template <typename T>
[[nodiscard]] inline Val<lse::f32> from_bits(
    const Val<ir::u32>& b) {
  return detail::invoke<lse::f32>(detail::bit_codec<T>::value_key, b);
}

// Branch-free choice. Both arms are evaluated when recording, so the losing
// arm must be defined (a division that yields inf is fine; a bad load is not).
inline Val<lse::f32> select(const Val<ir::boolean>& c,
                            const Val<lse::f32>& a, const Val<lse::f32>& b) {
  return ir::select(c, a, b);
}

// Explicit conversion that reads the same in both worlds: recording spells it
// from the backend's type table, the host does the plain static_cast.
template <typename To, typename From>
[[nodiscard]] inline Val<To> cast(const Val<From>& v) {
  return ir::cast<To>(v);
}
template <typename To, typename From>
  requires std::is_arithmetic_v<From>
[[nodiscard]] inline To cast(From v) {
  return static_cast<To>(v);
}

// Storage element -> accumulate type, in register. Exact for bf16 and f16:
// both are strict subsets of f32, so this adds no information and loses none.
// The identity overloads let one body serve every weight storage type.
template <typename T>
[[nodiscard]] inline Val<lse::f32> widen(const Val<T>& v) {
  if constexpr (std::is_same_v<T, lse::f32>) {
    return v;
  } else {
    return ir::cast<lse::f32>(v);
  }
}
[[nodiscard]] inline float widen(float v) { return v; }

// Accumulate type -> storage/operand element. Narrowing, so it rounds.
template <typename To, typename From>
[[nodiscard]] inline Val<To> narrow(const Val<From>& v) {
  if constexpr (std::is_same_v<To, From>) {
    return v;
  } else {
    return ir::cast<To>(v);
  }
}

inline Val<ir::u32> local_id() { return emit<op::LocalId>(); }
inline Val<ir::u32> workgroup_id_x() {
  return emit<op::WorkgroupIdX>();
}
inline Val<ir::u32> workgroup_id_y() {
  return emit<op::WorkgroupIdY>();
}
inline Val<ir::u32> workgroup_size() {
  return emit<op::WorkgroupSize>();
}
inline Val<ir::u32> grid_dim_x() { return emit<op::GridDimX>(); }
inline void barrier() {
  ir::KernelBody* body = ir::KernelBody::try_current();
  if (body != nullptr) body->barrier();
}

// `x` supplies the signed bytes, `codes` the unsigned ones. Both ride in an
// i32 lane; the byte pairing is the instruction's and is measured, not chosen
// here — see quant::dot4_operand_slot.
inline Val<ir::i32> dot4_iu8(const Val<ir::i32>& x, const Val<ir::i32>& codes,
                             const Val<ir::i32>& acc) {
  return emit<op::Dot4Iu8>(x, codes, acc);
}

inline Val<lse::f32> shfl_xor(const Val<lse::f32>& v,
                              const Val<ir::u32>& mask) {
  return emit<op::ShflXor>(v, mask);
}

// The workgroup scratch pad for the kernel being emitted. Allocations that
// would exceed the device's LDS budget fail instead of producing a bad launch.
inline ir::Lds& lds() {
  return ir::KernelBody::try_current()->lds();
}

// `max_bytes` is the live device property (HRX MAX_LOAD/STORE_BYTES). The
// call does not name 4 or 2; the recorder snaps that budget to an ISA op.
template <typename T>
inline ir::Pack<T> load(const ir::Buffer<T>& buf,
                                const ir::Val<ir::u32>& index,
                                std::uint32_t max_bytes) {
  return buf.load(index, max_bytes);
}

template <typename T>
inline void store(const ir::Buffer<T>& buf,
                  const ir::Val<ir::u32>& index,
                  const ir::Pack<T>& value, std::uint32_t max_bytes) {
  buf.store(index, value, max_bytes);
}

// ---------------------------------------------------------------------------
// The matrix core, as a descriptor rather than a name
// ---------------------------------------------------------------------------
//
// One shape carries many instructions whose per-lane fragments are all
// different widths, and the widths change again with the device generation. A
// kernel that spells any of them works for exactly one combination, so the
// table row carries all of it and the kernel asks the row.

// The formats a matrix core multiplies in. Deliberately wider than kir::Scalar:
// int4, fp8 and bf8 are operand formats with no scalar spelling — they ride
// packed inside an i32 lane — so keying on Scalar would make them unnameable.
enum class MatrixElem : std::uint8_t {
  kF32, kF16, kBF16, kI32, kI8, kI4, kFp8, kBf8,
};

// The ISA generation a row belongs to. The same (accumulator, operand, shape)
// is a different instruction with different per-lane widths on each, so the
// generation is part of the key rather than something a spelling hides.
enum class MatrixTarget : std::uint8_t { kRdna3, kRdna4, kCdna3 };

// What the live device must have for a row to be legal. Spelling and
// availability stay separate concerns: the dialect table says how a target
// spells the instruction, this says whether this device has it at all.
enum class MatrixCap : std::uint32_t {
  kNone = 0,
  kWmmaF16 = 1u << 0,
  kWmmaBf16 = 1u << 1,
  kWmmaInt8 = 1u << 2,
  kWmmaInt4 = 1u << 3,
  kWmma12F16 = 1u << 4,
  kWmma12Bf16 = 1u << 5,
  kWmma12Int8 = 1u << 6,
  kWmma12Int4 = 1u << 7,
  kWmma12Fp8 = 1u << 8,
  kMfmaF16 = 1u << 9,
  kMfmaBf16 = 1u << 10,
  kMfmaInt8 = 1u << 11,
  kMfmaFp8 = 1u << 12,
};

[[nodiscard]] constexpr std::uint32_t cap_bits(MatrixCap c) noexcept {
  return static_cast<std::uint32_t>(c);
}
[[nodiscard]] constexpr bool has_cap(std::uint32_t mask, MatrixCap c) noexcept {
  return (mask & cap_bits(c)) != 0u;
}

// How the operand fragment is spread over the wave. Measured on the device,
// not derived — the generations disagree and it is not guessable.
enum class OperandLayout : std::uint8_t {
  // No layout has been established for this row on real hardware. A row in
  // this state is a description, never something to emit.
  kUnmeasured = 0,
  // Lane L supplies row L of the operand and holds the whole k step
  // contiguously, so a[frag] = a[row * K + k0 + frag]. Measured on gfx1151.
  kLaneRowContiguousK,
};

// Where lane L's accumulator element e lands in the output tile.
enum class AccLayout : std::uint8_t {
  kUnmeasured = 0,
  // D[2*e + lane/N][lane%N]. Measured on gfx1151, 256/256 slots, max err 5e-7.
  kPairRowHalfWave,
};

// A row of the matrix-core table. Everything a kernel would otherwise have
// hardcoded: the widths, the wave, the K step, where the answer lands, and
// what the device has to have.
struct MatrixCoreRow {
  std::string_view key;   // dialect row; the backend table spells it

  MatrixTarget target;
  MatrixElem acc;         // logical accumulator format
  MatrixElem operand;     // logical operand format

  // Per-lane fragment registers. The element type is what the *register* holds,
  // which is not the operand format when the operand is packed: iu8 rides four
  // to an i32 lane, iu4 eight.
  ir::Scalar a_elem;
  int a_len;
  ir::Scalar b_elem;
  int b_len;
  ir::Scalar c_elem;
  int c_len;

  // Operand values per fragment element. 1 for a float operand; 4 for int8 in
  // an i32 lane, 8 for int4 or for fp8 in an i64 lane. A buffer holding the
  // operands in native format therefore has K/pack elements per row.
  int pack;

  int m, n, k;      // the instruction's own tile
  int wave;         // lanes that cooperate on one tile

  MatrixCap cap;

  // A row may be more than one instruction. Native 8-bit operands read 64 bits
  // per lane, which half-wastes a 128-bit load path, so the efficient form
  // chains two and steps K by twice the instruction's own K.
  int chained;
  int k_step;       // == k * chained; also the K this row is keyed by

  // Relative, per CU, with RDNA3 16-bit dense == 100. What a selector ranks by.
  int throughput;

  OperandLayout operands;
  AccLayout acc_layout;

  // A row that describes an instruction nobody has run here. It is still a
  // real row — name, widths, capability, shape — it simply must not be
  // emitted, because a guessed lane mapping is a silent wrong answer.
  [[nodiscard]] constexpr bool emittable() const noexcept {
    return operands != OperandLayout::kUnmeasured &&
           acc_layout != AccLayout::kUnmeasured;
  }
};

namespace detail {

using ir::Scalar;

// THE table. One place a row is added; one place a variant is looked up.
//
// Verified against this toolchain (clang for gfx1151 / gfx1201 / gfx942):
// every key below names a builtin that exists with the widths stated. What is
// NOT verified on hardware is the per-lane layout, and that is exactly what
// `operands` / `acc_layout` record — the gfx11 rows were measured on gfx1151,
// everything else says so and declines.
inline constexpr std::array<MatrixCoreRow, 23> kMatrixCore{{
    // -- RDNA3 / 3.5, wave32. A/B 16 elements per lane, whole K in each lane.
    {"wmma.f32.16x16x16.f16", MatrixTarget::kRdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 16, Scalar::kF16, 16, Scalar::kF32, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmmaF16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    {"wmma.f32.16x16x16.bf16", MatrixTarget::kRdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 16, Scalar::kBF16, 16, Scalar::kF32, 8,
     1, 16, 16, 16, 32, MatrixCap::kWmmaBf16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    // The narrow-accumulate forms take an opsel that selects which half of a
    // 16-wide accumulator the 8 results land in, so their D mapping is not the
    // f32 one and has never been measured here.
    {"wmma.f16.16x16x16.f16", MatrixTarget::kRdna3, MatrixElem::kF16,
     MatrixElem::kF16, Scalar::kF16, 16, Scalar::kF16, 16, Scalar::kF16, 16, 1,
     16, 16, 16, 32, MatrixCap::kWmmaF16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kUnmeasured},
    {"wmma.bf16.16x16x16.bf16", MatrixTarget::kRdna3, MatrixElem::kBF16,
     MatrixElem::kBF16, Scalar::kBF16, 16, Scalar::kBF16, 16, Scalar::kBF16, 16,
     1, 16, 16, 16, 32, MatrixCap::kWmmaBf16, 1, 16, 100,
     OperandLayout::kLaneRowContiguousK, AccLayout::kUnmeasured},
    // Four int8 to an i32 lane, eight int4. Same D layout as the f32 form —
    // measured, see tests/test_jit.cpp matrix_core_int8_*.
    {"wmma.i32.16x16x16.iu8", MatrixTarget::kRdna3, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmmaInt8, 1, 16, 200,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},
    {"wmma.i32.16x16x16.iu4", MatrixTarget::kRdna3, MatrixElem::kI32,
     MatrixElem::kI4, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 8, 16,
     16, 16, 32, MatrixCap::kWmmaInt4, 1, 16, 400,
     OperandLayout::kLaneRowContiguousK, AccLayout::kPairRowHalfWave},

    // -- RDNA4, wave32. Half the RDNA3 operand width: K splits across the
    // half-waves, which is a different fill, not just a narrower one. gfx1201
    // is offline here, so nothing below has a measured layout.
    {"wmma12.f32.16x16x16.f16", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 8, Scalar::kF16, 8, Scalar::kF32, 8, 1, 16,
     16, 16, 32, MatrixCap::kWmma12F16, 1, 16, 200, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"wmma12.f32.16x16x16.bf16", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 8, Scalar::kBF16, 8, Scalar::kF32, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmma12Bf16, 1, 16, 200,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.f16.16x16x16.f16", MatrixTarget::kRdna4, MatrixElem::kF16,
     MatrixElem::kF16, Scalar::kF16, 8, Scalar::kF16, 8, Scalar::kF16, 8, 1, 16,
     16, 16, 32, MatrixCap::kWmma12F16, 1, 16, 200, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"wmma12.bf16.16x16x16.bf16", MatrixTarget::kRdna4, MatrixElem::kBF16,
     MatrixElem::kBF16, Scalar::kBF16, 8, Scalar::kBF16, 8, Scalar::kBF16, 8, 1,
     16, 16, 16, 32, MatrixCap::kWmma12Bf16, 1, 16, 200,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.i32.16x16x16.iu8", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Int8, 1, 16, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    // The double-K form of the same instruction: 8 int8 per lane is a 64-bit
    // read on a 128-bit path, so two chained steps fill the load.
    {"wmma12.i32.16x16x16.iu8", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kI32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Int8, 2, 32, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.i32.16x16x32.iu4", MatrixTarget::kRdna4, MatrixElem::kI32,
     MatrixElem::kI4, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kI32, 8, 8, 16,
     16, 32, 32, MatrixCap::kWmma12Int4, 1, 32, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"wmma12.f32.16x16x16.fp8_fp8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 1, 16, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"wmma12.f32.16x16x16.bf8_bf8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kBf8, Scalar::kI32, 2, Scalar::kI32, 2, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 1, 16, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"wmma12.f32.16x16x16.fp8_fp8", MatrixTarget::kRdna4, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI32, 4, Scalar::kI32, 4, Scalar::kF32, 8, 4, 16,
     16, 16, 32, MatrixCap::kWmma12Fp8, 2, 32, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},

    // -- CDNA3, wave64, MFMA. A different instruction family, not a wider
    // WMMA: 64 lanes cooperate and a lane holds a slice of K, not all of it.
    // No MI300X here.
    {"mfma.f32.16x16x16.f16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 4, Scalar::kF16, 4, Scalar::kF32, 4, 1, 16,
     16, 16, 64, MatrixCap::kMfmaF16, 1, 16, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.32x32x8.f16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kF16, Scalar::kF16, 4, Scalar::kF16, 4, Scalar::kF32, 16, 1,
     32, 32, 8, 64, MatrixCap::kMfmaF16, 1, 8, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.16x16x16.bf16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 4, Scalar::kBF16, 4, Scalar::kF32, 4, 1,
     16, 16, 16, 64, MatrixCap::kMfmaBf16, 1, 16, 400,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
    {"mfma.f32.32x32x8.bf16", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kBF16, Scalar::kBF16, 4, Scalar::kBF16, 4, Scalar::kF32, 16, 1,
     32, 32, 8, 64, MatrixCap::kMfmaBf16, 1, 8, 400, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    // MFMA takes its packed operands as one i64 per lane, not a vector.
    {"mfma.i32.16x16x32.i8", MatrixTarget::kCdna3, MatrixElem::kI32,
     MatrixElem::kI8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kI32, 4, 8, 16,
     16, 32, 64, MatrixCap::kMfmaInt8, 1, 32, 800, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.16x16x32.fp8_fp8", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kF32, 4, 8, 16,
     16, 32, 64, MatrixCap::kMfmaFp8, 1, 32, 800, OperandLayout::kUnmeasured,
     AccLayout::kUnmeasured},
    {"mfma.f32.32x32x16.fp8_fp8", MatrixTarget::kCdna3, MatrixElem::kF32,
     MatrixElem::kFp8, Scalar::kI64, 1, Scalar::kI64, 1, Scalar::kF32, 16, 8,
     32, 32, 16, 64, MatrixCap::kMfmaFp8, 1, 16, 800,
     OperandLayout::kUnmeasured, AccLayout::kUnmeasured},
}};

}  // namespace detail

// The whole table, for a gate or a selector that has to rank rows at runtime.
[[nodiscard]] constexpr std::span<const MatrixCoreRow> matrix_core_table()
    noexcept {
  return detail::kMatrixCore;
}

// The row for one combination. Consteval, so a combination with no row cannot
// produce a value at all — the author asked for something that does not exist
// and the compiler says so, rather than falling through to whichever variant
// happened to be there first.
[[nodiscard]] consteval MatrixCoreRow matrix_core_row(MatrixTarget target,
                                                      MatrixElem acc,
                                                      MatrixElem operand, int M,
                                                      int N, int K) {
  for (const MatrixCoreRow& r : detail::kMatrixCore) {
    if (r.target == target && r.acc == acc && r.operand == operand &&
        r.m == M && r.n == N && r.k_step == K) {
      return r;
    }
  }
  throw "no matrix-core row for this (target, accumulator, operand, M, N, K)";
}

// The kir element type a fragment register holds. A Scalar with no entry is a
// compile error, not a default.
template <ir::Scalar S>
struct matrix_scalar_type;
template <>
struct matrix_scalar_type<ir::Scalar::kF16> {
  using type = lse::f16;
};
template <>
struct matrix_scalar_type<ir::Scalar::kBF16> {
  using type = lse::bf16;
};
template <>
struct matrix_scalar_type<ir::Scalar::kF32> {
  using type = lse::f32;
};
template <>
struct matrix_scalar_type<ir::Scalar::kI32> {
  using type = std::int32_t;
};
template <>
struct matrix_scalar_type<ir::Scalar::kI64> {
  using type = std::int64_t;
};
template <ir::Scalar S>
using matrix_scalar_t = typename matrix_scalar_type<S>::type;

namespace op {

// The variant op. The tile shape and the target are named because they are
// choices; the fragment types are not, because the row determines them.
template <MatrixTarget G, MatrixElem Acc, MatrixElem T, int M, int N, int K>
struct Mma {
  static consteval MatrixCoreRow row() {
    return matrix_core_row(G, Acc, T, M, N, K);
  }
  static constexpr MatrixCoreRow kRow = row();
  using result = lse::vec<matrix_scalar_t<kRow.c_elem>, kRow.c_len>;
};

}  // namespace op

// D = A*B + C on the matrix core. Everything about the shapes comes from
// `Op`'s row; a target whose table has no spelling for it yields an empty Val
// and the kernel declines — it never substitutes a format the hardware has.
template <class Op, typename A, typename B, typename C>
[[nodiscard]] inline Val<typename Op::result> mma(const Val<A>& a,
                                                  const Val<B>& b,
                                                  const Val<C>& c) {
  return emit<Op>(a, b, c);
}

}  // namespace lse::math

namespace lseMath = lse::math;
