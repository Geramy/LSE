// The kernel IR's type system, and the one table a target supplies for it.
//
// A type is (element, lanes, address space). Lanes is 1 for a scalar, N for a
// register vector, and the element count for a memory reference — an IR value
// that names storage rather than a number. The address space is what makes a
// memory pass able to ask "is this workgroup scratch?" without matching on a
// declaration's spelling.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace lse::ir {

enum class Scalar : std::uint8_t {
  kU8, kI8, kU16, kI16, kU32, kI32, kU64, kI64, kF16, kBF16, kF32, kBool,
};

// Where a memory reference lives. `kNone` is not memory at all — a value.
enum class Space : std::uint8_t {
  kNone,        // an SSA value in a register
  kGlobal,      // a kernel buffer parameter
  kWorkgroup,   // __shared__ / LDS
  kPrivate,     // a per-thread array or vector variable
};

struct Type {
  Scalar elem = Scalar::kF32;
  // Lanes for a vector value, element count for a memory reference, 1 for a
  // scalar, 0 for a buffer whose extent the kernel does not know.
  std::uint32_t lanes = 1;
  Space space = Space::kNone;

  [[nodiscard]] bool is_memory() const noexcept { return space != Space::kNone; }
  [[nodiscard]] bool is_vector() const noexcept {
    return space == Space::kNone && lanes > 1;
  }
  friend bool operator==(const Type&, const Type&) = default;
};

[[nodiscard]] constexpr Type scalar_type(Scalar s) noexcept {
  return Type{s, 1, Space::kNone};
}
[[nodiscard]] constexpr Type vector_type(Scalar s, std::uint32_t n) noexcept {
  return Type{s, n, Space::kNone};
}
[[nodiscard]] constexpr Type memory_type(Scalar s, std::uint32_t n,
                                         Space sp) noexcept {
  return Type{s, n, sp};
}

// How a tensor dimension reaches the generated kernel.
//
// `kLiteral` is the default and is what every extent in this engine is today:
// the number is baked into the source, so the shape is part of the JIT key and
// a new shape is a new kernel. That is deliberate — the inner loops are the
// whole cost model. A constant K is what lets the backend compiler unroll with
// a known trip count, strength-reduce the address chain to constant offsets,
// prove the alignment and width of a packed fetch, and drop boundary
// predication; the lm_head GEMV runs at 242 GB/s on this device, which is the
// hardware roof, and there is no headroom for a non-constant trip count to buy
// anything back.
//
// `kRuntime` says the opposite: this extent is a dispatch constant, so one
// code object serves every value of it. Legal for exactly the extents that
// only ever reach grid/block mapping, an outermost loop bound, or a guard —
// in practice the leading row/batch extent, which is what a continuous-batching
// batch size is. The verifier refuses a runtime extent anywhere else, because
// "K became an argument" is a silent 2x on the one loop that cannot afford it.
enum class ExtentBinding : std::uint8_t { kLiteral, kRuntime };

// What the number MEANS, which is what decides where it may appear. A size
// used in address arithmetic would make the stride itself vary — that is the
// unroll and the constant offsets gone. A window base only moves the origin:
// it enters with a constant coefficient, so the stride is untouched, which is
// why a kernel emitted for a slice of its iteration space is still the same
// inner loop. `ir::verify` holds both to their own rule.
enum class ExtentRole : std::uint8_t { kSize, kWindowBase };

// The whole of a backend's type knowledge. `vector_typedef` returns a
// declaration introducing `name`, because how a target spells a vector is not
// derivable from how it spells a scalar.
struct TypeTable {
  std::string_view (*scalar)(Scalar) noexcept = nullptr;
  std::string (*vector_typedef)(Scalar, int, std::string_view name) = nullptr;
};

// Names the element type in a generated vector typedef. Internal to the
// generated source, so it is the IR's spelling and not the backend's.
[[nodiscard]] std::string_view scalar_suffix(Scalar s) noexcept;

[[nodiscard]] std::uint32_t scalar_bytes(Scalar s) noexcept;

}  // namespace lse::ir
