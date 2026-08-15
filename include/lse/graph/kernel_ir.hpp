// Authoring a device kernel in C++ instead of in string literals.
//
// A kernel primitive used to build its body by concatenating HIP text, which
// put backend spellings — `unsigned int`, `_Float16`, `__builtin_amdgcn_...` —
// inside the kernel's own logic. That is the thing that has to be written once
// per backend, so it does not belong in the kernel.
//
// Here the kernel is written as ordinary C++. The values are proxies that
// record what is done to them, so *running* the C++ produces the source:
//
//   Val<u32> wave = k.thread_id() / 32u;
//   k.ret_if(wave >= tiles);
//   auto acc = k.local<vec<f32, 8>>("acc");
//   k.unroll(8, [&](Val<u32> e) { acc[e] = 0.0f; });
//   acc = math::wmma_f32_16x16x16(af, bf, acc);
//
// Two things stay backend-specific and both are tables, not code: TypeTable
// spells the scalar and vector types, and the intrinsic name is looked up in
// the backend's DialectSourceTable like any other op. Retarget by supplying
// both; the kernel above is untouched.
//
// This is deliberately a recorder, not a compiler. It has no constant folding
// and no type inference beyond what the C++ type system already does — the
// generated text mirrors the C++ statement for statement, which is what makes
// the output reviewable against the source that produced it.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "lse/core/elem.hpp"
#include "lse/graph/codegen.hpp"
#include "lse/graph/dialect_source.hpp"
#include "lse/graph/primitive.hpp"

namespace lse::graph::kir {

using lse::bf16;
using lse::f16;
using lse::f32;
using lse::vec;

// Recorder predicate, not an engine storage type.
struct boolean {};

enum class Scalar : std::uint8_t {
  kU8, kI8, kU16, kI16, kU32, kI32, kU64, kI64, kF16, kBF16, kF32, kBool,
};

template <typename T> struct scalar_of;
template <> struct scalar_of<std::uint8_t>  { static constexpr Scalar value = Scalar::kU8; };
template <> struct scalar_of<std::int8_t>   { static constexpr Scalar value = Scalar::kI8; };
template <> struct scalar_of<std::uint16_t> { static constexpr Scalar value = Scalar::kU16; };
template <> struct scalar_of<std::int16_t>  { static constexpr Scalar value = Scalar::kI16; };
template <> struct scalar_of<std::uint32_t> { static constexpr Scalar value = Scalar::kU32; };
template <> struct scalar_of<std::int32_t>  { static constexpr Scalar value = Scalar::kI32; };
template <> struct scalar_of<std::uint64_t> { static constexpr Scalar value = Scalar::kU64; };
template <> struct scalar_of<std::int64_t>  { static constexpr Scalar value = Scalar::kI64; };
template <> struct scalar_of<f16>           { static constexpr Scalar value = Scalar::kF16; };
template <> struct scalar_of<bf16>          { static constexpr Scalar value = Scalar::kBF16; };
template <> struct scalar_of<float>         { static constexpr Scalar value = Scalar::kF32; };
template <> struct scalar_of<boolean>       { static constexpr Scalar value = Scalar::kBool; };

// The whole of a backend's type knowledge. `vector_typedef` returns a
// declaration introducing `name`, because how a target spells a vector is not
// derivable from how it spells a scalar.
struct TypeTable {
  std::string_view (*scalar)(Scalar) noexcept = nullptr;
  std::string (*vector_typedef)(Scalar, int, std::string_view name) = nullptr;
};

using u32 = std::uint32_t;
using i32 = std::int32_t;

class KernelBody;

// A recorded expression. The type parameter is carried purely so C++ checks the
// kernel for us; only `text` survives into the output.
template <typename T>
class Val {
 public:
  Val() = default;
  Val(const TypeTable* tt, std::string text)
      : tt_(tt), text_(std::move(text)) {}

  [[nodiscard]] const std::string& text() const noexcept { return text_; }
  [[nodiscard]] const TypeTable* types() const noexcept { return tt_; }

 private:
  const TypeTable* tt_ = nullptr;
  std::string text_;
};

namespace detail {

[[nodiscard]] std::string literal_u32(std::uint32_t v);
[[nodiscard]] std::string literal_i32(std::int32_t v);
// Names the element type in a generated vector typedef. Internal to the
// generated source, so it is kir's spelling and not the backend's.
[[nodiscard]] std::string_view scalar_suffix(Scalar s) noexcept;

template <typename T>
[[nodiscard]] Val<T> lift(const TypeTable*, const Val<T>& v) { return v; }
[[nodiscard]] inline Val<u32> lift(const TypeTable* tt, std::uint32_t v) {
  return {tt, literal_u32(v)};
}
[[nodiscard]] inline Val<i32> lift(const TypeTable* tt, std::int32_t v) {
  return {tt, literal_i32(v)};
}
[[nodiscard]] inline Val<f32> lift(const TypeTable* tt, float v) {
  return {tt, float_literal(v)};
}

// Every binary result is parenthesized: the recorder does not track precedence,
// so it must not rely on it.
template <typename R, typename A, typename B>
[[nodiscard]] Val<R> binop(const A& a, const B& b, std::string_view op) {
  return {a.types(), "(" + a.text() + " " + std::string(op) + " " + b.text() + ")"};
}

}  // namespace detail

#define LSE_KIR_ARITH(op_)                                                     \
  template <typename T>                                                        \
  [[nodiscard]] Val<T> operator op_(const Val<T>& a, const Val<T>& b) {         \
    return detail::binop<T>(a, b, #op_);                                       \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<T> operator op_(const Val<T>& a, S b) {                     \
    return detail::binop<T>(a, detail::lift(a.types(), T(b)), #op_);           \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<T> operator op_(S a, const Val<T>& b) {                     \
    return detail::binop<T>(detail::lift(b.types(), T(a)), b, #op_);           \
  }

LSE_KIR_ARITH(+)
LSE_KIR_ARITH(-)
LSE_KIR_ARITH(*)
LSE_KIR_ARITH(/)
LSE_KIR_ARITH(%)
#undef LSE_KIR_ARITH

#define LSE_KIR_CMP(op_)                                                       \
  template <typename T>                                                        \
  [[nodiscard]] Val<boolean> operator op_(const Val<T>& a, const Val<T>& b) {   \
    return detail::binop<boolean>(a, b, #op_);                                 \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<boolean> operator op_(const Val<T>& a, S b) {               \
    return detail::binop<boolean>(a, detail::lift(a.types(), T(b)), #op_);     \
  }

LSE_KIR_CMP(<)
LSE_KIR_CMP(>)
LSE_KIR_CMP(<=)
LSE_KIR_CMP(>=)
LSE_KIR_CMP(==)
LSE_KIR_CMP(!=)
#undef LSE_KIR_CMP

[[nodiscard]] inline Val<boolean> operator&&(const Val<boolean>& a,
                                             const Val<boolean>& b) {
  return detail::binop<boolean>(a, b, "&&");
}
[[nodiscard]] inline Val<boolean> operator||(const Val<boolean>& a,
                                             const Val<boolean>& b) {
  return detail::binop<boolean>(a, b, "||");
}

// Conversion is explicit and spelled by the table: a kernel says "as f16",
// never "_Float16".
template <typename To, typename From>
[[nodiscard]] Val<To> cast(const Val<From>& v) {
  return {v.types(), "(" + std::string(v.types()->scalar(scalar_of<To>::value)) +
                         ")(" + v.text() + ")"};
}

template <typename T>
[[nodiscard]] Val<T> select(const Val<boolean>& cond, const Val<T>& a,
                            const Val<T>& b) {
  return {a.types(), "(" + cond.text() + " ? " + a.text() + " : " + b.text() + ")"};
}

// An addressable location. operator= records a store rather than rebinding, so
// a kernel assigns to a buffer slot or a local the way it would in plain C++.
template <typename T>
class LValue {
 public:
  LValue(KernelBody* body, const TypeTable* tt, std::string text)
      : body_(body), tt_(tt), text_(std::move(text)) {}
  // Explicit because operator= here is a *store*, not a rebind, and providing
  // it suppresses the implicit copy constructor that a helper returning an
  // accumulator needs.
  LValue(const LValue&) = default;

  const LValue& operator=(const Val<T>& v) const;
  const LValue& operator=(float v) const;
  const LValue& operator=(const LValue& other) const { return *this = Val<T>(other); }

  operator Val<T>() const { return {tt_, text_}; }  // NOLINT: reads as a value
  [[nodiscard]] Val<T> read() const { return {tt_, text_}; }
  [[nodiscard]] const std::string& text() const noexcept { return text_; }

 private:
  KernelBody* body_;
  const TypeTable* tt_;
  std::string text_;
};

// A register vector whose width was snapped from a live byte budget
// (HRX MAX_LOAD_BYTES / MAX_STORE_BYTES), not written as 4 or 2 at the
// call site. `width()` is how many T's that budget bought.
template <typename T>
class Pack {
 public:
  Pack() = default;
  Pack(const TypeTable* tt, std::string text, std::uint32_t n)
      : tt_(tt), text_(std::move(text)), n_(n) {}

  [[nodiscard]] std::uint32_t width() const noexcept { return n_; }
  [[nodiscard]] const std::string& text() const noexcept { return text_; }

  [[nodiscard]] Val<T> operator[](const Val<u32>& i) const {
    if (n_ <= 1) return {tt_, text_};
    return {tt_, text_ + "[" + i.text() + "]"};
  }
  [[nodiscard]] Val<T> operator[](int i) const {
    if (n_ <= 1) return {tt_, text_};
    return {tt_, text_ + "[" + std::to_string(i) + "]"};
  }

 private:
  const TypeTable* tt_ = nullptr;
  std::string text_;
  std::uint32_t n_ = 1;
};

template <typename T>
class Buffer {
 public:
  Buffer() = default;
  Buffer(KernelBody* body, const TypeTable* tt, std::string name)
      : body_(body), tt_(tt), name_(std::move(name)) {}

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, name_ + "[" + index.text() + "]"};
  }

  // `max_bytes` is the device property (16 on gfx1151). The instruction
  // width is derived from that, not passed as 4 or 2.
  [[nodiscard]] Pack<T> load(const Val<u32>& index,
                             std::uint32_t max_bytes) const;
  void store(const Val<u32>& index, const Pack<T>& value,
             std::uint32_t max_bytes) const;

  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  KernelBody* body_ = nullptr;
  const TypeTable* tt_ = nullptr;
  std::string name_;
};

template <typename T>
class Tile;

template <typename T>
constexpr std::uint32_t pack_elem_bytes() noexcept;

// Workgroup scratch. The generator owns one of these per kernel: every
// `__shared__` tile is reserved here, 16-byte aligned, and a request that
// would pass the device's LDS budget is refused so we never emit a launch
// the hardware cannot run. Budget 0 means "unknown" and is not enforced.
class Lds {
 public:
  static constexpr std::uint32_t kAlign = 16;

  Lds() = default;
  explicit Lds(std::uint32_t budget_bytes) : budget_(budget_bytes) {}

  void attach(KernelBody* body) noexcept { body_ = body; }
  void set_budget(std::uint32_t bytes) noexcept { budget_ = bytes; }

  [[nodiscard]] std::uint32_t budget() const noexcept { return budget_; }
  [[nodiscard]] std::uint32_t used() const noexcept { return used_; }
  [[nodiscard]] std::uint32_t remaining() const noexcept {
    return budget_ > used_ ? budget_ - used_ : 0;
  }
  [[nodiscard]] bool ok() const noexcept { return ok_; }

  // True when `bytes` (after alignment) fits. Budget 0 always fits.
  [[nodiscard]] bool fits(std::uint32_t bytes) const noexcept {
    const std::uint32_t need = align(bytes);
    return budget_ == 0 || used_ + need <= budget_;
  }

  // Reserve without emitting. False if it would overflow a known budget.
  bool reserve(std::uint32_t bytes);

  template <typename T>
  bool reserve_elems(std::uint32_t count) {
    return reserve(count * pack_elem_bytes<T>());
  }

  // Allocate `count` elements and emit the declaration. Empty on overflow.
  template <typename T>
  Tile<T> array(std::string_view name, std::uint32_t count);

  static constexpr std::uint32_t align(std::uint32_t bytes) noexcept {
    return (bytes + (kAlign - 1u)) & ~(kAlign - 1u);
  }

 private:
  KernelBody* body_ = nullptr;
  std::uint32_t budget_ = 0;
  std::uint32_t used_ = 0;
  bool ok_ = true;
};

// Runtime-sized LDS tile. `operator bool` is whether the reserve succeeded.
template <typename T>
class Tile {
 public:
  Tile() = default;
  Tile(KernelBody* body, const TypeTable* tt, std::string name,
       std::uint32_t count)
      : body_(body), tt_(tt), name_(std::move(name)), n_(count) {}

  [[nodiscard]] explicit operator bool() const noexcept {
    return body_ != nullptr;
  }
  [[nodiscard]] std::uint32_t size() const noexcept { return n_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, name_ + "[" + index.text() + "]"};
  }
  [[nodiscard]] LValue<T> operator[](int index) const {
    return {body_, tt_, name_ + "[" + std::to_string(index) + "]"};
  }

 private:
  KernelBody* body_ = nullptr;
  const TypeTable* tt_ = nullptr;
  std::string name_;
  std::uint32_t n_ = 0;
};

// A local whose elements are addressable — a fragment register, in practice.
template <typename T, int N>
class Local {
 public:
  Local(KernelBody* body, const TypeTable* tt, std::string name)
      : body_(body), tt_(tt), name_(std::move(name)) {}

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, name_ + "[" + index.text() + "]"};
  }
  [[nodiscard]] LValue<T> operator[](int index) const {
    return {body_, tt_, name_ + "[" + std::to_string(index) + "]"};
  }

  [[nodiscard]] Val<vec<T, N>> value() const { return {tt_, name_}; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

  // Whole-register assignment, which is how an accumulator takes the result of
  // a matrix instruction.
  const Local& operator=(const Val<vec<T, N>>& v) const;

 private:
  KernelBody* body_;
  const TypeTable* tt_;
  std::string name_;
};

class KernelBody {
 public:
  KernelBody(const TypeTable& types, const DialectSourceTable& intrinsics,
             std::uint32_t lds_budget = 0)
      : types_(&types), intrinsics_(&intrinsics), lds_(lds_budget) {
    lds_.attach(this);
    bind();
  }
  ~KernelBody() { unbind(); }
  KernelBody(const KernelBody&) = delete;
  KernelBody& operator=(const KernelBody&) = delete;

  // The body `lse::math::*` records into. Null outside emit_kernel.
  [[nodiscard]] static KernelBody* try_current() noexcept;

  // The flat thread id the emitter binds before the body runs.
  [[nodiscard]] Val<u32> thread_id() const { return {types_, "i"}; }

  template <typename T>
  [[deprecated("declare an env::In member and env::bind the args struct")]]
  [[nodiscard]] Buffer<T> input(std::size_t index) {
    return {this, types_, "in" + std::to_string(index)};
  }
  template <typename T>
  [[deprecated("declare an env::Out member and env::bind the args struct")]]
  [[nodiscard]] Buffer<T> output() {
    return {this, types_, "out"};
  }

  template <typename T>
  [[deprecated("use env::Emit::u32 or a bare literal in a Val expression")]]
  [[nodiscard]] Val<T> constant(std::uint32_t v) {
    return {types_, detail::literal_u32(v)};
  }
  [[nodiscard]] Val<f32> lit(float v) const { return {types_, float_literal(v)}; }

  // A named immutable binding, so a subexpression used twice is computed once
  // and the generated source stays readable.
  template <typename T>
  [[nodiscard]] Val<T> let(std::string_view name, const Val<T>& v);

  template <typename T, int N>
  [[nodiscard]] Local<T, N> local(std::string_view name);

  // Workgroup scratch. Goes through `lds()` so the byte budget is enforced.
  template <typename T, int N>
  [[nodiscard]] Local<T, N> shared(std::string_view name);

  [[nodiscard]] Lds& lds() noexcept { return lds_; }
  [[nodiscard]] const Lds& lds() const noexcept { return lds_; }

  // Workgroup barrier. Spelling is the `barrier` table row.
  void barrier();

  template <typename T>
  [[nodiscard]] Pack<T> load_pack(std::string_view buf, const Val<u32>& index,
                                  std::uint32_t max_bytes);
  template <typename T>
  void store_pack(std::string_view buf, const Val<u32>& index, const Pack<T>& v,
                  std::uint32_t max_bytes);

  // A mutable local — an accumulator. `let` is the immutable form.
  template <typename T>
  [[nodiscard]] LValue<T> var(std::string_view name, const Val<T>& init) {
    statement(std::string(types_->scalar(scalar_of<T>::value)) + " " +
              std::string(name) + " = " + init.text() + ";");
    return {this, types_, std::string(name)};
  }

  void ret();
  void ret_if(const Val<boolean>& cond);
  // The per-element form: a kernel primitive that is not self-indexing returns
  // the value for its output element rather than storing it.
  template <typename T>
  void ret(const Val<T>& v) {
    statement("return " + v.text() + ";");
  }

  template <typename F>
  [[deprecated("use if (auto g = e.when(cond)) { ... }")]]
  void when(const Val<boolean>& cond, F&& body);

  // A counted loop. `step` defaults to 1; `unroll` asks the target to unroll,
  // which is what makes fragment indices constant.
  template <typename F>
  [[deprecated("use range-for over env::Emit::range")]]
  void loop(std::string_view var, const Val<u32>& lo, const Val<u32>& hi,
            std::uint32_t step, F&& body);
  template <typename F>
  [[deprecated("use range-for over env::Emit::range")]]
  void loop(std::string_view var, const Val<u32>& lo, const Val<u32>& hi,
            const Val<u32>& step, F&& body);
  template <typename F>
  [[deprecated("use range-for over env::Emit::unroll")]]
  void unroll(std::string_view var, std::uint32_t count, F&& body);

  // An intrinsic, spelled by the backend's table. `$0`.. are the arguments, so
  // a target that needs a different call shape supplies a different entry
  // rather than different kernel code.
  template <typename R, typename... A>
  [[nodiscard]] Val<R> call(std::string_view intrinsic, const A&... args);

  // Installed by the emitter. A kernel stores through this rather than
  // assigning the output buffer, which is what lets a fused epilogue run on the
  // value before it reaches memory.
  using StoreFn =
      std::function<std::string(std::string_view index, std::string_view value)>;
  void set_store(StoreFn fn) { store_ = std::move(fn); }
  [[nodiscard]] bool has_store() const noexcept { return static_cast<bool>(store_); }
  void store(const Val<u32>& index, const Val<f32>& value);

  [[nodiscard]] const TypeTable& types() const noexcept { return *types_; }

  // The body text, with any vector typedefs the kernel used declared at the
  // top. Function-scope typedefs keep this self-contained: a kernel authored
  // here needs nothing added to the emitter's preamble.
  [[nodiscard]] std::string str() const;

  void statement(std::string text);

 private:
  void bind() noexcept;
  void unbind() noexcept;

  static thread_local KernelBody* current_;
  KernelBody* prev_ = nullptr;

  template <typename T, int N>
  [[nodiscard]] std::string vector_type();
  [[nodiscard]] std::string vector_type_n(Scalar s, int n);

  [[nodiscard]] std::string indent() const {
    return std::string(static_cast<std::size_t>(2 * depth_), ' ');
  }

  const TypeTable* types_;
  const DialectSourceTable* intrinsics_;
  std::vector<std::string> lines_;
  std::vector<std::string> typedefs_;
  StoreFn store_;
  Lds lds_;
  int depth_ = 1;
  int temp_ = 0;

  template <typename T> friend class LValue;
  friend class Lds;
};

template <typename T, int N>
const Local<T, N>& Local<T, N>::operator=(const Val<vec<T, N>>& v) const {
  body_->statement(name_ + " = " + v.text() + ";");
  return *this;
}

template <typename T>
const LValue<T>& LValue<T>::operator=(const Val<T>& v) const {
  body_->statement(text_ + " = " + v.text() + ";");
  return *this;
}

template <typename T>
const LValue<T>& LValue<T>::operator=(float v) const {
  body_->statement(text_ + " = " + float_literal(v) + ";");
  return *this;
}

template <typename T, int N>
[[nodiscard]] Val<T> lane(const Val<vec<T, N>>& v, int i) {
  return {v.types(), v.text() + "[" + std::to_string(i) + "]"};
}

template <typename T, int N>
[[nodiscard]] Val<T> lane(const Val<vec<T, N>>& v, const Val<u32>& i) {
  return {v.types(), v.text() + "[" + i.text() + "]"};
}

template <typename T>
constexpr std::uint32_t pack_elem_bytes() noexcept {
  if constexpr (std::is_same_v<T, f16> || std::is_same_v<T, bf16>) return 2;
  return 4;
}

inline std::uint32_t pack_n(std::uint32_t max_bytes,
                            std::uint32_t elem_bytes) noexcept {
  if (elem_bytes == 0 || max_bytes < elem_bytes) return 1;
  const std::uint32_t n = max_bytes / elem_bytes;
  if (n >= 4) return 4;
  if (n >= 2) return 2;
  return 1;
}

template <typename T>
Pack<T> Buffer<T>::load(const Val<u32>& index, std::uint32_t max_bytes) const {
  return body_->load_pack<T>(name_, index, max_bytes);
}

template <typename T>
void Buffer<T>::store(const Val<u32>& index, const Pack<T>& value,
                      std::uint32_t max_bytes) const {
  body_->store_pack<T>(name_, index, value, max_bytes);
}

template <typename T>
Val<T> KernelBody::let(std::string_view name, const Val<T>& v) {
  const std::string decl(name);
  statement("const " + std::string(types_->scalar(scalar_of<T>::value)) + " " +
            decl + " = " + v.text() + ";");
  return {types_, decl};
}

inline std::string KernelBody::vector_type_n(Scalar s, int n) {
  const std::string name =
      "lse_v" + std::to_string(n) + "_" + std::string(detail::scalar_suffix(s));
  const std::string decl = types_->vector_typedef(s, n, name);
  for (const std::string& t : typedefs_) {
    if (t == decl) return name;
  }
  typedefs_.push_back(decl);
  return name;
}

template <typename T, int N>
std::string KernelBody::vector_type() {
  return vector_type_n(scalar_of<T>::value, N);
}

template <typename T, int N>
Local<T, N> KernelBody::local(std::string_view name) {
  const std::string type = vector_type<T, N>();
  statement(type + " " + std::string(name) + ";");
  return {this, types_, std::string(name)};
}

inline bool Lds::reserve(std::uint32_t bytes) {
  const std::uint32_t need = align(bytes);
  if (budget_ != 0 && used_ + need > budget_) {
    ok_ = false;
    return false;
  }
  used_ += need;
  return true;
}

template <typename T>
Tile<T> Lds::array(std::string_view name, std::uint32_t count) {
  if (body_ == nullptr || !reserve(count * pack_elem_bytes<T>())) return {};
  // Uniqued per body, not per process: the generated text must be identical
  // for the same group on every run or the disk cache can never hit.
  const std::string ident =
      std::string(name) + "_" + std::to_string(body_->temp_++);
  const std::string_view storage = body_->intrinsics_->find("shared");
  body_->statement(std::string(storage) + " " +
                   std::string(body_->types_->scalar(scalar_of<T>::value)) +
                   " " + ident + "[" + std::to_string(count) + "];");
  return {body_, body_->types_, ident, count};
}

template <typename T, int N>
Local<T, N> KernelBody::shared(std::string_view name) {
  if (!lds_.reserve(static_cast<std::uint32_t>(N) * pack_elem_bytes<T>())) {
    return {this, types_, std::string(name)};
  }
  const std::string_view storage = intrinsics_->find("shared");
  statement(std::string(storage) + " " +
            std::string(types_->scalar(scalar_of<T>::value)) + " " +
            std::string(name) + "[" + std::to_string(N) + "];");
  return {this, types_, std::string(name)};
}

inline void KernelBody::barrier() {
  const std::string_view spell = intrinsics_->find("barrier");
  if (spell.empty()) return;
  statement(std::string(spell) + ";");
}

template <typename T>
Pack<T> KernelBody::load_pack(std::string_view buf, const Val<u32>& index,
                              std::uint32_t max_bytes) {
  const auto n = pack_n(max_bytes, pack_elem_bytes<T>());
  const std::string name = "ld" + std::to_string(temp_++);
  if (n <= 1) {
    statement("const " + std::string(types_->scalar(scalar_of<T>::value)) +
              " " + name + " = " + std::string(buf) + "[" + index.text() +
              "];");
    return {types_, name, 1};
  }
  const std::string type = vector_type_n(scalar_of<T>::value, static_cast<int>(n));
  statement("const " + type + " " + name + " = *(const " + type + "*)(&" +
            std::string(buf) + "[" + index.text() + "]);");
  return {types_, name, n};
}

template <typename T>
void KernelBody::store_pack(std::string_view buf, const Val<u32>& index,
                            const Pack<T>& v, std::uint32_t max_bytes) {
  const auto n = pack_n(max_bytes, pack_elem_bytes<T>());
  const auto use = n < v.width() ? n : v.width();
  if (use <= 1) {
    statement(std::string(buf) + "[" + index.text() + "] = " + v.text() + ";");
    return;
  }
  const std::string type = vector_type_n(scalar_of<T>::value, static_cast<int>(use));
  statement("*(" + type + "*)(&" + std::string(buf) + "[" + index.text() +
            "]) = " + v.text() + ";");
}

template <typename F>
void KernelBody::when(const Val<boolean>& cond, F&& body) {
  statement("if (" + cond.text() + ") {");
  ++depth_;
  std::forward<F>(body)();
  --depth_;
  statement("}");
}

template <typename F>
void KernelBody::loop(std::string_view var, const Val<u32>& lo,
                      const Val<u32>& hi, std::uint32_t step, F&& body) {
  loop(var, lo, hi, Val<u32>{types_, detail::literal_u32(step)},
       std::forward<F>(body));
}

template <typename F>
void KernelBody::loop(std::string_view var, const Val<u32>& lo,
                      const Val<u32>& hi, const Val<u32>& step, F&& body) {
  const std::string v(var);
  const std::string ty(types_->scalar(Scalar::kU32));
  statement("for (" + ty + " " + v + " = " + lo.text() + "; " + v + " < " +
            hi.text() + "; " + v + " += " + step.text() + ") {");
  ++depth_;
  std::forward<F>(body)(Val<u32>{types_, v});
  --depth_;
  statement("}");
}

template <typename F>
void KernelBody::unroll(std::string_view var, std::uint32_t count, F&& body) {
  lines_.push_back("#pragma unroll");
  loop(var, Val<u32>{types_, detail::literal_u32(0)},
       Val<u32>{types_, detail::literal_u32(count)}, 1, std::forward<F>(body));
}

template <typename R, typename... A>
Val<R> KernelBody::call(std::string_view intrinsic, const A&... args) {
  const std::string_view tmpl = intrinsics_->find(intrinsic);
  if (tmpl.empty()) return {};  // caller reports the gap; see emit_kernel
  const std::vector<std::string> argv{args.text()...};
  return {types_, substitute(tmpl, argv)};
}

}  // namespace lse::graph::kir
