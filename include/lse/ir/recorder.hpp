// Authoring a device kernel in C++ instead of in string literals.
//
// A kernel primitive used to build its body by concatenating HIP text, which
// put backend spellings — `unsigned int`, `_Float16`, `__builtin_amdgcn_...` —
// inside the kernel's own logic. That is the thing that has to be written once
// per backend, so it does not belong in the kernel.
//
// Here the kernel is written as ordinary C++. The values are proxies that
// record what is done to them, so *running* the C++ builds an IR body:
//
//   Val<u32> wave = k.thread_id() / 32u;
//   auto acc = k.local<vec<f32, 8>>("acc");
//   acc = math::wmma_16x16x16(af, bf, acc);
//
// The proxies below are a thin front end over `ir::Body`: each records an op
// and hands back the value it defined. Nothing here holds text. Turning the
// body into source is a separate step (`ir::lower`), and what a target *calls*
// a type or an intrinsic arrives there through two tables — `TypeTable` spells
// the scalars and vectors, `DialectSourceTable` spells the ops. Retarget by
// supplying both; the kernel above is untouched.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "lse/core/elem.hpp"
#include "lse/ir/body.hpp"
#include "lse/ir/dialect.hpp"
#include "lse/ir/lower.hpp"
#include "lse/ir/spell.hpp"
#include "lse/ir/types.hpp"

namespace lse::ir {

using lse::bf16;
using lse::f16;
using lse::f32;
using lse::vec;

// Recorder predicate, not an engine storage type.
struct boolean {};

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

// The IR type an author's C++ type stands for. A fragment register is
// `vec<T, N>` in the kernel and a vector value here; everything else is a
// scalar. A type with neither mapping is a compile error, not a default.
template <typename T>
struct ir_type_of {
  [[nodiscard]] static constexpr Type get() noexcept {
    return scalar_type(scalar_of<T>::value);
  }
};
template <typename T, int N>
struct ir_type_of<vec<T, N>> {
  [[nodiscard]] static constexpr Type get() noexcept {
    return vector_type(scalar_of<T>::value, static_cast<std::uint32_t>(N));
  }
};

using u32 = std::uint32_t;
using i32 = std::int32_t;

class KernelBody;

// A recorded expression: an IR value plus the C++ type the author gave it, so
// the C++ type system checks the kernel for us. Only the value survives.
template <typename T>
class Val {
 public:
  Val() = default;
  Val(const TypeTable* tt, Body* body, ValueId id)
      : tt_(tt), body_(body), id_(id) {}
  // The raw-text form: a kernel naming something the recorder did not define
  // (a dispatch constant). Recorded as a symbol so it is still a real value.
  Val(const TypeTable* tt, std::string text);

  [[nodiscard]] std::string text() const {
    return body_ == nullptr || id_ == kNoValue ? std::string{}
                                               : render(*body_, id_);
  }
  [[nodiscard]] const TypeTable* types() const noexcept { return tt_; }
  [[nodiscard]] Body* body() const noexcept { return body_; }
  [[nodiscard]] ValueId id() const noexcept { return id_; }
  [[nodiscard]] bool valid() const noexcept {
    return body_ != nullptr && id_ != kNoValue;
  }

 private:
  const TypeTable* tt_ = nullptr;
  Body* body_ = nullptr;
  ValueId id_ = kNoValue;
};

namespace detail {

template <typename T>
[[nodiscard]] Val<T> lift(const Val<T>&, const Val<T>& v) { return v; }

[[nodiscard]] Val<u32> lift_u32(Body* b, const TypeTable* tt, std::uint32_t v);
[[nodiscard]] Val<i32> lift_i32(Body* b, const TypeTable* tt, std::int32_t v);
[[nodiscard]] Val<f32> lift_f32(Body* b, const TypeTable* tt, float v);

template <typename T>
[[nodiscard]] Val<T> lift_scalar(Body* b, const TypeTable* tt, T v) {
  if constexpr (std::is_same_v<T, float>) return lift_f32(b, tt, v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return lift_i32(b, tt, v);
  else return lift_u32(b, tt, static_cast<std::uint32_t>(v));
}

// Every binary result is parenthesized: the recorder does not track precedence,
// so it must not rely on it.
template <typename R, typename A, typename B>
[[nodiscard]] Val<R> binop(const A& a, const B& b, std::string_view op) {
  Body* body = a.body();
  if (body == nullptr) return {};
  Operation o;
  o.kind = OpKind::kBinary;
  o.type = ir_type_of<R>::get();
  o.operands = {a.id(), b.id()};
  o.key = std::string(op);
  return {a.types(), body, body->add_value(std::move(o))};
}

}  // namespace detail

#define LSE_IR_ARITH(op_)                                                      \
  template <typename T>                                                        \
  [[nodiscard]] Val<T> operator op_(const Val<T>& a, const Val<T>& b) {        \
    return detail::binop<T>(a, b, #op_);                                       \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<T> operator op_(const Val<T>& a, S b) {                    \
    return detail::binop<T>(                                                   \
        a, detail::lift_scalar<T>(a.body(), a.types(), T(b)), #op_);           \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<T> operator op_(S a, const Val<T>& b) {                    \
    return detail::binop<T>(                                                   \
        detail::lift_scalar<T>(b.body(), b.types(), T(a)), b, #op_);           \
  }

LSE_IR_ARITH(+)
LSE_IR_ARITH(-)
LSE_IR_ARITH(*)
LSE_IR_ARITH(/)
LSE_IR_ARITH(%)
#undef LSE_IR_ARITH

#define LSE_IR_CMP(op_)                                                        \
  template <typename T>                                                        \
  [[nodiscard]] Val<boolean> operator op_(const Val<T>& a, const Val<T>& b) {  \
    return detail::binop<boolean>(a, b, #op_);                                 \
  }                                                                            \
  template <typename T, typename S>                                            \
  [[nodiscard]] Val<boolean> operator op_(const Val<T>& a, S b) {              \
    return detail::binop<boolean>(                                             \
        a, detail::lift_scalar<T>(a.body(), a.types(), T(b)), #op_);           \
  }

LSE_IR_CMP(<)
LSE_IR_CMP(>)
LSE_IR_CMP(<=)
LSE_IR_CMP(>=)
LSE_IR_CMP(==)
LSE_IR_CMP(!=)
#undef LSE_IR_CMP

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
  Body* body = v.body();
  if (body == nullptr) return {};
  Operation o;
  o.kind = OpKind::kCast;
  o.type = ir_type_of<To>::get();
  o.operands = {v.id()};
  o.cast_to = scalar_of<To>::value;
  return {v.types(), body, body->add_value(std::move(o))};
}

template <typename T>
[[nodiscard]] Val<T> select(const Val<boolean>& cond, const Val<T>& a,
                            const Val<T>& b) {
  Body* body = a.body();
  if (body == nullptr) return {};
  Operation o;
  o.kind = OpKind::kSelect;
  o.type = ir_type_of<T>::get();
  o.operands = {cond.id(), a.id(), b.id()};
  return {a.types(), body, body->add_value(std::move(o))};
}

// An addressable location. operator= records a store rather than rebinding, so
// a kernel assigns to a buffer slot or a local the way it would in plain C++.
template <typename T>
class LValue {
 public:
  LValue(KernelBody* body, const TypeTable* tt, Body* ir, ValueId id)
      : body_(body), tt_(tt), ir_(ir), id_(id) {}
  // Explicit because operator= here is a *store*, not a rebind, and providing
  // it suppresses the implicit copy constructor that a helper returning an
  // accumulator needs.
  LValue(const LValue&) = default;

  const LValue& operator=(const Val<T>& v) const;
  const LValue& operator=(float v) const;
  const LValue& operator=(const LValue& other) const { return *this = Val<T>(other); }

  operator Val<T>() const { return {tt_, ir_, id_}; }  // NOLINT: reads as a value
  [[nodiscard]] Val<T> read() const { return {tt_, ir_, id_}; }
  [[nodiscard]] std::string text() const { return read().text(); }
  [[nodiscard]] ValueId id() const noexcept { return id_; }

 private:
  KernelBody* body_;
  const TypeTable* tt_;
  Body* ir_;
  ValueId id_;
};

namespace detail {

// base[index], where base is either a memory reference or a register vector.
template <typename T>
[[nodiscard]] ValueId subscript(Body* body, ValueId base, ValueId index) {
  Operation o;
  o.kind = OpKind::kSubscript;
  o.type = ir_type_of<T>::get();
  o.operands = {base, index};
  return body->add_value(std::move(o));
}

[[nodiscard]] ValueId int_literal(Body* body, int i);

}  // namespace detail

// A register vector whose width was snapped from a live byte budget
// (HRX MAX_LOAD_BYTES / MAX_STORE_BYTES), not written as 4 or 2 at the
// call site. `width()` is how many T's that budget bought.
template <typename T>
class Pack {
 public:
  Pack() = default;
  Pack(const TypeTable* tt, Body* body, ValueId id, std::uint32_t n)
      : tt_(tt), body_(body), id_(id), n_(n) {}

  [[nodiscard]] std::uint32_t width() const noexcept { return n_; }
  [[nodiscard]] ValueId id() const noexcept { return id_; }
  [[nodiscard]] std::string text() const {
    return body_ == nullptr ? std::string{} : render(*body_, id_);
  }

  [[nodiscard]] Val<T> operator[](const Val<u32>& i) const {
    if (n_ <= 1) return {tt_, body_, id_};
    return {tt_, body_, detail::subscript<T>(body_, id_, i.id())};
  }
  [[nodiscard]] Val<T> operator[](int i) const {
    if (n_ <= 1) return {tt_, body_, id_};
    return {tt_, body_,
            detail::subscript<T>(body_, id_, detail::int_literal(body_, i))};
  }

 private:
  const TypeTable* tt_ = nullptr;
  Body* body_ = nullptr;
  ValueId id_ = kNoValue;
  std::uint32_t n_ = 1;
};

template <typename T>
class Buffer {
 public:
  Buffer() = default;
  Buffer(KernelBody* body, const TypeTable* tt, std::string name);
  Buffer(KernelBody* body, const TypeTable* tt, Body* ir, ValueId id)
      : body_(body), tt_(tt), ir_(ir), id_(id) {}

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, ir_, detail::subscript<T>(ir_, id_, index.id())};
  }

  // `max_bytes` is the device property (16 on gfx1151). The instruction
  // width is derived from that, not passed as 4 or 2.
  [[nodiscard]] Pack<T> load(const Val<u32>& index,
                             std::uint32_t max_bytes) const;
  void store(const Val<u32>& index, const Pack<T>& value,
             std::uint32_t max_bytes) const;

  [[nodiscard]] const std::string& name() const noexcept {
    return ir_->value(id_).name;
  }
  [[nodiscard]] ValueId id() const noexcept { return id_; }
  [[nodiscard]] Body* body() const noexcept { return ir_; }

 private:
  KernelBody* body_ = nullptr;
  const TypeTable* tt_ = nullptr;
  Body* ir_ = nullptr;
  ValueId id_ = kNoValue;
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
  Tile(KernelBody* body, const TypeTable* tt, Body* ir, ValueId id,
       std::uint32_t count)
      : body_(body), tt_(tt), ir_(ir), id_(id), n_(count) {}
  // Names an array the kernel declared itself through the raw statement
  // escape hatch (gdn's per-thread register tile). The recorder did not build
  // the declaration, so it is a symbol rather than an allocation.
  Tile(KernelBody* body, const TypeTable* tt, std::string name,
       std::uint32_t count);

  [[nodiscard]] explicit operator bool() const noexcept {
    return body_ != nullptr;
  }
  [[nodiscard]] std::uint32_t size() const noexcept { return n_; }
  [[nodiscard]] const std::string& name() const noexcept {
    return ir_->value(id_).name;
  }
  [[nodiscard]] ValueId id() const noexcept { return id_; }

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, ir_, detail::subscript<T>(ir_, id_, index.id())};
  }
  [[nodiscard]] LValue<T> operator[](int index) const {
    return {body_, tt_, ir_,
            detail::subscript<T>(ir_, id_, detail::int_literal(ir_, index))};
  }

 private:
  KernelBody* body_ = nullptr;
  const TypeTable* tt_ = nullptr;
  Body* ir_ = nullptr;
  ValueId id_ = kNoValue;
  std::uint32_t n_ = 0;
};

// A local whose elements are addressable — a fragment register, in practice.
template <typename T, int N>
class Local {
 public:
  Local(KernelBody* body, const TypeTable* tt, Body* ir, ValueId id)
      : body_(body), tt_(tt), ir_(ir), id_(id) {}

  template <typename I>
  [[nodiscard]] LValue<T> operator[](const Val<I>& index) const {
    return {body_, tt_, ir_, detail::subscript<T>(ir_, id_, index.id())};
  }
  [[nodiscard]] LValue<T> operator[](int index) const {
    return {body_, tt_, ir_,
            detail::subscript<T>(ir_, id_, detail::int_literal(ir_, index))};
  }

  [[nodiscard]] Val<vec<T, N>> value() const { return {tt_, ir_, id_}; }
  [[nodiscard]] const std::string& name() const noexcept {
    return ir_->value(id_).name;
  }
  [[nodiscard]] ValueId id() const noexcept { return id_; }

  // Whole-register assignment, which is how an accumulator takes the result of
  // a matrix instruction.
  const Local& operator=(const Val<vec<T, N>>& v) const;

 private:
  KernelBody* body_;
  const TypeTable* tt_;
  Body* ir_;
  ValueId id_;
};

// What the emitter tells the recorder about the kernel it is building: how the
// buffer parameters are named in this translation unit, and a prefix that keeps
// the generated names of concatenated sibling stages apart. Installed around an
// `emit_kernel` call, so a kernel primitive never mentions either.
struct RecordOptions {
  std::span<const std::string> input_names;
  std::string_view output_name;
  std::string_view name_prefix;
};

class KernelBody {
 public:
  KernelBody(const TypeTable& types, const DialectSourceTable& intrinsics,
             std::uint32_t lds_budget = 0);
  ~KernelBody() { unbind(); }
  KernelBody(const KernelBody&) = delete;
  KernelBody& operator=(const KernelBody&) = delete;

  // The body `lse::math::*` records into. Null outside emit_kernel.
  [[nodiscard]] static KernelBody* try_current() noexcept;

  // Installed by the emitter for the duration of one `emit_kernel` call.
  class Recording {
   public:
    explicit Recording(RecordOptions o) noexcept;
    ~Recording();
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;

   private:
    const RecordOptions* prev_;
    RecordOptions opts_;
  };

  // Takes ownership of the IR the next kernel body builds. An emitter that
  // fuses several primitives into one launch needs their IR, not their text:
  // the whole point of a middle end is that two stages can be looked at
  // together, and a string cannot be.
  class Capture {
   public:
    Capture() noexcept;
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    [[nodiscard]] bool has() const noexcept { return taken_.has_value(); }
    [[nodiscard]] Body& body() { return *taken_; }

   private:
    friend class KernelBody;
    Capture* prev_;
    std::optional<Body> taken_;
  };

  [[nodiscard]] Body& ir() noexcept { return ir_; }
  [[nodiscard]] const Body& ir() const noexcept { return ir_; }

  // The flat thread id the emitter binds before the body runs.
  [[nodiscard]] Val<u32> thread_id();

  // How this translation unit names input slot `index` and the output. The
  // emitter's convention, asked rather than restated by every kernel.
  [[nodiscard]] std::string input_name(std::size_t index) const;
  [[nodiscard]] std::string output_name() const;

  template <typename T>
  [[deprecated("declare an env::In member and env::bind the args struct")]]
  [[nodiscard]] Buffer<T> input(std::size_t index) {
    return {this, types_, input_name(index)};
  }
  template <typename T>
  [[deprecated("declare an env::Out member and env::bind the args struct")]]
  [[nodiscard]] Buffer<T> output() {
    return {this, types_, output_name()};
  }

  template <typename T>
  [[deprecated("use env::Emit::u32 or a bare literal in a Val expression")]]
  [[nodiscard]] Val<T> constant(std::uint32_t v) {
    return detail::lift_u32(&ir_, types_, v);
  }
  [[nodiscard]] Val<f32> lit(float v) { return detail::lift_f32(&ir_, types_, v); }

  // A tensor dimension, said to be one. `extent` bakes the number in, which is
  // what every shape in this engine does today and what keeps the inner loops
  // at a known trip count. `runtime_extent` says the opposite — the value
  // arrives in the dispatch constants, so one code object serves every value —
  // and the verifier then refuses it anywhere but a guard, a grid mapping or an
  // outermost loop bound. `field` is how the kernel reads it, e.g. "k.rows".
  [[nodiscard]] Val<u32> extent(std::string_view name, std::uint32_t value) {
    return {types_, &ir_,
            ir_.extent(name, ExtentBinding::kLiteral, ExtentRole::kSize,
                       literal_u32(value), value)};
  }
  [[nodiscard]] Val<u32> runtime_extent(std::string_view name,
                                        std::string_view field) {
    return {types_, &ir_,
            ir_.extent(name, ExtentBinding::kRuntime, ExtentRole::kSize,
                       std::string(field), 0)};
  }
  // The base index of the iteration-space window this launch covers. Unlike a
  // size it may enter an address — with a constant coefficient, so the stride
  // the inner loop walks is unchanged and only the origin moves.
  [[nodiscard]] Val<u32> window_base(std::string_view dim,
                                     std::string_view field) {
    return {types_, &ir_,
            ir_.extent(dim, ExtentBinding::kRuntime, ExtentRole::kWindowBase,
                       std::string(field), 0)};
  }

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
  [[nodiscard]] Pack<T> load_pack(ValueId buf, const Val<u32>& index,
                                  std::uint32_t max_bytes);
  template <typename T>
  void store_pack(ValueId buf, const Val<u32>& index, const Pack<T>& v,
                  std::uint32_t max_bytes);

  // A mutable local — an accumulator. `let` is the immutable form.
  template <typename T>
  [[nodiscard]] LValue<T> var(std::string_view name, const Val<T>& init) {
    Operation o;
    o.kind = OpKind::kMutable;
    o.type = scalar_type(scalar_of<T>::value);
    o.operands = {init.id()};
    return {this, types_, &ir_,
            ir_.add_value(std::move(o), std::string(name))};
  }

  void ret();
  void ret_if(const Val<boolean>& cond);
  // The per-element form: a kernel primitive that is not self-indexing returns
  // the value for its output element rather than storing it.
  template <typename T>
  void ret(const Val<T>& v) {
    Operation o;
    o.kind = OpKind::kReturn;
    o.operands = {v.id()};
    ir_.add(std::move(o));
  }

  template <typename F>
  [[deprecated("use if (auto g = e.when(cond)) { ... }")]]
  void when(const Val<boolean>& cond, F&& body) {
    begin_if(cond, true);
    std::forward<F>(body)();
    end_block();
  }

  // A counted loop. `step` defaults to 1; `unroll` asks the target to unroll,
  // which is what makes fragment indices constant.
  template <typename F>
  [[deprecated("use range-for over env::Emit::range")]]
  void loop(std::string_view var, const Val<u32>& lo, const Val<u32>& hi,
            std::uint32_t step, F&& body) {
    loop(var, lo, hi, detail::lift_u32(&ir_, types_, step),
         std::forward<F>(body));
  }
  template <typename F>
  [[deprecated("use range-for over env::Emit::range")]]
  void loop(std::string_view var, const Val<u32>& lo, const Val<u32>& hi,
            const Val<u32>& step, F&& body) {
    const Val<u32> iv = begin_for(var, lo, hi, step, false, true);
    std::forward<F>(body)(iv);
    end_block();
  }
  template <typename F>
  [[deprecated("use range-for over env::Emit::unroll")]]
  void unroll(std::string_view var, std::uint32_t count, F&& body) {
    const Val<u32> iv =
        begin_for(var, detail::lift_u32(&ir_, types_, 0),
                  detail::lift_u32(&ir_, types_, count),
                  detail::lift_u32(&ir_, types_, 1), true, true);
    std::forward<F>(body)(iv);
    end_block();
  }

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

  // One id sequence per body, shared by every generator layered on it, so
  // two recorders on the same body cannot mint the same name.
  [[nodiscard]] int fresh_id() noexcept {
    return static_cast<int>(ir_.fresh_id());
  }
  [[nodiscard]] std::string fresh_name(std::string_view stem) {
    return ir_.fresh_name(stem);
  }

  // The body as source, with any vector typedefs the kernel used declared at
  // the top. Function-scope typedefs keep this self-contained: a kernel
  // authored here needs nothing added to the emitter's preamble.
  //
  // The optimization pipeline runs first, and is not optional: a pass that is
  // not correct enough to be on is not done.
  [[nodiscard]] std::string str();

  // The raw escape hatch: a statement the recorder has no op for. It is opaque
  // to every pass, which is the price of using it.
  void statement(std::string text);

  // Structured control flow. The `env` surface drives these through its RAII
  // guard and its range-for; nothing else should call them directly.
  void begin_if(const Val<boolean>& cond, bool indent_body = false);
  [[nodiscard]] Val<u32> begin_for(std::string_view var, const Val<u32>& lo,
                                   const Val<u32>& hi, const Val<u32>& step,
                                   bool unroll, bool indent_body = false);
  void end_block();

 private:
  void bind() noexcept;
  void unbind() noexcept;

  static thread_local KernelBody* current_;
  static thread_local const RecordOptions* options_;
  static thread_local Capture* capture_;
  KernelBody* prev_ = nullptr;

  template <typename T, int N>
  [[nodiscard]] std::string vector_type() {
    return ir_.vector_typedef(scalar_of<T>::value, N);
  }

  const TypeTable* types_;
  const DialectSourceTable* intrinsics_;
  Body ir_;
  StoreFn store_;
  Lds lds_;

  template <typename T> friend class LValue;
  friend class Lds;
};

// --------------------------------------------------------------------------
// Out-of-line definitions
// --------------------------------------------------------------------------

template <typename T>
Val<T>::Val(const TypeTable* tt, std::string text) : tt_(tt) {
  KernelBody* k = KernelBody::try_current();
  if (k == nullptr) return;
  body_ = &k->ir();
  id_ = body_->symbol(text, scalar_type(scalar_of<T>::value));
}

template <typename T>
Buffer<T>::Buffer(KernelBody* body, const TypeTable* tt, std::string name)
    : body_(body), tt_(tt), ir_(&body->ir()) {
  id_ = ir_->symbol(name, memory_type(scalar_of<T>::value, 0, Space::kGlobal));
}

template <typename T>
Tile<T>::Tile(KernelBody* body, const TypeTable* tt, std::string name,
              std::uint32_t count)
    : body_(body), tt_(tt), ir_(&body->ir()), n_(count) {
  id_ = ir_->symbol(name,
                    memory_type(scalar_of<T>::value, count, Space::kPrivate));
}

template <typename T, int N>
const Local<T, N>& Local<T, N>::operator=(const Val<vec<T, N>>& v) const {
  Operation o;
  o.kind = OpKind::kAssign;
  o.operands = {id_, v.id()};
  ir_->add(std::move(o));
  return *this;
}

template <typename T>
const LValue<T>& LValue<T>::operator=(const Val<T>& v) const {
  Operation o;
  o.kind = OpKind::kAssign;
  o.operands = {id_, v.id()};
  ir_->add(std::move(o));
  return *this;
}

template <typename T>
const LValue<T>& LValue<T>::operator=(float v) const {
  Operation o;
  o.kind = OpKind::kAssign;
  o.operands = {id_, detail::lift_f32(ir_, tt_, v).id()};
  ir_->add(std::move(o));
  return *this;
}

template <typename T, int N>
[[nodiscard]] Val<T> lane(const Val<vec<T, N>>& v, int i) {
  return {v.types(), v.body(),
          detail::subscript<T>(v.body(), v.id(),
                               detail::int_literal(v.body(), i))};
}

template <typename T, int N>
[[nodiscard]] Val<T> lane(const Val<vec<T, N>>& v, const Val<u32>& i) {
  return {v.types(), v.body(),
          detail::subscript<T>(v.body(), v.id(), i.id())};
}

template <typename T>
constexpr std::uint32_t pack_elem_bytes() noexcept {
  if constexpr (std::is_same_v<T, f16> || std::is_same_v<T, bf16>) return 2;
  return 4;
}

// Widest power-of-two vector the byte budget buys. 8 is the cap because a
// 16-byte load of 2-byte elements is 8 wide; stopping at 4 there would leave
// half of every dwordx4 on the table (measured: MoE-shape gain 2.72x -> 2.25x).
inline std::uint32_t pack_n(std::uint32_t max_bytes,
                            std::uint32_t elem_bytes) noexcept {
  if (elem_bytes == 0 || max_bytes < elem_bytes) return 1;
  const std::uint32_t n = max_bytes / elem_bytes;
  if (n >= 8) return 8;
  if (n >= 4) return 4;
  if (n >= 2) return 2;
  return 1;
}

template <typename T>
Pack<T> Buffer<T>::load(const Val<u32>& index, std::uint32_t max_bytes) const {
  return body_->load_pack<T>(id_, index, max_bytes);
}

template <typename T>
void Buffer<T>::store(const Val<u32>& index, const Pack<T>& value,
                      std::uint32_t max_bytes) const {
  body_->store_pack<T>(id_, index, value, max_bytes);
}

template <typename T>
Val<T> KernelBody::let(std::string_view name, const Val<T>& v) {
  Operation o;
  o.kind = OpKind::kBind;
  o.type = scalar_type(scalar_of<T>::value);
  o.operands = {v.id()};
  return {types_, &ir_, ir_.add_value(std::move(o), std::string(name))};
}

template <typename T, int N>
Local<T, N> KernelBody::local(std::string_view name) {
  Operation o;
  o.kind = OpKind::kAlloc;
  o.type = memory_type(scalar_of<T>::value, N, Space::kPrivate);
  o.text = vector_type<T, N>();
  return {this, types_, &ir_, ir_.add_value(std::move(o), std::string(name))};
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
      std::string(name) + "_" + std::to_string(body_->ir_.fresh_id());
  Operation o;
  o.kind = OpKind::kAlloc;
  o.type = memory_type(scalar_of<T>::value, count, Space::kWorkgroup);
  o.imm = count;
  o.flags |= kFlagArray;
  return {body_, body_->types_, &body_->ir_,
          body_->ir_.add_value(std::move(o), ident), count};
}

template <typename T, int N>
Local<T, N> KernelBody::shared(std::string_view name) {
  if (!lds_.reserve(static_cast<std::uint32_t>(N) * pack_elem_bytes<T>())) {
    // Over budget: name it without declaring it. The kernel then declines on
    // `lds().ok()` and nothing this returns is ever emitted.
    return {this, types_, &ir_,
            ir_.symbol(name, memory_type(scalar_of<T>::value,
                                         static_cast<std::uint32_t>(N),
                                         Space::kWorkgroup))};
  }
  Operation o;
  o.kind = OpKind::kAlloc;
  o.type = memory_type(scalar_of<T>::value, static_cast<std::uint32_t>(N),
                       Space::kWorkgroup);
  o.imm = N;
  o.flags |= kFlagArray;
  return {this, types_, &ir_, ir_.add_value(std::move(o), std::string(name))};
}

template <typename T>
Pack<T> KernelBody::load_pack(ValueId buf, const Val<u32>& index,
                              std::uint32_t max_bytes) {
  const auto n = pack_n(max_bytes, pack_elem_bytes<T>());
  Operation o;
  o.kind = OpKind::kLoadVec;
  o.operands = {buf, index.id()};
  o.imm = n;
  if (n <= 1) {
    o.type = scalar_type(scalar_of<T>::value);
  } else {
    o.type = ir::vector_type(scalar_of<T>::value, n);
    o.text = ir_.vector_typedef(scalar_of<T>::value, static_cast<int>(n));
  }
  const ValueId v = ir_.add_value(std::move(o), fresh_name("ld"));
  return {types_, &ir_, v, n};
}

template <typename T>
void KernelBody::store_pack(ValueId buf, const Val<u32>& index,
                            const Pack<T>& v, std::uint32_t max_bytes) {
  const auto n = pack_n(max_bytes, pack_elem_bytes<T>());
  const auto use = n < v.width() ? n : v.width();
  Operation o;
  o.kind = OpKind::kStoreVec;
  o.operands = {buf, index.id(), v.id()};
  o.imm = use;
  if (use > 1) {
    o.text = ir_.vector_typedef(scalar_of<T>::value, static_cast<int>(use));
  }
  ir_.add(std::move(o));
}

template <typename R, typename... A>
Val<R> KernelBody::call(std::string_view intrinsic, const A&... args) {
  const std::string_view tmpl = intrinsics_->find(intrinsic);
  if (tmpl.empty()) return {};  // caller reports the gap; see emit_kernel
  Operation o;
  o.kind = OpKind::kCall;
  o.type = ir_type_of<R>::get();
  o.operands = {args.id()...};
  o.key = std::string(intrinsic);
  o.text = std::string(tmpl);
  return {types_, &ir_, ir_.add_value(std::move(o))};
}

}  // namespace lse::ir
