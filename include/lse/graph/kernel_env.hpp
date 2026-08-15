// Straight-C++ kernel authoring: one body, two instantiations.
//
//   template <class E>
//   void dot_row(E& e, DotArgs<E>& a, std::uint32_t cols) {
//     auto row = e.thread_id();
//     auto acc = e.var(0.0f);
//     for (auto t : e.range(cols)) {
//       acc = lse::math::fma(a.x[t], a.w[row * cols + t], acc);
//     }
//     a.out[row] = acc;
//   }
//
// Instantiated with `env::Emit`, every value is a kir proxy and running the
// body records the kernel source. Instantiated with `env::Cpu`, every value
// is a real scalar and running the body IS the kernel, so a device kernel is
// unit-testable on the host with no GPU and no compiler in the loop.
//
// The C++ control keywords do the recording work:
//   for (auto i : e.range(n))     — a device loop; range-for, no lambda
//   for (auto i : e.unroll(8))    — the same, asked to unroll
//   if (auto in = e.when(cond))   — a device conditional; the guard emits
//                                   `if (…) {` on entry and `}` when it dies
//
// Two authoring constraints the types cannot enforce: do not `break` or
// `return` out of an `e.range` body, and do not attach an `else` to an
// `e.when` — on the recording env the guard is always entered, so an else
// branch would be silently dropped. Use a second `when` on the negated
// condition.
#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "lse/graph/kernel_ir.hpp"

namespace lse::graph::env {

struct Emit;
struct Cpu;

// Device buffers as the body sees them. `In` reads produce values, so the
// common case never spells `.read()`; `Out` produces an assignable slot.
template <typename T, class E>
struct In;
template <typename T, class E>
struct Out;

template <typename T>
struct In<T, Emit> {
  kir::Buffer<T> b;
  const kir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] kir::Val<T> operator[](const kir::Val<I>& i) const {
    return b[i].read();
  }
  [[nodiscard]] kir::Val<T> operator[](std::uint32_t i) const {
    return b[kir::detail::lift(tt, i)].read();
  }
};

template <typename T>
struct Out<T, Emit> {
  kir::Buffer<T> b;
  const kir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] kir::LValue<T> operator[](const kir::Val<I>& i) const {
    return b[i];
  }
  [[nodiscard]] kir::LValue<T> operator[](std::uint32_t i) const {
    return b[kir::detail::lift(tt, i)];
  }
};

// An input-slot binding the kernel also writes — carried state, in practice.
template <typename T, class E>
struct InOut;

template <typename T>
struct InOut<T, Emit> {
  kir::Buffer<T> b;
  const kir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] kir::LValue<T> operator[](const kir::Val<I>& i) const {
    return b[i];
  }
  [[nodiscard]] kir::LValue<T> operator[](std::uint32_t i) const {
    return b[kir::detail::lift(tt, i)];
  }
};

template <typename T>
struct InOut<T, Cpu> {
  T* p = nullptr;
  [[nodiscard]] T& operator[](std::uint32_t i) const { return p[i]; }
};

template <typename T>
struct In<T, Cpu> {
  const T* p = nullptr;
  [[nodiscard]] T operator[](std::uint32_t i) const { return p[i]; }
};

template <typename T>
struct Out<T, Cpu> {
  T* p = nullptr;
  [[nodiscard]] T& operator[](std::uint32_t i) const { return p[i]; }
};

// --------------------------------------------------------------------------
// Emit: values are kir proxies, statements land in a KernelBody.
// --------------------------------------------------------------------------
struct Emit {
  kir::KernelBody* k = nullptr;

  Emit() = default;
  explicit Emit(kir::KernelBody* body) : k(body) {}

  [[nodiscard]] kir::Val<kir::u32> thread_id() const { return k->thread_id(); }

  // A mutable local. Names are generated, not authored.
  [[nodiscard]] kir::LValue<kir::f32> var(float init) {
    return k->var<kir::f32>(next("v"), k->lit(init));
  }
  template <typename T>
  [[nodiscard]] kir::Val<T> let(const kir::Val<T>& v) {
    return k->let<T>(next("t"), v);
  }

  [[nodiscard]] kir::Val<kir::u32> u32(std::uint32_t v) const {
    return lift(v);
  }
  [[nodiscard]] kir::Val<kir::f32> f32(float v) const { return k->lit(v); }

  // An addressable register vector — a fragment, in WMMA terms.
  template <typename T, int N>
  [[nodiscard]] kir::Local<T, N> local() {
    return k->local<T, N>(next("v"));
  }

  // Workgroup scratch; empty when the reserve would pass the LDS budget.
  template <typename T>
  [[nodiscard]] kir::Tile<T> lds(std::uint32_t count) {
    return k->lds().array<T>(next("s"), count);
  }
  void barrier() { k->barrier(); }

  // Guarded early exit that reads as one in both worlds:
  //   if (e.ret_if(wave >= tiles)) return;
  // Recording emits the guard and keeps going; Cpu actually returns.
  [[nodiscard]] bool ret_if(const kir::Val<kir::boolean>& cond) {
    k->ret_if(cond);
    return false;
  }

  // Vectorized memory access; width comes from the device budget.
  template <typename T>
  [[nodiscard]] kir::Pack<T> load(const In<T, Emit>& in,
                                  const kir::Val<kir::u32>& index,
                                  std::uint32_t max_bytes) {
    return in.b.load(index, max_bytes);
  }
  template <typename T>
  void store(const Out<T, Emit>& out, const kir::Val<kir::u32>& index,
             const kir::Pack<T>& v, std::uint32_t max_bytes) {
    out.b.store(index, v, max_bytes);
  }

  // Store through the emitter's hook so a fused epilogue can run on the
  // value in register before it reaches memory.
  void store(const kir::Val<kir::u32>& index, const kir::Val<kir::f32>& v) {
    k->store(index, v);
  }

  class [[nodiscard]] Guard {
   public:
    explicit Guard(kir::KernelBody* k, const kir::Val<kir::boolean>& cond)
        : k_(k) {
      k_->statement("if (" + cond.text() + ") {");
    }
    ~Guard() {
      if (k_ != nullptr) k_->statement("}");
    }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    // The body must record, so the recording guard is always entered.
    explicit operator bool() const { return true; }

   private:
    kir::KernelBody* k_;
  };
  [[nodiscard]] Guard when(const kir::Val<kir::boolean>& cond) {
    return Guard(k, cond);
  }

  class Range {
   public:
    Range(Emit* e, kir::Val<kir::u32> lo, kir::Val<kir::u32> hi,
          kir::Val<kir::u32> step, bool unroll)
        : e_(e), lo_(std::move(lo)), hi_(std::move(hi)),
          step_(std::move(step)), unroll_(unroll) {}

    class iterator {
     public:
      iterator(Range* r, bool live) : r_(r), live_(live) {}
      [[nodiscard]] kir::Val<kir::u32> operator*() const {
        return {&r_->e_->k->types(), r_->var_};
      }
      iterator& operator++() {
        r_->e_->k->statement("}");
        live_ = false;
        return *this;
      }
      [[nodiscard]] bool operator!=(const iterator&) const { return live_; }

     private:
      Range* r_;
      bool live_;
    };

    [[nodiscard]] iterator begin() {
      var_ = e_->next("i");
      kir::KernelBody* k = e_->k;
      if (unroll_) k->statement("#pragma unroll");
      const std::string ty(k->types().scalar(kir::Scalar::kU32));
      k->statement("for (" + ty + " " + var_ + " = " + lo_.text() + "; " +
                   var_ + " < " + hi_.text() + "; " + var_ + " += " +
                   step_.text() + ") {");
      return {this, true};
    }
    [[nodiscard]] iterator end() { return {this, false}; }

   private:
    friend class iterator;
    Emit* e_;
    kir::Val<kir::u32> lo_, hi_, step_;
    bool unroll_;
    std::string var_;
  };

  [[nodiscard]] Range range(const kir::Val<kir::u32>& hi) {
    return {this, lift(0u), hi, lift(1u), false};
  }
  [[nodiscard]] Range range(std::uint32_t hi) { return range(lift(hi)); }
  [[nodiscard]] Range range(const kir::Val<kir::u32>& lo,
                            const kir::Val<kir::u32>& hi,
                            std::uint32_t step = 1) {
    return {this, lo, hi, lift(step), false};
  }
  [[nodiscard]] Range range(std::uint32_t lo, std::uint32_t hi,
                            std::uint32_t step = 1) {
    return {this, lift(lo), lift(hi), lift(step), false};
  }
  [[nodiscard]] Range unroll(std::uint32_t count) {
    return {this, lift(0u), lift(count), lift(1u), true};
  }

  [[nodiscard]] kir::Val<kir::u32> lift(std::uint32_t v) const {
    return kir::detail::lift(&k->types(), v);
  }

 private:
  [[nodiscard]] std::string next(const char* stem) {
    return std::string(stem) + std::to_string(counter_++);
  }
  int counter_ = 0;
};

// --------------------------------------------------------------------------
// Cpu: values are real scalars, running the body executes the kernel for one
// thread. Wave intrinsics have no host model, so wave-cooperative bodies stay
// emit-only; everything one thread computes alone runs here as-is.
// --------------------------------------------------------------------------
struct Cpu {
  std::uint32_t tid = 0;

  [[nodiscard]] std::uint32_t thread_id() const { return tid; }

  struct Var {
    float v;
    operator float() const { return v; }  // NOLINT: reads as a value
    Var& operator=(float x) {
      v = x;
      return *this;
    }
  };
  [[nodiscard]] Var var(float init) const { return {init}; }
  template <typename T>
  [[nodiscard]] T let(T v) const {
    return v;
  }

  [[nodiscard]] std::uint32_t u32(std::uint32_t v) const { return v; }
  [[nodiscard]] float f32(float v) const { return v; }

  [[nodiscard]] bool ret_if(bool cond) const { return cond; }

  template <typename T>
  struct Pack {
    const T* p;
    std::uint32_t n;
    [[nodiscard]] std::uint32_t width() const { return n; }
    [[nodiscard]] T operator[](std::uint32_t i) const { return p[i]; }
  };
  template <typename T>
  [[nodiscard]] Pack<T> load(const In<T, Cpu>& in, std::uint32_t index,
                             std::uint32_t max_bytes) const {
    return {in.p + index, kir::pack_n(max_bytes, kir::pack_elem_bytes<T>())};
  }

  struct Guard {
    bool taken;
    explicit operator bool() const { return taken; }
  };
  [[nodiscard]] Guard when(bool cond) const { return {cond}; }

  struct Range {
    std::uint32_t lo, hi, step;
    struct iterator {
      std::uint32_t v, hi, step;
      [[nodiscard]] std::uint32_t operator*() const { return v; }
      iterator& operator++() {
        v += step;
        return *this;
      }
      [[nodiscard]] bool operator!=(const iterator&) const { return v < hi; }
    };
    [[nodiscard]] iterator begin() const { return {lo, hi, step}; }
    [[nodiscard]] iterator end() const { return {hi, hi, step}; }
  };
  [[nodiscard]] Range range(std::uint32_t hi) const { return {0, hi, 1}; }
  [[nodiscard]] Range range(std::uint32_t lo, std::uint32_t hi,
                            std::uint32_t step = 1) const {
    return {lo, hi, step};
  }
  [[nodiscard]] Range unroll(std::uint32_t count) const {
    return {0, count, 1};
  }
};

// Run a body for every thread of a flat launch, in thread order.
template <typename Body, typename... Args>
void run_flat(std::uint32_t threads, Body&& body, Args&&... args) {
  for (std::uint32_t i = 0; i < threads; ++i) {
    Cpu e{i};
    body(e, std::forward<Args>(args)...);
  }
}

}  // namespace lse::graph::env
