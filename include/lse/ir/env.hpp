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
// Instantiated with `env::Emit`, every value is an IR proxy and running the
// body records the kernel into an `ir::Body`. Instantiated with `env::Cpu`,
// every value is a real scalar and running the body IS the kernel, so a device
// kernel is unit-testable on the host with no GPU and no compiler in the loop.
//
// The C++ control keywords do the recording work:
//   for (auto i : e.range(n))     — a device loop; range-for, no lambda
//   for (auto i : e.unroll(8))    — the same, asked to unroll
//   if (auto in = e.when(cond))   — a device conditional; the guard opens an
//                                   `if` region on entry and closes it when
//                                   it dies
//
// Two authoring constraints the types cannot enforce: do not `break` or
// `return` out of an `e.range` body, and do not attach an `else` to an
// `e.when` — on the recording env the guard is always entered, so an else
// branch would be silently dropped. Use a second `when` on the negated
// condition.
//
// Names are qualified `ir::` throughout even though this namespace is nested
// inside `lse::ir`: `Emit::u32` and `Emit::f32` are part of the authoring
// surface, and an unqualified `f32` in a member's type would then mean two
// different things inside one class.
#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "lse/core/dtype.hpp"
#include "lse/ir/recorder.hpp"

namespace lse::ir::env {

struct Emit;
struct Cpu;

// How the host spells an IR element type. Reads always hand back a float, so a
// body written against the accumulate type runs unchanged over narrow storage.
template <typename T>
struct host_storage {
  using type = T;
};
template <>
struct host_storage<lse::f16> {
  using type = float16_t;
};
template <>
struct host_storage<lse::bf16> {
  using type = bfloat16_t;
};
template <typename T>
using host_storage_t = typename host_storage<T>::type;

// Device buffers as the body sees them. `In` reads produce values, so the
// common case never spells `.read()`; `Out` produces an assignable slot.
template <typename T, class E>
struct In;
template <typename T, class E>
struct Out;

template <typename T>
struct In<T, Emit> {
  ir::Buffer<T> b;
  const ir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] ir::Val<T> operator[](const ir::Val<I>& i) const {
    return b[i].read();
  }
  [[nodiscard]] ir::Val<T> operator[](std::uint32_t i) const {
    return b[ir::detail::lift_u32(b.body(), tt, i)].read();
  }
};

template <typename T>
struct Out<T, Emit> {
  ir::Buffer<T> b;
  const ir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] ir::LValue<T> operator[](const ir::Val<I>& i) const {
    return b[i];
  }
  [[nodiscard]] ir::LValue<T> operator[](std::uint32_t i) const {
    return b[ir::detail::lift_u32(b.body(), tt, i)];
  }
};

// An input-slot binding the kernel also writes — carried state, in practice.
template <typename T, class E>
struct InOut;

template <typename T>
struct InOut<T, Emit> {
  ir::Buffer<T> b;
  const ir::TypeTable* tt = nullptr;
  template <typename I>
  [[nodiscard]] ir::LValue<T> operator[](const ir::Val<I>& i) const {
    return b[i];
  }
  [[nodiscard]] ir::LValue<T> operator[](std::uint32_t i) const {
    return b[ir::detail::lift_u32(b.body(), tt, i)];
  }
};

template <typename T>
struct InOut<T, Cpu> {
  host_storage_t<T>* p = nullptr;
  [[nodiscard]] host_storage_t<T>& operator[](std::uint32_t i) const {
    return p[i];
  }
};

template <typename T>
struct In<T, Cpu> {
  const host_storage_t<T>* p = nullptr;
  [[nodiscard]] float operator[](std::uint32_t i) const {
    return static_cast<float>(p[i]);
  }
};

template <typename T>
struct Out<T, Cpu> {
  host_storage_t<T>* p = nullptr;
  [[nodiscard]] host_storage_t<T>& operator[](std::uint32_t i) const {
    return p[i];
  }
};

// --------------------------------------------------------------------------
// Emit: values are IR proxies, ops land in the KernelBody's ir::Body.
// --------------------------------------------------------------------------
struct Emit {
  ir::KernelBody* k = nullptr;

  Emit() = default;
  explicit Emit(ir::KernelBody* body) : k(body) {}

  [[nodiscard]] ir::Val<ir::u32> thread_id() const { return k->thread_id(); }

  // A mutable local. Names are generated, not authored.
  [[nodiscard]] ir::LValue<ir::f32> var(float init) {
    return k->var<ir::f32>(next("v"), k->lit(init));
  }
  template <typename T>
  [[nodiscard]] ir::LValue<T> var(const ir::Val<T>& init) {
    return k->var<T>(next("v"), init);
  }
  template <typename T>
  [[nodiscard]] ir::Val<T> let(const ir::Val<T>& v) {
    return k->let<T>(next("t"), v);
  }

  // The per-element protocol: a kernel that does not own indexing returns
  // the value for its output element instead of storing it.
  template <typename T>
  void ret(const ir::Val<T>& v) {
    k->ret(v);
  }

  [[nodiscard]] ir::Val<ir::u32> u32(std::uint32_t v) const { return lift(v); }

  // A shape extent, named as one. `extent` is the baked form every kernel
  // uses today; `runtime_extent` is the dispatch-constant form, legal only in
  // a guard or an outermost loop bound and checked by `ir::verify`.
  [[nodiscard]] ir::Val<ir::u32> extent(std::string_view name,
                                        std::uint32_t value) {
    return k->extent(name, value);
  }
  [[nodiscard]] ir::Val<ir::u32> runtime_extent(std::string_view name,
                                                std::string_view field) {
    return k->runtime_extent(name, field);
  }
  // Base index of the iteration-space window this launch covers.
  [[nodiscard]] ir::Val<ir::u32> window_base(std::string_view dim,
                                             std::string_view field) {
    return k->window_base(dim, field);
  }
  [[nodiscard]] ir::Val<ir::f32> f32(float v) const { return k->lit(v); }

  // An addressable register vector — a fragment, in WMMA terms.
  template <typename T, int N>
  [[nodiscard]] ir::Local<T, N> local() {
    return k->local<T, N>(next("v"));
  }

  // Workgroup scratch; empty when the reserve would pass the LDS budget.
  template <typename T>
  [[nodiscard]] ir::Tile<T> lds(std::uint32_t count) {
    return k->lds().array<T>(next("s"), count);
  }
  // Whether `count` elements would fit, asked without reserving — a kernel
  // that has a global-memory fallback must not trip the budget flag that
  // makes the whole emit decline.
  template <typename T>
  [[nodiscard]] bool lds_fits(std::uint32_t count) const {
    return k->lds().fits(count * ir::pack_elem_bytes<T>());
  }
  void barrier() { k->barrier(); }

  // Guarded early exit that reads as one in both worlds:
  //   if (e.ret_if(wave >= tiles)) return;
  // Recording emits the guard and keeps going; Cpu actually returns.
  [[nodiscard]] bool ret_if(const ir::Val<ir::boolean>& cond) {
    k->ret_if(cond);
    return false;
  }

  // Vectorized memory access; width comes from the device budget.
  template <typename T>
  [[nodiscard]] ir::Pack<T> load(const In<T, Emit>& in,
                                 const ir::Val<ir::u32>& index,
                                 std::uint32_t max_bytes) {
    return in.b.load(index, max_bytes);
  }
  template <typename T>
  void store(const Out<T, Emit>& out, const ir::Val<ir::u32>& index,
             const ir::Pack<T>& v, std::uint32_t max_bytes) {
    out.b.store(index, v, max_bytes);
  }

  // Store through the emitter's hook so a fused epilogue can run on the
  // value in register before it reaches memory.
  void store(const ir::Val<ir::u32>& index, const ir::Val<ir::f32>& v) {
    k->store(index, v);
  }

  class [[nodiscard]] Guard {
   public:
    explicit Guard(ir::KernelBody* k, const ir::Val<ir::boolean>& cond) : k_(k) {
      k_->begin_if(cond);
    }
    ~Guard() {
      if (k_ != nullptr) k_->end_block();
    }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    // The body must record, so the recording guard is always entered.
    explicit operator bool() const { return true; }

   private:
    ir::KernelBody* k_;
  };
  [[nodiscard]] Guard when(const ir::Val<ir::boolean>& cond) {
    return Guard(k, cond);
  }

  class Range {
   public:
    Range(Emit* e, ir::Val<ir::u32> lo, ir::Val<ir::u32> hi,
          ir::Val<ir::u32> step, bool unroll)
        : e_(e), lo_(std::move(lo)), hi_(std::move(hi)),
          step_(std::move(step)), unroll_(unroll) {}

    class iterator {
     public:
      iterator(Range* r, bool live) : r_(r), live_(live) {}
      [[nodiscard]] ir::Val<ir::u32> operator*() const {
        return r_->induction_;
      }
      iterator& operator++() {
        r_->e_->k->end_block();
        live_ = false;
        return *this;
      }
      [[nodiscard]] bool operator!=(const iterator&) const { return live_; }

     private:
      Range* r_;
      bool live_;
    };

    [[nodiscard]] iterator begin() {
      induction_ = e_->k->begin_for(e_->next("i"), lo_, hi_, step_, unroll_);
      return {this, true};
    }
    [[nodiscard]] iterator end() { return {this, false}; }

   private:
    friend class iterator;
    Emit* e_;
    ir::Val<ir::u32> lo_, hi_, step_;
    bool unroll_;
    ir::Val<ir::u32> induction_;
  };

  [[nodiscard]] Range range(const ir::Val<ir::u32>& hi) {
    return {this, lift(0u), hi, lift(1u), false};
  }
  [[nodiscard]] Range range(std::uint32_t hi) { return range(lift(hi)); }
  [[nodiscard]] Range range(const ir::Val<ir::u32>& lo,
                            const ir::Val<ir::u32>& hi,
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

  [[nodiscard]] ir::Val<ir::u32> lift(std::uint32_t v) const {
    return ir::detail::lift_u32(&k->ir(), &k->types(), v);
  }

 private:
  // Ids come from the body, not this object: a helper that builds its own
  // Emit over the same body (the shims do) stays collision-free.
  [[nodiscard]] std::string next(const char* stem) {
    return k->fresh_name(stem);
  }
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
    const host_storage_t<T>* p;
    std::uint32_t n;
    [[nodiscard]] std::uint32_t width() const { return n; }
    [[nodiscard]] float operator[](std::uint32_t i) const {
      return static_cast<float>(p[i]);
    }
  };
  template <typename T>
  [[nodiscard]] Pack<T> load(const In<T, Cpu>& in, std::uint32_t index,
                             std::uint32_t max_bytes) const {
    return {in.p + index, ir::pack_n(max_bytes, ir::pack_elem_bytes<T>())};
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

}  // namespace lse::ir::env
