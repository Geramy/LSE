// Kernel parameters as a plain struct, reflected instead of hand-bound.
//
//   template <class E>
//   struct DotArgs {
//     env::In<f32, E> x;
//     env::In<f32, E> w;
//     env::Out<f32, E> out;
//   };
//
// `bind(k, args)` walks the struct with P2996 and attaches each member to the
// recorder in declaration order — the struct IS the binding contract, so there
// is no `k.input<f32>(0)` to keep in sync with an arity() count by hand.
// `input_count` / `output_count` are consteval and come from the same walk,
// so the two cannot drift. On the Cpu env the same struct aggregate-
// initializes from host pointers and the body runs directly.
#pragma once

#if !defined(__cpp_impl_reflection) || __cpp_impl_reflection < 202603L
#error "kernel_args.hpp needs P2996 reflection (g++-16 -std=c++26 -freflection)"
#endif

#include <cstddef>
#include <string>
#include <meta>

#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"

namespace lse::graph::env {

namespace detail {

template <typename M>
struct arg_traits {
  static constexpr bool is_in = false;
  static constexpr bool is_out = false;
};
template <typename T>
struct arg_traits<In<T, Emit>> {
  static constexpr bool is_in = true;
  static constexpr bool is_out = false;
  using elem = T;
};
template <typename T>
struct arg_traits<Out<T, Emit>> {
  static constexpr bool is_in = false;
  static constexpr bool is_out = true;
  using elem = T;
};
// Takes an input slot like In; writability is between the kernel and the
// emitter's binding constness, exactly as it was with positional input().
template <typename T>
struct arg_traits<InOut<T, Emit>> {
  static constexpr bool is_in = true;
  static constexpr bool is_out = false;
  using elem = T;
};

template <class A>
consteval auto members() {
  return std::define_static_array(std::meta::nonstatic_data_members_of(
      ^^A, std::meta::access_context::current()));
}

}  // namespace detail

template <class A>
consteval std::size_t input_count() {
  std::size_t n = 0;
  template for (constexpr std::meta::info m : detail::members<A>()) {
    if constexpr (detail::arg_traits<
                      typename[:std::meta::type_of(m):]>::is_in) {
      ++n;
    }
  }
  return n;
}

template <class A>
consteval std::size_t output_count() {
  std::size_t n = 0;
  template for (constexpr std::meta::info m : detail::members<A>()) {
    if constexpr (detail::arg_traits<
                      typename[:std::meta::type_of(m):]>::is_out) {
      ++n;
    }
  }
  return n;
}

// Attach every member to the recorder: In members take input slots in
// declaration order, the Out member takes the output. A member that is
// neither is a mistake the compiler reports here, not a silent skip.
template <class A>
void bind(kir::KernelBody& k, A& args) {
  std::size_t index = 0;
  template for (constexpr std::meta::info m : detail::members<A>()) {
    using M = typename[:std::meta::type_of(m):];
    using Tr = detail::arg_traits<M>;
    static_assert(Tr::is_in || Tr::is_out,
                  "kernel args members must be env::In or env::Out");
    if constexpr (Tr::is_in) {
      // Names follow the emitter's binding convention directly; this is the
      // one place that convention is spelled for authored kernels.
      args.[:m:].b = kir::Buffer<typename Tr::elem>(
          &k, &k.types(), "in" + std::to_string(index++));
      args.[:m:].tt = &k.types();
    } else if constexpr (Tr::is_out) {
      args.[:m:].b = kir::Buffer<typename Tr::elem>(&k, &k.types(), "out");
      args.[:m:].tt = &k.types();
    }
  }
  // At most one: kernels that return their element value or store through
  // the emitter's hook declare no Out at all.
  static_assert(output_count<A>() <= 1,
                "a kernel stores through at most one output");
}

}  // namespace lse::graph::env
