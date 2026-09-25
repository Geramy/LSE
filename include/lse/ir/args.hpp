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

// The floor is what this tree has actually been compiled against, not the
// newest P2996 revision: gcc trunk r16-8246 reports 202506 and builds every
// reflection site here unchanged, while Ubuntu 26.04 ships no g++-16 newer
// than that. Requiring 202603 turned a compiler that works into one that
// stops at the first header, which is a worse answer than trying.
#if !defined(__cpp_impl_reflection) || __cpp_impl_reflection < 202506L
#define LSE_PORTABLE_AGGREGATE_ARGS 1
#endif

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#if !defined(LSE_PORTABLE_AGGREGATE_ARGS)
#include <meta>
#else
#include <tuple>
#include <type_traits>
#include <utility>
#endif

#include "lse/core/dtype.hpp"
#include "lse/ir/env.hpp"
#include "lse/ir/recorder.hpp"

namespace lse::ir::env {

// The storage dtype a kir element type is a view of. A slot declared over the
// wrong one does not fail to compile — it reinterprets the bytes — so this is
// what `bind` checks the bound buffer against.
template <typename T>
struct elem_dtype;
template <>
struct elem_dtype<f32> {
  static constexpr DType value = DType::kF32;
};
template <>
struct elem_dtype<lse::f16> {
  static constexpr DType value = DType::kF16;
};
template <>
struct elem_dtype<lse::bf16> {
  static constexpr DType value = DType::kBF16;
};
// A byte slot: the wire side of a block codec, where a "buffer element" is a
// packed byte rather than a number.
template <>
struct elem_dtype<std::uint8_t> {
  static constexpr DType value = DType::kU8;
};
// A 32-bit lane. Either a plain integer or four packed int8 / eight int4 on
// their way to an integer matrix core, which reads them as i32 registers.
template <>
struct elem_dtype<std::int32_t> {
  static constexpr DType value = DType::kI32;
};
// The packed plane of a group-affine weight: several codes to a lane, at a bit
// width the tag does not name (quant/group_affine.hpp).
template <>
struct elem_dtype<std::uint32_t> {
  static constexpr DType value = DType::kU32;
};

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

#ifndef LSE_PORTABLE_AGGREGATE_ARGS
template <class A>
consteval auto members() {
  return std::define_static_array(std::meta::nonstatic_data_members_of(
      ^^A, std::meta::access_context::current()));
}

#else
// The portable path supports plain aggregate parameter bundles only. Member
// order comes from C++ structured binding, retaining the reflected ABI order.
struct AggregateElement { template<class T> constexpr operator T() const noexcept; };
template<class A, std::size_t... I>
consteval bool initializes(std::index_sequence<I...>) {
  return requires { A{(static_cast<void>(I), AggregateElement{})...}; };
}
template<class A, std::size_t N=0> consteval std::size_t member_count() {
  static_assert(std::is_aggregate_v<A>, "kernel parameter bundle must be an aggregate");
  if constexpr (initializes<A>(std::make_index_sequence<N+1>{})) {
    static_assert(N<16, "portable kernel parameter bundle exceeds 16 members");
    return member_count<A,N+1>();
  } else return N;
}
template<class A> auto member_refs(A& value) {
  constexpr auto count=member_count<A>();
  if constexpr (count==0) return std::tuple<>();
  else if constexpr (count==1) { auto& [m0]=value; return std::tie(m0); }
  else if constexpr (count==2) { auto& [m0, m1]=value; return std::tie(m0, m1); }
  else if constexpr (count==3) { auto& [m0, m1, m2]=value; return std::tie(m0, m1, m2); }
  else if constexpr (count==4) { auto& [m0, m1, m2, m3]=value; return std::tie(m0, m1, m2, m3); }
  else if constexpr (count==5) { auto& [m0, m1, m2, m3, m4]=value; return std::tie(m0, m1, m2, m3, m4); }
  else if constexpr (count==6) { auto& [m0, m1, m2, m3, m4, m5]=value; return std::tie(m0, m1, m2, m3, m4, m5); }
  else if constexpr (count==7) { auto& [m0, m1, m2, m3, m4, m5, m6]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6); }
  else if constexpr (count==8) { auto& [m0, m1, m2, m3, m4, m5, m6, m7]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7); }
  else if constexpr (count==9) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8); }
  else if constexpr (count==10) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9); }
  else if constexpr (count==11) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10); }
  else if constexpr (count==12) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11); }
  else if constexpr (count==13) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12); }
  else if constexpr (count==14) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13); }
  else if constexpr (count==15) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14); }
  else if constexpr (count==16) { auto& [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15]=value; return std::tie(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15); }
}
#endif

}  // namespace detail

#ifndef LSE_PORTABLE_AGGREGATE_ARGS
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

#else
template<class A, bool Inputs> consteval std::size_t count_arguments() {
  using Fields=decltype(detail::member_refs(std::declval<A&>()));
  return []<std::size_t... I>(std::index_sequence<I...>) {
    return (std::size_t{0} + ... + (Inputs ?
      detail::arg_traits<std::remove_cvref_t<std::tuple_element_t<I,Fields>>>::is_in :
      detail::arg_traits<std::remove_cvref_t<std::tuple_element_t<I,Fields>>>::is_out));
  }(std::make_index_sequence<std::tuple_size_v<Fields>>{});
}
template<class A> consteval std::size_t input_count() { return count_arguments<A,true>(); }
template<class A> consteval std::size_t output_count() { return count_arguments<A,false>(); }

#endif

// Attach every member to the recorder: In members take input slots in
// declaration order, the Out member takes the output. A member that is
// neither is a mistake the compiler reports here, not a silent skip.
//
// Returns false — the kernel must then decline by returning an empty body —
// when a slot's declared element type does not match the dtype of the buffer
// the emitter will bind to it. Declining is loud: the group falls back and
// `--stats` counts it. Reinterpreting the buffer would not be.
//
// `in_dtypes` empty means the caller could not describe the buffers, and
// nothing is checked; a slot past its end is likewise unchecked, since the
// emitter may append bindings the primitive did not name.
template <class A>
[[nodiscard]] bool bind(KernelBody& k, A& args,
                        std::span<const DType> in_dtypes,
                        DType out_dtype) {
  std::size_t index = 0;
  bool ok = true;
#ifndef LSE_PORTABLE_AGGREGATE_ARGS
  template for (constexpr std::meta::info m : detail::members<A>()) {
    using M = typename[:std::meta::type_of(m):];
    using Tr = detail::arg_traits<M>;
    static_assert(Tr::is_in || Tr::is_out,
                  "kernel args members must be env::In or env::Out");
    if constexpr (Tr::is_in) {
      if (!in_dtypes.empty() && index < in_dtypes.size() &&
          in_dtypes[index] != elem_dtype<typename Tr::elem>::value) {
        ok = false;
      }
      // Names follow the emitter's binding convention directly; this is the
      // one place that convention is spelled for authored kernels.
      args.[:m:].b =
          Buffer<typename Tr::elem>(&k, &k.types(), k.input_name(index++));
      args.[:m:].tt = &k.types();
    } else if constexpr (Tr::is_out) {
      if (out_dtype != elem_dtype<typename Tr::elem>::value) ok = false;
      args.[:m:].b =
          Buffer<typename Tr::elem>(&k, &k.types(), k.output_name());
      args.[:m:].tt = &k.types();
    }
  }
#else
  const auto attach = [&](auto& member) {
    using M = std::remove_cvref_t<decltype(member)>;
    using Tr = detail::arg_traits<M>;
    static_assert(Tr::is_in || Tr::is_out,
                  "kernel args members must be env::In or env::Out");
    if constexpr (Tr::is_in) {
      if (!in_dtypes.empty() && index < in_dtypes.size() &&
          in_dtypes[index] != elem_dtype<typename Tr::elem>::value) {
        ok = false;
      }
      // Names follow the emitter's binding convention directly; this is the
      // one place that convention is spelled for authored kernels.
      member.b =
          Buffer<typename Tr::elem>(&k, &k.types(), k.input_name(index++));
      member.tt = &k.types();
    } else if constexpr (Tr::is_out) {
      if (out_dtype != elem_dtype<typename Tr::elem>::value) ok = false;
      member.b =
          Buffer<typename Tr::elem>(&k, &k.types(), k.output_name());
      member.tt = &k.types();
    }
  };
  std::apply([&](auto&... member) { (attach(member), ...); }, detail::member_refs(args));
#endif
  // At most one: kernels that return their element value or store through
  // the emitter's hook declare no Out at all.
  static_assert(output_count<A>() <= 1,
                "a kernel stores through at most one output");
  return ok;
}

template <class A, class S>
  requires requires(const S& s) { s.input_dtypes; s.output_dtype; }
[[nodiscard]] bool bind(KernelBody& k, A& args, const S& shapes) {
  return bind(k, args, shapes.input_dtypes, shapes.output_dtype);
}

}  // namespace lse::ir::env
