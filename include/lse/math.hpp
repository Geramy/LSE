// Fake device math library. Calling these records into the kir body that is
// currently being emitted; the HRX (or other) table supplies the spelling.
//
//   acc = lse::math::fma(x, y, acc);
//   acc = lse::math::wmma_f32_16x16x16(a, b, acc);
//
// The C++ function is the symbol. When the dialect key is that name, __func__
// is the key — no second string to drift. Dotted ISA names are not identifiers
// so those stay written down.
#pragma once

#include <cmath>

#include "lse/core/elem.hpp"
#include "lse/graph/kernel_ir.hpp"

namespace lse::math {

// Host overloads of the same spellings: a kernel body instantiated with
// env::Cpu calls these and executes directly instead of recording.
inline float exp(float x) { return std::exp(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float fma(float a, float b, float c) { return std::fma(a, b, c); }
inline float max(float a, float b) { return a > b ? a : b; }
inline float min(float a, float b) { return a < b ? a : b; }

// Expression type of this library. The recorder lives in graph::kir; authors
// include this header and never name that namespace.
template <typename T>
using Val = graph::kir::Val<T>;

namespace detail {

template <typename R, typename... A>
Val<R> invoke(std::string_view id, const A&... args) {
  graph::kir::KernelBody* body = graph::kir::KernelBody::try_current();
  if (body == nullptr) return {};
  return body->call<R>(id, args...);
}

}  // namespace detail

inline Val<lse::f32> exp(const Val<lse::f32>& x) {
  return detail::invoke<lse::f32>(__func__, x);
}
inline Val<lse::f32> sqrt(const Val<lse::f32>& x) {
  return detail::invoke<lse::f32>(__func__, x);
}
inline Val<lse::f32> rsqrt(const Val<lse::f32>& x) {
  return detail::invoke<lse::f32>(__func__, x);
}
inline Val<lse::f32> fma(const Val<lse::f32>& a, const Val<lse::f32>& b,
                         const Val<lse::f32>& c) {
  return detail::invoke<lse::f32>(__func__, a, b, c);
}
inline Val<lse::f32> max(const Val<lse::f32>& a, const Val<lse::f32>& b) {
  return detail::invoke<lse::f32>(__func__, a, b);
}
inline Val<lse::f32> min(const Val<lse::f32>& a, const Val<lse::f32>& b) {
  return detail::invoke<lse::f32>(__func__, a, b);
}
inline Val<lse::f32> neg_inf() {
  return detail::invoke<lse::f32>(__func__);
}

inline Val<graph::kir::u32> local_id() {
  return detail::invoke<graph::kir::u32>("thread.local_id");
}
inline Val<graph::kir::u32> workgroup_id_x() {
  return detail::invoke<graph::kir::u32>("thread.workgroup_id.x");
}
inline Val<graph::kir::u32> workgroup_id_y() {
  return detail::invoke<graph::kir::u32>("thread.workgroup_id.y");
}
inline Val<graph::kir::u32> workgroup_size() {
  return detail::invoke<graph::kir::u32>("thread.workgroup_size");
}
inline Val<graph::kir::u32> grid_dim_x() {
  return detail::invoke<graph::kir::u32>("thread.grid_dim.x");
}
inline void barrier() {
  graph::kir::KernelBody* body = graph::kir::KernelBody::try_current();
  if (body != nullptr) body->barrier();
}

inline Val<lse::f32> shfl_xor(const Val<lse::f32>& v,
                              const Val<graph::kir::u32>& mask) {
  return detail::invoke<lse::f32>("wave.shfl_xor", v, mask);
}

// The workgroup scratch pad for the kernel being emitted. Allocations that
// would exceed the device's LDS budget fail instead of producing a bad launch.
inline graph::kir::Lds& lds() {
  return graph::kir::KernelBody::try_current()->lds();
}

// `max_bytes` is the live device property (HRX MAX_LOAD/STORE_BYTES). The
// call does not name 4 or 2; the recorder snaps that budget to an ISA op.
template <typename T>
inline graph::kir::Pack<T> load(const graph::kir::Buffer<T>& buf,
                                const graph::kir::Val<graph::kir::u32>& index,
                                std::uint32_t max_bytes) {
  return buf.load(index, max_bytes);
}

template <typename T>
inline void store(const graph::kir::Buffer<T>& buf,
                  const graph::kir::Val<graph::kir::u32>& index,
                  const graph::kir::Pack<T>& value, std::uint32_t max_bytes) {
  buf.store(index, value, max_bytes);
}

inline Val<lse::vec<lse::f32, 8>> wmma_f32_16x16x16(
    const Val<lse::vec<lse::f16, 16>>& a,
    const Val<lse::vec<lse::f16, 16>>& b,
    const Val<lse::vec<lse::f32, 8>>& c) {
  return detail::invoke<lse::vec<lse::f32, 8>>("wmma.f32.16x16x16.f16", a, b,
                                               c);
}

}  // namespace lse::math

namespace lseMath = lse::math;
