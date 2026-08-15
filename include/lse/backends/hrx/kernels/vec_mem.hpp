#pragma once

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/math.hpp"

namespace lse::backend::hrx_kernels {

inline std::uint32_t device_load_bytes(const DeviceInfo* device) {
  return device != nullptr ? max_load_bytes(*device) : 4u;
}

inline std::uint32_t device_store_bytes(const DeviceInfo* device) {
  return device != nullptr ? max_store_bytes(*device) : 4u;
}

// acc += x[x0+t] * w[w0+t] for t in [0, kdim). Width comes from the live
// MAX_LOAD_BYTES property, not a hardcoded 4 or 2 at the call site.
inline void emit_dot_f32(graph::kir::KernelBody& k,
                         const graph::kir::Buffer<graph::kir::f32>& x,
                         const graph::kir::Buffer<graph::kir::f32>& w,
                         const graph::kir::Val<graph::kir::u32>& x0,
                         const graph::kir::Val<graph::kir::u32>& w0,
                         const graph::kir::LValue<graph::kir::f32>& acc,
                         std::uint32_t kdim, std::uint32_t max_load_bytes) {
  namespace kir = graph::kir;
  namespace math = lse::math;
  const auto vn = kir::pack_n(max_load_bytes, 4);
  const auto aligned = (kdim / vn) * vn;
  k.loop("t", k.constant<kir::u32>(0), k.constant<kir::u32>(aligned), vn,
         [&](kir::Val<kir::u32> t) {
           const auto xv = math::load(x, x0 + t, max_load_bytes);
           const auto wv = math::load(w, w0 + t, max_load_bytes);
           k.unroll("e", xv.width(), [&](kir::Val<kir::u32> e) {
             acc = math::fma(xv[e], wv[e], acc.read());
           });
         });
  if (aligned < kdim) {
    k.loop("t", k.constant<kir::u32>(aligned), k.constant<kir::u32>(kdim), 1,
           [&](kir::Val<kir::u32> t) {
             acc = math::fma(x[x0 + t].read(), w[w0 + t].read(), acc.read());
           });
  }
}

}  // namespace lse::backend::hrx_kernels
