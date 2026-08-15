#pragma once

#include "lse/backends/hrx/device_info.hpp"
#include "lse/graph/kernel_env.hpp"
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
inline void emit_dot_f32(
    graph::env::Emit& e,
    const graph::env::In<graph::kir::f32, graph::env::Emit>& x,
    const graph::env::In<graph::kir::f32, graph::env::Emit>& w,
    const graph::kir::Val<graph::kir::u32>& x0,
    const graph::kir::Val<graph::kir::u32>& w0,
    const graph::kir::LValue<graph::kir::f32>& acc, std::uint32_t kdim,
    std::uint32_t max_load_bytes) {
  namespace kir = graph::kir;
  namespace math = lse::math;
  const auto vn = kir::pack_n(max_load_bytes, 4);
  const auto aligned = (kdim / vn) * vn;
  for (auto t : e.range(0u, aligned, vn)) {
    const auto xv = e.load(x, x0 + t, max_load_bytes);
    const auto wv = e.load(w, w0 + t, max_load_bytes);
    for (auto lane : e.unroll(xv.width())) {
      acc = math::fma(xv[lane], wv[lane], acc.read());
    }
  }
  if (aligned < kdim) {
    for (auto t : e.range(aligned, kdim)) {
      acc = math::fma(x[x0 + t], w[w0 + t], acc.read());
    }
  }
}

// Recorder-signature shim for callers that still hold raw kir buffers.
inline void emit_dot_f32(graph::kir::KernelBody& k,
                         const graph::kir::Buffer<graph::kir::f32>& x,
                         const graph::kir::Buffer<graph::kir::f32>& w,
                         const graph::kir::Val<graph::kir::u32>& x0,
                         const graph::kir::Val<graph::kir::u32>& w0,
                         const graph::kir::LValue<graph::kir::f32>& acc,
                         std::uint32_t kdim, std::uint32_t max_load_bytes) {
  namespace env = graph::env;
  namespace kir = graph::kir;
  env::Emit e{&k};
  const env::In<kir::f32, env::Emit> xi{x, &k.types()};
  const env::In<kir::f32, env::Emit> wi{w, &k.types()};
  emit_dot_f32(e, xi, wi, x0, w0, acc, kdim, max_load_bytes);
}

}  // namespace lse::backend::hrx_kernels
