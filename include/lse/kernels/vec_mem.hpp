#pragma once

#include "lse/backends/hrx/device_info.hpp"
#include "lse/core/dtype.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/math.hpp"

namespace lse::kernels {

// These name device facts, which the backend supplies.
using backend::DeviceInfo;
using backend::max_load_bytes;
using backend::max_store_bytes;

inline std::uint32_t device_load_bytes(const DeviceInfo* device) {
  return device != nullptr ? max_load_bytes(*device) : 4u;
}

inline std::uint32_t device_store_bytes(const DeviceInfo* device) {
  return device != nullptr ? max_store_bytes(*device) : 4u;
}

// Widest pack a row of `kdim` elements can be read in. The budget alone is not
// enough: consecutive rows start at multiples of kdim, so a pack wider than the
// largest power of two dividing kdim would put some row's load off its natural
// alignment. Same rule at every element width — it is bytes that must line up.
inline std::uint32_t row_pack(std::uint32_t kdim, std::uint32_t max_bytes,
                              std::uint32_t elem_bytes) noexcept {
  std::uint32_t n = graph::kir::pack_n(max_bytes, elem_bytes);
  while (n > 1 && (kdim % n) != 0) n >>= 1;
  return n;
}

// Runs the body once for the kir element type that `dt` names. Returning {}
// for anything else is how a kernel declines a storage format it has no body
// for; a quantized block type joins by adding an arm here and there.
template <class F>
[[nodiscard]] std::string with_elem(DType dt, F&& fn) {
  switch (dt) {
    case DType::kF32: return fn.template operator()<graph::kir::f32>();
    case DType::kBF16: return fn.template operator()<lse::bf16>();
    case DType::kF16: return fn.template operator()<lse::f16>();
    default: return {};
  }
}

// acc += x[x0+t] * w[w0+t] for t in [0, kdim). Width comes from the live
// MAX_LOAD_BYTES property, not a hardcoded 4 or 2 at the call site. The weight
// arrives in its stored format and widens in register.
template <class W>
inline void emit_dot(graph::env::Emit& e,
                     const graph::env::In<graph::kir::f32, graph::env::Emit>& x,
                     const graph::env::In<W, graph::env::Emit>& w,
                     const graph::kir::Val<graph::kir::u32>& x0,
                     const graph::kir::Val<graph::kir::u32>& w0,
                     const graph::kir::LValue<graph::kir::f32>& acc,
                     std::uint32_t kdim, std::uint32_t max_load_bytes) {
  namespace kir = graph::kir;
  namespace math = lse::math;
  constexpr std::uint32_t we = kir::pack_elem_bytes<W>();
  const auto xn = row_pack(kdim, max_load_bytes, 4);
  const auto wn = row_pack(kdim, max_load_bytes, we);
  const auto vn = xn < wn ? xn : wn;
  const auto aligned = (kdim / vn) * vn;
  for (auto t : e.range(0u, aligned, vn)) {
    // Byte budgets, not widths: the recorder snaps each to an ISA op, and
    // asking for vn elements is how both packs come out the same width.
    const auto xv = e.load(x, x0 + t, vn * 4u);
    const auto wv = e.load(w, w0 + t, vn * we);
    for (auto lane : e.unroll(vn)) {
      acc = math::fma(xv[lane], math::widen(wv[lane]), acc.read());
    }
  }
  if (aligned < kdim) {
    for (auto t : e.range(aligned, kdim)) {
      acc = math::fma(x[x0 + t], math::widen(w[w0 + t]), acc.read());
    }
  }
}

}  // namespace lse::kernels
