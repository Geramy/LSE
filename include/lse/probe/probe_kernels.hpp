// The probe's device-side work, on the ordinary authoring surface.
//
// One body per measurement, written against `class E`, so the same source is
// what a GPU backend records into HIP and what a host backend executes
// directly. That is the point of the two envs, and it is why the probe needs no
// hand-written device text: a backend that can compile records these; a backend
// that cannot runs them under env::Cpu and reports what its own memory did.
//
// Matrix-core rates are not here. They are per-row, and a row's fragment widths
// and lane mapping are a property of one ISA generation, so that probe lives
// beside the tile that already knows how to read a row.
#pragma once

#include <cstdint>

#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_ir.hpp"
#include "lse/math.hpp"

namespace lse::probe {

namespace env = ::lse::graph::env;
namespace kir = ::lse::graph::kir;

template <class E>
struct StreamArgs {
  env::In<kir::f32, E> in;
  env::Out<kir::f32, E> out;
};

// A pure read stream.
//
// The sum is stored rather than dropped, because a load whose result nothing
// consumes is a load the compiler is entitled to delete — and this kernel
// exists only to make those loads happen. Consecutive threads take consecutive
// packs so the access is coalesced, which is the pattern a weight stream has;
// the number this yields is therefore the roofline a GEMV can reach, not a
// synthetic peak. Measured on gfx1151 it lands at 224-233 GB/s, against the
// 227 GB/s the engine's own lm_head GEMV reaches on the same part.
//
// One load per iteration is deliberate. Issuing four independent ones per
// thread was tried and is 5% SLOWER here (214 vs 229 GB/s): the extra
// registers cost occupancy, and occupancy is where this device's
// memory-level parallelism comes from.
//
// `elems` must be a whole number of `threads * width`, so every load lands
// inside the buffer and none of them needs a tail guard.
template <class E>
void stream_read(E& e, StreamArgs<E>& a, std::uint32_t elems,
                 std::uint32_t threads, std::uint32_t load_bytes) {
  const std::uint32_t width =
      kir::pack_n(load_bytes, kir::pack_elem_bytes<kir::f32>());
  const std::uint32_t span = threads * width;
  const auto tid = e.thread_id();
  auto acc = e.var(0.0f);
  for (auto k : e.range(tid * width, e.u32(elems), span)) {
    auto v = e.load(a.in, k, load_bytes);
    for (auto j : e.unroll(width)) {
      acc = lse::math::fma(v[j], e.f32(1.0f), acc);
    }
  }
  a.out[tid] = acc;
}

template <class E>
struct TouchArgs {
  env::Out<kir::f32, E> out;
};

// The smallest dispatch that still has to be dispatched. What is left when its
// time is divided by the launch count is submission cost and nothing else.
template <class E>
void touch_one(E& e, TouchArgs<E>& a) {
  const auto tid = e.thread_id();
  if (auto only = e.when(tid < 1u)) {
    a.out[0] = e.f32(1.0f);
  }
}

}  // namespace lse::probe
